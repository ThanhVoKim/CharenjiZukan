#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to sys.path so we can import modules properly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import tempfile
import shutil
import time
from typing import Optional

from utils.logger import get_logger, setup_logging
from utils.srt_parser import parse_srt_file

from sync_engine.analyzer import classify_and_compute_slots, compute_speeds, build_timeline_map
from sync_engine.video_processor import query_keyframes, process_video_chunks_parallel
from sync_engine.audio_assembler import assemble_audio_track
from sync_engine.timestamp_remapper import recalculate_srt, recalculate_ass
from sync_engine.renderer import render_final_video, DEFAULT_SUBTITLE_STYLE

logger = get_logger("sync_video")

def get_audio_start_time(video_path: str) -> float:
    """
    Trả về PTS (giây) của audio packet đầu tiên trong video.
    Dùng để phát hiện encoder delay / priming samples.
    Nếu PTS > 0 → audio bị trễ so với video → cần aresample=async=1:first_pts=0.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "packet=pts_time",
        "-of", "default=noprint_wrappers=1:nokey=1",
        "-read_intervals", "%+#1",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        pts = float(result.stdout.strip())
        return pts
    except (ValueError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return 0.0

def run_sync_pipeline(args):
    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create local tmp directory inside the project instead of system /tmp
    local_tmp_base = PROJECT_ROOT / "tmp"
    local_tmp_base.mkdir(parents=True, exist_ok=True)
    
    tmp_dir = tempfile.mkdtemp(prefix="sync_video_tmp_", dir=str(local_tmp_base))
    
    try:
        start_time = time.time()
        logger.info("=== BẮT ĐẦU TTS-VIDEO SYNC ===")
        logger.info(f"Video: {args.video}")
        logger.info(f"Subtitle: {args.subtitle}")
        if args.tts_provider == "edge":
            logger.info(f"TTS Voice: {args.tts_voice}")
        else:
            logger.info(f"Voice ID: {args.tts_voice}")
        
        # Parse inputs
        if not Path(args.subtitle).exists():
            raise FileNotFoundError(f"Không tìm thấy file subtitle: {args.subtitle}")

        subtitle_segments = parse_srt_file(args.subtitle)
        mute_segments = parse_srt_file(args.mute) if args.mute and Path(args.mute).exists() else []
        
        # Get video duration
        import wave
        
        # Lấy duration video từ ffprobe thay vì ffmpeg để nhanh hơn
        import subprocess
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", args.video
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        video_duration_ms = float(result.stdout.strip()) * 1000.0
        
        cmd_fps = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of",
            "default=noprint_wrappers=1:nokey=1", args.video
        ]
        result_fps = subprocess.run(cmd_fps, capture_output=True, text=True, check=True)
        fps_str = result_fps.stdout.strip()
        
        # Parse fps to float
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps_float = float(num) / float(den)
        else:
            fps_float = float(fps_str)
            
        logger.info(f"Video FPS detected: {fps_str} ({fps_float:.2f})")
        
        # PHASE 0: AUTO GENERATE TTS
        logger.info("\n--- PHASE 0: AUTO GENERATE TTS ---")
        tts_dir = str(Path(tmp_dir) / "tts_clips")
        Path(tts_dir).mkdir(parents=True, exist_ok=True)
        
        from sync_engine.analyzer import filter_tts_subtitles
        from tts.edgetts import EdgeTTSEngine
        from tts.voicevox import VoicevoxTTSEngine
        
        tts_only = filter_tts_subtitles(subtitle_segments, mute_segments)
        queue_tts = []
        for i, it in enumerate(tts_only):
            queue_tts.append({
                "text":       it["text"],
                "line":       it["line"],
                "start_time": it["start_time"],
                "end_time":   it["end_time"],
                "role":       args.tts_voice,
                "filename":   str(Path(tts_dir) / f"dubb-{i}.wav"),
            })
            
        logger.info(f"Đang sinh {len(queue_tts)} audio clips bằng {args.tts_provider.upper()}...")
        if args.tts_provider == "edge":
            engine = EdgeTTSEngine(
                queue_tts=queue_tts,
                voice=args.tts_voice,
                rate=args.tts_rate,
                volume=args.tts_volume,
                pitch=args.tts_pitch,
                strip_silence=True,
                max_concurrent=10,
                min_silence_len_ms=300
            )
        elif args.tts_provider == "voicevox":
            try:
                voice_id = int(args.tts_voice)
            except ValueError:
                raise ValueError(f"Với Voicevox, tham số --tts-voice phải là ID dạng số nguyên (ví dụ: 10008). Giá trị hiện tại: {args.tts_voice}")
            
            engine = VoicevoxTTSEngine(
                queue_tts=queue_tts,
                voice_id=voice_id,
                concurrent_requests=100,
                speed_scale=1.12,
                pitch_scale=-0.05,
            )
        else:
            raise ValueError(f"Provider không hợp lệ: {args.tts_provider}")

        tts_stats = engine.run()
        if tts_stats["ok"] == 0 and len(queue_tts) > 0:
            raise RuntimeError(f"{args.tts_provider.upper()} thất bại hoàn toàn — không có audio nào được tạo")
        logger.info(f"Tạo TTS hoàn tất: {tts_stats['ok']} thành công, {tts_stats['err']} lỗi")
        
        # PHASE 1: ANALYSIS
        logger.info("\n--- PHASE 1: ANALYSIS ---")
        blocks = classify_and_compute_slots(
            subtitle_segments, mute_segments, video_duration_ms, tts_dir=tts_dir
        )
        logger.info(f"Tìm thấy {len(blocks)} blocks (bao gồm tts, mute, gap).")
        
        is_voicevox = args.tts_provider == "voicevox"
        if is_voicevox:
            logger.info("Voicevox mode: no_cap=True, video có thể slow xuống dưới %.1fx", args.slow_cap)

        speeds = []
        for b in blocks:
            vs, as_, new_dur = compute_speeds(
                tts_ms        = b.tts_duration,
                slot_ms       = b.slot_duration,
                cap           = args.slow_cap,
                hard_limit_ms = b.hard_limit_ms,
                no_cap        = is_voicevox,
            )
            speeds.append((vs, as_, new_dur))
            
        timeline = build_timeline_map(blocks, speeds, video_duration_ms, fps_float=fps_float)
        logger.info(f"Timeline map đã tạo với {len(timeline)} segments (đã được Snap to Frames chuẩn xác).")
        
        # PHASE 2: VIDEO PROCESSING
        logger.info("\n--- PHASE 2: VIDEO PROCESSING ---")
        stretched_video = str(Path(tmp_dir) / "video_stretched.mp4")
        
        process_video_chunks_parallel(
            video_path=args.video,
            timeline=timeline,
            output_dir=tmp_dir, # put chunks in tmp
            max_workers=args.workers,
            use_gpu=not args.no_gpu,
            fps_str=fps_str,
            fps_float=fps_float
        )
        
        stretched_video_chunked = str(Path(tmp_dir) / "video_stretched.mp4")
        if not Path(stretched_video_chunked).exists():
            raise RuntimeError("Lỗi khi xử lý video chunks.")
            
        # PHASE 3: AUDIO ASSEMBLY
        logger.info("\n--- PHASE 3: AUDIO ASSEMBLY ---")

        source_audio_for_quotes = args.video

        if getattr(args, 'use_demucs', False):
            logger.info("Đang chạy Demucs để tách lời (vocals) từ video gốc...")
            from cli.demucs_audio import separate_audio

            # ── BƯỚC TIÊN QUYẾT: Pre-extract audio ra WAV bằng FFmpeg ──────────
            # Mục đích: Đảm bảo time reference của Demucs input ĐỒNG NHẤT với
            # FFmpeg. Khi torchaudio.load() đọc trực tiếp từ video (AAC/MP3),
            # nó có thể bao gồm priming samples (~1024 samples ≈ 23ms ở 44100Hz)
            # mà FFmpeg tự động bỏ qua qua edit list. Điều này gây lệch timestamp
            # ở ranh giới segment và dẫn đến "audio leak".
            # Pre-extract bằng FFmpeg giải quyết triệt để vấn đề này.
            raw_audio_for_demucs = str(Path(tmp_dir) / "raw_audio_for_demucs.wav")

            # Kiểm tra động: chỉ dùng aresample nếu audio packet đầu tiên có PTS > 0
            audio_start = get_audio_start_time(args.video)
            need_aresample = audio_start > 0.001  # Trễ hơn 1ms → cần bù trừ
            if need_aresample:
                logger.info(
                    "Phát hiện audio start time = %.6fs (> 0) → bật aresample=async=1:first_pts=0 để bù trừ.",
                    audio_start,
                )
            else:
                logger.info("Audio start time = %.6fs (~0) → không cần aresample.", audio_start)

            logger.info("Pre-extracting audio từ video bằng FFmpeg (đảm bảo time reference chuẩn)...")
            try:
                extract_cmd = [
                    "ffmpeg", "-y",
                    "-i", args.video,
                    "-vn",
                ]
                if need_aresample:
                    extract_cmd.extend(["-af", "aresample=async=1:first_pts=0"])
                extract_cmd.extend([
                    "-ar", "44100",   # giữ nguyên SR phổ biến cho Demucs
                    "-ac", "2",
                    "-c:a", "pcm_s16le",
                    raw_audio_for_demucs,
                ])
                subprocess.run(extract_cmd, check=True, capture_output=True)
                logger.info("Pre-extract hoàn tất: %s", raw_audio_for_demucs)
            except subprocess.CalledProcessError as e:
                stderr_msg = e.stderr.decode(errors="ignore")[-500:]
                logger.error("Lỗi pre-extract audio: %s", stderr_msg)
                logger.warning("Fallback: dùng video gốc làm source cho Demucs (có thể bị lệch timestamp).")
                raw_audio_for_demucs = args.video

            # ── Chạy Demucs trên WAV đã pre-extract ──────────────────────────
            vocals_path = str(Path(tmp_dir) / "vocals_only.wav")
            try:
                separate_audio(
                    input_path=raw_audio_for_demucs,   # WAV, không phải video
                    output_path=vocals_path,
                    model="htdemucs",
                    keep="vocals",
                    bitrate="192k",
                    device="cuda",
                    segment=7,
                )
                source_audio_for_quotes = vocals_path
                logger.info("Hoàn tất tách lời bằng Demucs: %s", vocals_path)
            except Exception as e:
                logger.error("Lỗi khi chạy Demucs, fallback về audio gốc: %s", e)
        
        mixed_audio = str(Path(tmp_dir) / "mixed_audio.wav")
        assemble_audio_track(
            timeline=timeline,
            video_path=source_audio_for_quotes,
            ambient_path=args.ambient,
            output_path=mixed_audio,
            tmp_dir=tmp_dir,
            use_demucs=args.use_demucs,
            tts_provider=args.tts_provider
        )
        
        # PHASE 4: RECALCULATE TIMESTAMPS
        logger.info("\n--- PHASE 4: RECALCULATE TIMESTAMPS ---")
        
        # 1. subtitle_tts_synced.srt
        subtitle_tts_synced = str(output_dir / f"{args.output_name}_tts_synced.srt")
        # filter mute blocks (tạo subtitle_tts)
        from sync_engine.analyzer import filter_tts_subtitles
        tts_only = filter_tts_subtitles(subtitle_segments, mute_segments)
        
        recalculate_srt(tts_only, timeline, subtitle_tts_synced, is_tts_track=True)
        logger.info(f"Đã tạo {subtitle_tts_synced}")
        
        # 2. subtitle_synced.srt (đầy đủ)
        subtitle_synced = str(output_dir / f"{args.output_name}_synced.srt")
        recalculate_srt(subtitle_segments, timeline, subtitle_synced, is_tts_track=False, max_chars=args.subtitle_max_chars)
        logger.info(f"Đã tạo {subtitle_synced}")
        
        # 3. mute_synced.srt (nếu có)
        if mute_segments:
            mute_synced = str(output_dir / f"{args.output_name}_mute_synced.srt")
            recalculate_srt(mute_segments, timeline, mute_synced, is_tts_track=False, max_chars=args.subtitle_max_chars)
            logger.info(f"Đã tạo {mute_synced}")
            
        # 4. note_overlay_synced.ass (nếu có)
        note_ass_synced = None
        if args.note_overlay_ass and Path(args.note_overlay_ass).exists():
            note_ass_synced = str(output_dir / f"{args.output_name}_note_synced.ass")
            recalculate_ass(args.note_overlay_ass, timeline, note_ass_synced, max_chars_per_line=args.note_max_chars)
            logger.info(f"Đã tạo {note_ass_synced}")
            
        # PHASE 5: FINAL RENDER
        logger.info("\n--- PHASE 5: FINAL RENDER ---")
        if not args.no_hardsub:
            final_video = str(output_dir / f"{args.output_name}.mp4")
            
            # Xây dựng subtitle style
            style_parts = [
                f"Fontname={args.subtitle_fontname}",
                r"\,Bold=1",
                f"\\,FontSize={args.subtitle_fontsize}",
                f"\\,PrimaryColour={args.subtitle_color}",
                r"\,OutlineColour=&H00000000",
                r"\,Outline=1",
                r"\,Shadow=1.5",
                r"\,BackColour=0xE6000000",
                r"\,Alignment=2",
                f"\\,MarginV={args.subtitle_margin_v}"
            ]
            custom_style = "".join(style_parts)
            
            render_final_video(
                stretched_video=stretched_video_chunked,
                mixed_audio=mixed_audio,
                subtitle_synced_srt=subtitle_synced,
                output_path=final_video,
                note_overlay_png=args.note_overlay_png,
                note_overlay_synced_ass=note_ass_synced,
                black_bg_path=args.black_bg,
                subtitle_style=custom_style,
                use_gpu=not args.no_gpu
            )
            logger.info(f"Render hoàn tất: {final_video}")
        else:
            logger.info("Bỏ qua bước Render do cờ --no-hardsub.")
            
        elapsed = time.time() - start_time
        logger.info(f"\n=== HOÀN TẤT SAU {elapsed:.1f} GIÂY ===")

    finally:
        # Cleanup
        if args.keep_tmp:
            logger.info("\n--- CLEANUP KHÔNG THỰC HIỆN ---")
            logger.info(f"Đã giữ lại thư mục tạm chứa video chunks và audio mix: {tmp_dir}")
        else:
            logger.info("\n--- CLEANUP ---")
            logger.info(f"Đang xóa thư mục tạm: {tmp_dir}...")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info("Hoàn tất dọn dẹp thư mục tạm.")

def main():
    parser = argparse.ArgumentParser(description="TTS-Video Sync - Chunk-Based Stretch")
    
    # Bắt buộc
    parser.add_argument("--video", required=True, help="File video gốc (.mp4, .mkv)")
    parser.add_argument("--subtitle", required=True, help="File subtitle.srt đầy đủ (kể cả vùng mute)")
    
    # TTS Settings
    parser.add_argument("--tts-provider", choices=["edge", "voicevox"], default="edge", help="Chọn TTS engine (mặc định: edge)")
    parser.add_argument("--tts-voice", default="vi-VN-HoaiMyNeural", help="Tên giọng EdgeTTS hoặc ID nhân vật Voicevox")
    parser.add_argument("--tts-rate", default="+0%", help="Tốc độ giọng đọc EdgeTTS")
    parser.add_argument("--tts-volume", default="+0%", help="Âm lượng EdgeTTS")
    parser.add_argument("--tts-pitch", default="+0Hz", help="Pitch EdgeTTS")
    
    # Tùy chọn - Input
    parser.add_argument("--mute", help="File mute.srt")
    parser.add_argument("--note-overlay-png", help="PNG tĩnh nền note")
    parser.add_argument("--note-overlay-ass", help="ASS text cho note")
    parser.add_argument("--black-bg", help="Dải đen 1920x80 (tự tạo nếu không có)")
    parser.add_argument("--ambient", default=str(PROJECT_ROOT / "assets" / "ambient.mp3"), help="Nhạc nền")
    
    # Algorithm
    parser.add_argument("--slow-cap", type=float, default=0.5, help="Video speed tối thiểu (mặc định: 0.5)")
    parser.add_argument("--use-demucs", action="store_true", help="Sử dụng Demucs để loại bỏ nhạc nền, chỉ giữ lại giọng nói cho các đoạn quoted audio")
    
    # Output
    parser.add_argument("--output-dir", default="./sync_output/", help="Thư mục output")
    parser.add_argument("--output-name", default="video_synced", help="Base name output")
    parser.add_argument("--no-hardsub", action="store_true", help="Chỉ output recalculated files")
    parser.add_argument("--keep-tmp", action="store_true", help="Giữ lại thư mục tạm chứa các chunks")
    
    # Performance
    parser.add_argument("--workers", type=int, default=4, help="FFmpeg workers song song")
    parser.add_argument("--no-gpu", action="store_true", help="Dùng libx264 thay vì h264_nvenc")
    
    # Subtitle Style
    parser.add_argument("--subtitle-fontname", default="Noto Sans CJK JP")
    parser.add_argument("--subtitle-fontsize", type=int, default=24)
    parser.add_argument("--subtitle-color", default="&H00EEF5FF")
    parser.add_argument("--subtitle-margin-v", type=int, default=7)
    parser.add_argument("--subtitle-max-chars", type=int, default=0, help="Ngắt dòng subtitle nếu dài hơn số ký tự này (0 = không ngắt)")
    parser.add_argument("--note-max-chars", type=int, default=16)
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        run_sync_pipeline(args)
    except Exception as e:
        logger.exception("Đã xảy ra lỗi:")
        sys.exit(1)

if __name__ == "__main__":
    main()
