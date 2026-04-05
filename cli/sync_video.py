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
        logger.info(f"TTS Voice: {args.tts_voice}")
        
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
        from tts_edgetts import EdgeTTSEngine
        
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
            
        logger.info(f"Đang sinh {len(queue_tts)} audio clips bằng EdgeTTS (voice={args.tts_voice})...")
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
        tts_stats = engine.run()
        if tts_stats["ok"] == 0 and len(queue_tts) > 0:
            raise RuntimeError("EdgeTTS thất bại hoàn toàn — không có audio nào được tạo")
        logger.info(f"Tạo TTS hoàn tất: {tts_stats['ok']} thành công, {tts_stats['err']} lỗi")
        
        # PHASE 1: ANALYSIS
        logger.info("\n--- PHASE 1: ANALYSIS ---")
        blocks = classify_and_compute_slots(
            subtitle_segments, mute_segments, video_duration_ms, tts_dir=tts_dir
        )
        logger.info(f"Tìm thấy {len(blocks)} blocks (bao gồm tts, mute, gap).")
        
        speeds = []
        for b in blocks:
            vs, as_, new_dur = compute_speeds(
                tts_ms=b.tts_duration,
                slot_ms=b.slot_duration,
                hard_limit_ms=b.hard_limit_ms,
                cap=args.slow_cap
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
        mixed_audio = str(Path(tmp_dir) / "mixed_audio.wav")
        assemble_audio_track(
            timeline=timeline,
            video_path=args.video,
            ambient_path=args.ambient,
            output_path=mixed_audio,
            tmp_dir=tmp_dir
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
    parser.add_argument("--tts-voice", default="vi-VN-HoaiMyNeural", help="Giọng đọc EdgeTTS (ví dụ: vi-VN-HoaiMyNeural)")
    parser.add_argument("--tts-rate", default="+0%", help="Tốc độ giọng đọc EdgeTTS")
    parser.add_argument("--tts-volume", default="+0%", help="Âm lượng EdgeTTS")
    parser.add_argument("--tts-pitch", default="+0Hz", help="Pitch EdgeTTS")
    
    # Tùy chọn - Input
    parser.add_argument("--mute", help="File mute.srt")
    parser.add_argument("--note-overlay-png", help="PNG tĩnh nền note")
    parser.add_argument("--note-overlay-ass", help="ASS text cho note")
    parser.add_argument("--black-bg", help="Dải đen 1920x80 (tự tạo nếu không có)")
    parser.add_argument("--ambient", default="assets/ambient.mp3", help="Nhạc nền")
    
    # Algorithm
    parser.add_argument("--slow-cap", type=float, default=0.5, help="Video speed tối thiểu (mặc định: 0.5)")
    
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
