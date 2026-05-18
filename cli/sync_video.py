#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to sys.path so we can import modules properly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import subprocess
import tempfile
import shutil
import time
import json

from utils.logger import get_logger, setup_logging
from utils.srt_parser import parse_srt_file

from sync_engine.analyzer import classify_and_compute_slots, compute_speeds, build_timeline_map
from sync_engine.video_processor import query_keyframes, process_video_chunks_parallel
from sync_engine.audio_assembler import assemble_audio_track
from sync_engine.timestamp_remapper import recalculate_srt, recalculate_ass
from sync_engine.renderer import render_final_video
from sync_engine.llm_metadata import apply_llm_metadata_override, run_llm_metadata_task
from sync_engine.forced_alignment_subtitle import run_forced_alignment_subtitle

logger = get_logger("sync_video")

def _load_render_config(config_path: str) -> dict:
    """Đọc file JSON cấu hình render.

    Hỗ trợ đường dẫn tương đối (từ PROJECT_ROOT) và tuyệt đối.
    Trả về dict rỗng nếu file không tồn tại.
    """
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = PROJECT_ROOT / config_file
    if not config_file.exists():
        logger.warning(f"Không tìm thấy file cấu hình render: {config_file}. Sử dụng cấu hình mặc định.")
        return {}
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info(f"Đã nạp cấu hình render từ: {config_file}")
    return config


def _load_tts_config(config_path: str) -> dict:
    """Đọc file YAML cấu hình TTS.

    Hỗ trợ đường dẫn tương đối (từ PROJECT_ROOT) và tuyệt đối.
    Trả về dict rỗng nếu file không tồn tại.
    """
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = PROJECT_ROOT / config_file
    if not config_file.exists():
        logger.warning(f"Không tìm thấy file cấu hình TTS: {config_file}. Sử dụng cấu hình mặc định.")
        return {}
    try:
        import yaml as yaml_lib
    except ImportError:
        logger.warning("PyYAML chưa cài nên không thể nạp TTS config. Sử dụng cấu hình mặc định.")
        return {}

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml_lib.safe_load(f) or {}
    logger.info(f"Đã nạp cấu hình TTS từ: {config_file}")
    return config


def _probe_video_duration(video_path: str) -> float:
    """Đo duration video bằng ffprobe, trả về ms."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip()) * 1000.0

def run_sync_pipeline(args):
    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create local tmp directory inside the project instead of system /tmp
    local_tmp_base = PROJECT_ROOT / "tmp"
    local_tmp_base.mkdir(parents=True, exist_ok=True)
    
    tmp_dir = tempfile.mkdtemp(prefix="sync_video_tmp_", dir=str(local_tmp_base))
    
    try:
        # Load render config từ JSON
        render_config = _load_render_config(args.render_config)
        render_config = apply_llm_metadata_override(render_config, getattr(args, "llm_metadata_override", None))

        start_time = time.time()
        logger.info("=== BẮT ĐẦU TTS-VIDEO SYNC ===")
        logger.info(f"Video: {args.video}")
        logger.info(f"Subtitle: {args.subtitle}")
        if args.tts_provider == "edge":
            logger.info(f"TTS Voice: {args.tts_voice}")
        elif args.tts_provider in ("voicevox", "voicevox_nemo"):
            logger.info(f"Voice ID: {args.tts_voice} (provider={args.tts_provider})")
        else:
            logger.info(f"TTS Provider: {args.tts_provider}")
        
        # Parse inputs
        if not Path(args.subtitle).exists():
            raise FileNotFoundError(f"Không tìm thấy file subtitle: {args.subtitle}")

        subtitle_segments = parse_srt_file(args.subtitle)
        mute_segments = parse_srt_file(args.mute) if args.mute and Path(args.mute).exists() else []

        # TẠO GENERIC TRANSCRIPT CHO PIPELINE
        from utils.srt_parser import write_segments_to_flat_text
        transcript_input_path = Path(tmp_dir) / "flat_transcript.txt"
        write_segments_to_flat_text(subtitle_segments, str(transcript_input_path))
        logger.info(f"Đã tạo flat transcript dùng chung: {transcript_input_path}")
        
        # Get video duration
        import wave
        
        # Lấy duration video từ ffprobe thay vì ffmpeg để nhanh hơn
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
        from tts.voicevox_nemo import VoicevoxNemoTTSEngine
        from tts.qwen import QwenTTSEngine
        
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
        
        # Load TTS config một lần duy nhất (dùng cho tất cả provider)
        tts_cfg_full = _load_tts_config(args.tts_config)
        
        if args.tts_provider == "edge":
            edge_cfg = tts_cfg_full.get("edge", {})
            engine = EdgeTTSEngine(
                queue_tts=queue_tts,
                voice=args.tts_voice or edge_cfg.get("voice", "vi-VN-HoaiMyNeural"),
                rate=edge_cfg.get("rate", "+0%"),
                volume=edge_cfg.get("volume", "+0%"),
                pitch=edge_cfg.get("pitch", "+0Hz"),
                strip_silence=edge_cfg.get("strip_silence", True),
                max_concurrent=edge_cfg.get("concurrent", 10),
                min_silence_len_ms=edge_cfg.get("min_silence_len_ms", 300),
            )
        elif args.tts_provider == "voicevox_nemo":
            vv_cfg = tts_cfg_full.get("voicevox_nemo", {})
            raw_voice = args.tts_voice or vv_cfg.get("voice_id", 10008)
            try:
                voice_id = int(raw_voice)
            except ValueError:
                raise ValueError(f"Với Voicevox Nemo, tham số --tts-voice phải là ID dạng số nguyên (ví dụ: 10008). Giá trị hiện tại: {raw_voice}")
            
            engine = VoicevoxNemoTTSEngine(
                queue_tts=queue_tts,
                voice_id=voice_id,
                concurrent_requests=vv_cfg.get("concurrent_requests", 100),
                speed_scale=vv_cfg.get("speed_scale", 1.12),
                pitch_scale=vv_cfg.get("pitch_scale", -0.05),
                intonation_scale=vv_cfg.get("intonation_scale", 1.0),
                volume_scale=vv_cfg.get("volume_scale", 2.0),
            )
        elif args.tts_provider == "voicevox":
            vv_cfg = tts_cfg_full.get("voicevox", {})
            raw_voice = args.tts_voice or vv_cfg.get("voice_id", 100)
            try:
                voice_id = int(raw_voice)
            except ValueError:
                raise ValueError(f"Với Voicevox, tham số --tts-voice phải là ID dạng số nguyên (ví dụ: 100). Giá trị hiện tại: {raw_voice}")
            
            engine = VoicevoxTTSEngine(
                queue_tts=queue_tts,
                voice_id=voice_id,
                concurrent_requests=vv_cfg.get("concurrent_requests", 100),
                speed_scale=vv_cfg.get("speed_scale", 1.12),
                pitch_scale=vv_cfg.get("pitch_scale", -0.05),
                intonation_scale=vv_cfg.get("intonation_scale", 1.0),
                volume_scale=vv_cfg.get("volume_scale", 2.0),
            )
        elif args.tts_provider == "qwen":
            qwen_cfg = tts_cfg_full.get("qwen", {})
            engine = QwenTTSEngine(
                queue_tts=queue_tts,
                **qwen_cfg
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
        
        is_voicevox_family = args.tts_provider in ("voicevox", "voicevox_nemo")
        if is_voicevox_family:
            logger.info("Voicevox family mode: no_cap=True, video có thể slow xuống dưới %.1fx", args.slow_cap)

        speeds = []
        for b in blocks:
            vs, as_, new_dur = compute_speeds(
                tts_ms        = b.tts_duration,
                slot_ms       = b.slot_duration,
                cap           = args.slow_cap,
                hard_limit_ms = b.hard_limit_ms,
                no_cap        = is_voicevox_family,
            )
            speeds.append((vs, as_, new_dur))
            
        timeline = build_timeline_map(blocks, speeds, video_duration_ms, fps_float=fps_float)
        logger.info(f"Timeline map đã tạo với {len(timeline)} segments (đã được Snap to Frames chuẩn xác).")
        
        # PHASE 2: VIDEO PROCESSING
        logger.info("\n--- PHASE 2: VIDEO PROCESSING ---")
        stretched_video = str(Path(tmp_dir) / "video_stretched.mp4")
        
        stretched_video_chunked, actual_durations = process_video_chunks_parallel(
            video_path=args.video,
            timeline=timeline,
            output_dir=tmp_dir, # put chunks in tmp
            max_workers=args.workers,
            use_gpu=not args.no_gpu,
            fps_str=fps_str,
            fps_float=fps_float,
            batch_size=args.batch_size,
        )
        
        if not Path(stretched_video_chunked).exists():
            raise RuntimeError("Lỗi khi xử lý video chunks.")
            
        # Đo duration thực tế của video đã concat
        video_actual_duration_ms = _probe_video_duration(stretched_video_chunked)
            
        # Lấy cấu hình tách audio
        audio_separator_cfg = render_config.get("audio_separator", {})
        extract_bgm = audio_separator_cfg.get("extract_bgm", False)
        extract_vocals = audio_separator_cfg.get("extract_vocals", False)

        # Cập nhật timeline dựa trên duration thực tế
        from sync_engine.analyzer import recalculate_timeline_from_actual_durations
        timeline = recalculate_timeline_from_actual_durations(timeline, actual_durations, fps_float)
        logger.info(f"Timeline đã cập nhật với {len(timeline)} segments (dựa trên duration thực tế).")
            
        # PHASE 2.5: BGM EXTRACTION (nếu được bật)
        bgm_path = None
        if extract_bgm:
            logger.info("\n--- PHASE 2.5: BGM EXTRACTION ---")
            from cli.audio_separator import separate_audio
            try:
                bgm_path = separate_audio(
                    input_path=args.video,
                    output_dir=str(tmp_dir),
                    preset="bgm_extraction",
                    override_kwargs=audio_separator_cfg
                )
                logger.info(f"Đã tách BGM từ video gốc: {bgm_path}")
            except Exception as e:
                logger.error(f"Lỗi khi tách BGM bằng audio-separator: {e}. Bỏ qua BGM track.")
                bgm_path = None

        # PHASE 3: AUDIO ASSEMBLY
        logger.info("\n--- PHASE 3: AUDIO ASSEMBLY ---")

        source_audio_for_quotes = args.video
        
        mixed_audio = str(Path(tmp_dir) / "mixed_audio.wav")
        assemble_audio_track(
            timeline=timeline,
            video_path=source_audio_for_quotes,
            ambient_path=args.ambient,
            output_path=mixed_audio,
            tmp_dir=tmp_dir,
            use_vocal_extraction=extract_vocals,
            tts_provider=args.tts_provider,
            video_duration_override=video_actual_duration_ms,
            bgm_path=bgm_path,
            audio_mix_config=render_config.get("audio_mix"),
            audio_separator_config=audio_separator_cfg,
        )
        
        # PHASE 3.5: FORCED ALIGNMENT SUBTITLE (nếu được bật)
        fa_cfg = render_config.get("forced_alignment_subtitle", {}) or {}
        fa_enabled = fa_cfg.get("enabled", False)

        subtitle_synced = str(output_dir / f"{args.output_name}_synced.srt")
        align_success = False

        if fa_enabled:
            logger.info("\n--- PHASE 3.5: FORCED ALIGNMENT SUBTITLE ---")
            align_result = run_forced_alignment_subtitle(
                audio_path=mixed_audio,
                transcript_path=str(transcript_input_path),
                output_srt_path=subtitle_synced,
                render_config=render_config,
            )
            align_success = align_result is not None

        # PHASE 4: RECALCULATE TIMESTAMPS
        logger.info("\n--- PHASE 4: RECALCULATE TIMESTAMPS ---")

        # subtitle_synced.srt — nếu forced alignment thất bại hoặc tắt, dùng remap
        if not align_success:
            recalculate_srt(subtitle_segments, timeline, subtitle_synced, is_tts_track=False, max_chars=args.subtitle_max_chars, fps_float=fps_float)
            logger.info(f"Đã tạo {subtitle_synced} (remap timestamp)")
        else:
            logger.info(f"Đã tạo {subtitle_synced} (forced alignment)")

        # subtitle_tts_synced.srt — chỉ tạo khi keep_tts_synced_debug=true
        keep_tts_debug = fa_cfg.get("keep_tts_synced_debug", False)
        if keep_tts_debug:
            subtitle_tts_synced = str(output_dir / f"{args.output_name}_tts_synced.srt")
            from sync_engine.analyzer import filter_tts_subtitles
            tts_only = filter_tts_subtitles(subtitle_segments, mute_segments)
            recalculate_srt(tts_only, timeline, subtitle_tts_synced, is_tts_track=True, max_chars=0, fps_float=fps_float)
            logger.info(f"Đã tạo {subtitle_tts_synced} (debug)")
        
        # 3. mute_synced.srt (nếu có)
        if mute_segments:
            mute_synced = str(output_dir / f"{args.output_name}_mute_synced.srt")
            recalculate_srt(mute_segments, timeline, mute_synced, is_tts_track=False, max_chars=args.subtitle_max_chars, fps_float=fps_float)
            logger.info(f"Đã tạo {mute_synced}")
            
        # 4. note_overlay_synced.ass (nếu được bật trong render_config)
        note_ass_synced = None
        note_overlay_cfg = render_config.get("note_overlay", {})
        if note_overlay_cfg.get("enabled"):
            if not args.note_overlay_ass:
                logger.info("Cấu hình note_overlay bật nhưng không truyền --note-overlay-ass. Bỏ qua note overlay.")
                # Override cấu hình để renderer bỏ qua
                note_overlay_cfg["enabled"] = False
            else:
                ass_input_path = Path(args.note_overlay_ass)
                if not ass_input_path.is_absolute():
                    ass_input_path = PROJECT_ROOT / ass_input_path
                
                if not ass_input_path.exists():
                    raise FileNotFoundError(f"Không tìm thấy file ASS: {ass_input_path}")
                    
                note_ass_synced = str(output_dir / f"{args.output_name}_note_synced.ass")
                recalculate_ass(str(ass_input_path), timeline, note_ass_synced, max_chars_per_line=args.note_max_chars, fps_float=fps_float)
                logger.info(f"Đã tạo {note_ass_synced}")
            
        # PHASE 5: FINAL RENDER
        logger.info("\n--- PHASE 5: FINAL RENDER ---")
        if not args.no_hardsub:
            final_video = str(output_dir / f"{args.output_name}.mp4")

            render_final_video(
                stretched_video=stretched_video_chunked,
                mixed_audio=mixed_audio,
                subtitle_synced_srt=subtitle_synced,
                output_path=final_video,
                note_overlay_synced_ass=note_ass_synced,
                render_config=render_config,
                use_gpu=not args.no_gpu,
            )
            logger.info(f"Render hoàn tất: {final_video}")
            if transcript_input_path:
                run_llm_metadata_task(
                    input_text_path=str(transcript_input_path),
                    render_config=render_config,
                    video_path=args.video,
                    output_name=args.output_name,
                )
        else:
            logger.info("Bỏ qua bước Render do cờ --no-hardsub.")
            logger.info("Bỏ qua LLM metadata vì bước này chỉ chạy sau final render.")
            
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

def worker_task(task_data: dict, base_args: argparse.Namespace):
    """
    Hàm này chạy trong một Process hoàn toàn độc lập.
    Khi hàm này return (hoặc crash), OS sẽ dọn sạch 100% RAM và VRAM.
    """
    try:
        import copy
        from utils.task_utils import resolve_output_dir_and_stem
        
        args = copy.deepcopy(base_args)
        
        # Ghi đè các tham số từ JSON
        args.video = task_data["input"]
        if "subtitle" not in task_data:
            raise ValueError("Task JSON thiếu 'subtitle'. Sync video bắt buộc phải có subtitle.")
        args.subtitle = task_data["subtitle"]
        
        if "mute" in task_data:
            args.mute = task_data["mute"]
        if "note_overlay_ass" in task_data:
            args.note_overlay_ass = task_data["note_overlay_ass"]
        if "render_config" in task_data:
            args.render_config = task_data["render_config"]
        if "llm_metadata" in task_data:
            args.llm_metadata_override = task_data["llm_metadata"]
            
        out_dir, stem = resolve_output_dir_and_stem(task_data)
        args.output_dir = str(out_dir)
        args.output_name = stem
        
        # Chạy pipeline
        run_sync_pipeline(args)
        
    except Exception as e:
        logger.exception(f"❌ [Worker] Lỗi nghiêm trọng khi chạy video {task_data.get('input')}: {e}")
        sys.exit(1)  # Báo cho OS biết process này fail

def main():
    parser = argparse.ArgumentParser(description="TTS-Video Sync - Chunk-Based Stretch")
    
    # Task File integration
    parser.add_argument("--task-file", type=str, help="Đường dẫn đến file JSON chứa danh sách task")
    
    # Bắt buộc (nếu không dùng --task-file)
    parser.add_argument("--video", help="File video gốc (.mp4, .mkv)")
    parser.add_argument("--subtitle", help="File subtitle.srt đầy đủ (kể cả vùng mute)")
    
    # TTS Settings
    parser.add_argument("--tts-provider", choices=["edge", "voicevox_nemo", "voicevox", "qwen"], default="edge", help="Chọn TTS engine: edge, voicevox_nemo, voicevox, qwen (mặc định: edge)")
    parser.add_argument("--tts-voice", default=None, help="Tên giọng EdgeTTS hoặc ID nhân vật Voicevox/Voicevox Nemo (ghi đè YAML)")
    parser.add_argument("--tts-config", default=str(PROJECT_ROOT / "config" / "tts_config.yaml"),
                        help="File YAML cấu hình TTS (mặc định: config/tts_config.yaml)")
    
    # Tùy chọn - Input
    parser.add_argument("--mute", help="File mute.srt")
    parser.add_argument("--note-overlay-ass", help="ASS text cho note")
    parser.add_argument("--ambient", default=str(PROJECT_ROOT / "assets" / "ambient.mp3"), help="Nhạc nền")

    # Render Config
    parser.add_argument("--render-config", default=str(PROJECT_ROOT / "assets" / "default_render_config.json"),
                        help="File JSON cấu hình render (mặc định: assets/default_render_config.json)")
    
    # Algorithm
    parser.add_argument("--slow-cap", type=float, default=0.5, help="Video speed tối thiểu (mặc định: 0.5)")
    
    # Output
    parser.add_argument("--output-dir", default="./sync_output/", help="Thư mục output")
    parser.add_argument("--output-name", default="video_synced", help="Base name output")
    parser.add_argument("--no-hardsub", action="store_true", help="Chỉ output recalculated files")
    parser.add_argument("--keep-tmp", action="store_true", help="Giữ lại thư mục tạm chứa các chunks")
    
    # Performance
    parser.add_argument("--workers", type=int, default=4, help="FFmpeg workers song song")
    parser.add_argument("--batch-size", type=int, default=100, help="Số segments mỗi batch Filter Complex (mặc định: 100)")
    parser.add_argument("--no-gpu", action="store_true", help="Dùng libx264 thay vì h264_nvenc")
    
    # Subtitle / Note Processing
    parser.add_argument("--subtitle-max-chars", type=int, default=0, help="Ngắt dòng subtitle nếu dài hơn số ký tự này (0 = không ngắt)")
    parser.add_argument("--note-max-chars", type=int, default=16)
    
    args = parser.parse_args()
    args.llm_metadata_override = None
    
    setup_logging()
    
    import multiprocessing as mp
    from utils.task_utils import resolve_cli_tasks
    
    if getattr(args, "task_file", None):
        # Resolve tasks
        tasks = resolve_cli_tasks(
            task_file=args.task_file,
            input_file=None,
            output_path=None,
            default_ext=".mp4" # Sync video default output ext is mp4
        )
        
        logger.info(f"Đã tải {len(tasks)} tasks từ JSON.")
        
        # Bắt buộc dùng 'spawn' cho CUDA
        ctx = mp.get_context('spawn')
        
        for idx, task_data in enumerate(tasks):
            logger.info("=============================================")
            logger.info(f"▶ Đang xử lý Task {idx + 1}/{len(tasks)}")
            logger.info("=============================================")
            
            p = ctx.Process(target=worker_task, args=(task_data, args))
            p.start()
            p.join()
            
            if p.exitcode == 0:
                logger.info(f"✅ Task {idx + 1} hoàn tất.")
            else:
                logger.warning(f"⚠️ Task {idx + 1} thất bại (Exit code: {p.exitcode}). Bỏ qua và chạy task tiếp theo.")
                
    else:
        # Fallback chạy đơn lẻ
        if not args.video or not args.subtitle:
            parser.error("Khi không dùng --task-file, phải cung cấp --video và --subtitle")
            
        try:
            run_sync_pipeline(args)
        except Exception as e:
            logger.exception("Đã xảy ra lỗi:")
            sys.exit(1)

if __name__ == "__main__":
    main()
