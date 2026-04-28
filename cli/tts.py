#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli/tts.py — CLI: TTS từ .srt hoặc .txt

Hỗ trợ 3 engine: EdgeTTS, Voicevox, Qwen3-TTS.
Cấu hình qua YAML (--config), hỗ trợ batch qua JSON (--task-file).

Ví dụ:
    uv run tts --input video.srt --config config/tts_config.yaml
    uv run tts --input script.txt --provider qwen --config config/tts_config.yaml
    uv run tts --task-file tasks.json --config config/tts_config.yaml
"""

import sys
import json
import shutil
import time
import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from pydub import AudioSegment

from utils.logger import setup_logging, get_logger
from utils.srt_parser import parse_srt
from utils.task_utils import resolve_cli_tasks
from speed_rate import SpeedRate
from tts.edgetts import EdgeTTSEngine
from tts.voicevox import VoicevoxTTSEngine
from tts.qwen import QwenTTSEngine

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    if not config_path or not Path(config_path).exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ─────────────────────────────────────────────────────────────────────
# QUEUE BUILDERS
# ─────────────────────────────────────────────────────────────────────
def build_queue_from_srt(input_file: str, cache_folder: str):
    raw = Path(input_file).read_text(encoding="utf-8", errors="ignore")
    srt_list = parse_srt(raw)
    if not srt_list:
        raise ValueError(f"Không đọc được subtitle nào từ: {input_file}")
    queue = []
    for i, it in enumerate(srt_list):
        queue.append({
            "text": it["text"],
            "line": it["line"],
            "start_time": it["start_time"],
            "end_time": it["end_time"],
            "filename": str(Path(cache_folder) / f"dubb-{i}.wav"),
        })
    return queue, srt_list[-1]["end_time"]


def build_queue_from_txt(input_file: str, cache_folder: str):
    text = Path(input_file).read_text(encoding="utf-8", errors="ignore")
    # Không phân chia — đưa nguyên khối text vào 1 item
    queue = [{
        "text": text,
        "line": 1,
        "start_time": 0,
        "end_time": 0,
        "filename": str(Path(cache_folder) / "dubb-0.wav"),
    }]
    return queue, 0


# ─────────────────────────────────────────────────────────────────────
# ENGINE FACTORY
# ─────────────────────────────────────────────────────────────────────
def get_engine(provider: str, queue_tts: list, config: dict):
    provider_cfg = config.get(provider, {})
    if provider == "edge":
        return EdgeTTSEngine(queue_tts, **provider_cfg)
    elif provider == "voicevox":
        return VoicevoxTTSEngine(queue_tts, **provider_cfg)
    elif provider == "qwen":
        return QwenTTSEngine(queue_tts, **provider_cfg)
    else:
        raise ValueError(f"Provider không hợp lệ: {provider}")


# ─────────────────────────────────────────────────────────────────────
# AUDIO UTILS
# ─────────────────────────────────────────────────────────────────────
def concat_wav_files(file_list: list, output_path: str, silence_ms: int = 0):
    """Nối các file wav với khoảng lặng giữa các file (dùng cho .txt mode hoặc .srt không autorate)."""
    combined = AudioSegment.empty()
    for i, f in enumerate(file_list):
        if Path(f).exists():
            seg = AudioSegment.from_file(f, format="wav")
            combined += seg
            if i < len(file_list) - 1 and silence_ms > 0:
                combined += AudioSegment.silent(duration=silence_ms)
    combined.export(output_path, format="wav")


# ─────────────────────────────────────────────────────────────────────
# LIST VOICES (EdgeTTS only)
# ─────────────────────────────────────────────────────────────────────
async def _list_voices_async(locale_filter: str = None):
    from edge_tts import VoicesManager
    vm = await VoicesManager.create()
    voices = vm.voices
    if locale_filter:
        voices = [v for v in voices if locale_filter.lower() in v["Locale"].lower()]
    return sorted(voices, key=lambda v: v["Locale"])


def list_voices(locale_filter: str = None):
    import asyncio
    voices = asyncio.run(_list_voices_async(locale_filter))
    if not voices:
        print(f"Không tìm thấy giọng nào với filter: '{locale_filter}'")
        return
    print(f"\n{'Locale':<20} {'ShortName':<45} {'Gender'}")
    print("─" * 80)
    for v in voices:
        print(f"{v['Locale']:<20} {v['ShortName']:<45} {v['Gender']}")
    print(f"\nTổng: {len(voices)} giọng\n")


# ─────────────────────────────────────────────────────────────────────
# SINGLE TASK RUNNER
# ─────────────────────────────────────────────────────────────────────
def run_task(task: dict, config: dict, args) -> dict:
    input_file = task["input"]
    output_file = task["output"]
    provider = args.provider or config.get("provider", "edge")

    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"File không tồn tại: {input_file}")

    suffix = input_path.suffix.lower()
    is_txt_mode = suffix == ".txt"
    is_srt_mode = suffix == ".srt"

    if not is_txt_mode and not is_srt_mode:
        raise ValueError(f"File phải có đuôi .srt hoặc .txt: {input_file}")

    # Cache folder
    cache_folder = args.cache
    _tmp_created = False
    if not cache_folder:
        stem = input_path.stem
        cache_folder = str(PROJECT_ROOT / "tmp" / f"{stem}_{int(time.time())}")
        _tmp_created = True
    Path(cache_folder).mkdir(parents=True, exist_ok=True)

    # Build queue
    if is_srt_mode:
        queue_tts, raw_total_time = build_queue_from_srt(input_file, cache_folder)
    else:
        queue_tts, raw_total_time = build_queue_from_txt(input_file, cache_folder)

    total = len(queue_tts)
    print(f"\n{'='*55}")
    print(f"  🎙  TTS ({provider.upper()})")
    print(f"{'='*55}")
    print(f"  Input   : {input_file}")
    print(f"  Output  : {output_file}")
    print(f"  Mode    : {'TXT' if is_txt_mode else 'SRT'}")
    print(f"  Provider: {provider}")
    print(f"{'='*55}\n")

    # Run TTS
    print(f"🔊 Phase 1/2 — Tạo audio ({total} items)...")
    engine = get_engine(provider, queue_tts, config)
    tts_stats = engine.run()
    print(f"   ✅ {provider.upper()}: {tts_stats['ok']} OK | {tts_stats['err']} lỗi\n")

    if tts_stats["ok"] == 0:
        raise RuntimeError(f"{provider.upper()} thất bại hoàn toàn — không có audio nào được tạo")

    # Post-process
    target_wav = str(Path(cache_folder) / "_target.wav")

    silence_ms = getattr(args, "silence_ms", 0)

    if is_txt_mode or not args.autorate:
        mode_str = "TXT mode" if is_txt_mode else f"SRT không autorate (silence={silence_ms}ms)"
        print(f"🔗 Phase 2/2 — Ghép audio ({mode_str})...")
        wav_files = [it["filename"] for it in queue_tts if Path(it["filename"]).exists()]
        concat_wav_files(wav_files, target_wav, silence_ms=silence_ms)
    else:
        print("🔗 Phase 2/2 — Ghép audio (autorate=ON)...")
        sr = SpeedRate(
            queue_tts=queue_tts,
            target_audio=target_wav,
            cache_folder=cache_folder,
            voice_autorate=args.autorate,
            raw_total_time=raw_total_time,
            max_speed_rate=args.max_speed,
        )
        sr.run()

    if not Path(target_wav).exists():
        raise RuntimeError(f"Không tạo được file audio: {target_wav}")

    # Convert output format
    out_ext = Path(output_file).suffix.lower()
    if not out_ext:
        output_file = output_file + ".wav"
        out_ext = ".wav"

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    if out_ext == ".wav":
        shutil.move(target_wav, output_file)
    else:
        import subprocess
        cmd = ["ffmpeg", "-y", "-i", target_wav, "-b:a", "192k", output_file]
        subprocess.run(cmd, check=True, capture_output=True)

    # Cleanup
    if _tmp_created and not getattr(args, "keep_cache", False):
        shutil.rmtree(cache_folder, ignore_errors=True)
    elif getattr(args, "keep_cache", False):
        print(f"ℹ️ Đã giữ lại thư mục cache: {cache_folder}")

    print(f"\n{'─'*55}")
    print(f"✅ Hoàn thành: {output_file}")
    print(f"   Items: {total} | TTS OK: {tts_stats['ok']} | Err: {tts_stats['err']}")
    print(f"{'─'*55}")

    return {
        "input": input_file,
        "output": output_file,
        "ok": tts_stats["ok"],
        "err": tts_stats["err"],
    }


# ─────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tts",
        description="TTS: chuyển file .srt hoặc .txt thành audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ nhanh:
  uv run tts --input video.srt --config config/tts_config.yaml
  uv run tts --input script.txt --provider qwen --config config/tts_config.yaml
  uv run tts --task-file tasks.json --config config/tts_config.yaml

Xem danh sách giọng EdgeTTS:
  uv run tts --list-voices vi
        """,
    )

    io = parser.add_argument_group("Input / Output")
    io.add_argument("--input", "-i", metavar="FILE",
                    help="File .srt hoặc .txt đầu vào")
    io.add_argument("--output", "-o", metavar="FILE",
                    help="File audio đầu ra (mặc định: PROJ/output/<stem>.wav)")
    io.add_argument("--task-file", "-t", metavar="JSON",
                    help="File JSON chứa danh sách [{'input': '...', 'output': '...'}]")

    cfg = parser.add_argument_group("Config")
    cfg.add_argument("--config", "-c", default="config/tts_config.yaml", metavar="YAML",
                     help="File cấu hình YAML (mặc định: config/tts_config.yaml)")
    cfg.add_argument("--provider", "-p", choices=["edge", "voicevox", "qwen"],
                     help="TTS engine (ghi đè giá trị trong config)")

    proc = parser.add_argument_group("Processing")
    proc.add_argument("--autorate", action="store_true",
                      help="Bật autorate: tự động nén audio khớp với slot SRT (chỉ .srt)")
    proc.add_argument("--max-speed", type=float, default=100.0, metavar="X",
                      help="Giới hạn tốc độ tăng tối đa cho autorate (mặc định: 100)")
    proc.add_argument("--silence-ms", type=int, default=0, metavar="MS",
                      help="Độ dài silence được chèn giữa các dòng khi không dùng autorate (mặc định: 0)")
    proc.add_argument("--cache", metavar="DIR",
                      help="Thư mục cache audio tạm (mặc định: PROJ/tmp/<stem>_<ts>/)")
    proc.add_argument("--keep-cache", action="store_true",
                      help="Giữ lại thư mục cache tạm sau khi xử lý xong")

    misc = parser.add_argument_group("Misc")
    misc.add_argument("--list-voices", metavar="LOCALE",
                      help="Liệt kê giọng EdgeTTS, vd: --list-voices vi")
    misc.add_argument("--verbose", "-v", action="store_true",
                      help="Bật logging debug")

    return parser


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    # List voices mode
    if args.list_voices is not None:
        list_voices(args.list_voices or None)
        return

    try:
        tasks = resolve_cli_tasks(
            task_file=args.task_file,
            input_file=args.input,
            output_path=args.output,
            default_ext=".wav",
            default_out_dir=PROJECT_ROOT / "output"
        )
    except ValueError as e:
        parser.error(str(e))

    config = load_config(args.config)

    ok_tasks = 0
    for task in tasks:
        try:
            run_task(task, config, args)
            ok_tasks += 1
        except Exception as e:
            logger.error(f"❌ Lỗi xử lý {task['input']}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    print(f"\n{'='*55}")
    print(f"  Tổng kết: {ok_tasks}/{len(tasks)} task thành công")
    print(f"{'='*55}")

    sys.exit(0 if ok_tasks == len(tasks) else 1)


if __name__ == "__main__":
    main()
