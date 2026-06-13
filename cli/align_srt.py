#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli/align_srt.py — CLI: Forced-alignment SRT từ transcript (text OCR đã có dấu) + audio nguồn.

Flow OCR-centric (xem docs):
    video-ocr → (punctuation) → _flat.txt  ─┐
    video/audio source ──────────────────────┴─► align-srt → _aligned.srt

align-srt:
  1. Lấy audio từ --video (tự trích bằng ffmpeg nếu là video),
  2. TỰ tách vocal (reuse cli/audio_separator.separate_audio, preset vocal_extraction),
  3. Forced-align text transcript với vocals → SRT (text giữ nguyên, timing từ aligner).

Tham số forced-alignment cấu hình hoàn toàn bằng CLI args (đồng nhất idiom repo:
qwen3-asr / audio-separator). Lõi align ở utils/forced_aligner.py.
"""

import argparse
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger, setup_logging
from utils.task_utils import resolve_cli_tasks
from utils.forced_aligner import execute_forced_alignment

logger = get_logger(__name__)

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}

# Default forced-alignment cho flow OCR (khác sync-video: text Trung ngắn).
_ALIGN_DEFAULTS: dict[str, Any] = {
    "model_path": None,        # None → Qwen/Qwen3-ForcedAligner-0.6B
    "device": None,            # None → cuda:0
    "dtype": None,             # None → bfloat16
    "attn_implementation": None,
    "language": "Chinese",
    "max_chars": 18,
    "min_chars": 0,
    "split_on_comma": True,
    "offset_seconds": 0.0,
}


# ═════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════

def resolve_align_cfg(args: argparse.Namespace) -> dict[str, Any]:
    """Build align_cfg từ CLI args (đã có default từ argparse)."""
    cfg = dict(_ALIGN_DEFAULTS)
    cfg.update({
        "language": args.language,
        "max_chars": args.max_chars,
        "min_chars": args.min_chars,
        "offset_seconds": args.offset_seconds,
        "model_path": args.model_path,
        "device": args.device,
        "split_on_comma": not args.no_split_on_comma,
    })
    return cfg


# ═════════════════════════════════════════════════════════════════════
# Audio: extract + tách vocal
# ═════════════════════════════════════════════════════════════════════

def _extract_audio(source: str, out_dir: Path) -> str:
    """Trích audio WAV từ video. Nếu source đã là audio → trả về nguyên path."""
    src = Path(source)
    if src.suffix.lower() in _AUDIO_EXTS:
        return str(src)
    out_wav = out_dir / f"{src.stem}.source.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vn", "-acodec", "pcm_s16le", "-ac", "2",
        str(out_wav),
    ]
    logger.info(f"🎬 Trích audio từ video: {src.name} → {out_wav.name}")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return str(out_wav)


def _get_vocals(
    *,
    video: Optional[str],
    vocals: Optional[str],
    no_separate: bool,
    preset: str,
    out_dir: Path,
) -> str:
    """Trả về đường dẫn audio dùng để align (vocals đã tách, hoặc audio gốc)."""
    if vocals:
        logger.info(f"🎙️ Dùng vocals có sẵn: {vocals}")
        return vocals
    if not video:
        raise ValueError("Cần --video (hoặc --vocals) để forced-align.")

    audio_path = _extract_audio(video, out_dir)

    if no_separate:
        logger.info("⏭️ Bỏ qua tách vocal (align thẳng với audio nguồn).")
        return audio_path

    from cli.audio_separator import separate_audio
    logger.info(f"🔊 Tách vocal (preset={preset})...")
    vocals_path = separate_audio(
        audio_path,
        str(out_dir),
        preset=preset,
        config_path=str(PROJECT_ROOT / "config/audio_separator_config.yaml"),
    )
    return vocals_path


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="align-srt",
        description="Forced-alignment SRT từ transcript (text OCR đã có dấu) + audio nguồn.",
    )
    io = parser.add_argument_group("Input / Output")
    io.add_argument("transcript", nargs="?", default=None,
                    help="File text PHẲNG 1 dòng (đã có dấu), vd <stem>_flat.txt")
    io.add_argument("--input", "-i", default=None, metavar="FILE",
                    help="Alias của transcript (dùng khi không truyền positional)")
    io.add_argument("--video", "-V", default=None, metavar="FILE",
                    help="Video/audio nguồn để tách vocal & align")
    io.add_argument("--vocals", default=None, metavar="WAV",
                    help="File vocals có sẵn → bỏ qua bước tách")
    io.add_argument("--output", "-o", default=None, metavar="FILE_OR_DIR",
                    help="File .srt hoặc thư mục output")
    io.add_argument("--task-file", "-t", default=None, metavar="JSON",
                    help="JSON [{'input': transcript, 'video': ..., 'output': ...}]")

    seg = parser.add_argument_group("Forced alignment")
    seg.add_argument("--language", "-l", default=_ALIGN_DEFAULTS["language"],
                     help="Ngôn ngữ nguồn (default: Chinese)")
    seg.add_argument("--max-chars", type=int, default=_ALIGN_DEFAULTS["max_chars"],
                     help="Số ký tự tối đa mỗi block (default: 18)")
    seg.add_argument("--min-chars", type=int, default=_ALIGN_DEFAULTS["min_chars"],
                     help="Số ký tự tối thiểu mỗi block (default: 0)")
    seg.add_argument("--offset-seconds", type=float, default=_ALIGN_DEFAULTS["offset_seconds"],
                     help="Offset timestamp (giây, default: 0.0)")
    seg.add_argument("--no-split-on-comma", action="store_true",
                     help="Tắt cắt block tại dấu phẩy (default: bật)")
    seg.add_argument("--model-path", default=_ALIGN_DEFAULTS["model_path"],
                     help="Đường dẫn model aligner (default: Qwen/Qwen3-ForcedAligner-0.6B)")
    seg.add_argument("--device", "-d", default=_ALIGN_DEFAULTS["device"],
                     help="Device (default: cuda:0)")

    sep = parser.add_argument_group("Vocal separation")
    sep.add_argument("--separator-preset", default="vocal_extraction",
                     help="Preset trong config/audio_separator_config.yaml (default: vocal_extraction)")
    sep.add_argument("--no-separate", action="store_true",
                     help="Không tách vocal, align thẳng audio nguồn")

    parser.add_argument("--verbose", "-v", action="store_true", help="Bật logging DEBUG")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(level=10 if args.verbose else 20)

    transcript_input = args.transcript or args.input
    try:
        tasks = resolve_cli_tasks(
            task_file=args.task_file,
            input_file=transcript_input,
            output_path=args.output,
            default_ext="_aligned.srt",
        )
    except ValueError as exc:
        parser.error(str(exc))
        return

    align_cfg = resolve_align_cfg(args)

    ok = 0
    try:
        for task in tasks:
            transcript_path = task["input"]
            output_srt = task["output"]
            video = task.get("video") or args.video
            vocals = task.get("vocals") or args.vocals

            print("=" * 55)
            print("  🎯 Forced Alignment SRT")
            print(f"  Transcript : {transcript_path}")
            print(f"  Source     : {vocals or video}")
            print(f"  Output     : {output_srt}")
            print(f"  Language   : {align_cfg['language']} | max_chars={align_cfg['max_chars']}")
            print("=" * 55)

            with tempfile.TemporaryDirectory(prefix="align_srt_") as tmp:
                audio_for_align = _get_vocals(
                    video=video,
                    vocals=vocals,
                    no_separate=args.no_separate,
                    preset=args.separator_preset,
                    out_dir=Path(tmp),
                )
                stats = execute_forced_alignment(
                    audio_path=audio_for_align,
                    transcript_path=transcript_path,
                    output_srt_path=output_srt,
                    align_cfg=align_cfg,
                )
            ok += 1
            print(f"✅ {output_srt} ({stats['subtitle_blocks']} blocks / {stats['total_words']} words)")

    except KeyboardInterrupt:
        print("\n⚠️ Dừng bởi người dùng")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Lỗi: {exc}", exc_info=args.verbose)
        sys.exit(1)

    print(f"\nTổng kết: {ok}/{len(tasks)} task thành công")
    sys.exit(0 if ok == len(tasks) else 2)


if __name__ == "__main__":
    main()
