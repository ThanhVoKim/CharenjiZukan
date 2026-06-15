#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli/align_srt.py — CLI: Tách câu SRT theo dấu ngắt câu (không dùng AI model).

Flow OCR-centric (xem docs/colab-guide.md Mục 2.0c):
    video-ocr → punctuate-srt → _punct.srt ──► align-srt → _seg.srt

align-srt (v2 — thuần CPU):
    Đọc _punct.srt (đã có dấu câu từ punctuate-srt), gom các block liên tiếp
    cho tới khi gặp block kết thúc bằng dấu ngắt câu → tạo 1 block câu mới.
    Timestamp lấy thẳng từ biên block OCR gốc (không nội suy, không model, không GPU).

NOTE (hoãn lại — chưa implement ở v1):
    Dấu ngắt câu nằm GIỮA block subtitle (vd "去学校。然后") → v1 mặc kệ, không cắt.
    Khi cần: nội suy timestamp theo tỉ lệ ký tự trong block (dùng start_time/end_time
    thật của block đó), không nội suy qua nhiều block.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger, setup_logging
from utils.srt_parser import is_cjk, parse_srt, segments_to_srt
from utils.srt_sentence_segmenter import resegment_srt_by_sentence
from utils.task_utils import resolve_cli_tasks
from utils.text_segmenter import DEFAULT_GRAMMAR_SPLIT_CHARS, EXTENDED_GRAMMAR_SPLIT_CHARS

logger = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════
# Core
# ═════════════════════════════════════════════════════════════════════

def resegment_srt_file(
    input_srt: str,
    output_srt: str,
    offset_seconds: float = 0.0,
    split_on_comma: bool = False,
) -> dict:
    """Đọc _punct.srt, gom theo câu, ghi ra output_srt.

    Args:
        split_on_comma: Nếu True dùng EXTENDED_GRAMMAR_SPLIT_CHARS (gồm dấu phẩy);
                        mặc định False — chỉ cắt tại .!?:。！？：；.

    Returns:
        {"input_blocks": int, "output_blocks": int, "lang": "CJK" | "Latin"}
    """
    content = Path(input_srt).read_text(encoding="utf-8", errors="ignore")
    segments = parse_srt(content)
    if not segments:
        raise ValueError(f"Không parse được block nào từ: {input_srt}")

    # Auto-detect CJK vs Latin một lần từ toàn bộ text
    all_text = "".join(seg["text"] for seg in segments)
    lang = "CJK" if is_cjk(all_text) else "Latin"

    split_chars = EXTENDED_GRAMMAR_SPLIT_CHARS if split_on_comma else DEFAULT_GRAMMAR_SPLIT_CHARS
    blocks = resegment_srt_by_sentence(segments, split_chars=split_chars)

    if offset_seconds:
        off = int(offset_seconds * 1000)
        for b in blocks:
            b["start_time"] = max(0, b["start_time"] + off)
            b["end_time"] = max(b["start_time"], b["end_time"] + off)

    Path(output_srt).parent.mkdir(parents=True, exist_ok=True)
    Path(output_srt).write_text(segments_to_srt(blocks), encoding="utf-8")

    return {"input_blocks": len(segments), "output_blocks": len(blocks), "lang": lang}


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="align-srt",
        description="Tách câu SRT theo dấu ngắt câu từ output của punctuate-srt.",
    )
    io = parser.add_argument_group("Input / Output")
    io.add_argument("input_srt", nargs="?", default=None,
                    help="File _punct.srt (đã có dấu câu từ punctuate-srt)")
    io.add_argument("--input", "-i", default=None, metavar="FILE",
                    help="Alias của input_srt (dùng khi không truyền positional)")
    io.add_argument("--output", "-o", default=None, metavar="FILE_OR_DIR",
                    help="File .srt hoặc thư mục output (mặc định: <input>_seg.srt)")
    io.add_argument("--task-file", "-t", default=None, metavar="JSON",
                    help='JSON [{"input": "..._punct.srt", "output": "..."}]')

    seg = parser.add_argument_group("Segmentation")
    seg.add_argument("--offset-seconds", type=float, default=0.0,
                     help="Offset cộng vào timestamp (giây, mặc định: 0.0)")
    seg.add_argument("--split-on-comma", action="store_true", default=False,
                     help="Cắt câu tại dấu phẩy (，、,;) — mặc định tắt")

    parser.add_argument("--verbose", "-v", action="store_true", help="Bật logging DEBUG")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(level=10 if args.verbose else 20)

    input_file = args.input_srt or args.input
    try:
        tasks = resolve_cli_tasks(
            task_file=args.task_file,
            input_file=input_file,
            output_path=args.output,
            default_ext="_seg.srt",
        )
    except ValueError as exc:
        parser.error(str(exc))
        return

    ok = 0
    try:
        for task in tasks:
            input_srt = task["input"]
            output_srt = task["output"]

            print("=" * 55)
            print("  ✂️  SRT Sentence Segmentation")
            print(f"  Input  : {input_srt}")
            print(f"  Output : {output_srt}")
            if args.offset_seconds:
                print(f"  Offset : {args.offset_seconds:+.3f}s")
            if args.split_on_comma:
                print("  Comma  : split bật (，、,;)")
            print("=" * 55)

            stats = resegment_srt_file(
                input_srt=input_srt,
                output_srt=output_srt,
                offset_seconds=args.offset_seconds,
                split_on_comma=args.split_on_comma,
            )
            ok += 1
            print(
                f"✅ {output_srt} "
                f"[{stats['lang']}] "
                f"({stats['input_blocks']} block → {stats['output_blocks']} câu)"
            )

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
