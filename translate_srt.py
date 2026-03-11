#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translate_srt.py — CLI: Dịch file .srt bằng Gemini API

Dịch subtitle sang ngôn ngữ đích sử dụng Google Gemini API.
Hỗ trợ batch processing và rotate API keys.

Xem hướng dẫn chi tiết tại: docs/colab-usage.md

Ví dụ nhanh:
    uv run translate_srt.py --input video.srt --keys "AIza..."
    uv run translate_srt.py --input video.srt --lang "Japanese" --keys "AIza..."
"""

import sys
import logging
import argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Thêm thư mục chứa script vào sys.path để import translator
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from translator import translate_srt_file  # noqa: E402


# ─────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="translate_srt",
        description="Dịch file .srt bằng Gemini API (pyvideotrans logic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ nhanh:
  python translate_srt.py --input video.srt --keys "AIzaSy..."

Ví dụ đầy đủ:
  python translate_srt.py \\
      --input   /content/video.srt         \\
      --output  /content/video_vi.srt      \\
      --lang    "Vietnamese"               \\
      --keys    "AIza...k1,AIza...k2"      \\
      --model   gemini-2.5-flash           \\
      --batch   30                         \\
      --budget  8192
        """,
    )

    # Bắt buộc
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="FILE",
        help="Đường dẫn file SRT gốc (bắt buộc)",
    )
    parser.add_argument(
        "--keys", "-k",
        required=True,
        metavar="KEY[,KEY2,...]",
        help="Gemini API key(s), nhiều key cách nhau dấu phẩy (bắt buộc)",
    )

    # Tuỳ chọn
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="File SRT đầu ra (mặc định: <input>_<lang>.srt)",
    )
    parser.add_argument(
        "--lang", "-l",
        default="Vietnamese",
        metavar="LANGUAGE",
        help="Ngôn ngữ đích — tên tiếng Anh đầy đủ (mặc định: Vietnamese)",
    )
    parser.add_argument(
        "--model", "-m",
        default="gemini-2.5-flash",
        metavar="MODEL",
        help="Gemini model (mặc định: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        metavar="FILE",
        help="Đường dẫn tới gemini.txt (mặc định: gemini.txt cùng thư mục script)",
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=30,
        metavar="N",
        help="Số SRT block mỗi batch (mặc định: 30)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=24576,
        metavar="TOKENS",
        help="Thinking budget tokens, 0 để tắt (mặc định: 24576)",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Tắt full-context: không gửi toàn bộ SRT làm context nền",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=0.0,
        metavar="SEC",
        help="Giây chờ giữa mỗi batch (mặc định: 0)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Bật logging debug chi tiết",
    )

    return parser


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = build_parser()
    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"File không tồn tại: {args.input}")
    if input_path.suffix.lower() != ".srt":
        parser.error(f"File phải có đuôi .srt: {args.input}")

    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    else:
        lang_slug = args.lang.lower().replace(" ", "_")
        output_path = input_path.with_stem(f"{input_path.stem}_{lang_slug}")
    output_file = str(output_path)

    # Resolve prompt file
    if args.prompt:
        prompt_file = args.prompt
        if not Path(prompt_file).exists():
            parser.error(f"Prompt file không tồn tại: {prompt_file}")
    else:
        prompt_file = str(SCRIPT_DIR / "gemini.txt")
        if not Path(prompt_file).exists():
            parser.error(
                f"Không tìm thấy gemini.txt tại: {prompt_file}\n"
                "Hãy đặt gemini.txt cùng thư mục với translate_srt.py "
                "hoặc dùng --prompt <đường_dẫn>"
            )

    # Parse API keys
    api_keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    if not api_keys:
        parser.error("Không có API key hợp lệ nào")

    # In tóm tắt cấu hình
    print("=" * 55)
    print("  🎬  SRT Translator — Gemini API")
    print("=" * 55)
    print(f"  Input  : {args.input}")
    print(f"  Output : {output_file}")
    print(f"  Lang   : {args.lang}")
    print(f"  Model  : {args.model}")
    print(f"  Batch  : {args.batch}  |  Budget: {args.budget}")
    print(f"  Context: {'OFF' if args.no_context else 'ON'}")
    print(f"  Keys   : {len(api_keys)} key(s)")
    print("=" * 55)

    # Gọi pipeline dịch
    try:
        stats = translate_srt_file(
            input_file      = str(input_path),
            output_file     = output_file,
            prompt_file     = prompt_file,
            api_keys        = api_keys,
            model           = args.model,
            target_language = args.lang,
            batch_size      = args.batch,
            thinking_budget = args.budget,
            use_full_context= not args.no_context,
            wait_sec        = args.wait,
        )
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n⚠️  Đã dừng bởi người dùng")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Lỗi nghiêm trọng: {e}")
        logging.exception(e)
        sys.exit(2)


if __name__ == "__main__":
    main()
