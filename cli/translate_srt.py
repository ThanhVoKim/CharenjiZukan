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
# Add project root to path for imports
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from translation.translator import translate_srt_file  # noqa: E402
from translation.factory import create_provider  # noqa: E402
from utils.logger import setup_logging, get_logger  # noqa: E402


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
      --model   gemini-3-flash-preview           \\
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

    # Provider Settings
    parser.add_argument(
        "--provider", "-p",
        default="gemini",
        choices=["gemini", "openai", "vertexai"],
        metavar="PROVIDER",
        help="Translation provider (mặc định: gemini). Các lựa chọn: gemini | openai | vertexai",
    )
    parser.add_argument(
        "--provider-config",
        default=None,
        metavar="FILE",
        help="Đường dẫn tới file config YAML của provider. Mặc định: config/openai_compat_translate.yaml (openai) hoặc config/vertexai_translate.yaml (vertexai).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        metavar="URL",
        help="[openai provider only] Override base_url trong config YAML. Ví dụ: http://localhost:1234/v1",
    )

    parser.add_argument(
        "--keys", "-k",
        default=None,
        metavar="KEY[,KEY2,...]",
        help="API key(s) (bắt buộc cho gemini và openai, nhiều key cách nhau dấu phẩy đối với gemini)",
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
        default=None,
        metavar="MODEL",
        help="Model name cho provider (ưu tiên: CLI > config > default provider)",
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
        default=None,
        metavar="N",
        help="Số SRT block mỗi batch (ưu tiên: CLI > config > mặc định 30)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        metavar="TOKENS",
        help="Thinking budget tokens (gemini), ưu tiên: CLI > config > mặc định 24576",
    )
    parser.add_argument(
        "--context",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Bật/tắt full-context (ưu tiên: CLI > config > mặc định bật). Dùng --no-context để tắt.",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=None,
        metavar="SEC",
        help="Giây chờ giữa mỗi batch (ưu tiên: CLI > config > mặc định 0)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Bật logging debug chi tiết",
    )

    return parser


def resolve_by_priority(cli_value, config: dict, config_keys: list[str], default_value):
    """Ưu tiên giá trị: CLI > YAML config > default."""
    if cli_value is not None:
        return cli_value

    for key in config_keys:
        if key in config and config.get(key) is not None:
            return config.get(key)

    return default_value


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = build_parser()
    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = get_logger(__name__)

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
        # Mặc định: prompts/gemini.txt trong project root
        prompt_file = str(PROJECT_ROOT / "prompts" / "gemini.txt")
        if not Path(prompt_file).exists():
            parser.error(
                f"Không tìm thấy gemini.txt tại: {prompt_file}\n"
                "Hãy đặt gemini.txt trong thư mục prompts/ "
                "hoặc dùng --prompt <đường_dẫn>"
            )

    # Parse Provider Config & Keys
    provider_type = args.provider
    config_path = args.provider_config
    base_url = args.base_url

    secrets = {}
    if provider_type in ["gemini", "openai"]:
        if not args.keys:
            parser.error(f"--keys là bắt buộc đối với provider {provider_type}")
        if provider_type == "gemini":
            secrets["api_keys"] = [k.strip() for k in args.keys.split(",") if k.strip()]
            if not secrets["api_keys"]:
                parser.error("Không có API key hợp lệ nào cho Gemini")
        else:
            secrets["api_key"] = args.keys.strip()

    provider_config = {}
    if provider_type == "openai":
        config_path = config_path or "config/openai_compat_translate.yaml"
    elif provider_type == "vertexai":
        config_path = config_path or "config/vertexai_translate.yaml"

    if config_path:
        from translation.factory import load_provider_config
        try:
            provider_config = load_provider_config(config_path)
        except Exception as e:
            logger.warning(f"Không thể tải config {config_path}: {e}")

    if base_url and provider_type == "openai":
        provider_config["base_url"] = base_url

    default_model_by_provider = {
        "gemini": "gemini-3-flash-preview",
        "openai": "gpt-5.4",
        "vertexai": "gemini-3-flash-preview",
    }

    provider_config["model"] = resolve_by_priority(
        args.model,
        provider_config,
        ["model"],
        default_model_by_provider.get(provider_type, "gemini-3-flash-preview"),
    )

    if provider_type == "gemini":
        provider_config["thinking_budget"] = resolve_by_priority(
            args.budget,
            provider_config,
            ["thinking_budget", "budget"],
            24576,
        )

    batch_size = resolve_by_priority(args.batch, provider_config, ["batch", "batch_size"], 30)
    wait_sec = resolve_by_priority(args.wait, provider_config, ["wait", "wait_sec"], 0.0)
    use_full_context = resolve_by_priority(
        args.context,
        provider_config,
        ["use_full_context", "full_context"],
        True,
    )

    provider = create_provider(provider_type, provider_config, secrets)

    # In tóm tắt cấu hình
    print("=" * 55)
    print("  🎬  SRT Translator — Multi-Provider")
    print("=" * 55)
    print(f"  Input    : {args.input}")
    print(f"  Output   : {output_file}")
    print(f"  Provider : {provider.name}")
    print(f"  Lang     : {args.lang}")
    print(f"  Batch    : {batch_size}")
    print(f"  Context  : {'ON' if use_full_context else 'OFF'}")
    print("=" * 55)

    # Gọi pipeline dịch
    try:
        stats = translate_srt_file(
            input_file      = str(input_path),
            output_file     = output_file,
            prompt_file     = prompt_file,
            provider        = provider,
            target_language = args.lang,
            batch_size      = batch_size,
            use_full_context= use_full_context,
            wait_sec        = wait_sec,
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
