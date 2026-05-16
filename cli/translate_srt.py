#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translate_srt.py — CLI: Dịch file .srt bằng multi-provider LLM

Dịch subtitle sang ngôn ngữ đích sử dụng provider từ llm_ai.
Hỗ trợ batch processing, full-context, task-file và provider_chain fallback.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────
# Add project root to path for imports
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm_ai.base import BaseLLMProvider  # noqa: E402
from llm_ai.factory import create_provider, load_provider_config  # noqa: E402
from llm_ai.provider_chain import (  # noqa: E402
    FallbackLLMProvider,
    apply_provider_chain_entry_overrides,
    normalize_provider_chain,
)
from llm_ai.tasks.generic_text_task import load_task_config  # noqa: E402
from translation.srt_translator import translate_srt_file  # noqa: E402
from utils.logger import setup_logging, get_logger  # noqa: E402
from utils.task_utils import resolve_cli_tasks  # noqa: E402

DEFAULT_PROVIDER_CONFIGS = {
    "gemini": "config/llm/gemini.yaml",
    "openai": "config/llm/openai_compat.yaml",
    "vertexai": "config/llm/vertexai.yaml",
}


def _resolve_project_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ─────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="translate_srt",
        description="Dịch file .srt bằng multi-provider LLM (llm_ai)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ nhanh:
  python translate_srt.py --input video.srt --keys "AIzaSy..."

Ví dụ đầy đủ:
  python translate_srt.py \
      --input   /content/video.srt         \
      --output  /content/video_vi.srt      \
      --lang    "Vietnamese"               \
      --provider gemini                    \
      --keys    "AIza...k1,AIza...k2"      \
      --model   gemini-3-flash-preview     \
      --batch   30                         \
      --budget  8192
        """,
    )

    parser.add_argument(
        "--input", "-i",
        default=None,
        metavar="FILE",
        help="Đường dẫn file SRT gốc (bắt buộc nếu không dùng --task-file)",
    )
    parser.add_argument(
        "--task-file", "-t",
        default=None,
        metavar="JSON_FILE",
        help="File JSON chứa danh sách [{'input': '...', 'output': '...'}]",
    )

    parser.add_argument(
        "--provider", "-p",
        default=None,
        choices=["gemini", "openai", "vertexai"],
        metavar="PROVIDER",
        help="LLM provider. Nếu task config có provider_chain và không truyền --provider, sẽ dùng provider_chain.",
    )
    parser.add_argument(
        "--provider-config",
        default=None,
        metavar="FILE",
        help=(
            "Đường dẫn tới provider YAML. Mặc định: "
            "config/llm/gemini.yaml, config/llm/openai_compat.yaml hoặc config/llm/vertexai.yaml. "
            "Nếu truyền flag này sẽ dùng single-provider mode thay cho provider_chain."
        ),
    )
    parser.add_argument(
        "--task-config",
        default=str(PROJECT_ROOT / "config/llm_tasks/srt_translation.yaml"),
        metavar="FILE",
        help="Đường dẫn tới task YAML của SRT translation.",
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
        help=(
            "API key(s). Có thể bỏ qua nếu đã set GEMINI_API_KEY hoặc OPENAI_API_KEY. "
            "Nhiều key cách nhau dấu phẩy đối với gemini."
        ),
    )

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
        help="Model name cho single-provider mode (ưu tiên: CLI > config > default provider)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        metavar="FILE",
        help="Đường dẫn prompt dịch SRT (mặc định lấy từ config/llm_tasks/srt_translation.yaml)",
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=None,
        metavar="N",
        help="Số SRT block mỗi batch (ưu tiên: CLI > task config > mặc định 30)",
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
        help="Bật/tắt full-context (ưu tiên: CLI > task config > mặc định bật). Dùng --no-context để tắt.",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=None,
        metavar="SEC",
        help="Giây chờ giữa mỗi batch (ưu tiên: CLI > task config > mặc định 0)",
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


def _resolve_provider_config_path(provider_type: str, config_path: str | None) -> Path | None:
    raw_path = config_path or DEFAULT_PROVIDER_CONFIGS.get(provider_type)
    return _resolve_project_path(raw_path)


def _resolve_chain_provider_config_path(entry: dict[str, Any], provider_type: str) -> Path | None:
    raw_path = entry.get("provider_config") or DEFAULT_PROVIDER_CONFIGS.get(provider_type)
    return _resolve_project_path(str(raw_path) if raw_path else None)


def _load_translation_task_config(config_path: str | None) -> dict[str, Any]:
    resolved = _resolve_project_path(config_path)
    if not resolved or not resolved.exists():
        return {}
    return load_task_config(str(resolved))


def _build_secrets(provider_type: str, keys_arg: str | None) -> dict[str, Any]:
    secrets: dict[str, Any] = {}

    if provider_type == "gemini":
        raw_keys = keys_arg or os.getenv("GEMINI_API_KEY", "")
        api_keys = [key.strip() for key in raw_keys.split(",") if key.strip()]
        if not api_keys:
            raise ValueError("--keys hoặc GEMINI_API_KEY là bắt buộc đối với provider gemini")
        secrets["api_keys"] = api_keys
    elif provider_type == "openai":
        api_key = keys_arg or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("--keys hoặc OPENAI_API_KEY là bắt buộc đối với provider openai")
        secrets["api_key"] = api_key.strip()

    return secrets


def _load_provider_config(provider_type: str, config_path: str | None, logger: logging.Logger) -> dict[str, Any]:
    resolved_config_path = _resolve_provider_config_path(provider_type, config_path)
    if not resolved_config_path:
        return {}

    try:
        return load_provider_config(str(resolved_config_path))
    except Exception as exc:
        logger.warning(f"Không thể tải config {resolved_config_path}: {exc}")
        return {}


def _apply_single_provider_overrides(
    provider_type: str,
    provider_config: dict[str, Any],
    args: argparse.Namespace,
    task_config: dict[str, Any],
) -> dict[str, Any]:
    cfg = dict(provider_config)

    if args.base_url and provider_type == "openai":
        cfg["base_url"] = args.base_url

    task_system_prompt = task_config.get("system_prompt")
    if task_system_prompt:
        cfg["system_prompt"] = task_system_prompt

    default_model_by_provider = {
        "gemini": "gemini-3-flash-preview",
        "openai": "gpt-5.4",
        "vertexai": "gemini-3-flash-preview",
    }

    cfg["model"] = resolve_by_priority(
        args.model,
        cfg,
        ["model"],
        default_model_by_provider.get(provider_type, "gemini-3-flash-preview"),
    )

    if provider_type == "gemini":
        cfg["thinking_budget"] = resolve_by_priority(
            args.budget,
            cfg,
            ["thinking_budget", "budget"],
            24576,
        )

    return cfg


def _create_single_provider(
    provider_type: str,
    args: argparse.Namespace,
    task_config: dict[str, Any],
    logger: logging.Logger,
) -> BaseLLMProvider:
    provider_config = _load_provider_config(provider_type, args.provider_config, logger)
    provider_config = _apply_single_provider_overrides(provider_type, provider_config, args, task_config)
    secrets = _build_secrets(provider_type, args.keys)
    return create_provider(provider_type, provider_config, secrets)


def _create_provider_from_chain_entry(
    entry: dict[str, Any],
    args: argparse.Namespace,
    task_config: dict[str, Any],
    logger: logging.Logger,
) -> BaseLLMProvider:
    provider_type = entry["provider"]
    resolved_config_path = _resolve_chain_provider_config_path(entry, provider_type)
    if not resolved_config_path:
        raise ValueError(f"Không xác định được provider_config cho provider_chain entry: {entry}")

    try:
        provider_config = load_provider_config(str(resolved_config_path))
    except Exception as exc:
        logger.warning(f"Không thể tải provider_chain config {resolved_config_path}: {exc}")
        provider_config = {}

    provider_config = apply_provider_chain_entry_overrides(
        provider_config,
        entry,
        task_system_prompt=task_config.get("system_prompt"),
    )
    secrets = _build_secrets(provider_type, args.keys)
    return create_provider(provider_type, provider_config, secrets)


def _create_task_provider(
    args: argparse.Namespace,
    task_config: dict[str, Any],
    logger: logging.Logger,
) -> BaseLLMProvider:
    raw_chain = normalize_provider_chain(task_config.get("provider_chain"))
    if args.provider or args.provider_config or not raw_chain:
        provider_type = (args.provider or task_config.get("provider") or "gemini").lower().strip()
        return _create_single_provider(provider_type, args, task_config, logger)

    factories = [
        (lambda entry=entry: _create_provider_from_chain_entry(entry, args, task_config, logger))
        for entry in raw_chain
    ]
    names = [str(entry.get("name") or entry["provider"]) for entry in raw_chain]
    return FallbackLLMProvider(factories, names)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = get_logger(__name__)

    task_config = _load_translation_task_config(args.task_config)

    lang_slug = args.lang.lower().replace(" ", "_")
    try:
        tasks = resolve_cli_tasks(
            task_file=args.task_file,
            input_file=args.input,
            output_path=args.output,
            default_ext=f"_{lang_slug}.srt",
        )
    except ValueError as exc:
        parser.error(str(exc))

    for task in tasks:
        inp = Path(task["input"])
        if inp.suffix.lower() != ".srt":
            parser.error(f"File phải có đuôi .srt: {task['input']}")

    if args.prompt:
        prompt_path = _resolve_project_path(args.prompt)
    else:
        prompt_path = _resolve_project_path(
            task_config.get("prompt_file") or "prompts/translation/srt_translate.txt"
        )

    if not prompt_path or not prompt_path.exists():
        parser.error(
            f"Prompt file không tồn tại: {prompt_path}. "
            "Hãy dùng --prompt <đường_dẫn> hoặc cập nhật task config."
        )
    prompt_file = str(prompt_path)

    try:
        provider = _create_task_provider(args, task_config, logger)
    except ValueError as exc:
        parser.error(str(exc))

    batch_size = resolve_by_priority(args.batch, task_config, ["batch", "batch_size"], 30)
    wait_sec = resolve_by_priority(args.wait, task_config, ["wait", "wait_sec"], 0.0)
    use_full_context = resolve_by_priority(
        args.context,
        task_config,
        ["use_full_context", "full_context"],
        True,
    )

    ok_tasks = 0
    for task in tasks:
        input_file = task["input"]
        output_file = task["output"]

        print("=" * 55)
        print("  🎬  SRT Translator — llm_ai Multi-Provider")
        print("=" * 55)
        print(f"  Input    : {input_file}")
        print(f"  Output   : {output_file}")
        print(f"  Provider : {provider.name}")
        print(f"  Lang     : {args.lang}")
        print(f"  Batch    : {batch_size}")
        print(f"  Context  : {'ON' if use_full_context else 'OFF'}")
        print("=" * 55)

        try:
            translate_srt_file(
                input_file=input_file,
                output_file=output_file,
                prompt_file=prompt_file,
                provider=provider,
                target_language=args.lang,
                batch_size=batch_size,
                use_full_context=use_full_context,
                wait_sec=wait_sec,
            )
            ok_tasks += 1
        except KeyboardInterrupt:
            print("\n⚠️  Đã dừng bởi người dùng")
            sys.exit(1)
        except Exception as exc:
            print(f"\n❌ Lỗi nghiêm trọng khi xử lý {input_file}: {exc}")
            logging.exception(exc)

    print(f"\n{'='*55}")
    print(f"  Tổng kết: {ok_tasks}/{len(tasks)} task thành công")
    print(f"{'='*55}")

    sys.exit(0 if ok_tasks == len(tasks) else 2)


if __name__ == "__main__":
    main()
