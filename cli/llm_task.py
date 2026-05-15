#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli/llm_task.py
================
Generic LLM text task runner.

Cấu trúc layers khi test:
  Layer 1 — Unit Tests          (prompt render, response parser)
  Layer 2 — Component Tests     (generic task với mocked provider)
  Layer 3 — Integration         (CLI/config wiring, không gọi API thật)
  Layer 4 — Real API Tests      (tuỳ chọn, đọc API key từ environment)
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm_ai.factory import create_provider, load_provider_config  # noqa: E402
from llm_ai.tasks.generic_text_task import (  # noqa: E402
    GenericTextTaskConfig,
    load_task_config,
    run_generic_text_task,
)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-task",
        description="Chạy generic LLM task bằng prompt template + input text + provider config.",
    )

    parser.add_argument("--input", "-i", default=None, metavar="FILE", help="File text input.")
    parser.add_argument(
        "--task-file",
        "-t",
        default=None,
        metavar="JSON_FILE",
        help="File JSON chứa danh sách task [{'input': '...', 'output': '...'}].",
    )
    parser.add_argument("--output", "-o", default=None, metavar="FILE_OR_DIR", help="Output file hoặc thư mục.")

    parser.add_argument(
        "--task-config",
        default=str(PROJECT_ROOT / "config/llm_tasks/seo_metadata.yaml"),
        metavar="YAML",
        help="Config YAML cho generic LLM task.",
    )
    parser.add_argument("--prompt", default=None, metavar="FILE", help="Override prompt_file trong task config.")
    parser.add_argument(
        "--placeholder",
        default=None,
        metavar="TEXT",
        help="Override input_placeholder trong task config, ví dụ: [Video Content].",
    )
    parser.add_argument(
        "--parser",
        default=None,
        choices=["raw", "markdown", "tag", "json"],
        help="Override output_parser trong task config.",
    )

    parser.add_argument(
        "--provider",
        "-p",
        default=None,
        choices=["gemini", "openai", "vertexai"],
        help="Provider LLM. Mặc định lấy từ task config hoặc openai.",
    )
    parser.add_argument(
        "--provider-config",
        default=None,
        metavar="YAML",
        help="Config YAML cho provider LLM.",
    )
    parser.add_argument("--base-url", default=None, metavar="URL", help="Override base_url cho OpenAI-compatible.")
    parser.add_argument("--keys", "-k", default=None, metavar="KEY[,KEY2]", help="API key(s).")
    parser.add_argument("--model", "-m", default=None, metavar="MODEL", help="Override model name.")
    parser.add_argument("--system-prompt", default=None, metavar="TEXT", help="Override system prompt.")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max output tokens.")
    parser.add_argument("--request-timeout", type=int, default=None, help="Override request timeout, đơn vị giây.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Bật logging debug chi tiết.")

    return parser


def _resolve_provider_config_path(args: argparse.Namespace, task_cfg: dict[str, Any], provider_type: str) -> Path:
    raw_path = (
        args.provider_config
        or task_cfg.get("provider_config")
        or DEFAULT_PROVIDER_CONFIGS.get(provider_type)
    )
    resolved = _resolve_project_path(str(raw_path) if raw_path else None)
    if not resolved:
        raise ValueError(f"Không xác định được provider config cho provider: {provider_type}")
    return resolved


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


def _apply_provider_overrides(provider_config: dict[str, Any], args: argparse.Namespace, task_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(provider_config)

    if task_cfg.get("system_prompt") is not None:
        cfg["system_prompt"] = task_cfg.get("system_prompt")

    if args.base_url:
        cfg["base_url"] = args.base_url
    if args.model:
        cfg["model"] = args.model
    if args.system_prompt is not None:
        cfg["system_prompt"] = args.system_prompt
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if args.max_tokens is not None:
        cfg["max_tokens"] = args.max_tokens
        cfg["max_output_tokens"] = args.max_tokens
    if args.request_timeout is not None:
        cfg["request_timeout"] = args.request_timeout

    return cfg


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = get_logger(__name__)

    try:
        task_config_path = _resolve_project_path(args.task_config)
        task_cfg = load_task_config(str(task_config_path))

        if args.prompt:
            task_cfg["prompt_file"] = str(_resolve_project_path(args.prompt))
        else:
            task_cfg["prompt_file"] = str(_resolve_project_path(task_cfg.get("prompt_file")))

        if args.placeholder:
            task_cfg["input_placeholder"] = args.placeholder
        if args.parser:
            task_cfg["output_parser"] = args.parser

        generic_cfg = GenericTextTaskConfig.from_dict(task_cfg)

        provider_type = (args.provider or task_cfg.get("provider") or "openai").lower().strip()
        provider_config_path = _resolve_provider_config_path(args, task_cfg, provider_type)
        provider_config = load_provider_config(str(provider_config_path))
        provider_config = _apply_provider_overrides(provider_config, args, task_cfg)
        secrets = _build_secrets(provider_type, args.keys)
        provider = create_provider(provider_type, provider_config, secrets)

        tasks = resolve_cli_tasks(
            task_file=args.task_file,
            input_file=args.input,
            output_path=args.output,
            default_ext=generic_cfg.default_ext,
        )
    except Exception as exc:
        parser.error(str(exc))
        return

    ok_tasks = 0
    for task in tasks:
        input_file = Path(task["input"])
        if not input_file.exists():
            logger.error(f"File input không tồn tại: {input_file}")
            continue

        output_file = task["output"]
        print("=" * 55)
        print("  🧠  Generic LLM Task")
        print("=" * 55)
        print(f"  Task     : {generic_cfg.task_name}")
        print(f"  Input    : {input_file}")
        print(f"  Output   : {output_file}")
        print(f"  Provider : {provider.name}")
        print("=" * 55)

        try:
            task_prompt_file = task.get("prompt_file")
            task_placeholder = task.get("input_placeholder") or task.get("placeholder")
            stats = run_generic_text_task(
                input_file=str(input_file),
                output_file=output_file,
                provider=provider,
                task_config=generic_cfg,
                prompt_file=str(_resolve_project_path(task_prompt_file)) if task_prompt_file else None,
                input_placeholder=task_placeholder,
            )
            ok_tasks += 1
            print(f"✅ Hoàn thành: {stats['output']} ({stats['output_chars']} ký tự)")
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
