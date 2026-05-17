#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_ai/task_runner.py
=====================
Shared LLM provider creation logic, extracted from cli/llm_task.py.

Cung cấp các hàm public để tạo provider LLM từ config và argparse.Namespace.
Được dùng bởi cả `cli/llm_task.py` (CLI entrypoint) và `sync_engine/llm_metadata.py`
(post-render metadata task).

Tất cả các hàm trong module này đều là public API, không có prefix `_`.
"""

import argparse
import os
from pathlib import Path
from typing import Any

from llm_ai.base import BaseLLMProvider
from llm_ai.factory import create_provider, load_provider_config
from llm_ai.provider_chain import (
    FallbackLLMProvider,
    apply_provider_chain_entry_overrides,
    normalize_provider_chain,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PROVIDER_CONFIGS = {
    "gemini": "config/llm/gemini.yaml",
    "openai": "config/llm/openai_compat.yaml",
    "vertexai": "config/llm/vertexai.yaml",
}


def resolve_project_path(path_value: str | None) -> Path | None:
    """Resolve đường dẫn project-relative thành absolute."""
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_provider_config_path(
    args: argparse.Namespace,
    task_cfg: dict[str, Any],
    provider_type: str,
) -> Path:
    """Tìm đường dẫn file config YAML cho provider LLM."""
    raw_path = (
        args.provider_config
        or task_cfg.get("provider_config")
        or DEFAULT_PROVIDER_CONFIGS.get(provider_type)
    )
    resolved = resolve_project_path(str(raw_path) if raw_path else None)
    if not resolved:
        raise ValueError(f"Không xác định được provider config cho provider: {provider_type}")
    return resolved


def resolve_chain_provider_config_path(entry: dict[str, Any], provider_type: str) -> Path:
    """Tìm đường dẫn file config YAML cho một entry trong provider_chain."""
    raw_path = entry.get("provider_config") or DEFAULT_PROVIDER_CONFIGS.get(provider_type)
    resolved = resolve_project_path(str(raw_path) if raw_path else None)
    if not resolved:
        raise ValueError(f"Không xác định được provider config cho provider_chain entry: {entry}")
    return resolved


def build_secrets(provider_type: str, keys_arg: str | None) -> dict[str, Any]:
    """Xây dựng dict secrets (API key) cho provider LLM."""
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


def apply_provider_overrides(
    provider_config: dict[str, Any],
    args: argparse.Namespace,
    task_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Áp dụng CLI override vào provider config."""
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


def create_single_provider(args: argparse.Namespace, task_cfg: dict[str, Any]) -> BaseLLMProvider:
    """Tạo một provider LLM đơn (không qua provider_chain)."""
    provider_type = (args.provider or task_cfg.get("provider") or "openai").lower().strip()
    provider_config_path = resolve_provider_config_path(args, task_cfg, provider_type)
    provider_config = load_provider_config(str(provider_config_path))
    provider_config = apply_provider_overrides(provider_config, args, task_cfg)
    secrets = build_secrets(provider_type, args.keys)
    return create_provider(provider_type, provider_config, secrets)


def create_provider_from_chain_entry(
    entry: dict[str, Any],
    args: argparse.Namespace,
    task_cfg: dict[str, Any],
) -> BaseLLMProvider:
    """Tạo một provider LLM từ một entry trong provider_chain."""
    provider_type = entry["provider"]
    provider_config_path = resolve_chain_provider_config_path(entry, provider_type)
    provider_config = load_provider_config(str(provider_config_path))
    provider_config = apply_provider_chain_entry_overrides(
        provider_config,
        entry,
        task_system_prompt=task_cfg.get("system_prompt"),
    )

    if args.system_prompt is not None:
        provider_config["system_prompt"] = args.system_prompt

    secrets = build_secrets(provider_type, args.keys)
    return create_provider(provider_type, provider_config, secrets)


def create_task_provider(args: argparse.Namespace, task_cfg: dict[str, Any]) -> BaseLLMProvider:
    """Tạo provider LLM cho một task, hỗ trợ cả provider đơn và provider_chain.

    Nếu task_cfg có `provider_chain` và không có CLI override (--provider, --provider-config),
    tạo FallbackLLMProvider với danh sách provider dự phòng.
    Ngược lại, tạo single provider như bình thường.
    """
    raw_chain = normalize_provider_chain(task_cfg.get("provider_chain"))
    if args.provider or args.provider_config or not raw_chain:
        return create_single_provider(args, task_cfg)

    factories = [
        (lambda entry=entry: create_provider_from_chain_entry(entry, args, task_cfg))
        for entry in raw_chain
    ]
    names = [str(entry.get("name") or entry["provider"]) for entry in raw_chain]
    return FallbackLLMProvider(factories, names)
