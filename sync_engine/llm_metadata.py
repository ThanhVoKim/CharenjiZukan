#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sync_engine/llm_metadata.py
===========================
Post-render LLM metadata helper cho `cli/sync_video.py`.

Module này nằm trong `sync_engine/` vì nó phụ thuộc vào `llm_ai` (provider, task runner)
nhưng không phụ thuộc ngược vào `cli/`. `cli/sync_video.py` import module này như một
thư viện domain.

Schema `render_config.json` — xem `docs/sync-video-guide.md`.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("sync_video")


# ═════════════════════════════════════════════════════════════════════
# Config helpers
# ═════════════════════════════════════════════════════════════════════

def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge đệ quy 2 dict, dùng cho override cấu hình task-file."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_llm_metadata_override(render_config: dict[str, Any], override: Any) -> dict[str, Any]:
    """Áp dụng override `llm_metadata` từ task JSON nếu có."""
    if override is None:
        return render_config
    if isinstance(override, bool):
        override = {"enabled": override}
    if not isinstance(override, dict):
        raise ValueError("llm_metadata override phải là object hoặc boolean trong task JSON")
    return deep_merge_dict(render_config, {"llm_metadata": override})


# ═════════════════════════════════════════════════════════════════════
# Path resolution
# ═════════════════════════════════════════════════════════════════════

def resolve_input_video_path(video_path: str) -> Path:
    """Resolve đường dẫn video input thành absolute path."""
    path = Path(video_path)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def format_llm_metadata_template(template: str, video_path: Path, output_name: Optional[str] = None) -> str:
    """Format template string với các biến từ video path."""
    values = {
        "video_stem": video_path.stem,
        "video_name": video_path.name,
        "video_suffix": video_path.suffix,
        "output_name": output_name or video_path.stem,
    }
    return str(template).format(**values)


def resolve_llm_metadata_output_dir(video_path: str, directory_policy: str) -> Path:
    """Resolve output dir cho LLM metadata.

    Policy `/` trong schema không phải filesystem root; nó nghĩa là thư mục
    chứa input video. Ví dụ `content/a/b.mp4` -> `content/a/`.
    """
    video_file = resolve_input_video_path(video_path)
    policy = str(directory_policy or "/").strip()
    if policy == "/":
        return video_file.parent

    policy_path = Path(policy)
    return policy_path if policy_path.is_absolute() else _PROJECT_ROOT / policy_path


def resolve_llm_metadata_paths(
    metadata_cfg: dict[str, Any],
    video_path: str,
    output_name: Optional[str] = None,
) -> tuple[Path, Optional[Path]]:
    """Tính đường dẫn output metadata và debug input theo schema render_config."""
    output_cfg = metadata_cfg.get("output", {}) or {}
    input_cfg = metadata_cfg.get("input", {}) or {}
    video_file = resolve_input_video_path(video_path)
    output_dir = resolve_llm_metadata_output_dir(
        video_path,
        output_cfg.get("directory_policy", "/"),
    )

    output_filename = format_llm_metadata_template(
        output_cfg.get("filename_template", "{video_stem}_metadata.md"),
        video_file,
        output_name,
    )
    output_path = output_dir / output_filename

    debug_input_path = None
    if input_cfg.get("write_debug_input", False):
        debug_filename = format_llm_metadata_template(
            input_cfg.get("debug_input_filename_template", "{video_stem}_metadata_input.txt"),
            video_file,
            output_name,
        )
        debug_input_path = output_dir / debug_filename

    return output_path, debug_input_path


# ═════════════════════════════════════════════════════════════════════
# Provider args builder
# ═════════════════════════════════════════════════════════════════════

def build_llm_provider_args(metadata_cfg: dict[str, Any]) -> argparse.Namespace:
    """Xây dựng argparse.Namespace từ metadata_cfg để truyền vào create_task_provider."""
    overrides = metadata_cfg.get("provider_overrides", {}) or {}

    def pick(key: str) -> Any:
        if key in metadata_cfg:
            return metadata_cfg.get(key)
        return overrides.get(key)

    return argparse.Namespace(
        provider=pick("provider"),
        provider_config=pick("provider_config"),
        base_url=pick("base_url"),
        keys=pick("keys"),
        model=pick("model"),
        system_prompt=pick("system_prompt"),
        temperature=pick("temperature"),
        max_tokens=pick("max_tokens"),
        request_timeout=pick("request_timeout"),
    )


# ═════════════════════════════════════════════════════════════════════
# Core execution
# ═════════════════════════════════════════════════════════════════════

def execute_llm_metadata_task(
    *,
    metadata_cfg: dict[str, Any],
    input_text_path: str,
    video_path: str,
    output_name: Optional[str] = None,
) -> dict[str, Any]:
    """Thực thi LLM metadata task từ file input text đã được chuẩn bị sẵn.

    Args:
        metadata_cfg: Dict `llm_metadata` từ render_config.
        input_text_path: Đường dẫn tới file .txt chứa raw subtitle text.
        video_path: Đường dẫn video gốc (để resolve output path).
        output_name: Tên output (để format template).

    Returns:
        Dict stats từ run_generic_text_task.
    """
    input_path = Path(input_text_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input text cho LLM metadata không tồn tại: {input_path}")

    raw_text = input_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError("Input text cho LLM metadata rỗng")

    output_path, debug_input_path = resolve_llm_metadata_paths(metadata_cfg, video_path, output_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Nếu debug_input_path khác với input_path, copy nội dung sang debug path
    if debug_input_path and debug_input_path.resolve() != input_path.resolve():
        debug_input_path.parent.mkdir(parents=True, exist_ok=True)
        debug_input_path.write_text(raw_text, encoding="utf-8")
        logger.info(f"Đã ghi debug input cho LLM metadata: {debug_input_path}")

    from llm_ai.task_runner import create_task_provider, resolve_project_path
    from llm_ai.tasks.generic_text_task import (
        GenericTextTaskConfig,
        load_task_config,
        run_generic_text_task,
    )

    task_config = metadata_cfg.get("task_config", "config/llm_tasks/seo_metadata.yaml")
    task_config_path = resolve_project_path(str(task_config))
    task_cfg = load_task_config(str(task_config_path))
    task_cfg["prompt_file"] = str(resolve_project_path(task_cfg.get("prompt_file")))

    generic_cfg = GenericTextTaskConfig.from_dict(task_cfg)
    provider_args = build_llm_provider_args(metadata_cfg)
    provider = create_task_provider(provider_args, task_cfg)

    logger.info(f"Đang tạo LLM metadata: input={input_path}, output={output_path}, provider={provider.name}")
    stats = run_generic_text_task(
        input_file=str(input_path),
        output_file=str(output_path),
        provider=provider,
        task_config=generic_cfg,
    )
    logger.info(f"Đã tạo LLM metadata: {output_path} ({stats['output_chars']} ký tự)")
    return stats


def run_llm_metadata_task(
    *,
    input_text_path: str,
    render_config: dict[str, Any],
    video_path: str,
    output_name: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Entry point chính cho bước LLM metadata post-render.

    Được gọi từ `cli/sync_video.py` sau khi final render hoàn tất.
    Tự kiểm tra `enabled` và xử lý `fail_policy`.

    Args:
        input_text_path: Đường dẫn tới file .txt chứa raw subtitle text.
        render_config: Toàn bộ render config dict.
        video_path: Đường dẫn video gốc.
        output_name: Tên output (để format template).

    Returns:
        Dict stats nếu thành công, None nếu bị skip hoặc lỗi với fail_policy=warn.
    """
    metadata_cfg = render_config.get("llm_metadata", {}) or {}
    if not metadata_cfg.get("enabled", False):
        logger.info("LLM metadata không được bật trong render_config. Bỏ qua.")
        return None

    logger.info("\n--- PHASE 6: LLM METADATA (POST-RENDER) ---")
    fail_policy = str(metadata_cfg.get("fail_policy", "warn")).lower().strip()
    try:
        return execute_llm_metadata_task(
            metadata_cfg=metadata_cfg,
            input_text_path=input_text_path,
            video_path=video_path,
            output_name=output_name,
        )
    except Exception as exc:
        if fail_policy in {"raise", "error", "fail"}:
            raise
        logger.warning(f"LLM metadata thất bại ({exc}). Bỏ qua do fail_policy={fail_policy}.", exc_info=True)
        return None
