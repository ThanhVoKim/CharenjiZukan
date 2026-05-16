#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli/sync_video_llm_metadata.py
==============================
Post-render LLM metadata helper cho `cli/sync_video.py`.

Schema `render_config.json` hỗ trợ:

```json
{
  "llm_metadata": {
    "enabled": true,
    "task_config": "config/llm_tasks/seo_metadata.yaml",
    "input": {
      "write_debug_input": true,
      "debug_input_filename_template": "{video_stem}_metadata_input.txt"
    },
    "output": {
      "directory_policy": "/",
      "filename_template": "{video_stem}_metadata.md"
    },
    "fail_policy": "warn"
  }
}
```

Ý nghĩa schema:
- `enabled`: bật/tắt bước tạo metadata sau khi render final video.
- `task_config`: YAML config cho generic LLM task, ví dụ `config/llm_tasks/seo_metadata.yaml`.
- `input.write_debug_input`: nếu true, ghi raw text đã đưa vào LLM ra file `.txt` để kiểm tra prompt input.
- `input.debug_input_filename_template`: tên file debug input. Template hỗ trợ `{video_stem}`, `{video_name}`, `{video_suffix}`, `{output_name}`.
- `output.directory_policy`: policy `/` nghĩa là thư mục chứa input video, không phải filesystem root. Ví dụ input `content/a/b.mp4` -> output dir `content/a/`.
- `output.filename_template`: tên metadata output. Template hỗ trợ `{video_stem}`, `{video_name}`, `{video_suffix}`, `{output_name}`.
- `fail_policy`: `warn` log warning và không làm fail pipeline; `raise`/`error`/`fail` sẽ raise lỗi LLM.

Input LLM luôn là raw text phẳng được gom từ toàn bộ `segment["text"]` của subtitle SRT đã parse:
không timestamp, không line number, không giữ line break theo từng block.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("sync_video")


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


def resolve_input_video_path(video_path: str) -> Path:
    path = Path(video_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def format_llm_metadata_template(template: str, video_path: Path, output_name: Optional[str] = None) -> str:
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
    return policy_path if policy_path.is_absolute() else PROJECT_ROOT / policy_path


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


def build_llm_metadata_input_text(subtitle_segments: list[dict]) -> str:
    """Gom toàn bộ subtitle text thành raw text phẳng cho LLM.

    Không thêm timestamp, không thêm line number, không giữ line break theo block.
    """
    chunks = []
    for segment in subtitle_segments:
        text = " ".join(str(segment.get("text", "")).split())
        if text:
            chunks.append(text)
    return " ".join(chunks).strip()


def build_llm_provider_args(metadata_cfg: dict[str, Any]) -> argparse.Namespace:
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


def execute_llm_metadata_task(
    *,
    metadata_cfg: dict[str, Any],
    subtitle_segments: list[dict],
    video_path: str,
    tmp_dir: str,
    output_name: Optional[str] = None,
) -> dict[str, Any]:
    raw_text = build_llm_metadata_input_text(subtitle_segments)
    if not raw_text:
        raise ValueError("Không có subtitle text để đưa vào LLM metadata")

    output_path, debug_input_path = resolve_llm_metadata_paths(metadata_cfg, video_path, output_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if debug_input_path:
        debug_input_path.parent.mkdir(parents=True, exist_ok=True)
        debug_input_path.write_text(raw_text, encoding="utf-8")
        llm_input_path = debug_input_path
        logger.info(f"Đã ghi debug input cho LLM metadata: {debug_input_path}")
    else:
        llm_input_path = Path(tmp_dir) / "llm_metadata_input.txt"
        llm_input_path.write_text(raw_text, encoding="utf-8")

    from cli.llm_task import _create_task_provider, _resolve_project_path
    from llm_ai.tasks.generic_text_task import (
        GenericTextTaskConfig,
        load_task_config,
        run_generic_text_task,
    )

    task_config = metadata_cfg.get("task_config", "config/llm_tasks/seo_metadata.yaml")
    task_config_path = _resolve_project_path(str(task_config))
    task_cfg = load_task_config(str(task_config_path))
    task_cfg["prompt_file"] = str(_resolve_project_path(task_cfg.get("prompt_file")))

    generic_cfg = GenericTextTaskConfig.from_dict(task_cfg)
    provider_args = build_llm_provider_args(metadata_cfg)
    provider = _create_task_provider(provider_args, task_cfg)

    logger.info(f"Đang tạo LLM metadata: input={llm_input_path}, output={output_path}, provider={provider.name}")
    stats = run_generic_text_task(
        input_file=str(llm_input_path),
        output_file=str(output_path),
        provider=provider,
        task_config=generic_cfg,
    )
    logger.info(f"Đã tạo LLM metadata: {output_path} ({stats['output_chars']} ký tự)")
    return stats


def run_llm_metadata_task(
    *,
    subtitle_segments: list[dict],
    render_config: dict[str, Any],
    video_path: str,
    tmp_dir: str,
    output_name: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    metadata_cfg = render_config.get("llm_metadata", {}) or {}
    if not metadata_cfg.get("enabled", False):
        logger.info("LLM metadata không được bật trong render_config. Bỏ qua.")
        return None

    logger.info("\n--- PHASE 6: LLM METADATA (POST-RENDER) ---")
    fail_policy = str(metadata_cfg.get("fail_policy", "warn")).lower().strip()
    try:
        return execute_llm_metadata_task(
            metadata_cfg=metadata_cfg,
            subtitle_segments=subtitle_segments,
            video_path=video_path,
            tmp_dir=tmp_dir,
            output_name=output_name,
        )
    except Exception as exc:
        if fail_policy in {"raise", "error", "fail"}:
            raise
        logger.warning(f"LLM metadata thất bại ({exc}). Bỏ qua do fail_policy={fail_policy}.", exc_info=True)
        return None
