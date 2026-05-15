import json
import re
from typing import Any


def strip_think_blocks(result: str) -> str:
    return re.sub(r"<think>.*?</think>", "", result, flags=re.I | re.S)


def parse_raw_response(result: str) -> str:
    return strip_think_blocks(result).strip()


def parse_tag_response(result: str, tag: str) -> str:
    cleaned = strip_think_blocks(result)
    tag_name = re.escape(tag)
    match = re.search(
        rf"<{tag_name}>(.*?)(?:</{tag_name}>|$)",
        cleaned,
        re.S | re.I,
    )
    if match:
        return match.group(1).strip()
    raise RuntimeError(
        f"Không tìm thấy <{tag}> trong response. "
        f"Model có thể không follow prompt format. "
        f"Preview: {result[:200]}..."
    )


def _strip_markdown_json_fence(text: str) -> str:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.I | re.S)
    if fence:
        return fence.group(1).strip()
    return cleaned


def parse_json_response(result: str) -> str:
    cleaned = _strip_markdown_json_fence(parse_raw_response(result))
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Không parse được JSON từ response: {exc}") from exc
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def parse_task_response(result: str, parser_config: str | dict[str, Any] | None = None) -> str:
    if parser_config is None:
        return parse_raw_response(result)

    if isinstance(parser_config, str):
        parser_type = parser_config
        parser_options: dict[str, Any] = {}
    else:
        parser_type = str(parser_config.get("type") or parser_config.get("parser") or "raw")
        parser_options = parser_config

    parser_type = parser_type.lower().strip()
    if parser_type in {"raw", "markdown", "md", "text", "txt"}:
        return parse_raw_response(result)
    if parser_type == "tag":
        tag = parser_options.get("tag")
        if not tag:
            raise ValueError("Parser tag yêu cầu cấu hình 'tag'")
        return parse_tag_response(result, str(tag))
    if parser_type == "json":
        return parse_json_response(result)

    raise ValueError(f"Output parser không hỗ trợ: {parser_type}")
