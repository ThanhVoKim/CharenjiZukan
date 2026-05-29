from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("llm_ai")

API_MODE_CHAT_COMPLETIONS = "chat_completions"
API_MODE_RESPONSES = "responses"

STRUCTURED_MODE_NONE = "none"
STRUCTURED_MODE_PROMPT_JSON = "prompt_json"
STRUCTURED_MODE_API_SCHEMA = "api_schema"
STRUCTURED_MODE_CHAT_RESPONSE_FORMAT = "chat_response_format"
STRUCTURED_MODE_RESPONSES_TEXT_FORMAT = "responses_text_format"

PROBE_STATUS_UNSUPPORTED = "unsupported"
PROBE_STATUS_ACCEPTED = "accepted"
PROBE_STATUS_VERIFIED = "verified"
PROBE_STATUS_SKIPPED = "skipped"
PROBE_STATUS_ERROR = "error"
VALID_PROBE_STATUSES = {
    PROBE_STATUS_UNSUPPORTED,
    PROBE_STATUS_ACCEPTED,
    PROBE_STATUS_VERIFIED,
    PROBE_STATUS_SKIPPED,
    PROBE_STATUS_ERROR,
}


class OpenAICompatCapabilityError(RuntimeError):
    """Base error cho capability gating của OpenAI-compatible provider."""

    def __init__(
        self,
        message: str,
        *,
        profile_name: str | None = None,
        feature: str | None = None,
        api_mode: str | None = None,
    ):
        self.profile_name = profile_name
        self.feature = feature
        self.api_mode = api_mode
        super().__init__(message)


class CapabilityNotEnabledError(OpenAICompatCapabilityError):
    """Raise khi request yêu cầu capability nhưng flag tương ứng đang tắt."""


class CapabilityRejectedError(OpenAICompatCapabilityError):
    """Raise khi endpoint thật reject tham số/endpoint đã được config bật."""


class CapabilityModeError(OpenAICompatCapabilityError):
    """Raise khi tính năng yêu cầu API mode khác với profile hiện tại."""


class CapabilityProbeRequiredError(OpenAICompatCapabilityError):
    """Raise khi capability high-risk chưa có probe metadata hợp lệ."""


@dataclass(frozen=True)
class StructuredOutputCapabilityFlags:
    supports_prompt_json: bool = True
    supports_chat_response_format: bool = False
    supports_responses_text_format: bool = False

    @classmethod
    def from_config(cls, raw: Any) -> "StructuredOutputCapabilityFlags":
        cfg = _mapping(raw)
        return cls(
            supports_prompt_json=_as_bool(cfg.get("supports_prompt_json"), True),
            supports_chat_response_format=_as_bool(cfg.get("supports_chat_response_format"), False),
            supports_responses_text_format=_as_bool(cfg.get("supports_responses_text_format"), False),
        )


@dataclass(frozen=True)
class OpenAICompatCapabilityFlags:
    supports_chat_completions: bool = True
    supports_responses_api: bool = False
    supports_reasoning_effort: bool = False
    supports_verbosity: bool = False
    supports_prompt_cache_key: bool = False
    supports_previous_response_state: bool = False
    supports_compaction: bool = False
    structured_output: StructuredOutputCapabilityFlags = field(
        default_factory=StructuredOutputCapabilityFlags
    )

    @classmethod
    def from_config(cls, raw: Any) -> "OpenAICompatCapabilityFlags":
        cfg = _mapping(raw)
        return cls(
            supports_chat_completions=_as_bool(cfg.get("supports_chat_completions"), True),
            supports_responses_api=_as_bool(cfg.get("supports_responses_api"), False),
            supports_reasoning_effort=_as_bool(cfg.get("supports_reasoning_effort"), False),
            supports_verbosity=_as_bool(cfg.get("supports_verbosity"), False),
            supports_prompt_cache_key=_as_bool(cfg.get("supports_prompt_cache_key"), False),
            supports_previous_response_state=_as_bool(
                cfg.get("supports_previous_response_state"), False
            ),
            supports_compaction=_as_bool(cfg.get("supports_compaction"), False),
            structured_output=StructuredOutputCapabilityFlags.from_config(
                cfg.get("structured_output", {})
            ),
        )


@dataclass(frozen=True)
class StructuredOutputOptions:
    mode: str = STRUCTURED_MODE_NONE
    schema_name: str | None = None
    schema: dict[str, Any] | None = None
    strict: bool = True

    @classmethod
    def from_config(cls, raw: Any) -> "StructuredOutputOptions":
        if isinstance(raw, str):
            return cls(mode=normalize_structured_output_mode(raw))
        cfg = _mapping(raw)
        schema = cfg.get("schema")
        if schema is not None and not isinstance(schema, dict):
            raise ValueError("request_options.structured_output.schema phải là object/dict hoặc null")
        return cls(
            mode=normalize_structured_output_mode(cfg.get("mode", STRUCTURED_MODE_NONE)),
            schema_name=_optional_str(cfg.get("schema_name")),
            schema=copy.deepcopy(schema) if schema is not None else None,
            strict=_as_bool(cfg.get("strict"), True),
        )


@dataclass(frozen=True)
class OpenAICompatRequestOptions:
    reasoning_effort: str | None = None
    verbosity: str | None = None
    prompt_cache_key: str | None = None
    structured_output: StructuredOutputOptions = field(default_factory=StructuredOutputOptions)

    @classmethod
    def from_config(cls, raw: Any) -> "OpenAICompatRequestOptions":
        cfg = _mapping(raw)
        return cls(
            reasoning_effort=_optional_str(cfg.get("reasoning_effort")),
            verbosity=_optional_str(cfg.get("verbosity")),
            prompt_cache_key=_optional_str(cfg.get("prompt_cache_key")),
            structured_output=StructuredOutputOptions.from_config(cfg.get("structured_output", {})),
        )


@dataclass(frozen=True)
class OpenAICompatStatefulOptions:
    store: bool = False
    use_previous_response_id: bool = False
    compact_threshold: int | None = None

    @classmethod
    def from_config(cls, raw: Any) -> "OpenAICompatStatefulOptions":
        cfg = _mapping(raw)
        return cls(
            store=_as_bool(cfg.get("store"), False),
            use_previous_response_id=_as_bool(cfg.get("use_previous_response_id"), False),
            compact_threshold=_optional_int(cfg.get("compact_threshold")),
        )


@dataclass(frozen=True)
class OpenAICompatTelemetryConfig:
    enabled: bool = False
    capture_usage: bool = True
    capture_cache_headers: bool = True
    capture_raw_headers: bool = False
    log_level: str = "summary"
    output_path: str = "logs/llm_telemetry.jsonl"

    @classmethod
    def from_config(cls, raw: Any) -> "OpenAICompatTelemetryConfig":
        cfg = _mapping(raw)
        return cls(
            enabled=_as_bool(cfg.get("enabled"), False),
            capture_usage=_as_bool(cfg.get("capture_usage"), True),
            capture_cache_headers=_as_bool(cfg.get("capture_cache_headers"), True),
            capture_raw_headers=_as_bool(cfg.get("capture_raw_headers"), False),
            log_level=str(cfg.get("log_level") or "summary"),
            output_path=str(cfg.get("output_path") or "logs/llm_telemetry.jsonl"),
        )


@dataclass(frozen=True)
class OpenAICompatProfile:
    provider: str = "openai"
    profile_name: str = "openai_compat_default"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_mode: str = API_MODE_CHAT_COMPLETIONS
    temperature: float = 1
    max_tokens: int = 8192
    capability_flags: OpenAICompatCapabilityFlags = field(default_factory=OpenAICompatCapabilityFlags)
    request_options: OpenAICompatRequestOptions = field(default_factory=OpenAICompatRequestOptions)
    stateful_options: OpenAICompatStatefulOptions = field(default_factory=OpenAICompatStatefulOptions)
    telemetry: OpenAICompatTelemetryConfig = field(default_factory=OpenAICompatTelemetryConfig)
    task_name: str | None = None

    @classmethod
    def from_config(cls, raw: Mapping[str, Any] | None) -> "OpenAICompatProfile":
        cfg = dict(raw or {})
        return cls(
            provider=str(cfg.get("provider") or "openai"),
            profile_name=sanitize_profile_name(cfg.get("profile_name") or "openai_compat_default"),
            base_url=str(cfg.get("base_url") or "https://api.openai.com/v1"),
            model=str(cfg.get("model") or "gpt-4o-mini"),
            api_mode=normalize_api_mode(cfg.get("api_mode") or API_MODE_CHAT_COMPLETIONS),
            temperature=float(cfg.get("temperature", 1)),
            max_tokens=int(cfg.get("max_tokens", cfg.get("max_output_tokens", 8192))),
            capability_flags=OpenAICompatCapabilityFlags.from_config(
                cfg.get("capability_flags", {})
            ),
            request_options=OpenAICompatRequestOptions.from_config(cfg.get("request_options", {})),
            stateful_options=OpenAICompatStatefulOptions.from_config(
                cfg.get("stateful_options", {})
            ),
            telemetry=OpenAICompatTelemetryConfig.from_config(cfg.get("telemetry", {})),
            task_name=_optional_str(cfg.get("task_name")),
        )


def normalize_api_mode(raw: Any) -> str:
    value = str(raw or API_MODE_CHAT_COMPLETIONS).strip().lower().replace("-", "_")
    if value in {"chat", "chat_completion", "chat_completions"}:
        return API_MODE_CHAT_COMPLETIONS
    if value in {"response", "responses", "responses_api"}:
        return API_MODE_RESPONSES
    raise ValueError(
        f"api_mode không hợp lệ: {raw!r}. Giá trị hợp lệ: chat_completions, responses"
    )


def normalize_structured_output_mode(raw: Any) -> str:
    value = str(raw or STRUCTURED_MODE_NONE).strip().lower().replace("-", "_")
    aliases = {
        "": STRUCTURED_MODE_NONE,
        "none": STRUCTURED_MODE_NONE,
        "off": STRUCTURED_MODE_NONE,
        "disabled": STRUCTURED_MODE_NONE,
        "prompt_json": STRUCTURED_MODE_PROMPT_JSON,
        "json_prompt": STRUCTURED_MODE_PROMPT_JSON,
        "prompt_based_json": STRUCTURED_MODE_PROMPT_JSON,
        "api_schema": STRUCTURED_MODE_API_SCHEMA,
        "json_schema": STRUCTURED_MODE_API_SCHEMA,
        "schema": STRUCTURED_MODE_API_SCHEMA,
        "api_enforced_schema": STRUCTURED_MODE_API_SCHEMA,
        "chat_response_format": STRUCTURED_MODE_CHAT_RESPONSE_FORMAT,
        "response_format": STRUCTURED_MODE_CHAT_RESPONSE_FORMAT,
        "responses_text_format": STRUCTURED_MODE_RESPONSES_TEXT_FORMAT,
        "text_format": STRUCTURED_MODE_RESPONSES_TEXT_FORMAT,
    }
    if value not in aliases:
        raise ValueError(
            "structured_output.mode không hợp lệ: "
            f"{raw!r}. Giá trị hợp lệ: none, prompt_json, api_schema, "
            "chat_response_format, responses_text_format"
        )
    return aliases[value]


def build_chat_messages(system_prompt: str | None, message: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": message})
    return messages


def build_responses_input(system_prompt: str | None, message: str) -> list[dict[str, str]]:
    return build_chat_messages(system_prompt, message)


def build_chat_completions_payload(
    profile: OpenAICompatProfile,
    system_prompt: str | None,
    message: str,
) -> dict[str, Any]:
    if profile.api_mode != API_MODE_CHAT_COMPLETIONS:
        _raise_mode_error(profile, "chat_completions", API_MODE_CHAT_COMPLETIONS)
    _require_capability(
        profile,
        profile.capability_flags.supports_chat_completions,
        "chat_completions_basic",
        "capability_flags.supports_chat_completions",
    )
    _ensure_chat_stateful_options_are_disabled(profile)

    payload: dict[str, Any] = {
        "model": profile.model,
        "messages": build_chat_messages(system_prompt, message),
        "temperature": profile.temperature,
        "max_tokens": profile.max_tokens,
    }
    _apply_common_chat_options(payload, profile)
    return payload


def build_responses_payload(
    profile: OpenAICompatProfile,
    system_prompt: str | None,
    message: str,
    *,
    previous_response_id: str | None = None,
) -> dict[str, Any]:
    if profile.api_mode != API_MODE_RESPONSES:
        _raise_mode_error(profile, "responses_api", API_MODE_RESPONSES)
    _require_capability(
        profile,
        profile.capability_flags.supports_responses_api,
        "responses_api",
        "capability_flags.supports_responses_api",
    )

    payload: dict[str, Any] = {
        "model": profile.model,
        "input": build_responses_input(system_prompt, message),
        "temperature": profile.temperature,
        "max_output_tokens": profile.max_tokens,
    }
    _apply_common_responses_options(payload, profile, previous_response_id=previous_response_id)
    return payload


def build_compaction_payload(profile: OpenAICompatProfile, response_id: str) -> dict[str, Any]:
    if profile.api_mode != API_MODE_RESPONSES:
        _raise_mode_error(profile, "compaction", API_MODE_RESPONSES)
    _require_capability(
        profile,
        profile.capability_flags.supports_compaction,
        "compaction",
        "capability_flags.supports_compaction",
    )
    if not response_id or not response_id.strip():
        raise ValueError("response_id là bắt buộc để compact Responses API state")

    payload: dict[str, Any] = {"response_id": response_id.strip()}
    if profile.stateful_options.compact_threshold is not None:
        payload["max_tokens"] = profile.stateful_options.compact_threshold
    return payload


def extract_chat_completion_text(response: Any) -> str:
    choices = _get_value(response, "choices", []) or []
    if not choices:
        return ""
    first = choices[0]
    message = _get_value(first, "message", {})
    content = _get_value(message, "content", "")
    return _content_to_text(content)


def extract_responses_text(response: Any) -> str:
    output_text = _get_value(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text

    output = _get_value(response, "output", None)
    if output:
        parts: list[str] = []
        for item in output:
            content = _get_value(item, "content", None)
            if not content:
                continue
            if isinstance(content, str):
                parts.append(content)
                continue
            for chunk in content:
                text = _get_value(chunk, "text", None) or _get_value(chunk, "output_text", None)
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)

    return extract_chat_completion_text(response)


def extract_response_id(response: Any) -> str | None:
    response_id = _get_value(response, "id", None)
    return response_id if isinstance(response_id, str) and response_id.strip() else None


def extract_usage_metrics(response: Any) -> dict[str, int | None]:
    usage = _get_value(response, "usage", None)
    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cached_tokens": None,
        }

    input_tokens = _first_int(usage, ("input_tokens", "prompt_tokens"))
    output_tokens = _first_int(usage, ("output_tokens", "completion_tokens"))
    total_tokens = _first_int(usage, ("total_tokens",))

    cached_tokens = None
    for details_name in (
        "prompt_tokens_details",
        "input_tokens_details",
        "input_token_details",
        "cache_creation_input_tokens_details",
    ):
        details = _get_value(usage, details_name, None)
        if details is None:
            continue
        cached_tokens = _first_int(
            details,
            (
                "cached_tokens",
                "cache_read_input_tokens",
                "cached_input_tokens",
            ),
        )
        if cached_tokens is not None:
            break

    if cached_tokens is None:
        cached_tokens = _first_int(usage, ("cached_tokens", "cache_read_input_tokens"))

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
    }


def normalize_headers(headers: Any) -> dict[str, str]:
    if not headers:
        return {}
    items: Any
    if isinstance(headers, Mapping):
        items = headers.items()
    elif hasattr(headers, "items"):
        items = headers.items()
    else:
        return {}

    normalized: dict[str, str] = {}
    for raw_key, raw_value in items:
        key = str(raw_key).strip().lower()
        if not key or key in {"authorization", "proxy-authorization", "api-key", "x-api-key"}:
            continue
        normalized[key] = str(raw_value)
    return normalized


def build_telemetry_record(
    profile: OpenAICompatProfile,
    response: Any,
    *,
    headers: Any = None,
    latency_ms: int | None = None,
    retry_count: int = 0,
    prompt_cache_key: str | None = None,
) -> dict[str, Any]:
    normalized_headers = normalize_headers(headers)
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "profile_name": profile.profile_name,
        "base_url_hash": hash_base_url(profile.base_url),
        "model": profile.model,
        "api_mode": profile.api_mode,
        "task_name": profile.task_name,
        "prompt_cache_key": prompt_cache_key or profile.request_options.prompt_cache_key,
        "latency_ms": latency_ms,
        "retry_count": retry_count,
    }

    if profile.telemetry.capture_cache_headers:
        cache_headers = {
            key: value
            for key, value in normalized_headers.items()
            if "cache" in key or key in {"x-request-id", "request-id", "openai-request-id"}
        }
        record["cache_status"] = _first_header_value(
            normalized_headers,
            (
                "x-cache-status",
                "x-prompt-cache-status",
                "openai-cache-status",
                "cf-cache-status",
            ),
        )
        if cache_headers:
            record["cache_headers"] = cache_headers

    if profile.telemetry.capture_usage:
        record.update(extract_usage_metrics(response))

    request_id = _first_header_value(
        normalized_headers,
        ("x-request-id", "request-id", "openai-request-id"),
    ) or _optional_str(_get_value(response, "_request_id", None))
    if request_id:
        record["request_id"] = request_id

    resolved_model = _optional_str(_get_value(response, "model", None))
    if resolved_model:
        record["resolved_model"] = resolved_model

    system_fingerprint = _optional_str(_get_value(response, "system_fingerprint", None))
    if system_fingerprint:
        record["system_fingerprint"] = system_fingerprint

    if profile.telemetry.capture_raw_headers and normalized_headers:
        record["headers"] = normalized_headers

    return {key: value for key, value in record.items() if value is not None}


def write_telemetry_record(profile: OpenAICompatProfile, record: Mapping[str, Any]) -> Path | None:
    if not profile.telemetry.enabled:
        return None
    try:
        output_path = _resolve_project_path(profile.telemetry.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
        return output_path
    except Exception as exc:  # pragma: no cover - best-effort logging only
        logger.warning("[OpenAICompatTelemetry] Không thể ghi telemetry: %s", exc)
        return None


def sanitize_profile_name(raw: Any) -> str:
    value = str(raw or "openai_compat_profile").strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._-")
    return value or "openai_compat_profile"


def hash_base_url(base_url: str | None) -> str:
    digest = hashlib.sha256(str(base_url or "").encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def build_capability_report(
    *,
    profile_name: str,
    base_url: str,
    model: str,
    capabilities: Mapping[str, Any],
    telemetry_summary: Mapping[str, Any] | None = None,
    errors_sanitized: list[str] | None = None,
    sdk_version: str | None = None,
    timestamp_utc: str | None = None,
    probe_schema_version: str = "1.0",
) -> dict[str, Any]:
    report = {
        "probe_schema_version": probe_schema_version,
        "timestamp_utc": timestamp_utc
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "profile_name": sanitize_profile_name(profile_name),
        "base_url_hash": hash_base_url(base_url),
        "model": model,
        "sdk_version": sdk_version,
        "capabilities": _sanitize_capabilities(capabilities),
        "telemetry_summary": dict(telemetry_summary or {}),
        "errors_sanitized": list(errors_sanitized or []),
    }
    return {key: value for key, value in report.items() if value is not None}


def write_capability_report(
    *,
    profile_name: str,
    base_url: str,
    model: str,
    capabilities: Mapping[str, Any],
    telemetry_summary: Mapping[str, Any] | None = None,
    errors_sanitized: list[str] | None = None,
    sdk_version: str | None = None,
    output_root: str | Path = "tests/test_reports/openai_compat_capabilities",
    timestamp_utc: str | None = None,
) -> Path:
    safe_profile = sanitize_profile_name(profile_name)
    report = build_capability_report(
        profile_name=safe_profile,
        base_url=base_url,
        model=model,
        capabilities=capabilities,
        telemetry_summary=telemetry_summary,
        errors_sanitized=errors_sanitized,
        sdk_version=sdk_version,
        timestamp_utc=timestamp_utc,
    )
    file_stamp = _timestamp_for_filename(report["timestamp_utc"])
    root = _resolve_project_path(output_root) / safe_profile
    root.mkdir(parents=True, exist_ok=True)

    report_path = root / f"{file_stamp}.json"
    latest_path = root / "latest.json"
    content = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    report_path.write_text(content + "\n", encoding="utf-8")
    latest_path.write_text(content + "\n", encoding="utf-8")
    return report_path


def requested_payload_feature(api_mode: str, payload: Mapping[str, Any]) -> str | None:
    if api_mode == API_MODE_RESPONSES:
        if "previous_response_id" in payload:
            return "previous_response_state"
        if "context_management" in payload:
            return "compaction"
        if "responses" in payload:
            return "responses_api"

    feature_keys = (
        ("reasoning_effort", "reasoning_effort"),
        ("verbosity", "verbosity"),
        ("prompt_cache_key", "prompt_cache_key"),
        ("response_format", "structured_output"),
        ("reasoning", "reasoning_effort"),
        ("text", "verbosity_or_structured_output"),
        ("previous_response_id", "previous_response_state"),
        ("context_management", "compaction"),
    )
    for key, feature in feature_keys:
        if key in payload:
            return feature
    if api_mode == API_MODE_RESPONSES:
        return "responses_api"
    return None


def _apply_common_chat_options(payload: dict[str, Any], profile: OpenAICompatProfile) -> None:
    options = profile.request_options
    if options.reasoning_effort:
        _require_capability(
            profile,
            profile.capability_flags.supports_reasoning_effort,
            "reasoning_effort",
            "capability_flags.supports_reasoning_effort",
        )
        payload["reasoning_effort"] = options.reasoning_effort

    if options.verbosity:
        _require_capability(
            profile,
            profile.capability_flags.supports_verbosity,
            "verbosity",
            "capability_flags.supports_verbosity",
        )
        payload["verbosity"] = options.verbosity

    if options.prompt_cache_key:
        _require_capability(
            profile,
            profile.capability_flags.supports_prompt_cache_key,
            "prompt_cache_key",
            "capability_flags.supports_prompt_cache_key",
        )
        payload["prompt_cache_key"] = options.prompt_cache_key

    structured = options.structured_output
    if structured.mode == STRUCTURED_MODE_NONE:
        return
    if structured.mode == STRUCTURED_MODE_PROMPT_JSON:
        _require_capability(
            profile,
            profile.capability_flags.structured_output.supports_prompt_json,
            "structured_output.prompt_json",
            "capability_flags.structured_output.supports_prompt_json",
        )
        return
    if structured.mode == STRUCTURED_MODE_RESPONSES_TEXT_FORMAT:
        _raise_mode_error(profile, "structured_output.responses_text_format", API_MODE_RESPONSES)

    _require_capability(
        profile,
        profile.capability_flags.structured_output.supports_chat_response_format,
        "structured_output.api_schema",
        "capability_flags.structured_output.supports_chat_response_format",
    )
    payload["response_format"] = _build_chat_response_format(structured)


def _apply_common_responses_options(
    payload: dict[str, Any],
    profile: OpenAICompatProfile,
    *,
    previous_response_id: str | None,
) -> None:
    options = profile.request_options
    text_options: dict[str, Any] = {}

    if options.reasoning_effort:
        _require_capability(
            profile,
            profile.capability_flags.supports_reasoning_effort,
            "reasoning_effort",
            "capability_flags.supports_reasoning_effort",
        )
        payload["reasoning"] = {"effort": options.reasoning_effort}

    if options.verbosity:
        _require_capability(
            profile,
            profile.capability_flags.supports_verbosity,
            "verbosity",
            "capability_flags.supports_verbosity",
        )
        text_options["verbosity"] = options.verbosity

    if options.prompt_cache_key:
        _require_capability(
            profile,
            profile.capability_flags.supports_prompt_cache_key,
            "prompt_cache_key",
            "capability_flags.supports_prompt_cache_key",
        )
        payload["prompt_cache_key"] = options.prompt_cache_key

    structured = options.structured_output
    if structured.mode == STRUCTURED_MODE_PROMPT_JSON:
        _require_capability(
            profile,
            profile.capability_flags.structured_output.supports_prompt_json,
            "structured_output.prompt_json",
            "capability_flags.structured_output.supports_prompt_json",
        )
    elif structured.mode in {STRUCTURED_MODE_API_SCHEMA, STRUCTURED_MODE_RESPONSES_TEXT_FORMAT}:
        _require_capability(
            profile,
            profile.capability_flags.structured_output.supports_responses_text_format,
            "structured_output.api_schema",
            "capability_flags.structured_output.supports_responses_text_format",
        )
        text_options["format"] = _build_responses_text_format(structured)
    elif structured.mode == STRUCTURED_MODE_CHAT_RESPONSE_FORMAT:
        _raise_mode_error(profile, "structured_output.chat_response_format", API_MODE_CHAT_COMPLETIONS)

    stateful = profile.stateful_options
    if stateful.store or stateful.use_previous_response_id or previous_response_id:
        _require_capability(
            profile,
            profile.capability_flags.supports_previous_response_state,
            "previous_response_state",
            "capability_flags.supports_previous_response_state",
        )
    if stateful.store:
        payload["store"] = True
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id

    if stateful.compact_threshold is not None:
        _require_capability(
            profile,
            profile.capability_flags.supports_compaction,
            "compaction",
            "capability_flags.supports_compaction",
        )
        payload["context_management"] = {
            "type": "auto",
            "max_tokens": stateful.compact_threshold,
        }

    if text_options:
        payload["text"] = text_options


def _ensure_chat_stateful_options_are_disabled(profile: OpenAICompatProfile) -> None:
    stateful = profile.stateful_options
    if stateful.store or stateful.use_previous_response_id:
        _raise_mode_error(profile, "previous_response_state", API_MODE_RESPONSES)
    if stateful.compact_threshold is not None:
        _raise_mode_error(profile, "compaction", API_MODE_RESPONSES)


def _build_chat_response_format(structured: StructuredOutputOptions) -> dict[str, Any]:
    if structured.schema:
        json_schema: dict[str, Any] = {
            "name": structured.schema_name or "structured_output",
            "schema": copy.deepcopy(structured.schema),
            "strict": structured.strict,
        }
        return {"type": "json_schema", "json_schema": json_schema}
    return {"type": "json_object"}


def _build_responses_text_format(structured: StructuredOutputOptions) -> dict[str, Any]:
    if structured.schema:
        return {
            "type": "json_schema",
            "name": structured.schema_name or "structured_output",
            "schema": copy.deepcopy(structured.schema),
            "strict": structured.strict,
        }
    return {"type": "json_object"}


def _require_capability(
    profile: OpenAICompatProfile,
    enabled: bool,
    feature: str,
    required_flag: str,
) -> None:
    if enabled:
        return
    raise CapabilityNotEnabledError(
        f"{feature} was requested for profile {profile.profile_name}, "
        f"but {required_flag} is false. Enable this flag only after probe verifies support, "
        "or disable the corresponding request_options/stateful_options entry.",
        profile_name=profile.profile_name,
        feature=feature,
        api_mode=profile.api_mode,
    )


def _raise_mode_error(profile: OpenAICompatProfile, feature: str, required_api_mode: str) -> None:
    raise CapabilityModeError(
        f"{feature} requires api_mode={required_api_mode}, but profile {profile.profile_name} "
        f"uses api_mode={profile.api_mode}.",
        profile_name=profile.profile_name,
        feature=feature,
        api_mode=profile.api_mode,
    )


def _sanitize_capabilities(capabilities: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in capabilities.items():
        if isinstance(value, Mapping):
            item = dict(value)
            status = item.get("status")
            if status is not None and status not in VALID_PROBE_STATUSES:
                item["status"] = PROBE_STATUS_ERROR
                item["status_note"] = f"invalid status sanitized from {status!r}"
            sanitized[str(key)] = item
        else:
            sanitized[str(key)] = value
    return sanitized


def _timestamp_for_filename(timestamp_utc: str) -> str:
    compact = timestamp_utc.replace("-", "").replace(":", "")
    compact = compact.replace("+0000", "Z").replace("+00:00", "Z")
    compact = compact.replace(".", "")
    compact = re.sub(r"[^0-9TZ]", "", compact)
    if compact.endswith("Z"):
        return compact
    return f"{compact}Z"


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _as_bool(raw: Any, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _optional_str(raw: Any) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _optional_int(raw: Any) -> int | None:
    if raw is None or raw == "":
        return None
    return int(raw)


def _mapping(raw: Any) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return raw
    raise ValueError(f"Expected object/dict config, got {type(raw).__name__}")


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _first_int(obj: Any, keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _get_value(obj, key, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_header_value(headers: Mapping[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = headers.get(key.lower())
        if value:
            return value
    return None


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = _get_value(item, "text", None) or _get_value(item, "content", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return str(content)
