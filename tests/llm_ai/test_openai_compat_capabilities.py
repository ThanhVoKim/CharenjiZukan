#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/llm_ai/test_openai_compat_capabilities.py
================================================
Test OpenAI-compatible provider capability flags, request builders, telemetry,
and mock client integration.

Cấu trúc layers:
  Layer 1 — Unit Tests          (config parsing, payload builders, reports)
  Layer 2 — Component Tests     (provider with mocked OpenAI-compatible client)
  Layer 3 — Integration         (không cần trong file này)
  Layer 4 — Real API Tests      (tách riêng trong test_openai_compat_capability_probe.py)

Cách chạy:
    pytest tests/llm_ai/test_openai_compat_capabilities.py -v -k "Layer1"
    pytest tests/llm_ai/test_openai_compat_capabilities.py -v -k "Layer2"
"""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_ai.factory import create_provider
from llm_ai.openai_compat import (  # noqa: E402
    API_MODE_CHAT_COMPLETIONS,
    API_MODE_RESPONSES,
    CapabilityModeError,
    CapabilityNotEnabledError,
    CapabilityRejectedError,
    OpenAICompatProfile,
    build_capability_report,
    build_chat_completions_payload,
    build_responses_payload,
    build_telemetry_record,
    write_capability_report,
)


def _base_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "provider": "openai",
        "profile_name": "unit_profile",
        "base_url": "https://example.test/v1",
        "model": "gpt-test",
        "api_mode": "chat_completions",
        "temperature": 0.5,
        "max_tokens": 123,
    }
    config.update(overrides)
    return config


def _fake_chat_response(text: str = "OK", **extra: Any) -> SimpleNamespace:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        **extra,
    )
    return response


@pytest.fixture()
def fake_openai_modules(monkeypatch):
    """Stub openai/httpx modules so Layer 2 tests never call network or require API key."""

    fake_openai = ModuleType("openai")

    class AuthenticationError(Exception):
        pass

    class BadRequestError(Exception):
        pass

    class PermissionDeniedError(Exception):
        pass

    class APIStatusError(Exception):
        def __init__(self, message: str, *, status_code: int = 400, text: str = "bad request"):
            super().__init__(message)
            self.status_code = status_code
            self.response = SimpleNamespace(text=text)

    class OpenAI:
        def __init__(self, **kwargs: Any):
            self.init_kwargs = kwargs
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **payload: _fake_chat_response())
            )
            self.responses = SimpleNamespace(
                create=lambda **payload: SimpleNamespace(output_text="OK", id="resp_mock")
            )

    fake_openai.AuthenticationError = AuthenticationError
    fake_openai.BadRequestError = BadRequestError
    fake_openai.PermissionDeniedError = PermissionDeniedError
    fake_openai.APIStatusError = APIStatusError
    fake_openai.OpenAI = OpenAI
    fake_openai.__version__ = "fake-openai-1.0"

    fake_httpx = ModuleType("httpx")

    class Client:
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs

    fake_httpx.Client = Client

    fake_tenacity = ModuleType("tenacity")

    class Attempt:
        def __init__(self, attempt_number: int = 1):
            self.retry_state = SimpleNamespace(attempt_number=attempt_number)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class Retrying:
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs

        def __iter__(self):
            yield Attempt()

    def retry_if_not_exception_type(exceptions: Any):
        return exceptions

    def stop_after_attempt(attempts: int):
        return attempts

    fake_tenacity.Retrying = Retrying
    fake_tenacity.retry_if_not_exception_type = retry_if_not_exception_type
    fake_tenacity.stop_after_attempt = stop_after_attempt

    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    monkeypatch.setitem(sys.modules, "tenacity", fake_tenacity)
    return fake_openai


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# ═════════════════════════════════════════════════════════════════════


class TestLayer1_OpenAICompatConfigParsing:
    """Test config profile giữ backward-compatible và normalize field mới."""

    def test_profile_defaults_keep_chat_completions_baseline(self):
        profile = OpenAICompatProfile.from_config(
            {
                "provider": "openai",
                "base_url": "https://api.example.test/v1",
                "model": "gpt-basic",
                "temperature": 1,
                "max_tokens": 8192,
            }
        )

        assert profile.api_mode == API_MODE_CHAT_COMPLETIONS
        assert profile.capability_flags.supports_chat_completions is True
        assert profile.capability_flags.supports_responses_api is False
        assert profile.capability_flags.structured_output.supports_prompt_json is True
        assert profile.request_options.reasoning_effort is None
        assert profile.telemetry.enabled is False

    def test_profile_sanitizes_name_and_parses_nested_options(self):
        profile = OpenAICompatProfile.from_config(
            _base_config(
                profile_name="open router / GPT 5.5",
                api_mode="responses-api",
                capability_flags={
                    "supports_responses_api": True,
                    "supports_previous_response_state": True,
                    "structured_output": {"supports_responses_text_format": True},
                },
                request_options={
                    "reasoning_effort": "low",
                    "verbosity": "medium",
                    "structured_output": {"mode": "json_schema", "schema_name": "demo"},
                },
                stateful_options={"store": True, "use_previous_response_id": True},
                telemetry={"enabled": True, "output_path": "logs/custom.jsonl"},
            )
        )

        assert profile.profile_name == "open_router_GPT_5.5"
        assert profile.api_mode == API_MODE_RESPONSES
        assert profile.request_options.structured_output.mode == "api_schema"
        assert profile.stateful_options.store is True
        assert profile.telemetry.output_path == "logs/custom.jsonl"


class TestLayer1_OpenAICompatPayloadBuilder:
    """Test payload builders và capability gating không gọi network."""

    def test_chat_payload_contains_only_baseline_when_no_advanced_options(self):
        profile = OpenAICompatProfile.from_config(_base_config())

        payload = build_chat_completions_payload(profile, "System", "Hello")

        assert payload == {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Hello"},
            ],
            "temperature": 0.5,
            "max_tokens": 123,
        }

    def test_capability_false_raises_clear_exception_before_payload_is_sent(self):
        profile = OpenAICompatProfile.from_config(
            _base_config(request_options={"reasoning_effort": "medium"})
        )

        with pytest.raises(CapabilityNotEnabledError, match="supports_reasoning_effort"):
            build_chat_completions_payload(profile, None, "Hello")

    def test_capability_true_injects_advanced_chat_parameters(self):
        profile = OpenAICompatProfile.from_config(
            _base_config(
                capability_flags={
                    "supports_reasoning_effort": True,
                    "supports_verbosity": True,
                    "supports_prompt_cache_key": True,
                },
                request_options={
                    "reasoning_effort": "low",
                    "verbosity": "medium",
                    "prompt_cache_key": "stable-cache-key",
                },
            )
        )

        payload = build_chat_completions_payload(profile, None, "Hello")

        assert payload["reasoning_effort"] == "low"
        assert payload["verbosity"] == "medium"
        assert payload["prompt_cache_key"] == "stable-cache-key"

    def test_api_schema_requires_chat_response_format_capability(self):
        profile = OpenAICompatProfile.from_config(
            _base_config(
                request_options={
                    "structured_output": {
                        "mode": "api_schema",
                        "schema_name": "demo",
                        "schema": {
                            "type": "object",
                            "properties": {"title": {"type": "string"}},
                            "required": ["title"],
                        },
                    }
                }
            )
        )

        with pytest.raises(CapabilityNotEnabledError, match="supports_chat_response_format"):
            build_chat_completions_payload(profile, None, "Hello")

    def test_api_schema_chat_payload_uses_response_format_when_enabled(self):
        profile = OpenAICompatProfile.from_config(
            _base_config(
                capability_flags={
                    "structured_output": {"supports_chat_response_format": True}
                },
                request_options={
                    "structured_output": {
                        "mode": "api_schema",
                        "schema_name": "demo",
                        "schema": {
                            "type": "object",
                            "properties": {"title": {"type": "string"}},
                            "required": ["title"],
                        },
                    }
                },
            )
        )

        payload = build_chat_completions_payload(profile, None, "Hello")

        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["name"] == "demo"

    def test_responses_payload_maps_stateful_options_and_text_format(self):
        profile = OpenAICompatProfile.from_config(
            _base_config(
                api_mode="responses",
                capability_flags={
                    "supports_responses_api": True,
                    "supports_reasoning_effort": True,
                    "supports_verbosity": True,
                    "supports_prompt_cache_key": True,
                    "supports_previous_response_state": True,
                    "supports_compaction": True,
                    "structured_output": {"supports_responses_text_format": True},
                },
                request_options={
                    "reasoning_effort": "medium",
                    "verbosity": "low",
                    "prompt_cache_key": "cache-key",
                    "structured_output": {"mode": "api_schema", "schema": {"type": "object"}},
                },
                stateful_options={
                    "store": True,
                    "use_previous_response_id": True,
                    "compact_threshold": 4096,
                },
            )
        )

        payload = build_responses_payload(profile, "System", "Hello", previous_response_id="resp_1")

        assert payload["model"] == "gpt-test"
        assert payload["input"][0] == {"role": "system", "content": "System"}
        assert payload["max_output_tokens"] == 123
        assert payload["reasoning"] == {"effort": "medium"}
        assert payload["text"]["verbosity"] == "low"
        assert payload["text"]["format"]["type"] == "json_schema"
        assert payload["prompt_cache_key"] == "cache-key"
        assert payload["store"] is True
        assert payload["previous_response_id"] == "resp_1"
        assert payload["context_management"]["max_tokens"] == 4096

    def test_stateful_options_fail_fast_in_chat_mode(self):
        profile = OpenAICompatProfile.from_config(
            _base_config(stateful_options={"use_previous_response_id": True})
        )

        with pytest.raises(CapabilityModeError, match="previous_response_state"):
            build_chat_completions_payload(profile, None, "Hello")


class TestLayer1_OpenAICompatTelemetryAndReport:
    """Test telemetry metadata và versioned capability report."""

    def test_telemetry_record_extracts_usage_cache_headers_and_sanitizes_secrets(self):
        profile = OpenAICompatProfile.from_config(
            _base_config(
                telemetry={
                    "enabled": True,
                    "capture_usage": True,
                    "capture_cache_headers": True,
                    "capture_raw_headers": True,
                },
                request_options={"prompt_cache_key": "cache-key"},
            )
        )
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                prompt_tokens_details=SimpleNamespace(cached_tokens=80),
            ),
            model="resolved-model",
            system_fingerprint="fp-test",
        )

        record = build_telemetry_record(
            profile,
            response,
            headers={
                "x-cache-status": "hit",
                "x-request-id": "req_123",
                "authorization": "Bearer secret",
            },
            latency_ms=250,
            retry_count=1,
        )

        assert record["profile_name"] == "unit_profile"
        assert record["cache_status"] == "hit"
        assert record["cached_tokens"] == 80
        assert record["input_tokens"] == 100
        assert record["output_tokens"] == 20
        assert record["request_id"] == "req_123"
        assert record["latency_ms"] == 250
        assert "authorization" not in record.get("headers", {})

    def test_capability_report_schema_and_versioned_writer(self, tmp_path: Path):
        report = build_capability_report(
            profile_name="profile/demo",
            base_url="https://secret-endpoint.example/v1",
            model="gpt-test",
            capabilities={"chat_completions_basic": {"status": "verified"}},
            telemetry_summary={"usage_seen": True},
            timestamp_utc="2026-05-27T19:06:00Z",
        )

        assert report["profile_name"] == "profile_demo"
        assert report["base_url_hash"].startswith("sha256:")
        assert report["capabilities"]["chat_completions_basic"]["status"] == "verified"

        path = write_capability_report(
            profile_name="profile/demo",
            base_url="https://secret-endpoint.example/v1",
            model="gpt-test",
            capabilities={"prompt_cache_key": {"status": "accepted"}},
            telemetry_summary={"cache_headers_seen": False},
            output_root=tmp_path,
            timestamp_utc="2026-05-27T19:06:00Z",
        )

        latest = tmp_path / "profile_demo" / "latest.json"
        assert path.exists()
        assert latest.exists()
        loaded = json.loads(latest.read_text(encoding="utf-8"))
        assert loaded["capabilities"]["prompt_cache_key"]["status"] == "accepted"


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — COMPONENT TESTS
# ═════════════════════════════════════════════════════════════════════


class TestLayer2_OpenAICompatChatCompletionsClient:
    """Test provider Chat Completions bằng mocked client."""

    def test_provider_call_uses_chat_completions_payload(self, fake_openai_modules):
        provider = create_provider(
            "openai",
            _base_config(system_prompt="You are concise."),
            {"api_key": "test-key"},
        )
        captured: dict[str, Any] = {}

        def fake_create(**payload: Any):
            captured.update(payload)
            return _fake_chat_response("Mocked answer")

        provider._client.chat.completions.create = fake_create

        assert provider.call("Hello") == "Mocked answer"
        assert captured["model"] == "gpt-test"
        assert captured["messages"] == [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Hello"},
        ]
        assert "reasoning_effort" not in captured

    def test_factory_preserves_profile_config(self, fake_openai_modules):
        provider = create_provider(
            "openai",
            _base_config(profile_name="custom_profile"),
            {"api_key": "test-key"},
        )

        assert provider.profile.profile_name == "custom_profile"
        assert provider.profile.base_url == "https://example.test/v1"


class TestLayer2_OpenAICompatResponsesClient:
    """Test Responses API opt-in bằng mocked client."""

    def test_provider_call_uses_responses_payload_when_enabled(self, fake_openai_modules):
        provider = create_provider(
            "openai",
            _base_config(
                api_mode="responses",
                capability_flags={
                    "supports_responses_api": True,
                    "supports_previous_response_state": True,
                },
                stateful_options={"store": True, "use_previous_response_id": True},
            ),
            {"api_key": "test-key"},
        )
        captured: dict[str, Any] = {}

        def fake_response_create(**payload: Any):
            captured.update(payload)
            return SimpleNamespace(output_text="Response answer", id="resp_2")

        provider._client.responses.create = fake_response_create

        assert provider.call("Hello") == "Response answer"
        assert captured["input"] == [{"role": "user", "content": "Hello"}]
        assert captured["store"] is True
        assert provider.last_response_id == "resp_2"


class TestLayer2_OpenAICompatUnsupportedCapabilityHandling:
    """Test endpoint rejection được wrap thành custom exception rõ nguyên nhân."""

    def test_endpoint_bad_request_for_enabled_capability_wraps_rejected_error(
        self,
        fake_openai_modules,
    ):
        provider = create_provider(
            "openai",
            _base_config(
                capability_flags={"supports_reasoning_effort": True},
                request_options={"reasoning_effort": "low"},
            ),
            {"api_key": "test-key"},
        )

        def fake_create(**payload: Any):
            raise fake_openai_modules.APIStatusError(
                "bad request",
                status_code=400,
                text="unsupported parameter: reasoning_effort",
            )

        provider._client.chat.completions.create = fake_create

        with pytest.raises(CapabilityRejectedError, match="reasoning_effort"):
            provider.call("Hello")
