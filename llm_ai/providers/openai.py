from __future__ import annotations

import logging
import time
from typing import Any, Mapping

from llm_ai.base import BaseLLMProvider
from llm_ai.openai_compat import (
    API_MODE_CHAT_COMPLETIONS,
    API_MODE_RESPONSES,
    CapabilityRejectedError,
    OpenAICompatCapabilityError,
    OpenAICompatProfile,
    build_chat_completions_payload,
    build_compaction_payload,
    build_responses_payload,
    build_telemetry_record,
    extract_chat_completion_text,
    extract_response_id,
    extract_responses_text,
    requested_payload_feature,
    write_telemetry_record,
)
from llm_ai.retry import build_linear_retry_wait


class OpenAICompatibleProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
        request_timeout: int,
        retry_attempts: int,
        retry_wait_seconds: int,
        profile_config: Mapping[str, Any] | None = None,
    ):
        try:
            import httpx
            import openai
        except ImportError as exc:
            raise ImportError("openai package chưa cài. Chạy: pip install openai>=1.35.0") from exc

        merged_config = dict(profile_config or {})
        merged_config.setdefault("provider", "openai")
        merged_config["base_url"] = base_url
        merged_config["model"] = model
        merged_config["temperature"] = temperature
        merged_config["max_tokens"] = max_tokens

        self._profile = OpenAICompatProfile.from_config(merged_config)

        http_client = httpx.Client()
        self._client = openai.OpenAI(
            base_url=self._profile.base_url,
            api_key=api_key,
            timeout=request_timeout,
            http_client=http_client,
        )
        self._model = self._profile.model
        self._base_url = self._profile.base_url
        self._temperature = self._profile.temperature
        self._max_tokens = self._profile.max_tokens
        self._system_prompt = system_prompt
        self._retry_attempts = retry_attempts
        self._retry_wait_seconds = retry_wait_seconds
        self._previous_response_id: str | None = None
        self._last_response_id: str | None = None
        self._last_telemetry_record: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return f"OpenAI-Compatible ({self._profile.profile_name}: {self._model} @ {self._base_url})"

    @property
    def profile(self) -> OpenAICompatProfile:
        return self._profile

    @property
    def last_response_id(self) -> str | None:
        return self._last_response_id

    @property
    def last_telemetry_record(self) -> dict[str, Any] | None:
        return dict(self._last_telemetry_record) if self._last_telemetry_record else None

    def reset_state(self) -> None:
        """Xóa previous response state local của provider."""
        self._previous_response_id = None
        self._last_response_id = None

    def call(self, message: str) -> str:
        from openai import AuthenticationError, BadRequestError, PermissionDeniedError
        from tenacity import Retrying, retry_if_not_exception_type, stop_after_attempt

        no_retry_errors = (
            AuthenticationError,
            BadRequestError,
            PermissionDeniedError,
            OpenAICompatCapabilityError,
        )

        for attempt in Retrying(
            retry=retry_if_not_exception_type(no_retry_errors),
            stop=stop_after_attempt(self._retry_attempts),
            wait=build_linear_retry_wait(self._retry_wait_seconds),
            reraise=True,
        ):
            with attempt:
                try:
                    attempt_number = attempt.retry_state.attempt_number
                    return self._call_once(message, attempt_number=attempt_number)
                except OpenAICompatCapabilityError:
                    raise
                except Exception as exc:
                    self._log_provider_error(exc)
                    raise

        return ""

    def compact_state(self, response_id: str | None = None) -> Any:
        """Gọi compaction endpoint nếu OpenAI-compatible gateway có hỗ trợ."""
        target_response_id = response_id or self._last_response_id or self._previous_response_id
        payload = build_compaction_payload(self._profile, target_response_id or "")
        responses_resource = getattr(self._client, "responses", None)
        compact_resource = getattr(responses_resource, "compact", None) if responses_resource else None
        if compact_resource is None or not hasattr(compact_resource, "create"):
            raise CapabilityRejectedError(
                f"compaction was requested for profile {self._profile.profile_name}, "
                "but client.responses.compact.create is not available.",
                profile_name=self._profile.profile_name,
                feature="compaction",
                api_mode=self._profile.api_mode,
            )
        return compact_resource.create(**payload)

    def _call_once(self, message: str, *, attempt_number: int) -> str:
        started = time.perf_counter()
        headers: Any = None

        if self._profile.api_mode == API_MODE_RESPONSES:
            previous_response_id = (
                self._previous_response_id
                if self._profile.stateful_options.use_previous_response_id
                else None
            )
            payload = build_responses_payload(
                self._profile,
                self._system_prompt,
                message,
                previous_response_id=previous_response_id,
            )
            try:
                response, headers = self._create_response(payload)
            except Exception as exc:
                wrapped = self._maybe_wrap_endpoint_rejection(exc, payload)
                if wrapped:
                    raise wrapped from exc
                raise

            text = extract_responses_text(response)
            self._capture_state(response)
        else:
            payload = build_chat_completions_payload(
                self._profile,
                self._system_prompt,
                message,
            )
            try:
                response, headers = self._create_chat_completion(payload)
            except Exception as exc:
                wrapped = self._maybe_wrap_endpoint_rejection(exc, payload)
                if wrapped:
                    raise wrapped from exc
                raise

            text = extract_chat_completion_text(response)
            self._capture_state(response)

        latency_ms = int((time.perf_counter() - started) * 1000)
        self._capture_telemetry(
            response,
            headers=headers,
            latency_ms=latency_ms,
            retry_count=max(0, attempt_number - 1),
        )
        return text

    def _create_chat_completion(self, payload: Mapping[str, Any]) -> tuple[Any, Any]:
        completions = self._client.chat.completions
        raw_completions = getattr(completions, "with_raw_response", None)
        if self._should_capture_raw_response() and raw_completions is not None:
            raw_response = raw_completions.create(**payload)
            return raw_response.parse(), getattr(raw_response, "headers", None)

        response = completions.create(**payload)
        return response, self._extract_response_headers(response)

    def _create_response(self, payload: Mapping[str, Any]) -> tuple[Any, Any]:
        responses_resource = getattr(self._client, "responses", None)
        if responses_resource is None or not hasattr(responses_resource, "create"):
            raise CapabilityRejectedError(
                f"responses_api was requested for profile {self._profile.profile_name}, "
                "but client.responses.create is not available in this OpenAI SDK/client.",
                profile_name=self._profile.profile_name,
                feature="responses_api",
                api_mode=self._profile.api_mode,
            )

        raw_responses = getattr(responses_resource, "with_raw_response", None)
        if self._should_capture_raw_response() and raw_responses is not None:
            raw_response = raw_responses.create(**payload)
            return raw_response.parse(), getattr(raw_response, "headers", None)

        response = responses_resource.create(**payload)
        return response, self._extract_response_headers(response)

    def _should_capture_raw_response(self) -> bool:
        telemetry = self._profile.telemetry
        return telemetry.enabled and (telemetry.capture_cache_headers or telemetry.capture_raw_headers)

    def _capture_state(self, response: Any) -> None:
        response_id = extract_response_id(response)
        if not response_id:
            return
        self._last_response_id = response_id
        if self._profile.stateful_options.use_previous_response_id:
            self._previous_response_id = response_id

    def _capture_telemetry(
        self,
        response: Any,
        *,
        headers: Any,
        latency_ms: int,
        retry_count: int,
    ) -> None:
        if not self._profile.telemetry.enabled:
            return
        record = build_telemetry_record(
            self._profile,
            response,
            headers=headers,
            latency_ms=latency_ms,
            retry_count=retry_count,
            prompt_cache_key=self._profile.request_options.prompt_cache_key,
        )
        self._last_telemetry_record = record
        write_telemetry_record(self._profile, record)

    def _maybe_wrap_endpoint_rejection(
        self,
        exc: Exception,
        payload: Mapping[str, Any],
    ) -> CapabilityRejectedError | None:
        if isinstance(exc, OpenAICompatCapabilityError):
            return None

        try:
            import openai
        except ImportError:
            return None

        if not isinstance(exc, getattr(openai, "APIStatusError", tuple())):
            return None

        status_code = getattr(exc, "status_code", None)
        if status_code not in {400, 404, 422}:
            return None

        feature = requested_payload_feature(self._profile.api_mode, payload)
        if not feature:
            return None

        provider_text = self._provider_error_text(exc)
        return CapabilityRejectedError(
            f"{feature} was enabled for profile {self._profile.profile_name}, "
            f"but endpoint rejected the request with HTTP {status_code}. "
            f"Provider error: {provider_text}. Run capability probe and update this profile.",
            profile_name=self._profile.profile_name,
            feature=feature,
            api_mode=self._profile.api_mode,
        )

    def _extract_response_headers(self, response: Any) -> Any:
        direct_headers = getattr(response, "headers", None)
        if direct_headers is not None:
            return direct_headers
        raw_response = getattr(response, "response", None) or getattr(response, "_response", None)
        if raw_response is not None:
            return getattr(raw_response, "headers", None)
        return None

    def _log_provider_error(self, exc: Exception) -> None:
        try:
            import openai
        except ImportError:
            openai = None  # type: ignore[assignment]

        if openai is not None and isinstance(exc, openai.APIStatusError):
            logging.error(
                "[OpenAI Provider] HTTP Error %s: %s",
                getattr(exc, "status_code", "unknown"),
                self._provider_error_text(exc),
            )
        else:
            logging.error(
                "[OpenAI Provider] Lỗi hệ thống: %s - %s",
                type(exc).__name__,
                exc,
            )

    def _provider_error_text(self, exc: Exception) -> str:
        response = getattr(exc, "response", None)
        text = getattr(response, "text", None)
        if text:
            return str(text)
        return str(exc)
