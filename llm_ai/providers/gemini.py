import logging
from typing import Any, List

from llm_ai.base import BaseLLMProvider

logger = logging.getLogger("llm_ai")


class GeminiProvider(BaseLLMProvider):
    def __init__(
        self,
        api_keys: List[str],
        model: str = "gemini-3-flash-preview",
        thinking_budget: int = 8192,
        system_prompt: str = "",
        temperature: float = 1,
        max_output_tokens: int = 65530,
        retry_attempts: int = 3,
        retry_wait_seconds: int = 5,
        safety_settings: dict[str, Any] | None = None,
    ):
        if not api_keys:
            raise ValueError("Cần ít nhất 1 Gemini API key")
        self.api_keys = api_keys[:]
        self.model = model
        self.thinking_budget = thinking_budget
        self._system_prompt = (system_prompt or "").strip()
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._retry_attempts = retry_attempts
        self._retry_wait_seconds = retry_wait_seconds
        self._safety_settings_raw = safety_settings or {}

    @property
    def name(self) -> str:
        return f"Gemini ({self.model})"

    def _next_key(self) -> str:
        key = self.api_keys.pop(0)
        self.api_keys.append(key)
        return key

    def _default_safety_settings(self, types: object) -> list[object]:
        return [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

    def _parse_safety_settings(self, types: object) -> list[object]:
        if not self._safety_settings_raw:
            return self._default_safety_settings(types)

        parsed: list[object] = []
        for category_name, threshold_name in self._safety_settings_raw.items():
            category = getattr(types.HarmCategory, str(category_name))
            threshold = getattr(types.HarmBlockThreshold, str(threshold_name))
            parsed.append(types.SafetySetting(category=category, threshold=threshold))
        return parsed

    def _build_generate_config(self, types: object) -> object:
        cfg: dict[str, Any] = {
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
            "safety_settings": self._parse_safety_settings(types),
        }

        if self._system_prompt:
            cfg["system_instruction"] = [types.Part.from_text(text=self._system_prompt)]

        if not (self.model.startswith("gemini-1.") or self.model.startswith("gemini-2.0")):
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)

        return types.GenerateContentConfig(**cfg)

    def call(self, message: str) -> str:
        try:
            from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_fixed
        except ImportError as exc:
            raise ImportError("tenacity chưa cài. Chạy: pip install tenacity>=8.0.0") from exc

        from google import genai
        from google.genai import types

        for attempt in Retrying(
            retry=retry_if_exception_type(RuntimeError),
            stop=stop_after_attempt(self._retry_attempts),
            wait=wait_fixed(self._retry_wait_seconds),
            reraise=True,
        ):
            with attempt:
                api_key = self._next_key()
                client = genai.Client(api_key=api_key)

                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=message)],
                    )
                ]

                try:
                    result = ""
                    for chunk in client.models.generate_content_stream(
                        model=self.model,
                        contents=contents,
                        config=self._build_generate_config(types),
                    ):
                        result += chunk.text if chunk.text else ""

                    if not result:
                        raise RuntimeError("[Gemini] Response rỗng — sẽ retry")
                    return result
                except Exception as exc:
                    from google.genai import errors

                    if isinstance(exc, errors.APIError):
                        logger.error(f"[Gemini Provider] API Error {exc.code}: {exc.message}")
                    else:
                        logger.error(
                            f"[Gemini Provider] Lỗi hệ thống: {type(exc).__name__} - {exc}"
                        )
                    raise
