from typing import List
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type,
    before_log,
    after_log,
)
from translation.base import BaseTranslationProvider

logger = logging.getLogger("srt_translator")

class GeminiProvider(BaseTranslationProvider):
    def __init__(self, api_keys: List[str], model: str = "gemini-3-flash-preview", thinking_budget: int = 8192):
        if not api_keys:
            raise ValueError("Cần ít nhất 1 Gemini API key")
        self.api_keys = api_keys[:]
        self.model = model
        self.thinking_budget = thinking_budget

    @property
    def name(self) -> str:
        return f"Gemini ({self.model})"

    def _next_key(self) -> str:
        key = self.api_keys.pop(0)
        self.api_keys.append(key)
        return key

    @retry(
        retry=retry_if_exception_type(RuntimeError),
        stop=stop_after_attempt(3),
        wait=wait_fixed(5),
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.DEBUG),
    )
    def call(self, message: str) -> str:
        from google import genai
        from google.genai import types

        api_key = self._next_key()
        client = genai.Client(api_key=api_key)
        model = self.model

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=message)],
            )
        ]

        gen_config = types.GenerateContentConfig(
            temperature=1,
            max_output_tokens=65530,
            safety_settings=[
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
            ],
            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
            system_instruction=[
                types.Part.from_text(text="You are a top-tier Subtitle Translation Engine.")
            ],
        )

        if model.startswith("gemini-1.") or model.startswith("gemini-2.0"):
            gen_config = types.GenerateContentConfig(temperature=1, max_output_tokens=65530)

        try:
            result = ""
            for chunk in client.models.generate_content_stream(
                model=model, contents=contents, config=gen_config
            ):
                result += chunk.text if chunk.text else ""

            if not result:
                raise RuntimeError("[Gemini] Response rỗng — sẽ retry")
            return result
        except Exception as e:
            from google.genai import errors
            if isinstance(e, errors.APIError):
                logger.error(f"[Gemini Provider] API Error {e.code}: {e.message}")
            else:
                logger.error(f"[Gemini Provider] Lỗi hệ thống: {type(e).__name__} - {str(e)}")
            raise e
