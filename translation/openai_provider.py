import logging
from translation.base import BaseTranslationProvider

class OpenAICompatibleProvider(BaseTranslationProvider):
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
    ):
        try:
            import httpx
            import openai
        except ImportError:
            raise ImportError("openai package chưa cài. Chạy: pip install openai>=1.35.0")
        
        http_client = httpx.Client(headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=request_timeout, http_client=http_client
)
        self._model = model
        self._base_url = base_url
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._retry_attempts = retry_attempts
        self._retry_wait_seconds = retry_wait_seconds

    @property
    def name(self) -> str:
        return f"OpenAI-Compatible ({self._model} @ {self._base_url})"

    def call(self, message: str) -> str:
        from openai import AuthenticationError, BadRequestError, PermissionDeniedError
        from tenacity import Retrying, stop_after_attempt, wait_exponential, retry_if_not_exception_type
        
        NO_RETRY_ERRORS = (AuthenticationError, BadRequestError, PermissionDeniedError)
        
        for attempt in Retrying(
            retry=retry_if_not_exception_type(NO_RETRY_ERRORS),
            stop=stop_after_attempt(self._retry_attempts),
            wait=wait_exponential(multiplier=1, min=self._retry_wait_seconds, max=self._retry_wait_seconds * 2),
            reraise=True
        ):
            with attempt:
                messages = []
                if self._system_prompt and self._system_prompt.strip():
                    messages.append({"role": "system", "content": self._system_prompt.strip()})
                messages.append({"role": "user", "content": message})

                try:
                    response = self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                    )
                    return response.choices[0].message.content or ""
                except Exception as e:
                    import openai
                    if isinstance(e, openai.APIStatusError):
                        logging.error(f"[OpenAI Provider] HTTP Error {e.status_code}: {e.response.text}")
                    else:
                        logging.error(f"[OpenAI Provider] Lỗi hệ thống: {type(e).__name__} - {str(e)}")
                    raise e
