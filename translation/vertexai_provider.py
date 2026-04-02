import logging

from translation.base import BaseTranslationProvider

logger = logging.getLogger("srt_translator")


class VertexAIProvider(BaseTranslationProvider):
    def __init__(
        self,
        project_id: str,
        location: str,
        model: str,
        generation_config: dict,
        safety_settings: dict,
        system_prompt: str,
        request_timeout: int,
        retry_attempts: int,
        retry_wait_seconds: int,
        cache_ttl_seconds: int = 3600,
    ):
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("google-genai chưa cài. Chạy: pip install google-genai>=0.3.0")

        self._genai = genai
        self._types = types

        self._project_id = project_id
        self._location = location
        self._model_name = model
        self._retry_attempts = retry_attempts
        self._retry_wait_seconds = retry_wait_seconds
        self._request_timeout = request_timeout
        self._cache_ttl_seconds = max(1, int(cache_ttl_seconds))

        self._generation_config_raw = generation_config or {}
        self._system_prompt = (system_prompt or "").strip()

        self._client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
            http_options={"timeout": request_timeout},
        )

        self._safety_settings = self._parse_safety_settings(safety_settings)
        self._cached_content_name: str | None = None

    @property
    def name(self) -> str:
        return f"Vertex AI ({self._model_name}, project={self._project_id})"

    def _parse_safety_settings(self, safety_settings: dict) -> list[object] | None:
        """
        Convert YAML safety settings dạng string sang list SafetySetting cho google-genai SDK.

        Input YAML dạng:
            {
              "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
              ...
            }
        """
        if not safety_settings:
            return None

        if not isinstance(safety_settings, dict):
            raise ValueError(
                "safety_settings phải là object/dict trong YAML, ví dụ: "
                "{HARM_CATEGORY_HATE_SPEECH: BLOCK_NONE}"
            )

        parsed: list[object] = []

        for category_name, threshold_name in safety_settings.items():
            try:
                category = getattr(self._types.HarmCategory, str(category_name))
            except AttributeError as exc:
                valid_categories = [name for name in dir(self._types.HarmCategory) if name.startswith("HARM_CATEGORY_")]
                raise ValueError(
                    f"Giá trị category không hợp lệ: '{category_name}'. "
                    f"Các giá trị hợp lệ: {valid_categories}"
                ) from exc

            try:
                threshold = getattr(self._types.HarmBlockThreshold, str(threshold_name))
            except AttributeError as exc:
                valid_thresholds = [name for name in dir(self._types.HarmBlockThreshold) if name.startswith("BLOCK_")]
                raise ValueError(
                    f"Giá trị threshold không hợp lệ: '{threshold_name}' cho category '{category_name}'. "
                    f"Các giá trị hợp lệ: {valid_thresholds}"
                ) from exc

            parsed.append(self._types.SafetySetting(category=category, threshold=threshold))

        return parsed

    def _build_generate_config(self) -> object:
        cfg = dict(self._generation_config_raw)

        if self._safety_settings is not None:
            cfg["safety_settings"] = self._safety_settings

        if self._cached_content_name:
            cfg["cached_content"] = self._cached_content_name

        if self._system_prompt:
            cfg["system_instruction"] = self._system_prompt

        return self._types.GenerateContentConfig(**cfg)

    def _build_cache_system_instruction(self) -> str:
        base_system = self._system_prompt or "You are a top-tier Subtitle Translation Engine."
        return (
            f"{base_system}\n\n"
            "The request may use cached global subtitle context. "
            "Treat that cache as read-only reference and DO NOT translate it directly."
        )

    def set_global_context(self, context: str) -> bool:
        """
        Thử tạo explicit context cache trên Vertex AI.

        Returns:
            - True: cache tạo thành công, caller có thể bỏ context ra khỏi prompt batch.
            - False: cache không dùng được (ví dụ context quá ngắn / API lỗi), caller nên fallback
                     chèn context vào prompt theo cách cũ.
        """
        if not context or not context.strip():
            return False

        try:
            cached_content = self._client.caches.create(
                model=self._model_name,
                config=self._types.CreateCachedContentConfig(
                    contents=[context],
                    system_instruction=self._build_cache_system_instruction(),
                    ttl=f"{self._cache_ttl_seconds}s",
                ),
            )
            self._cached_content_name = cached_content.name
            logger.info(f"[VertexAI Provider] Context cache created: {self._cached_content_name}")
            return True
        except Exception as e:
            error_text = str(e).lower()
            if "2048" in error_text or "minimum" in error_text and "token" in error_text:
                logger.info(
                    "[VertexAI Provider] Context cache không đủ token tối thiểu, fallback prompt-inline"
                )
            else:
                logger.warning(
                    f"[VertexAI Provider] Không tạo được context cache, fallback prompt-inline. "
                    f"Chi tiết: {type(e).__name__} - {e}"
                )
            self._cached_content_name = None
            return False

    def call(self, message: str) -> str:
        from google.api_core import exceptions as gex
        from tenacity import Retrying, retry_if_not_exception_type, stop_after_attempt, wait_exponential

        NO_RETRY_ERRORS = (
            gex.PermissionDenied,
            gex.Unauthenticated,
            gex.InvalidArgument,
        )

        for attempt in Retrying(
            retry=retry_if_not_exception_type(NO_RETRY_ERRORS),
            stop=stop_after_attempt(self._retry_attempts),
            wait=wait_exponential(multiplier=1, min=self._retry_wait_seconds, max=self._retry_wait_seconds * 2),
            reraise=True,
        ):
            with attempt:
                try:
                    response = self._client.models.generate_content(
                        model=self._model_name,
                        contents=message,
                        config=self._build_generate_config(),
                    )

                    if not response.candidates:
                        raise RuntimeError(
                            f"Vertex AI response bị block. Feedback: {response.prompt_feedback}"
                        )

                    text = getattr(response, "text", "") or ""
                    if not text.strip():
                        raise RuntimeError("Vertex AI trả response rỗng")
                    return text
                except Exception as e:
                    logging.error(f"[VertexAI Provider] Lỗi API: {type(e).__name__} - {str(e)}")
                    raise e
