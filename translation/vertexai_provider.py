import logging
from translation.base import BaseTranslationProvider


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
    ):
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except ImportError:
            raise ImportError("google-cloud-aiplatform chưa cài. Chạy: pip install google-cloud-aiplatform>=1.60.0")

        vertexai.init(project=project_id, location=location)
        self._project_id = project_id
        self._model_name = model
        self._generation_config = generation_config
        self._retry_attempts = retry_attempts
        self._retry_wait_seconds = retry_wait_seconds

        self._safety_settings = self._parse_safety_settings(safety_settings)

        system_instruction = [system_prompt] if system_prompt and system_prompt.strip() else None
        self._model = GenerativeModel(model, system_instruction=system_instruction)

    @property
    def name(self) -> str:
        return f"Vertex AI ({self._model_name}, project={self._project_id})"

    def _parse_safety_settings(self, safety_settings: dict) -> dict[object, object] | None:
        """
        Convert YAML safety settings dạng string sang enum map mà Vertex AI SDK yêu cầu.

        Input YAML dạng:
            {
              "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
              ...
            }

        Output cho generate_content:
            {
              HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
              ...
            }
        """
        if not safety_settings:
            return None

        try:
            from vertexai.generative_models import HarmCategory, HarmBlockThreshold
        except ImportError:
            raise ImportError("google-cloud-aiplatform chưa cài. Chạy: pip install google-cloud-aiplatform>=1.60.0")

        if not isinstance(safety_settings, dict):
            raise ValueError(
                "safety_settings phải là object/dict trong YAML, ví dụ: "
                "{HARM_CATEGORY_HATE_SPEECH: BLOCK_NONE}"
            )

        parsed: dict[object, object] = {}

        for category_name, threshold_name in safety_settings.items():
            try:
                category = getattr(HarmCategory, str(category_name))
            except AttributeError as exc:
                valid_categories = [name for name in dir(HarmCategory) if name.startswith("HARM_CATEGORY_")]
                raise ValueError(
                    f"Giá trị category không hợp lệ: '{category_name}'. "
                    f"Các giá trị hợp lệ: {valid_categories}"
                ) from exc

            try:
                threshold = getattr(HarmBlockThreshold, str(threshold_name))
            except AttributeError as exc:
                valid_thresholds = [name for name in dir(HarmBlockThreshold) if name.startswith("BLOCK_")]
                raise ValueError(
                    f"Giá trị threshold không hợp lệ: '{threshold_name}' cho category '{category_name}'. "
                    f"Các giá trị hợp lệ: {valid_thresholds}"
                ) from exc

            parsed[category] = threshold

        return parsed

    def call(self, message: str) -> str:
        from google.api_core import exceptions as gex
        from vertexai.generative_models import GenerationConfig
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
                    gen_config = GenerationConfig(**self._generation_config)

                    response = self._model.generate_content(
                        message,
                        generation_config=gen_config,
                        safety_settings=self._safety_settings,
                    )

                    if not response.candidates:
                        raise RuntimeError(
                            f"Vertex AI response bị block. Feedback: {response.prompt_feedback}"
                        )

                    return response.text
                except Exception as e:
                    logging.error(f"[VertexAI Provider] Lỗi API: {type(e).__name__} - {str(e)}")
                    raise e
