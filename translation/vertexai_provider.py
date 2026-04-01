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
        self._safety_settings = safety_settings
        self._retry_attempts = retry_attempts
        self._retry_wait_seconds = retry_wait_seconds

        system_instruction = [system_prompt] if system_prompt and system_prompt.strip() else None
        self._model = GenerativeModel(model, system_instruction=system_instruction)

    @property
    def name(self) -> str:
        return f"Vertex AI ({self._model_name}, project={self._project_id})"

    def call(self, message: str) -> str:
        from google.api_core import exceptions as gex
        from vertexai.generative_models import GenerationConfig
        from tenacity import Retrying, stop_after_attempt, wait_exponential, retry_if_not_exception_type

        NO_RETRY_ERRORS = (
            gex.PermissionDenied,
            gex.Unauthenticated,
            gex.InvalidArgument,
        )

        for attempt in Retrying(
            retry=retry_if_not_exception_type(NO_RETRY_ERRORS),
            stop=stop_after_attempt(self._retry_attempts),
            wait=wait_exponential(multiplier=1, min=self._retry_wait_seconds, max=self._retry_wait_seconds * 2),
            reraise=True
        ):
            with attempt:
                try:
                    gen_config = GenerationConfig(**self._generation_config)
                    
                    response = self._model.generate_content(
                        message,
                        generation_config=gen_config,
                        safety_settings=self._safety_settings if self._safety_settings else None,
                    )

                    if not response.candidates:
                        raise RuntimeError(
                            f"Vertex AI response bị block. Feedback: {response.prompt_feedback}"
                        )

                    return response.text
                except Exception as e:
                    import logging
                    logging.error(f"[VertexAI Provider] Lỗi API: {type(e).__name__} - {str(e)}")
                    raise e
