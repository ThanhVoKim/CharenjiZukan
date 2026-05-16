from llm_ai.base import BaseLLMProvider, BaseTranslationProvider


def create_provider(*args, **kwargs):
    """Lazy import để tránh kéo dependency nặng tại import time."""
    from llm_ai.factory import create_provider as _create_provider

    return _create_provider(*args, **kwargs)


def create_llm_provider(*args, **kwargs):
    """Alias lazy cho factory provider generic trong llm_ai."""
    from llm_ai.factory import create_llm_provider as _create_llm_provider

    return _create_llm_provider(*args, **kwargs)


def load_provider_config(*args, **kwargs):
    """Lazy load provider config từ llm_ai.factory."""
    from llm_ai.factory import load_provider_config as _load_provider_config

    return _load_provider_config(*args, **kwargs)


def translate_srt_file(*args, **kwargs):
    """Lazy import workflow dịch SRT hiện hành."""
    from translation.srt_translator import translate_srt_file as _translate_srt_file

    return _translate_srt_file(*args, **kwargs)


__all__ = [
    "BaseLLMProvider",
    "BaseTranslationProvider",
    "create_provider",
    "create_llm_provider",
    "load_provider_config",
    "translate_srt_file",
]
