from llm_ai.base import BaseLLMProvider, BaseTranslationProvider
from llm_ai.provider_chain import FallbackLLMProvider, ProviderChainError


def create_provider(*args, **kwargs):
    """Lazy import để tránh kéo dependency provider nặng tại import time."""
    from llm_ai.factory import create_provider as _create_provider

    return _create_provider(*args, **kwargs)


def create_llm_provider(*args, **kwargs):
    """Alias mô tả rõ hơn cho create_provider()."""
    from llm_ai.factory import create_llm_provider as _create_llm_provider

    return _create_llm_provider(*args, **kwargs)


__all__ = [
    "BaseLLMProvider",
    "BaseTranslationProvider",
    "FallbackLLMProvider",
    "ProviderChainError",
    "create_provider",
    "create_llm_provider",
]
