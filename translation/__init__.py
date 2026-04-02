from translation.base import BaseTranslationProvider


def create_provider(*args, **kwargs):
    """Lazy import để tránh kéo dependency nặng tại import time."""
    from translation.factory import create_provider as _create_provider
    return _create_provider(*args, **kwargs)


__all__ = ["BaseTranslationProvider", "create_provider"]
