# -*- coding: utf-8 -*-
from typing import Optional
from .base import BaseOCR

def create_ocr_backend(
    model_name: str,
    device: str = "cuda",
    hf_token: Optional[str] = None,
    **kwargs
) -> BaseOCR:
    """Factory: Tạo Qwen3VLOCR backend."""
    from .qwen3vl import Qwen3VLOCR
    return Qwen3VLOCR(
        model_name=model_name,
        device=device,
        hf_token=hf_token,
        max_new_tokens=kwargs.get("qwen_max_new_tokens", 256),
        min_pixels=kwargs.get("qwen_min_pixels", 256 * 28 * 28),
        max_pixels=kwargs.get("qwen_max_pixels", 1280 * 28 * 28),
    )
