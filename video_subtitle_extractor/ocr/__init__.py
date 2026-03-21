# -*- coding: utf-8 -*-
from .base import BaseOCR
from .deepseek import DeepSeekOCR
from .qwen3vl import Qwen3VLOCR
from .factory import create_ocr_backend

__all__ = ["BaseOCR", "DeepSeekOCR", "Qwen3VLOCR", "create_ocr_backend"]
