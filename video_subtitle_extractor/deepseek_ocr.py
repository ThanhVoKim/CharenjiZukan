# -*- coding: utf-8 -*-
"""
Backward compatibility alias cho DeepSeekOCR
Vui lòng sử dụng video_subtitle_extractor.ocr.DeepSeekOCR hoặc factory function tạo OCR backend
"""

from .ocr.deepseek import DeepSeekOCR

__all__ = ["DeepSeekOCR"]
