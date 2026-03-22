# -*- coding: utf-8 -*-
"""
Video Subtitle Extractor - Trích xuất subtitle tiếng Trung từ video

Sử dụng DeepSeek-OCR-2 với các kỹ thuật tối ưu:
- Frame Sampling
- Multi-Box ROI Cropping
- Selective Scene Detection
- Chinese Filter

Usage:
    from video_subtitle_extractor import VideoSubtitleExtractor
    from video_subtitle_extractor.box_manager import OcrBox
    
    boxes = [OcrBox(name="subtitle", x=0, y=800, w=1920, h=280)]
    extractor = VideoSubtitleExtractor(
        boxes=boxes,
        frame_interval=30,
        scene_threshold=30.0
    )
    result = extractor.extract("video.mp4")
"""

from .extractor import VideoSubtitleExtractor, ExtractionResult
from .native_video_extractor import NativeVideoSubtitleExtractor, NativeExtractionResult
from .frame_processor import FrameProcessor
from .chinese_filter import ChineseFilter, ChineseText
from .subtitle_writer import SubtitleWriter, SubtitleEntry
from .box_manager import OcrBox, BoxState, parse_boxes_file
from .ocr import BaseOCR, Qwen3VLOCR, create_ocr_backend

__version__ = "1.1.0"
__all__ = [
    "VideoSubtitleExtractor",
    "ExtractionResult",
    "NativeVideoSubtitleExtractor",
    "NativeExtractionResult",
    "FrameProcessor",
    "ChineseFilter",
    "ChineseText",
    "SubtitleWriter",
    "SubtitleEntry",
    "OcrBox",
    "BoxState",
    "parse_boxes_file",
    "BaseOCR",
    "Qwen3VLOCR",
    "create_ocr_backend"
]
