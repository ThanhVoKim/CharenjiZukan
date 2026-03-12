# -*- coding: utf-8 -*-
"""
Video Subtitle Extractor - Trích xuất subtitle tiếng Trung từ video

Sử dụng DeepSeek-OCR-2 với các kỹ thuật tối ưu:
- Frame Sampling
- ROI Cropping
- Scene Detection
- Chinese Filter

Usage:
    from video_subtitle_extractor import VideoSubtitleExtractor
    
    extractor = VideoSubtitleExtractor(
        frame_interval=30,
        roi_y_start=0.85,
        scene_threshold=30.0
    )
    result = extractor.extract("video.mp4", "output.srt")
"""

from .extractor import VideoSubtitleExtractor, ExtractionResult
from .frame_processor import FrameProcessor, FrameInfo
from .chinese_filter import ChineseFilter, ChineseText
from .subtitle_writer import SubtitleWriter, SubtitleEntry

__version__ = "1.0.0"
__all__ = [
    "VideoSubtitleExtractor",
    "ExtractionResult",
    "FrameProcessor",
    "FrameInfo",
    "ChineseFilter",
    "ChineseText",
    "SubtitleWriter",
    "SubtitleEntry",
]