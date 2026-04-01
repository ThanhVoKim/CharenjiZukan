# -*- coding: utf-8 -*-
"""
Box Manager - Quản lý các vùng ảnh cần OCR (ROI Boxes)

Hỗ trợ:
- Định nghĩa cấu trúc OcrBox
- Đọc cấu hình các box từ file txt
- Quản lý trạng thái xử lý độc lập của từng box (BoxState)
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path

# Thêm project root vào path để import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger
from .subtitle_writer import SubtitleEntry

logger = get_logger(__name__)


@dataclass
class OcrBox:
    """Định nghĩa một vùng ảnh cần OCR"""
    name: str
    x: int
    y: int
    w: int
    h: int


@dataclass
class BoxState:
    """Trạng thái xử lý của một box trong quá trình OCR"""
    box: OcrBox
    prev_roi: Optional[np.ndarray] = None
    last_scene_frame: int = 0
    entries: List[SubtitleEntry] = field(default_factory=list)
    current_text: Optional[str] = None
    text_start_time: float = 0.0
    prev_hash: Optional[Any] = None  # Dùng cho imagehash.dhash


def parse_boxes_file(file_path: str) -> List[OcrBox]:
    """
    Đọc cấu hình các box từ file text.
    Format mỗi dòng: "tên_box x y w h"
    Ví dụ: "subtitle 370 930 1180 140"
    
    Args:
        file_path: Đường dẫn tới file txt chứa thông tin các box
        
    Returns:
        List các đối tượng OcrBox
    """
    boxes = []
    if not os.path.exists(file_path):
        logger.warning(f"File boxes configuration không tồn tại: {file_path}")
        return boxes
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            line = line.strip()
            # Bỏ qua dòng trống hoặc comment
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if len(parts) >= 5:
                try:
                    name = parts[0]
                    x = int(parts[1])
                    y = int(parts[2])
                    w = int(parts[3])
                    h = int(parts[4])
                    boxes.append(OcrBox(name=name, x=x, y=y, w=w, h=h))
                except ValueError as e:
                    logger.error(f"Lỗi parse dữ liệu dòng {i+1} trong {file_path}: {line}. Lỗi: {e}")
            else:
                logger.warning(f"Dòng {i+1} trong {file_path} không đúng format (cần 5 phần tử): {line}")
                
        logger.info(f"Đã tải {len(boxes)} boxes từ {file_path}")
    except Exception as e:
        logger.error(f"Lỗi khi đọc file boxes {file_path}: {e}")
        
    return boxes
