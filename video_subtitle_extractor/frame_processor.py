# -*- coding: utf-8 -*-
"""
Frame Processor - Xử lý frame video với các kỹ thuật tối ưu

Bao gồm:
- Frame Sampling: Kiểm tra frame cần xử lý
- ROI Cropping: Cắt vùng quan tâm (subtitle area)
- Scene Detection: Phát hiện chuyển cảnh trên từng vùng box
"""

import cv2
import numpy as np
import hashlib
from typing import Optional
import sys
from pathlib import Path

# Thêm project root vào path để import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger
from .box_manager import OcrBox

logger = get_logger(__name__)


class FrameProcessor:
    """
    Xử lý frame video với các kỹ thuật tối ưu:
    
    1. Frame Sampling - Kiểm tra frame nào cần xử lý
    2. ROI Cropping - Cắt theo tọa độ tuyệt đối của từng box
    3. Scene Detection - Phát hiện chuyển cảnh cho từng box
    
    Attributes:
        frame_interval: Số frame bỏ qua giữa mỗi lần xử lý (default: 30)
        scene_threshold: Ngưỡng phát hiện chuyển cảnh (default: 30.0)
    """
    
    def __init__(
        self,
        frame_interval: int = 30,
        scene_threshold: float = 30.0,
    ):
        self.frame_interval = frame_interval
        self.scene_threshold = scene_threshold
        
        logger.info(f"FrameProcessor initialized: interval={frame_interval}, "
                   f"threshold={scene_threshold}")
    
    def should_process_frame(self, frame_number: int) -> bool:
        """
        Kiểm tra frame có nên được xử lý không (Frame Sampling)
        
        Args:
            frame_number: Số thứ tự frame
            
        Returns:
            True nếu frame nên được xử lý
        """
        return frame_number % self.frame_interval == 0
    
    def crop_roi(self, frame: np.ndarray, box: OcrBox) -> np.ndarray:
        """
        Cắt vùng quan tâm (ROI) dựa vào tọa độ tuyệt đối của box
        
        Args:
            frame: Video frame (BGR format)
            box: OcrBox chứa tọa độ
            
        Returns:
            Vùng ROI đã cắt
        """
        height, width = frame.shape[:2]
        
        # Đảm bảo bounds hợp lệ
        x1 = max(0, min(box.x, width))
        y1 = max(0, min(box.y, height))
        x2 = max(0, min(box.x + box.w, width))
        y2 = max(0, min(box.y + box.h, height))
        
        if x1 >= x2 or y1 >= y2:
            logger.warning(f"Invalid box bounds for '{box.name}': x={x1}-{x2}, y={y1}-{y2}")
            return np.zeros((1, 1, 3), dtype=np.uint8) # Trả về ảnh 1x1 đen nếu lỗi
            
        return frame[y1:y2, x1:x2]
    
    def detect_scene_change_for_box(
        self, 
        curr_roi: np.ndarray,
        prev_roi: Optional[np.ndarray]
    ) -> bool:
        """
        Phát hiện chuyển cảnh giữa 2 ROI (của cùng 1 box qua 2 frame)
        Sử dụng hash-based comparison kết hợp pixel diff nếu cần,
        hoặc đơn giản tính pixel diff.
        
        Args:
            curr_roi: ROI hiện tại
            prev_roi: ROI trước đó
            
        Returns:
            True nếu có chuyển cảnh, False nếu không
        """
        if prev_roi is None:
            return True
            
        try:
            # 1. Fast check bằng Hash (nếu ROI hoàn toàn giống nhau từng pixel)
            # Tối ưu cho trường hợp video tĩnh hoàn toàn tại vùng box
            curr_hash = hashlib.md5(curr_roi.tobytes()).hexdigest()
            prev_hash = hashlib.md5(prev_roi.tobytes()).hexdigest()
            if curr_hash == prev_hash:
                return False
                
            # 2. Chuyển sang grayscale để so sánh
            curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
            
            # Resize nếu kích thước khác nhau
            if curr_gray.shape != prev_gray.shape:
                prev_gray = cv2.resize(prev_gray, (curr_gray.shape[1], curr_gray.shape[0]))
            
            # Tính độ khác biệt tuyệt đối
            diff = cv2.absdiff(prev_gray, curr_gray)
            
            # Tính trung bình độ khác biệt
            mean_diff = np.mean(diff)
            
            return mean_diff > self.scene_threshold
            
        except Exception as e:
            logger.error(f"Error in per-box scene detection: {e}")
            return True  # Mặc định xử lý nếu có lỗi
