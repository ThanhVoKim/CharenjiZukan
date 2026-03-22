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
from typing import Optional, Iterator, Tuple
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

    def has_text_content(
        self,
        roi: np.ndarray,
        min_edge_density: float = 0.03,
        low_threshold: int = 50,
        high_threshold: int = 150
    ) -> bool:
        """
        Tiền lọc ROI bằng OpenCV để ước lượng xem có khả năng chứa text hay không.

        Ý tưởng:
        - Chữ thường tạo nhiều cạnh sắc nét (edges)
        - ROI không có text thường có mật độ cạnh rất thấp

        Args:
            roi: Ảnh ROI của box hiện tại (BGR)
            min_edge_density: Ngưỡng tối thiểu mật độ edge (0.0-1.0)
            low_threshold: Ngưỡng thấp cho Canny
            high_threshold: Ngưỡng cao cho Canny

        Returns:
            True nếu ROI có khả năng chứa text, False nếu khả năng cao là frame trống
        """
        if roi is None or roi.size == 0:
            return False

        # Bảo vệ cấu hình sai
        if low_threshold < 0:
            low_threshold = 0
        if high_threshold <= low_threshold:
            high_threshold = low_threshold + 1

        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Giảm nhiễu nhẹ để edge ổn định hơn
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, low_threshold, high_threshold)

            total_pixels = edges.size
            if total_pixels == 0:
                return False

            edge_pixels = int(np.count_nonzero(edges))
            edge_density = edge_pixels / float(total_pixels)

            logger.debug(
                "CV prefilter edge_density=%.4f (min=%.4f, low=%d, high=%d)",
                edge_density,
                min_edge_density,
                low_threshold,
                high_threshold,
            )

            return edge_density >= min_edge_density

        except Exception as e:
            # Fail-open: khi prefilter lỗi thì vẫn cho OCR chạy để tránh mất subtitle thật
            logger.warning(f"CV prefilter error, fallback to OCR: {e}")
            return True


def iter_sampled_frames(
    video_path: str,
    frame_interval: int = 6,
) -> Iterator[Tuple[int, float, np.ndarray]]:
    """
    Generator duyệt video và yield các frame được lấy mẫu.

    Args:
        video_path: Đường dẫn tới video local.
        frame_interval: Lấy 1 frame sau mỗi N frame (mặc định: 6).

    Yields:
        Tuple gồm (frame_number, timestamp_seconds, frame_bgr_numpy)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_number = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % frame_interval == 0:
                timestamp = frame_number / fps
                yield frame_number, timestamp, frame

            frame_number += 1
    finally:
        cap.release()
