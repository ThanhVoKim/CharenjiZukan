# -*- coding: utf-8 -*-
"""
Frame Processor - Xử lý frame video với các kỹ thuật tối ưu

Bao gồm:
- Frame Sampling: Lấy mỗi N frame
- ROI Cropping: Cắt vùng quan tâm (subtitle area)
- Scene Detection: Phát hiện chuyển cảnh
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
import sys
from pathlib import Path

# Thêm project root vào path để import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FrameInfo:
    """Thông tin về một frame cần OCR"""
    frame_number: int
    timestamp: float  # seconds
    frame_data: np.ndarray
    is_scene_change: bool = False
    roi_data: Optional[np.ndarray] = None


class FrameProcessor:
    """
    Xử lý frame video với các kỹ thuật tối ưu:
    
    1. Frame Sampling - Lấy mỗi N frame thay vì tất cả
    2. ROI Cropping - Chỉ xử lý vùng subtitle (thường ở dưới video)
    3. Scene Detection - Chỉ OCR khi có chuyển cảnh
    
    Attributes:
        frame_interval: Số frame bỏ qua giữa mỗi lần xử lý (default: 30)
        roi_y_start: Vị trí bắt đầu ROI theo tỷ lệ (0-1, default: 0.85)
        roi_y_end: Vị trí kết thúc ROI theo tỷ lệ (0-1, default: 1.0)
        scene_threshold: Ngưỡng phát hiện chuyển cảnh (default: 30.0)
        min_scene_frames: Số frame tối thiểu giữa 2 scene (default: 10)
    """
    
    def __init__(
        self,
        frame_interval: int = 30,
        roi_y_start: float = 0.85,
        roi_y_end: float = 1.0,
        scene_threshold: float = 30.0,
        min_scene_frames: int = 10
    ):
        self.frame_interval = frame_interval
        self.roi_y_start = roi_y_start
        self.roi_y_end = roi_y_end
        self.scene_threshold = scene_threshold
        self.min_scene_frames = min_scene_frames
        
        # Internal state
        self._prev_frame: Optional[np.ndarray] = None
        self._last_scene_frame: int = 0
        
        logger.info(f"FrameProcessor initialized: interval={frame_interval}, "
                   f"roi={roi_y_start}-{roi_y_end}, threshold={scene_threshold}")
    
    def should_process_frame(self, frame_number: int) -> bool:
        """
        Kiểm tra frame có nên được xử lý không (Frame Sampling)
        
        Args:
            frame_number: Số thứ tự frame
            
        Returns:
            True nếu frame nên được xử lý
        """
        return frame_number % self.frame_interval == 0
    
    def crop_roi(self, frame: np.ndarray) -> np.ndarray:
        """
        Cắt vùng quan tâm (ROI) - vùng subtitle
        
        Subtitle thường nằm ở dưới cùng của video (khoảng 85-100% chiều cao)
        
        Args:
            frame: Video frame (BGR format)
            
        Returns:
            Vùng ROI đã cắt
        """
        height, width = frame.shape[:2]
        
        # Tính toán vị trí cắt
        y_start = int(height * self.roi_y_start)
        y_end = int(height * self.roi_y_end)
        
        # Đảm bảo bounds hợp lệ
        y_start = max(0, min(y_start, height))
        y_end = max(0, min(y_end, height))
        
        if y_start >= y_end:
            logger.warning(f"Invalid ROI bounds: y_start={y_start}, y_end={y_end}")
            return frame
        
        return frame[y_start:y_end, 0:width]
    
    def detect_scene_change(
        self, 
        curr_frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None
    ) -> bool:
        """
        Phát hiện chuyển cảnh giữa 2 frame
        
        Sử dụng độ khác biệt trung bình giữa 2 frame grayscale.
        Nếu độ khác biệt > threshold => có chuyển cảnh.
        
        Args:
            curr_frame: Frame hiện tại
            prev_frame: Frame trước đó (nếu None, coi như có chuyển cảnh)
            
        Returns:
            True nếu có chuyển cảnh, False nếu không
        """
        if prev_frame is None:
            return True
        
        try:
            # Chuyển sang grayscale để so sánh
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
            # Resize nếu kích thước khác nhau
            if curr_gray.shape != prev_gray.shape:
                prev_gray = cv2.resize(prev_gray, (curr_gray.shape[1], curr_gray.shape[0]))
            
            # Tính độ khác biệt tuyệt đối
            diff = cv2.absdiff(prev_gray, curr_gray)
            
            # Tính trung bình độ khác biệt
            mean_diff = np.mean(diff)
            
            return mean_diff > self.scene_threshold
            
        except Exception as e:
            logger.error(f"Error in scene detection: {e}")
            return True  # Mặc định xử lý nếu có lỗi
    
    def process_frame(
        self, 
        frame: np.ndarray, 
        frame_number: int,
        fps: float
    ) -> Optional[FrameInfo]:
        """
        Xử lý một frame với tất cả các kỹ thuật tối ưu
        
        Workflow:
        1. Frame Sampling - Skip nếu không phải frame cần xử lý
        2. Scene Detection - Skip nếu không có chuyển cảnh
        3. ROI Cropping - Cắt vùng subtitle
        
        Args:
            frame: Video frame (BGR format)
            frame_number: Số thứ tự frame
            fps: Frames per second của video
            
        Returns:
            FrameInfo nếu frame cần OCR, None nếu skip
        """
        # 1. Frame Sampling
        if not self.should_process_frame(frame_number):
            return None
        
        # 2. Scene Detection
        is_scene_change = self.detect_scene_change(frame, self._prev_frame)
        
        # Skip nếu không có chuyển cảnh và đã xử lý gần đây
        frames_since_last_scene = frame_number - self._last_scene_frame
        if not is_scene_change and frames_since_last_scene < self.min_scene_frames:
            self._prev_frame = frame.copy()
            return None
        
        if is_scene_change:
            self._last_scene_frame = frame_number
            logger.debug(f"Scene change detected at frame {frame_number}")
        
        # 3. ROI Cropping
        roi = self.crop_roi(frame)
        
        # Tạo FrameInfo
        frame_info = FrameInfo(
            frame_number=frame_number,
            timestamp=frame_number / fps,
            frame_data=frame,
            is_scene_change=is_scene_change,
            roi_data=roi
        )
        
        self._prev_frame = frame.copy()
        return frame_info
    
    def extract_frames(
        self, 
        video_path: str,
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[FrameInfo], dict]:
        """
        Trích xuất các frame từ video cần OCR
        
        Args:
            video_path: Đường dẫn file video
            progress_callback: Callback function cho progress update
                              callback(current_frame, total_frames, frames_to_ocr)
            
        Returns:
            Tuple của (List FrameInfo, metadata dict)
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"📹 Video: {video_path}")
        logger.info(f"   FPS: {fps:.2f}, Total frames: {total_frames}")
        logger.info(f"   Resolution: {width}x{height}")
        logger.info(f"   Processing every {self.frame_interval} frames "
                   f"(~{total_frames // self.frame_interval} frames to process)")
        
        frames: List[FrameInfo] = []
        frame_number = 0
        scene_changes = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_info = self.process_frame(frame, frame_number, fps)
            if frame_info:
                frames.append(frame_info)
                if frame_info.is_scene_change:
                    scene_changes += 1
            
            frame_number += 1
            
            # Progress callback
            if progress_callback and frame_number % 100 == 0:
                progress_callback(frame_number, total_frames, len(frames))
        
        cap.release()
        
        # Metadata
        metadata = {
            "video_path": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "processed_frames": len(frames),
            "scene_changes": scene_changes,
            "frame_interval": self.frame_interval,
            "roi_y_start": self.roi_y_start,
            "roi_y_end": self.roi_y_end,
        }
        
        logger.info(f"✅ Extracted {len(frames)} frames for OCR "
                   f"({scene_changes} scene changes)")
        
        return frames, metadata
    
    def reset(self):
        """Reset internal state for processing new video"""
        self._prev_frame = None
        self._last_scene_frame = 0
        logger.debug("FrameProcessor state reset")