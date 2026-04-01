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
from typing import Optional, Iterator, Tuple, Any
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
        scene_threshold: float = 1.5,
        phash_threshold: int = 4,
        noise_threshold: int = 25,
    ):
        self.frame_interval = frame_interval
        self.scene_threshold = scene_threshold
        self.phash_threshold = phash_threshold
        self.noise_threshold = noise_threshold
        
        logger.info(f"FrameProcessor initialized: interval={frame_interval}, "
                   f"scene_threshold={scene_threshold}%, phash_threshold={phash_threshold}, noise_threshold={noise_threshold}")
    
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
        prev_roi: Optional[np.ndarray],
        prev_hash: Optional[Any] = None
    ) -> Tuple[bool, Optional[Any]]:
        """
        Phát hiện chuyển cảnh giữa 2 ROI (của cùng 1 box qua 2 frame)
        Sử dụng cấu trúc 3 lớp: Exact match (MD5) -> Perceptual Hash (DHash) -> Diff Threshold
        
        Args:
            curr_roi: ROI hiện tại
            prev_roi: ROI trước đó
            prev_hash: Hash của ROI trước đó (nếu có)
            
        Returns:
            Tuple(True/False: có chuyển cảnh không, curr_hash: hash của frame hiện tại)
        """
        if prev_roi is None:
            return True, None
            
        try:
            # ── LAYER 1: Fast exact-match (MD5) ──────────────────────────
            # Giữ lại: nếu frame giống 100% pixel thì chắc chắn không đổi
            if curr_roi.tobytes() == prev_roi.tobytes():
                return False, prev_hash
                
            # Tính hash của current ROI
            curr_hash = None
            try:
                import imagehash
                from PIL import Image
                
                # Resize nhỏ trước khi hash để tăng tốc
                h_curr = imagehash.dhash(
                    Image.fromarray(cv2.cvtColor(curr_roi, cv2.COLOR_BGR2RGB))
                )
                curr_hash = h_curr
                
                # Nếu không truyền prev_hash vào, tính lại
                if prev_hash is None:
                    h_prev = imagehash.dhash(
                        Image.fromarray(cv2.cvtColor(prev_roi, cv2.COLOR_BGR2RGB))
                    )
                else:
                    h_prev = prev_hash
                    
                # ── LAYER 2: Perceptual Hash (dhash) ─────────────────────────
                hamming_dist = h_curr - h_prev
                # Hamming distance = 0: giống hệt về cấu trúc
                # Hamming distance > phash_threshold: cấu trúc thay đổi đáng kể (chữ đổi)
                if hamming_dist <= self.phash_threshold:
                    return False, curr_hash
                # Nếu hash đã khác → bỏ qua layer 3, xác nhận thay đổi luôn
                return True, curr_hash
                
            except ImportError:
                # imagehash chưa cài → fallback xuống layer 3
                pass

            # ── LAYER 3: Threshold + Non-zero count (fix root cause) ─────
            # Thay thế np.mean(diff) để tránh bị loãng trên box lớn
            curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
            
            # Resize nếu kích thước khác nhau
            if curr_gray.shape != prev_gray.shape:
                prev_gray = cv2.resize(prev_gray, (curr_gray.shape[1], curr_gray.shape[0]))
            
            # Tính độ khác biệt tuyệt đối
            diff = cv2.absdiff(prev_gray, curr_gray)
            
            # Loại nhiễu nén video (JPEG artifact, codec noise)
            _, diff_bin = cv2.threshold(diff, self.noise_threshold, 255, cv2.THRESH_BINARY)
            
            # Tỉ lệ % pixel thực sự thay đổi trên tổng diện tích box
            changed_pixels = np.count_nonzero(diff_bin)
            percentage_change = (changed_pixels / diff_bin.size) * 100
            
            logger.debug(
                "Scene change box: changed=%.2f%% threshold=%.2f%% noise_threshold=%d",
                percentage_change,
                self.scene_threshold,
                self.noise_threshold
            )
            
            return percentage_change > self.scene_threshold, curr_hash
            
        except Exception as e:
            logger.error(f"Error in per-box scene detection: {e}")
            return True, None  # Mặc định xử lý nếu có lỗi

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
