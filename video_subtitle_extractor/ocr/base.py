# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np
from typing import List

class BaseOCR(ABC):
    """Abstract base class cho tất cả OCR backends."""

    @abstractmethod
    def recognize(self, image: np.ndarray) -> str:
        """Nhận dạng text từ ảnh numpy array (BGR format từ OpenCV)."""
        pass

    def recognize_batch(self, images: List[np.ndarray]) -> List[str]:
        """Batch OCR - mặc định gọi recognize từng ảnh một."""
        return [self.recognize(img) for img in images]
