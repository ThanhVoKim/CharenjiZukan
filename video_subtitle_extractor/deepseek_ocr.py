# -*- coding: utf-8 -*-
"""
DeepSeek-OCR-2 Wrapper

Bọc logic gọi model DeepSeek-OCR-2 thông qua thư viện transformers.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

class DeepSeekOCR:
    """
    Wrapper cho model DeepSeek-OCR-2 trên Hugging Face.
    """
    
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-OCR-2", device: str = "cuda", hf_token: Optional[str] = None):
        """
        Khởi tạo model DeepSeek-OCR-2.
        
        Args:
            model_name: Tên model trên Hugging Face.
            device: Thiết bị chạy model ("cuda" hoặc "cpu").
            hf_token: Hugging Face Token để tải model (nếu cần).
        """
        self.model_name = model_name
        self.device = device
        self.hf_token = hf_token
        self.model = None
        self.tokenizer = None
        
        # Prompt mặc định cho OCR không có layout (phù hợp cho subtitle)
        self.prompt = "<image>\nFree OCR. "
        
        self._load_model()
        
    def _load_model(self):
        """Load model và tokenizer từ Hugging Face."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            
            logger.info(f"Loading tokenizer from {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                token=self.hf_token
            )
            
            logger.info(f"Loading model from {self.model_name}...")
            
            # Cấu hình model
            model_kwargs = {
                "trust_remote_code": True,
                "use_safetensors": True,
                "token": self.hf_token
            }
            
            # Sử dụng flash_attention_2 nếu có GPU và thư viện flash_attn
            if self.device.startswith("cuda"):
                try:
                    import flash_attn
                    model_kwargs["_attn_implementation"] = "flash_attention_2"
                    logger.info("Using flash_attention_2 for optimization.")
                except ImportError:
                    logger.warning("flash_attn not found. Falling back to default attention.")
                    logger.warning("To speed up, install: pip install flash-attn --no-build-isolation")
            
            self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
            
            # Chuyển model sang device và dtype phù hợp
            if self.device.startswith("cuda"):
                self.model = self.model.eval().cuda().to(torch.bfloat16)
            else:
                self.model = self.model.eval().to(self.device)
                
            logger.info("DeepSeek-OCR-2 loaded successfully.")
            
        except ImportError as e:
            logger.error(f"Missing required libraries for DeepSeek-OCR-2: {e}")
            logger.error("Please install: pip install transformers torch torchvision einops")
            raise
        except Exception as e:
            logger.error(f"Error loading DeepSeek-OCR-2: {e}")
            raise

    def recognize(self, image: np.ndarray) -> str:
        """
        Nhận dạng text từ ảnh (numpy array).
        
        Args:
            image: Ảnh numpy array (BGR format từ OpenCV).
            
        Returns:
            Text được nhận dạng.
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model not loaded.")
            return ""
            
        if image is None or image.size == 0:
            return ""
            
        # Tạo file ảnh tạm vì model.infer yêu cầu đường dẫn file
        temp_file = None
        try:
            # Tạo file tạm với đuôi .jpg
            fd, temp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            temp_file = temp_path
            
            # Lưu ảnh numpy array ra file tạm
            cv2.imwrite(temp_file, image)
            
            # Gọi model infer
            # Lưu ý: output_path='' để không lưu kết quả render ra file
            res = self.model.infer(
                self.tokenizer, 
                prompt=self.prompt, 
                image_file=temp_file, 
                output_path="", 
                base_size=1024, 
                image_size=768, 
                crop_mode=True, 
                save_results=False
            )
            
            # Trích xuất text từ kết quả trả về
            # Tùy thuộc vào format trả về của model.infer, thường là string hoặc dict
            if isinstance(res, str):
                return res
            elif isinstance(res, dict) and "text" in res:
                return res["text"]
            else:
                return str(res)
                
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logger.error("CUDA out of memory during OCR. Try reducing batch size or using CPU.")
            else:
                logger.error(f"Error during OCR recognition: {e}")
            return ""
            
        finally:
            # Xóa file tạm
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {temp_file}: {e}")
