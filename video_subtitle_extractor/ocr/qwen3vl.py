# -*- coding: utf-8 -*-
"""
Qwen3-VL Wrapper

Bọc logic gọi model Qwen3-VL thông qua thư viện transformers và qwen_vl_utils.
"""

import os
import re
from typing import Optional, List
import cv2
import numpy as np
from PIL import Image

from utils.logger import get_logger
from .base import BaseOCR

logger = get_logger(__name__)

class Qwen3VLOCR(BaseOCR):
    """
    Wrapper cho model Qwen3-VL trên Hugging Face.
    """
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen3-VL-8B-Thinking", 
        device: str = "cuda", 
        hf_token: Optional[str] = None,
        max_new_tokens: int = 256,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28
    ):
        """
        Khởi tạo model Qwen3-VL.
        
        Args:
            model_name: Tên model trên Hugging Face.
            device: Thiết bị chạy model ("cuda" hoặc "cpu").
            hf_token: Hugging Face Token để tải model (nếu cần).
            max_new_tokens: Số lượng token tối đa tạo ra trong một lần inference.
            min_pixels: Số lượng pixel tối thiểu (VRAM control).
            max_pixels: Số lượng pixel tối đa (VRAM control).
        """
        self.model_name = model_name
        self.device = device
        self.hf_token = hf_token
        self.max_new_tokens = max_new_tokens
        
        # Nếu user truyền số block (256, 1280) thay vì pixel, ta tự nhân với 28*28
        self.min_pixels = min_pixels if min_pixels > 10000 else min_pixels * 28 * 28
        self.max_pixels = max_pixels if max_pixels > 10000 else max_pixels * 28 * 28
        
        self.model = None
        self.processor = None
        
        # Prompt đơn giản để bắt text từ image
        self.prompt = "Read and transcribe all visible text in this image exactly as it appears. Output only the text, nothing else."
        
        self._load_model()
        
    def _load_model(self):
        """Load model và processor từ Hugging Face."""
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            
            logger.info(f"Loading processor from {self.model_name}...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                token=self.hf_token,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels
            )
            
            # Tắt tính năng tự động resize trong bộ xử lý để tránh thay đổi kích thước ảnh hai lần
            if hasattr(self.processor, "image_processor"):
                self.processor.image_processor.do_resize = False
                
            # Đảm bảo padding side bằng left để hoạt động đúng với batch generation
            if hasattr(self.processor, "tokenizer") and self.processor.tokenizer:
                self.processor.tokenizer.padding_side = "left"

            logger.info(f"Loading model from {self.model_name}...")
            
            model_kwargs = {
                "torch_dtype": torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
                "token": self.hf_token
            }
            
            if self.device.startswith("cuda"):
                try:
                    import flash_attn
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Using flash_attention_2 for optimization.")
                except ImportError:
                    logger.warning("flash_attn not found. Falling back to default attention.")
                    
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name, **model_kwargs
            )
            self.model = self.model.eval().to(self.device)
            
            logger.info(f"Qwen3-VL loaded successfully on {self.device}.")
            
        except ImportError as e:
            logger.error(f"Missing required libraries for Qwen3-VL: {e}")
            logger.error("Please install: pip install transformers>=4.57.0 qwen-vl-utils==0.0.14 torch")
            raise
        except Exception as e:
            logger.error(f"Error loading Qwen3-VL: {e}")
            raise

    def _build_messages(self, pil_image: Image.Image) -> list:
        """Xây dựng message chat từ hình ảnh."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

    def _strip_thinking(self, text: str) -> str:
        """Loại bỏ khối <think>...</think> đặc trưng của phiên bản Qwen3-VL-Thinking."""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _apply_hallucination_filter(self, text_result: str) -> str:
        """Bộ lọc các mô tả hình ảnh vô nghĩa thường thấy từ AI khi không tìm thấy text."""
        if not text_result:
            return ""
            
        # Tránh lỗi trả về chuỗi "None" (từ str(None))
        if text_result.strip().lower() == "none":
            return ""
            
        empty_phrases = [
            "图片中没有",
            "没有可见的文字",
            "没有可识别的文字",
            "no visible text",
            "no text found",
            "no text in the image",
            "không có văn bản",
            "文件(f)",
            "编辑(e)",
            "视图(v)",
            "书签(o)"
        ]
        
        lower_result = text_result.lower()
        for phrase in empty_phrases:
            if phrase in lower_result:
                return ""
                
        # Bộ lọc Regex cho các Ảo giác (Hallucination)
        # 1. Ảo giác sinh ra URL
        if re.search(r'https?://[^\s]+', text_result):
            return ""
        
        # 2. Ảo giác sinh ra chuỗi đếm số liên tục (vd: 1\n2\n3\n4...)
        if re.fullmatch(r'[\d\s]+', text_result) and '\n' in text_result:
            return ""
            
        # 3. Ảo giác sinh ra 1 con số vô nghĩa (vd: "1")
        if text_result.strip().isdigit() and len(text_result.strip()) <= 2:
            return ""
            
        return text_result.strip()

    def recognize(self, image: np.ndarray) -> str:
        """Nhận dạng text từ 1 ảnh đơn (numpy array BGR format từ OpenCV)."""
        return self.recognize_batch([image])[0] if image is not None and image.size > 0 else ""
        
    def recognize_batch(self, images: List[np.ndarray]) -> List[str]:
        """
        Nhận dạng text từ một batch các ảnh.
        
        Args:
            images: List ảnh numpy array (BGR format từ OpenCV).
            
        Returns:
            List các chuỗi text được nhận dạng tương ứng.
        """
        if self.model is None or self.processor is None:
            logger.error("Model not loaded.")
            return [""] * len(images)
            
        valid_images = [img for img in images if img is not None and img.size > 0]
        if not valid_images:
            return [""] * len(images)
            
        try:
            from qwen_vl_utils import process_vision_info
            
            # Chuyển OpenCV BGR sang PIL RGB
            pil_images = []
            for img in valid_images:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(img_rgb))
            
            messages_list = [self._build_messages(img) for img in pil_images]
            
            # Xử lý text prompt
            text_inputs = self.processor.apply_chat_template(
                messages_list,
                tokenize=False,
                add_generation_prompt=True,
                padding=True
            )
            
            # Xử lý vision inputs (yêu cầu image_patch_size=16)
            image_inputs, video_inputs = process_vision_info(messages_list, image_patch_size=16)
            
            # Preprocess vào tensors
            inputs = self.processor(
                text=text_inputs,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            # Run inference
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens
            )
            
            # Cắt bỏ phần prompt input ra khỏi output generated_ids
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # Decode outputs
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            # Hậu xử lý kết quả
            results = []
            valid_idx = 0
            for img in images:
                if img is None or img.size == 0:
                    results.append("")
                else:
                    raw_text = output_texts[valid_idx]
                    # Loại bỏ block thinking của Qwen3-VL-Thinking
                    no_think_text = self._strip_thinking(raw_text)
                    # Lọc hallucination
                    filtered_text = self._apply_hallucination_filter(no_think_text)
                    results.append(filtered_text)
                    valid_idx += 1
                    logger.debug(f"OCR Extracted: '{filtered_text[:50]}...'")
                    
            return results
            
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logger.error("CUDA out of memory during Qwen3-VL OCR. Try reducing batch size.")
            else:
                logger.error(f"Error during Qwen3-VL OCR batch recognition: {e}")
            return [""] * len(images)
