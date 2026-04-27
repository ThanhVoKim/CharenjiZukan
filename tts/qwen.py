# -*- coding: utf-8 -*-
"""
tts/qwen.py — Qwen3-TTS Engine

Lazy-load torch & qwen_tts để tránh ngốn RAM khi không dùng.
"""

import sys
import gc
import logging
from pathlib import Path
from typing import List, Dict, Any

from tts.base import BaseTTSEngine

logger = logging.getLogger("qwen_tts")


class QwenTTSEngine(BaseTTSEngine):
    """
    Engine TTS dùng Qwen3-TTS (HuggingFace Transformers).
    Hỗ trợ voice-clone qua ref_audio + ref_text.
    """

    def __init__(
        self,
        queue_tts: List[Dict[str, Any]],
        model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        ref_audio: str = "",
        ref_text: str = "",
        batch_size: int = 32,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        gen_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(queue_tts, **kwargs)
        self.model_path = model_path
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.gen_kwargs = gen_kwargs or {
            "max_new_tokens": 2048,
            "do_sample": True,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.9,
            "repetition_penalty": 1.05,
            "subtalker_dosample": True,
            "subtalker_top_k": 50,
            "subtalker_top_p": 1.0,
            "subtalker_temperature": 0.9,
        }

    def run(self) -> Dict[str, int]:
        # Lazy import — chỉ load khi thực sự chạy
        try:
            import torch
            import soundfile as sf
            import numpy as np
            from qwen_tts import Qwen3TTSModel
        except ImportError as e:
            logger.error(f"❌ Thiếu thư viện cho QwenTTS: {e}")
            logger.error("   Cài đặt: pip install qwen-tts transformers accelerate soundfile")
            return {"ok": 0, "err": len(self.queue_tts)}

        model = None
        ok_count = 0
        err_count = 0

        try:
            logger.info(f"[QwenTTS] Khởi tạo model: {self.model_path}")
            torch_dtype = getattr(torch, self.dtype, torch.bfloat16)
            model = Qwen3TTSModel.from_pretrained(
                self.model_path,
                device_map=self.device,
                dtype=torch_dtype,
                attn_implementation=self.attn_implementation,
            )

            # Tạo voice clone prompt nếu có ref_audio hợp lệ
            voice_prompt = None
            if self.ref_audio and Path(self.ref_audio).exists() and self.ref_text:
                logger.info("[QwenTTS] Đang trích xuất đặc trưng giọng mẫu...")
                voice_prompt = model.create_voice_clone_prompt(
                    ref_audio=self.ref_audio,
                    ref_text=self.ref_text,
                    x_vector_only_mode=False,
                )

            # Lọc item có text
            valid_items = [it for it in self.queue_tts if it.get("text", "").strip()]
            total = len(valid_items)
            logger.info(f"[QwenTTS] Bắt đầu xử lý {total} dòng, batch_size={self.batch_size}")

            for i in range(0, total, self.batch_size):
                batch_items = valid_items[i : i + self.batch_size]
                batch_texts = [it["text"] for it in batch_items]
                batch_langs = ["Auto"] * len(batch_texts)

                logger.info(
                    f"[QwenTTS] Batch {i + 1}-{min(i + self.batch_size, total)} / {total}"
                )

                with torch.no_grad():
                    kwargs = {}
                    if voice_prompt is not None:
                        kwargs["voice_clone_prompt"] = voice_prompt

                    wav_outputs, sr = model.generate_voice_clone(
                        text=batch_texts,
                        language=batch_langs,
                        ref_audio=self.ref_audio if voice_prompt is None else None,
                        ref_text=self.ref_text if voice_prompt is None else None,
                        x_vector_only_mode=False,
                        **self.gen_kwargs,
                        **kwargs,
                    )

                for j, wav_data in enumerate(wav_outputs):
                    item = batch_items[j]
                    wav_path = item["filename"]
                    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)

                    if isinstance(wav_data, torch.Tensor):
                        wav_data_safe = wav_data.detach().cpu().numpy()
                    else:
                        wav_data_safe = np.copy(wav_data)

                    sf.write(wav_path, wav_data_safe, sr)
                    ok_count += 1

                # Dọn VRAM sau mỗi batch
                del wav_outputs
                gc.collect()
                torch.cuda.empty_cache()

        except Exception as e:
            logger.exception(f"[QwenTTS] Lỗi nghiêm trọng: {e}")
            err_count = len(valid_items) - ok_count
        finally:
            if model is not None:
                del model
            gc.collect()
            if "torch" in sys.modules:
                import torch

                torch.cuda.empty_cache()
            logger.info(f"[QwenTTS] Xong: {ok_count} thành công, {err_count} lỗi")

        return {"ok": ok_count, "err": err_count}
