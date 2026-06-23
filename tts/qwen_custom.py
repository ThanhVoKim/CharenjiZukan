# -*- coding: utf-8 -*-
"""
tts/qwen_custom.py — Qwen3-TTS Custom Voice Engine (checkpoint fine-tune)

Dùng cho checkpoint đã full-SFT thành 1 giọng cố định (vd Karlsson DE). Khác QwenTTSEngine
(voice-clone qua ref_audio/ref_text): engine này gọi `generate_custom_voice(speaker=...)`.

Đặc thù Colab: checkpoint nằm trên Google Drive (bền) nhưng load thẳng từ Drive FUSE rất chậm +
dễ lỗi I/O với file ~3.5GB → engine COPY Drive→/content 1 lần rồi load từ local.

Tái dùng từ tts/qwen.py: DEFAULT_QWEN_GEN_KWARGS, _clean_tail / _postprocess (hậu xử lý đuôi clip).
"""

import sys
import gc
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any

from tts.base import BaseTTSEngine
from tts.qwen import QwenTTSEngine, DEFAULT_QWEN_GEN_KWARGS

logger = logging.getLogger("qwen_custom_tts")


class QwenCustomTTSEngine(BaseTTSEngine):
    """Engine TTS dùng checkpoint Qwen3-TTS fine-tune (1 giọng cố định, generate_custom_voice)."""

    def __init__(
        self,
        queue_tts: List[Dict[str, Any]],
        drive_ckpt: str = "",
        local_ckpt: str = "",
        speaker: str = "",
        language: str = "German",
        batch_size: int = 32,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        pre_phoneme_length: float = 0.0,
        post_phoneme_length: float = 0.0,
        clean_tail: bool = True,
        fade_ms: float = 8.0,
        trim_top_db: float = 30.0,
        speed_scale: float = 1.0,
        gen_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(queue_tts, **kwargs)
        self.drive_ckpt = drive_ckpt
        self.local_ckpt = local_ckpt
        self.speaker = speaker
        self.language = language
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.pre_phoneme_length = pre_phoneme_length
        self.post_phoneme_length = post_phoneme_length
        self.clean_tail = clean_tail
        self.fade_ms = fade_ms
        self.trim_top_db = trim_top_db
        self.speed_scale = speed_scale
        self.gen_kwargs = gen_kwargs or dict(DEFAULT_QWEN_GEN_KWARGS)

    def _resolve_model_path(self) -> str:
        """Trả về path local để load. Đã có local → dùng luôn; chưa có → copy Drive→local.

        Bắt buộc có Drive: nếu cả local_ckpt lẫn drive_ckpt đều không tồn tại → raise lỗi rõ ràng.
        """
        local = Path(self.local_ckpt) if self.local_ckpt else None
        if local and local.exists():
            logger.info(f"[QwenCustom] Dùng checkpoint local sẵn có: {local}")
            return str(local)

        drive = Path(self.drive_ckpt) if self.drive_ckpt else None
        if not (drive and drive.exists()):
            raise FileNotFoundError(
                "[QwenCustom] Không tìm thấy checkpoint. Cần local_ckpt tồn tại, hoặc drive_ckpt "
                f"trỏ tới thư mục checkpoint trên Drive (Colab). drive_ckpt={self.drive_ckpt!r} "
                f"local_ckpt={self.local_ckpt!r}"
            )
        if not local:
            # Không khai báo local_ckpt → load thẳng từ Drive (chậm hơn, nhưng vẫn chạy).
            logger.warning("[QwenCustom] Thiếu local_ckpt → load thẳng từ Drive (chậm hơn).")
            return str(drive)

        logger.info(f"[QwenCustom] Copy checkpoint Drive→local: {drive} → {local}")
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(drive, local)
        return str(local)

    def run(self) -> Dict[str, int]:
        # Lazy import — chỉ load khi thực sự chạy
        try:
            import torch
            import soundfile as sf
            import numpy as np
            from qwen_tts import Qwen3TTSModel
        except ImportError as e:
            logger.error(f"❌ Thiếu thư viện cho QwenCustom: {e}")
            logger.error("   Cài đặt: pip install qwen-tts transformers accelerate soundfile")
            return {"ok": 0, "err": len(self.queue_tts)}

        if self.clean_tail:
            try:
                import librosa  # noqa: F401
            except Exception:
                logger.warning(
                    "[QwenCustom] clean_tail bật nhưng thiếu librosa → bỏ trim, chỉ fade+pad "
                    "(cài: pip install librosa)."
                )

        model = None
        ok_count = 0
        err_count = 0
        valid_items = []

        try:
            model_path = self._resolve_model_path()
            logger.info(f"[QwenCustom] Khởi tạo model: {model_path}")
            torch_dtype = getattr(torch, self.dtype, torch.bfloat16)
            model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=self.device,
                dtype=torch_dtype,
                attn_implementation=self.attn_implementation,
            )

            # Lọc item có text
            valid_items = [it for it in self.queue_tts if it.get("text", "").strip()]
            total = len(valid_items)
            logger.info(
                f"[QwenCustom] Bắt đầu xử lý {total} dòng, speaker={self.speaker}, "
                f"batch_size={self.batch_size}"
            )

            for i in range(0, total, self.batch_size):
                batch_items = valid_items[i : i + self.batch_size]
                batch_texts = [it["text"] for it in batch_items]
                batch_langs = [self.language] * len(batch_texts)

                logger.info(
                    f"[QwenCustom] Batch {i + 1}-{min(i + self.batch_size, total)} / {total}"
                )

                with torch.no_grad():
                    wav_outputs, sr = model.generate_custom_voice(
                        text=batch_texts,
                        speaker=self.speaker,
                        language=batch_langs,
                        **self.gen_kwargs,
                    )

                for j, wav_data in enumerate(wav_outputs):
                    item = batch_items[j]
                    wav_path = item["filename"]
                    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)

                    if isinstance(wav_data, torch.Tensor):
                        wav_data_safe = wav_data.detach().cpu().numpy()
                    else:
                        wav_data_safe = np.copy(wav_data)
                    wav_data_safe = np.asarray(wav_data_safe, dtype="float32").reshape(-1)

                    # Hậu xử lý đuôi clip (Design A) — dùng chung helper với QwenTTSEngine.
                    # Pad pre/post được thêm SAU speedup (xem bên dưới) để giữ đúng độ dài.
                    wav_data_safe = QwenTTSEngine._postprocess(
                        wav_data_safe, sr, self.clean_tail,
                        self.pre_phoneme_length, self.post_phoneme_length,
                        fade_ms=self.fade_ms, top_db=self.trim_top_db,
                        include_pad=False,
                    )

                    sf.write(wav_path, wav_data_safe, sr)
                    ok_count += 1

                # Dọn VRAM sau mỗi batch
                del wav_outputs
                gc.collect()
                torch.cuda.empty_cache()

            # Tăng tốc theo speed_scale, rồi pad pre/post phoneme silence — thứ tự này đảm bảo
            # padding không bị nén theo speed_scale (padding = silence cố định sau dub).
            self.apply_speed_scale()
            if self.pre_phoneme_length > 0 or self.post_phoneme_length > 0:
                for it in valid_items:
                    p = Path(it["filename"])
                    if p.exists():
                        QwenTTSEngine._pad_file(str(p), self.pre_phoneme_length, self.post_phoneme_length)

        except Exception as e:
            logger.exception(f"[QwenCustom] Lỗi nghiêm trọng: {e}")
            err_count = len(valid_items) - ok_count
        finally:
            if model is not None:
                del model
            gc.collect()
            if "torch" in sys.modules:
                import torch
                torch.cuda.empty_cache()
            logger.info(f"[QwenCustom] Xong: {ok_count} thành công, {err_count} lỗi")

        return {"ok": ok_count, "err": err_count}
