import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class BaseTTSEngine(ABC):
    """
    Abstract base class for all TTS engines (EdgeTTS, Voicevox, Voicevox Nemo, Qwen3-TTS, etc.)
    """
    def __init__(self, queue_tts: List[Dict[str, Any]], **kwargs):
        self.queue_tts = queue_tts
        self.kwargs = kwargs

    @abstractmethod
    def run(self) -> Dict[str, int]:
        """
        Thực thi TTS, ghi file ra đĩa và trả về thống kê.
        Returns:
            Dict[str, int]: Thống kê kết quả, ví dụ {"ok": 10, "err": 0}
        """
        pass

    @staticmethod
    def _clean_tail(wav, sr, pre: float = 0.0, post: float = 0.0, fade_ms: float = 8.0, top_db: float = 30.0):
        """Trim silence 2 đầu + fade edges + optional pad. Xử lý cả mono và stereo.

        wav shape: mono=(n_samples,), stereo=(n_samples, n_channels).
        Dùng librosa.effects.trim (lazy import, fallback nếu thiếu).
        Thứ tự: trim → fade → pad (pre/post mặc định 0 để caller quyết định khi nào pad).
        """
        import numpy as np
        y = np.asarray(wav, dtype="float32")
        is_stereo = y.ndim == 2  # shape: (n_samples, n_channels)
        if not is_stereo:
            y = y.reshape(-1)

        if y.shape[0] > 0:
            try:
                import librosa
                y_mono = y.mean(axis=1) if is_stereo else y
                _, (s, e) = librosa.effects.trim(y_mono, top_db=top_db)
                if e > s:
                    y = y[s:e]
            except Exception:
                pass  # thiếu librosa hoặc lỗi → bỏ trim, chỉ fade + pad
            y = y.copy()
            n = y.shape[0]
            f = min(int(sr * fade_ms / 1000), n)
            if f > 0:
                fade_out = np.linspace(1.0, 0.0, f, dtype="float32")
                fade_in  = np.linspace(0.0, 1.0, f, dtype="float32")
                if is_stereo:
                    y[-f:] *= fade_out[:, np.newaxis]
                    y[:f]  *= fade_in[:, np.newaxis]
                else:
                    y[-f:] *= fade_out
                    y[:f]  *= fade_in

        pre_s, post_s = int(sr * pre), int(sr * post)
        if pre_s > 0 or post_s > 0:
            if is_stereo:
                n_ch = y.shape[1]
                return np.concatenate([
                    np.zeros((pre_s, n_ch), dtype="float32"),
                    y,
                    np.zeros((post_s, n_ch), dtype="float32"),
                ])
            return np.pad(y, (pre_s, post_s))
        return y

    @staticmethod
    def _pad_file(wav_path: str, pre: float, post: float) -> None:
        """Thêm silence padding vào đầu/cuối file wav đã ghi (dùng sau speedup)."""
        if pre <= 0 and post <= 0:
            return
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(wav_path)
        pre_s, post_s = int(sr * pre), int(sr * post)
        if data.ndim == 1:
            padded = np.pad(data, (pre_s, post_s), mode="constant")
        else:
            padded = np.pad(data, ((pre_s, post_s), (0, 0)), mode="constant")
        sf.write(wav_path, padded, sr)

    def apply_speed_scale(self) -> None:
        """Tăng tốc các clip đã sinh theo `self.speed_scale` (>1.0 = nhanh hơn, GIỮ pitch).

        Dùng cho engine KHÔNG tăng tốc khi synth (edge/qwen/qwen_custom) — gọi ở CUỐI run()
        sau khi đã ghi wav. Voicevox/Voicevox Nemo áp speedScale ngay lúc synth qua API nên
        KHÔNG gọi hàm này. speed_scale <= 1.0 → no-op (chỉ tăng tốc, không làm chậm).
        """
        speed = getattr(self, "speed_scale", 1.0) or 1.0
        if speed <= 1.0:
            return
        from speed_rate import speedup_to_factor  # import trễ: tránh kéo pydub/ffmpeg khi không cần
        ok = 0
        for it in self.queue_tts:
            f = it.get("filename")
            if f and Path(f).exists() and Path(f).stat().st_size > 0:
                if speedup_to_factor(f, speed):
                    ok += 1
        logger.info(f"[speed_scale] Đã tăng tốc {ok} clip theo x{speed:.2f}")
