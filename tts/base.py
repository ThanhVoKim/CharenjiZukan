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
