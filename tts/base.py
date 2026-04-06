from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseTTSEngine(ABC):
    """
    Abstract base class for all TTS engines (EdgeTTS, Voicevox, etc.)
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
