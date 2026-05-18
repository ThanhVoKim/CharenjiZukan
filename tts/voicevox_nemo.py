from typing import List, Dict, Any
from tts.voicevox_base import VoicevoxRestTTSEngine


class VoicevoxNemoTTSEngine(VoicevoxRestTTSEngine):
    """
    Async engine tạo audio từng dòng subtitle bằng Voicevox Nemo.

    Voicevox Nemo là phiên bản tối ưu cho GPU NVIDIA, chạy server tại cổng 50121.
    Cài đặt trên Google Colab bằng lệnh: !python setup_voicevox_nemo.py
    """

    def __init__(
        self,
        queue_tts: List[Dict[str, Any]],
        voice_id: int = 10008,
        host: str = "127.0.0.1",
        port: int = 50121,
        speed_scale: float = 1.12,
        pitch_scale: float = -0.05,
        intonation_scale: float = 1.0,
        volume_scale: float = 2.0,
        pre_phoneme_length: float = None,
        post_phoneme_length: float = None,
        output_sampling_rate: int = 48000,
        concurrent_requests: int = 100,
        max_retries: int = 3,
        **kwargs,
    ):
        super().__init__(
            queue_tts=queue_tts,
            engine_label="VoicevoxNemoTTS",
            voice_id=voice_id,
            host=host,
            port=port,
            speed_scale=speed_scale,
            pitch_scale=pitch_scale,
            intonation_scale=intonation_scale,
            volume_scale=volume_scale,
            pre_phoneme_length=pre_phoneme_length,
            post_phoneme_length=post_phoneme_length,
            output_sampling_rate=output_sampling_rate,
            concurrent_requests=concurrent_requests,
            max_retries=max_retries,
            **kwargs,
        )
