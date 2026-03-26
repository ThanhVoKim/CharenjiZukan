from dataclasses import dataclass
from typing import Optional

@dataclass
class SubBlock:
    type:          str            # "tts" | "mute" | "gap" | "tail"
    start_time:    float          # ms (gốc)
    end_time:      float          # ms (gốc)
    slot_duration: float          # ms — khoảng từ start[i] đến next event start
    hard_limit_ms: Optional[float] # ms — giới hạn cứng khi chạm mute zone
    tts_clip_path: Optional[str]  # Path tới dubb-N.wav
    tts_duration:  float          # ms — độ dài TTS sau strip_silence (0 nếu không phải tts)

@dataclass
class TimelineSegment:
    orig_start:    float          # ms
    orig_end:      float          # ms — end of chunk (= start of next chunk)
    new_start:     float          # ms
    new_end:       float          # ms
    video_speed:   float
    audio_speed:   float
    new_chunk_dur: float          # ms
    block_type:    str            # "tts" | "mute" | "gap" | "tail"
    tts_clip_path: Optional[str]
    tts_duration:  float          # ms
