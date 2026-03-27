import wave
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from sync_engine.models import SubBlock, TimelineSegment

logger = logging.getLogger(__name__)

def filter_tts_subtitles(
    subtitle_segments: List[dict],
    mute_segments: List[dict],
) -> List[dict]:
    """
    Xóa bỏ bất kỳ subtitle block nào có overlap với mute region.
    Đánh lại line number từ 1.
    Output mapping: dubb-0.wav <-> tts_blocks[0], dubb-1.wav <-> tts_blocks[1], ...
    """
    tts_blocks = []
    for seg in subtitle_segments:
        overlaps_mute = any(
            seg["start_time"] < mute["end_time"] and
            seg["end_time"]   > mute["start_time"]
            for mute in mute_segments
        )
        if not overlaps_mute:
            tts_blocks.append(seg)

    for i, seg in enumerate(tts_blocks, 1):
        seg["line"] = i
    return tts_blocks

def _get_wav_duration_ms(wav_path: str) -> float:
    """Đọc duration của WAV file (ms), không cần ffprobe."""
    try:
        from pydub import AudioSegment
        return float(len(AudioSegment.from_file(wav_path)))
    except ImportError:
        # Fallback to wave if pydub is not available
        if not Path(wav_path).exists():
            return 0.0
        with wave.open(wav_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return (frames / float(rate)) * 1000.0
    except Exception as e:
        logger.warning(f"Không đọc được duration {wav_path}: {e}")
        return 0.0

def classify_and_compute_slots(
    subtitle_segments: List[dict],
    mute_segments: List[dict],
    video_duration_ms: float,
    tts_dir: Optional[str] = None
) -> List[SubBlock]:
    """
    Phân loại block: "tts" | "mute" | "gap"
    Tính slot_duration và hard_limit_ms.
    """
    events = []
    
    # 1. Thu thập tts blocks
    tts_blocks = filter_tts_subtitles(subtitle_segments, mute_segments)
    for i, seg in enumerate(tts_blocks):
        tts_clip = None
        tts_dur = 0.0
        if tts_dir:
            clip_path = Path(tts_dir) / f"dubb-{i}.wav"
            if clip_path.exists():
                tts_clip = str(clip_path)
                tts_dur = _get_wav_duration_ms(tts_clip)
            else:
                logger.warning(f"Không tìm thấy TTS clip: {clip_path}")
        
        events.append({
            "type": "tts",
            "start": seg["start_time"],
            "end": seg["end_time"],
            "tts_clip_path": tts_clip,
            "tts_duration": tts_dur
        })
        
    # 2. Thu thập mute blocks
    for mute in mute_segments:
        events.append({
            "type": "mute",
            "start": mute["start_time"],
            "end": mute["end_time"],
            "tts_clip_path": None,
            "tts_duration": 0.0
        })
        
    # Sắp xếp events theo start_time
    events.sort(key=lambda x: x["start"])
    
    blocks: List[SubBlock] = []
    
    # 3. Tính gap và tạo blocks
    cursor = 0.0
    for i, ev in enumerate(events):
        if ev["start"] > cursor:
            # Có gap
            blocks.append(SubBlock(
                type="gap",
                start_time=cursor,
                end_time=ev["start"],
                slot_duration=ev["start"] - cursor,
                hard_limit_ms=None,
                tts_clip_path=None,
                tts_duration=0.0
            ))
            
        slot_ms = 0.0
        hard_limit = None
        
        if ev["type"] == "mute":
            slot_ms = ev["end"] - ev["start"]
            hard_limit = None
            
        elif ev["type"] == "tts":
            # Tìm event tiếp theo
            next_start = video_duration_ms
            if i + 1 < len(events):
                next_start = events[i+1]["start"]
                
            # Kiểm tra next mute
            next_mute_start = None
            for j in range(i+1, len(events)):
                if events[j]["type"] == "mute":
                    next_mute_start = events[j]["start"]
                    break
                    
            if next_mute_start is None or next_start < next_mute_start:
                slot_ms = next_start - ev["start"]
                hard_limit = None
            else:
                slot_ms = next_mute_start - ev["start"]
                hard_limit = next_mute_start - ev["start"]
                
            # Nếu block cuối không có event sau, fallback
            if slot_ms <= 0:
                slot_ms = ev["end"] - ev["start"]
                
        blocks.append(SubBlock(
            type=ev["type"],
            start_time=ev["start"],
            end_time=ev["end"] if ev["type"] == "mute" else (ev["start"] + slot_ms), # Tạm thời end_time cho tts
            slot_duration=slot_ms,
            hard_limit_ms=hard_limit,
            tts_clip_path=ev["tts_clip_path"],
            tts_duration=ev["tts_duration"]
        ))
        
        cursor = ev["start"] + slot_ms if ev["type"] == "tts" else ev["end"]

    return blocks

def compute_speeds(
    tts_ms: float,
    slot_ms: float,
    hard_limit_ms: Optional[float] = None,
    cap: float = 0.5
) -> Tuple[float, float, float]:
    """
    Returns (video_speed, audio_speed, new_chunk_duration_ms).
    """
    if tts_ms <= 0:
        return 1.0, 1.0, slot_ms
        
    effective = hard_limit_ms if hard_limit_ms is not None else slot_ms
    max_str   = effective / cap

    if tts_ms <= effective:
        return 1.0, 1.0, slot_ms

    if tts_ms <= max_str:
        return effective / tts_ms, 1.0, tts_ms

    return cap, tts_ms / max_str, max_str

def build_timeline_map(
    all_blocks: List[SubBlock],
    speeds: List[Tuple[float, float, float]],  # Per block: (vs, as_, new_dur)
    video_duration_ms: float,
) -> List[TimelineSegment]:

    segments: List[TimelineSegment] = []
    cursor = 0.0

    # ── Head gap (trước subtitle đầu tiên) ───────────────────────────
    if all_blocks and all_blocks[0].start_time > 0:
        head = all_blocks[0].start_time
        # Tuy nhiên nếu block đầu là gap thì không cần xử lý thêm vì classify đã thêm gap rồi
        if all_blocks[0].type != "gap":
            segments.append(TimelineSegment(
                orig_start=0.0, orig_end=head,
                new_start=0.0,  new_end=head,
                video_speed=1.0, audio_speed=1.0, new_chunk_dur=head,
                block_type="gap", tts_clip_path=None, tts_duration=0.0,
            ))
            cursor = head

    # ── Các block chính ──────────────────────────────────────────────
    for i, block in enumerate(all_blocks):
        vs, as_, new_dur = speeds[i]
        orig_end = block.start_time + block.slot_duration
        segments.append(TimelineSegment(
            orig_start=block.start_time,
            orig_end=orig_end,
            new_start=cursor,
            new_end=cursor + new_dur,
            video_speed=vs,
            audio_speed=as_,
            new_chunk_dur=new_dur,
            block_type=block.type,
            tts_clip_path=block.tts_clip_path,
            tts_duration=block.tts_duration,
        ))
        cursor += new_dur

    # ── Tail: phần video sau block cuối cùng ← BUG FIX #1 ───────────
    if all_blocks:
        tail_start = all_blocks[-1].start_time + all_blocks[-1].slot_duration
        if tail_start < video_duration_ms:
            tail_dur = video_duration_ms - tail_start
            segments.append(TimelineSegment(
                orig_start=tail_start, orig_end=video_duration_ms,
                new_start=cursor,      new_end=cursor + tail_dur,
                video_speed=1.0, audio_speed=1.0, new_chunk_dur=tail_dur,
                block_type="tail", tts_clip_path=None, tts_duration=0.0,
            ))

    return segments

def remap_timestamp(orig_ms: float, timeline: List[TimelineSegment]) -> float:
    """Nội suy tuyến tính trong segment chứa timestamp."""
    if not timeline:
        return orig_ms
        
    for seg in timeline:
        if seg.orig_start <= orig_ms <= seg.orig_end:
            span = seg.orig_end - seg.orig_start
            if span <= 0:
                return seg.new_start
            ratio = (orig_ms - seg.orig_start) / span
            return seg.new_start + ratio * seg.new_chunk_dur
            
    # Extrapolate từ segment cuối
    last = timeline[-1]
    if orig_ms > last.orig_end:
        return last.new_end + (orig_ms - last.orig_end)
    
    # Trước segment đầu
    first = timeline[0]
    return first.new_start - (first.orig_start - orig_ms)
