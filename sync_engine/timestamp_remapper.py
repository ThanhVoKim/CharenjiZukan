from typing import List, Dict

from sync_engine.models import TimelineSegment
from sync_engine.analyzer import remap_timestamp
from utils.srt_parser import segments_to_srt, wrap_subtitle_text
from utils.media_utils import parse_ass_timestamp_to_ms, ms_to_ass_timestamp
from utils.ass_utils import wrap_text

def recalculate_srt(
    segments: List[dict],
    timeline: List[TimelineSegment],
    output_path: str,
    is_tts_track: bool = False,
    max_chars: int = 0,
    fps_float: float = 30.0,
) -> None:
    """
    Sửa lỗi #4: Với TTS track, end_time được neo theo tts_duration thực tế.
    Tránh subtitle biến mất sớm trong khi audio vẫn đang phát.
    """
    # Lookup: orig_start_ms → TimelineSegment (chỉ với TTS)
    tts_seg_lookup: Dict[int, TimelineSegment] = {
        int(round(seg.orig_start)): seg
        for seg in timeline if seg.block_type == "tts"
    }

    # Cần copy segments để không sửa đổi inplace list gốc
    new_segments = []
    
    for sub in segments:
        new_sub = sub.copy()
        new_start = remap_timestamp(new_sub["start_time"], timeline, fps_float)

        if is_tts_track:
            key = int(round(new_sub["start_time"]))
            # Cần tìm segment chứa timestamp này (do filter có thể đổi start_time một chút, hoặc do precision)
            # Tốt hơn là loop tìm segment có khoảng chứa start_time
            ts_seg = None
            for seg in timeline:
                if seg.block_type == "tts" and abs(seg.orig_start - new_sub["start_time"]) < 100:
                    ts_seg = seg
                    break
                    
            if ts_seg and ts_seg.tts_duration > 0:
                # Anchor: subtitle hiển thị đúng bằng thời gian audio phát
                new_end = new_start + ts_seg.tts_duration
                # Không vượt quá new_end của slot
                new_end = min(new_end, ts_seg.new_end)
            else:
                new_end = remap_timestamp(new_sub["end_time"], timeline, fps_float)
        else:
            new_end = remap_timestamp(new_sub["end_time"], timeline, fps_float)

        if new_end <= new_start:
            new_end = new_start + 100  # Min 100ms

        new_sub["start_time"] = int(round(new_start))
        new_sub["end_time"]   = int(round(new_end))
        
        if max_chars > 0:
            new_sub["text"] = wrap_subtitle_text(new_sub["text"], max_chars)
            
        new_segments.append(new_sub)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(segments_to_srt(new_segments))

def recalculate_ass(
    input_path: str,
    timeline: List[TimelineSegment],
    output_path: str,
    max_chars_per_line: int = 15,
    fps_float: float = 30.0,
) -> None:
    """
    Remap timestamps ASS + wrap text tại max_chars_per_line.
    wrap_text() từ utils/ass_utils.py (đã có sẵn, chỉ truyền max_chars=15).
    """
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    out_lines = []
    for line in lines:
        if line.startswith("Dialogue:"):
            parts = line.rstrip("\n").split(",", 9)
            if len(parts) >= 10:
                start_ms = parse_ass_timestamp_to_ms(parts[1].strip())
                end_ms   = parse_ass_timestamp_to_ms(parts[2].strip())

                parts[1] = ms_to_ass_timestamp(int(round(remap_timestamp(start_ms, timeline, fps_float))))
                parts[2] = ms_to_ass_timestamp(int(round(remap_timestamp(end_ms,   timeline, fps_float))))

                # wrap_text có sẵn trong ass_utils, chỉ đổi max_chars=15
                parts[9] = wrap_text(parts[9].rstrip("\n"), max_chars=max_chars_per_line)

                line = ",".join(parts) + "\n"
        out_lines.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
