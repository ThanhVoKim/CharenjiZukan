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
    Xử lý subtitle blocks overlap với mute:
    - Overlap hoàn toàn (start >= mute_start) → drop.
    - Overlap phần đuôi (start < mute_start <= end) → clip end_time về mute_start.
    - Không overlap → giữ nguyên.
    """
    mute_sorted = sorted(mute_segments, key=lambda m: m["start_time"])
    tts_blocks = []

    for seg in subtitle_segments:
        clipped = dict(seg)  # copy để không sửa dữ liệu gốc
        drop = False

        for mute in mute_sorted:
            mute_start = mute["start_time"]
            mute_end   = mute["end_time"]

            # Không overlap gì cả → bỏ qua mute này
            if clipped["end_time"] <= mute_start or clipped["start_time"] >= mute_end:
                continue

            # Overlap hoàn toàn: start của sub nằm trong vùng mute → drop
            if clipped["start_time"] >= mute_start:
                drop = True
                break

            # Overlap phần đuôi: start < mute_start < end → clip end về mute_start
            if clipped["start_time"] < mute_start < clipped["end_time"]:
                clipped["end_time"] = mute_start
                logger.debug(
                    "Clipped sub %s end_time %s → %s (mute starts at %s)",
                    clipped.get("line"),
                    seg["end_time"],
                    mute_start,
                    mute_start,
                )
                break  # Một mute đã clip rồi, không cần kiểm tra mute tiếp theo

        if drop:
            logger.debug("Dropped sub %s (fully inside mute region)", seg.get("line"))
            continue

        # Sau khi clip, nếu duration quá ngắn (< 100ms) thì drop
        if clipped["end_time"] - clipped["start_time"] < 100:
            logger.debug(
                "Dropped sub %s after clip: duration too short (%sms)",
                seg.get("line"),
                clipped["end_time"] - clipped["start_time"],
            )
            continue

        tts_blocks.append(clipped)

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
    Tính slot_duration và hard_limit_ms theo logic Sequential Builder (chỉ tiến không lùi).
    Đảm bảo 100% các block nối tiếp nhau khít rịt, không chồng lấn, không sai thứ tự.
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
        
    # Sắp xếp events:
    # Cấp 1: start_time tăng dần (làm tròn để khử sai số float)
    # Cấp 2: Nếu start trùng, ưu tiên "mute" xếp trước (khung xương cứng)
    events.sort(key=lambda x: (round(x["start"]), 0 if x["type"] == "mute" else 1))
    
    blocks: List[SubBlock] = []
    cursor = 0.0
    
    # 3. Tính gap và tạo blocks tuần tự (Sequential State Machine)
    for i, ev in enumerate(events):
        # Bỏ qua các event nằm hoàn toàn trong quá khứ do bị overlap
        if ev["end"] <= cursor:
            continue
            
        # Ép event bắt đầu từ cursor nếu nó bị lấn vào quá khứ (Overlap xử lý lỗi)
        actual_start = max(cursor, ev["start"])
        
        # Nếu có khoảng trống từ cursor đến actual_start, lấp đầy bằng một block GAP
        if actual_start > cursor:
            gap_dur = actual_start - cursor
            blocks.append(SubBlock(
                type="gap",
                start_time=cursor,
                end_time=actual_start,
                slot_duration=gap_dur,
                hard_limit_ms=None,
                tts_clip_path=None,
                tts_duration=0.0
            ))
            cursor = actual_start
            
        # Xử lý nội dung của chính event đó
        if ev["type"] == "mute":
            # Mute là cứng, bắt buộc phải lấy hết duration thực tế của nó (end - start gốc)
            mute_dur = ev["end"] - ev["start"]
            actual_end = actual_start + mute_dur
            blocks.append(SubBlock(
                type="mute",
                start_time=actual_start,
                end_time=actual_end,
                slot_duration=mute_dur,
                hard_limit_ms=None,
                tts_clip_path=None,
                tts_duration=0.0
            ))
            cursor = actual_end
            
        elif ev["type"] == "tts":
            # Không gian của TTS có thể bị giới hạn bởi event kế tiếp.
            # Ta tìm xem event tiếp theo (dù là TTS hay MUTE) sẽ bắt đầu khi nào.
            next_start = video_duration_ms
            next_mute_start = None
            
            for next_ev in events[i+1:]:
                # Chỉ lấy những event thực sự nằm trong tương lai
                if next_ev["start"] >= actual_start:
                    if next_start == video_duration_ms: # Chỉ cập nhật next_start lần đầu tiên gặp
                        next_start = next_ev["start"]
                        
                    if next_ev["type"] == "mute" and next_mute_start is None:
                        next_mute_start = next_ev["start"]
                        
                    if next_start != video_duration_ms and next_mute_start is not None:
                        break # Đã tìm đủ thông tin
            
            # Slot duration của TTS là khoảng trống từ lúc nó bắt đầu cho đến event kế tiếp
            slot_ms = next_start - actual_start
            
            # Hard limit là khoảng trống cho đến block mute kế tiếp
            # Tuy nhiên, nếu next event KHÔNG PHẢI là mute, ta không nên set hard_limit
            # vì hard_limit chỉ dùng để giới hạn việc giãn video không vượt qua mute.
            hard_limit = None
            if next_mute_start is not None and next_mute_start == next_start:
                hard_limit = next_mute_start - actual_start
                
            # Fallback nếu slot_ms <= 0 (do các event trùng đè nhau quá gắt)
            if slot_ms <= 0:
                slot_ms = ev["end"] - ev["start"]
                if hard_limit is not None and hard_limit < slot_ms:
                    slot_ms = hard_limit
                    
            if slot_ms > 0:
                blocks.append(SubBlock(
                    type="tts",
                    start_time=actual_start,
                    end_time=actual_start + slot_ms,
                    slot_duration=slot_ms,
                    hard_limit_ms=hard_limit,
                    tts_clip_path=ev["tts_clip_path"],
                    tts_duration=ev["tts_duration"]
                ))
                cursor = actual_start + slot_ms

    # Thêm gap cuối cùng nếu còn khoảng trống đến cuối video
    if cursor < video_duration_ms:
        gap_dur = video_duration_ms - cursor
        blocks.append(SubBlock(
            type="gap",
            start_time=cursor,
            end_time=video_duration_ms,
            slot_duration=gap_dur,
            hard_limit_ms=None,
            tts_clip_path=None,
            tts_duration=0.0
        ))

    return blocks

def compute_speeds(
    tts_ms: float,
    slot_ms: float,
    cap: float = 0.5,
    hard_limit_ms: Optional[float] = None,
    no_cap: bool = False,       # True → bỏ Case 3, video slow thoải mái
) -> Tuple[float, float, float]:
    """
    no_cap=True (Voicevox mode):
        Chỉ có Case 1 và Case 2. Không bao giờ compress audio.
        Video slow bao nhiêu cũng được để khớp TTS.
    """
    effective = hard_limit_ms if hard_limit_ms is not None else slot_ms

    if tts_ms <= effective:
        # Case 1: TTS vừa slot, không cần làm gì
        return 1.0, 1.0, slot_ms

    if no_cap:
        # Voicevox mode: kéo video chậm thoải mái, không compress audio
        # video_speed = effective / tts_ms (có thể rất nhỏ, e.g. 0.1x)
        video_speed = effective / tts_ms
        return video_speed, 1.0, tts_ms

    # EdgeTTS mode: có cap
    max_str = effective / cap

    if tts_ms <= max_str:
        # Case 2: slow video vừa đủ
        return effective / tts_ms, 1.0, tts_ms

    # Case 3: slow max + compress audio
    return cap, tts_ms / max_str, max_str

def build_timeline_map(
    all_blocks: List[SubBlock],
    speeds: List[Tuple[float, float, float]],  # Per block: (vs, as_, new_dur)
    video_duration_ms: float,
    fps_float: float = 30.0,
) -> List[TimelineSegment]:

    segments: List[TimelineSegment] = []
    cursor = 0.0

    # ── Head gap (trước subtitle đầu tiên) ───────────────────────────
    if all_blocks and all_blocks[0].start_time > 0:
        head = all_blocks[0].start_time
        # Tuy nhiên nếu block đầu là gap thì không cần xử lý thêm vì classify đã thêm gap rồi
        if all_blocks[0].type != "gap":
            head_frames = round((head / 1000.0) * fps_float)
            head_dur_snapped = (head_frames / fps_float) * 1000.0
            segments.append(TimelineSegment(
                orig_start=0.0, orig_end=head,
                new_start=0.0,  new_end=head_dur_snapped,
                video_speed=1.0, audio_speed=1.0, new_chunk_dur=head_dur_snapped,
                block_type="gap", tts_clip_path=None, tts_duration=0.0,
            ))
            cursor = head_dur_snapped

    # ── Các block chính ──────────────────────────────────────────────
    for i, block in enumerate(all_blocks):
        vs, as_, _ = speeds[i]  # Bỏ qua new_dur cũ chưa làm tròn
        orig_end = block.start_time + block.slot_duration
        
        # Snap time của block này thành chuẩn frame giống hệt lúc tạo chunk video
        duration_frames = round((block.slot_duration / 1000.0) * fps_float)
        pts_factor = 1.0 / vs
        new_duration_frames = round(duration_frames * pts_factor)
        new_dur_snapped = (new_duration_frames / fps_float) * 1000.0
        
        segments.append(TimelineSegment(
            orig_start=block.start_time,
            orig_end=orig_end,
            new_start=cursor,
            new_end=cursor + new_dur_snapped,
            video_speed=vs,
            audio_speed=as_,
            new_chunk_dur=new_dur_snapped,
            block_type=block.type,
            tts_clip_path=block.tts_clip_path,
            tts_duration=block.tts_duration,
        ))
        cursor += new_dur_snapped

    # ── Tail: phần video sau block cuối cùng ← BUG FIX #1 ───────────
    if all_blocks:
        tail_start = all_blocks[-1].start_time + all_blocks[-1].slot_duration
        if tail_start < video_duration_ms:
            tail_dur = video_duration_ms - tail_start
            tail_frames = round((tail_dur / 1000.0) * fps_float)
            tail_dur_snapped = (tail_frames / fps_float) * 1000.0
            
            segments.append(TimelineSegment(
                orig_start=tail_start, orig_end=video_duration_ms,
                new_start=cursor,      new_end=cursor + tail_dur_snapped,
                video_speed=1.0, audio_speed=1.0, new_chunk_dur=tail_dur_snapped,
                block_type="tail", tts_clip_path=None, tts_duration=0.0,
            ))

    return segments

def recalculate_timeline_from_actual_durations(
    timeline: List[TimelineSegment],
    actual_durations: List[float],
    fps_float: float,
) -> List[TimelineSegment]:
    """
    Cập nhật new_chunk_dur, new_start, new_end dựa trên duration thực tế từ FFmpeg.
    Giữ nguyên orig_start, orig_end, video_speed, audio_speed, block_type, tts_...
    """
    updated = []
    cursor = 0.0
    for seg, actual_dur in zip(timeline, actual_durations):
        new_seg = TimelineSegment(
            orig_start=seg.orig_start,
            orig_end=seg.orig_end,
            new_start=cursor,
            new_end=cursor + actual_dur,
            video_speed=seg.video_speed,
            audio_speed=seg.audio_speed,
            new_chunk_dur=actual_dur,
            block_type=seg.block_type,
            tts_clip_path=seg.tts_clip_path,
            tts_duration=seg.tts_duration,
        )
        updated.append(new_seg)
        cursor += actual_dur
    return updated

def remap_timestamp(
    orig_ms: float,
    timeline: List[TimelineSegment],
    fps_float: float = 30.0,
) -> float:
    """Nội suy tuyến tính trong segment chứa timestamp, snap về frame."""
    if not timeline:
        return orig_ms

    for seg in timeline:
        if seg.orig_start <= orig_ms <= seg.orig_end:
            span = seg.orig_end - seg.orig_start
            if span <= 0:
                return seg.new_start
            ratio = (orig_ms - seg.orig_start) / span
            raw_new = seg.new_start + ratio * seg.new_chunk_dur
            # Snap về frame-aligned
            new_frames = round((raw_new / 1000.0) * fps_float)
            return (new_frames / fps_float) * 1000.0

    # Extrapolate từ segment cuối
    last = timeline[-1]
    if orig_ms > last.orig_end:
        raw_new = last.new_end + (orig_ms - last.orig_end)
        new_frames = round((raw_new / 1000.0) * fps_float)
        return (new_frames / fps_float) * 1000.0

    # Trước segment đầu
    first = timeline[0]
    raw_new = first.new_start - (first.orig_start - orig_ms)
    new_frames = round((raw_new / 1000.0) * fps_float)
    return (new_frames / fps_float) * 1000.0
