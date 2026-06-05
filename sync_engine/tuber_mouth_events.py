"""
sync_engine/tuber_mouth_events.py
==================================
Analyze TTS audio clips → mouthEvents lists per segment.

Phase V2-P1: thay mouth mode ``cue`` bằng ``amplitude``. Đọc RMS amplitude
từng TTS WAV → danh sách {frame, state} cho mỗi segment TTS. Segment không TTS
(mute/gap/tail) → mouth state always 'closed', không sinh events.

Dùng soundfile (numpy) hoặc wave stdlib + struct.
"""
from __future__ import annotations

import logging
import math
import struct
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sync_video")

# ── Constants ─────────────────────────────────────────────────────────────
MOUTH_STATES = ("closed", "half", "open")
"""Các state discrete. Mặc định 3 state. Config mouthStates mở rộng sau."""

_DEFAULT_SILENCE_DB = -40.0
_DEFAULT_MIN_SILENCE_MS = 200.0
_DEFAULT_CADENCE_MS = 150.0


# ── Audio helpers ─────────────────────────────────────────────────────────

def _read_wav_rms(
    wav_path: Path,
    fps: float,
) -> List[float]:
    """Đọc file WAV mono → RMS per frame (window=fps). Trả về list RMS.

    WAV có thể là int16 (pcm_s16le). Tính RMS đơn giản:
      sum(samples**2) / len(samples) → sqrt → RMS.

    Chia audio thành khung = 1/fps giây. Mỗi khung trả về 1 RMS.
    Nếu file không đọc được → raise RuntimeError.
    """
    try:
        with wave.open(str(wav_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except (wave.Error, OSError) as exc:
        raise RuntimeError(f"Không đọc được WAV {wav_path}: {exc}")

    if n_frames == 0:
        return []  # silent clip

    # Chỉ xử lý pcm_s16le (sampwidth=2) — phổ biến nhất cho TTS.
    fmt = "<" + "h" * (n_channels * n_frames)
    try:
        samples = struct.unpack(fmt, raw)
    except struct.error:
        raise RuntimeError(f"Không parse được sample: {wav_path} (sampwidth={sampwidth})")

    # Trộn channels về mono nếu stereo: trung bình 2 kênh
    if n_channels > 1:
        mono = [
            sum(samples[i * n_channels:(i + 1) * n_channels]) / n_channels
            for i in range(n_frames)
        ]
    else:
        mono = list(samples)

    if not mono:
        return []

    # Chia thành window = 1/fps giây
    win_size = max(1, round(framerate / fps))
    rms_list = []
    for start in range(0, len(mono), win_size):
        chunk = mono[start:start + win_size]
        sq_sum = sum(s * s for s in chunk)
        rms = math.sqrt(sq_sum / len(chunk)) if sq_sum > 0 else 0.0
        rms_list.append(rms)

    return rms_list


def _rms_to_db(rms: float) -> float:
    """Chuyển RMS amplitude → dB (âm thanh, RMS normalized [0, 1])."""
    if rms <= 0:
        return -float("inf")
    return 20.0 * math.log10(rms / 32768.0) if rms > 0 else -float("inf")


def _rms_normalized(rms_list: List[float], min_db: float) -> float:
    """Trả về amplitude tương đối [0,1] từ dB trung bình.

    min_db: ngưỡng coi là silence (dB dưới mức này → amplitude 0).
    """
    if not rms_list:
        return 0.0
    db_values = [_rms_to_db(r) for r in rms_list if r > 0]
    if not db_values:
        return 0.0
    avg_db = sum(db_values) / len(db_values)
    if avg_db <= min_db:
        return 0.0
    # Map [min_db, 0] → [0, 1]
    return min(1.0, max(0.0, (avg_db - min_db) / (-min_db)))


# ── Mouth event generation ────────────────────────────────────────────────

def _state_from_amplitude(
    amplitude: float,
    num_states: int = 3,
) -> str:
    """Map amplitude [0,1] → mouth state string.

    Với 3 state: [0..0.01)=closed, [0.01..0.5)=half, [0.5..1]=open.
    Với n state: chia đều khoảng.
    """
    if amplitude < 0.01:
        return "closed"
    if num_states <= 2:
        return "half" if amplitude < 0.5 else "open"
    if num_states == 3:
        if amplitude < 0.2:
            return "half"
        return "open"
    # N > 3 state: chia đều
    idx = min(num_states - 1, int(amplitude * num_states))
    if idx < num_states - 1 and amplitude < (idx + 0.5) / num_states:
        idx = max(0, idx - 1)
    return MOUTH_STATES[min(idx, len(MOUTH_STATES) - 1)]


def analyze_tts_amplitude(
    tts_clip_path: Path,
    fps: float,
    *,
    silence_db: float = _DEFAULT_SILENCE_DB,
    min_silence_ms: float = _DEFAULT_MIN_SILENCE_MS,
    cadence_ms: float = _DEFAULT_CADENCE_MS,
    num_mouth_states: int = 3,
    mode: str = "amplitude",
) -> List[Dict[str, Any]]:
    """Phân tích RMS amplitude TTS audio → danh sách {frame, state}.

    Args:
        tts_clip_path: Path tới TTS WAV.
        fps: Frame rate của video.
        silence_db: Ngưỡng dB — dưới mức này = im lặng.
        min_silence_ms: Bỏ qua khoảng silence ngắn hơn (tránh nhấp nháy).
        cadence_ms: Tốc độ cadence khi đang nói (chỉ dùng mode hybrid).
        num_mouth_states: Số mouth states (mặc định 3: closed/half/open).
        mode: "amplitude" (mặc định) | "hybrid" (debounce bằng cadence_ms).

    Returns:
        List[{frame, state}]: mỗi entry = mouth state tại frame đó.
        Frame liên tục; frame nằm giữa 2 event → dùng state của event gần nhất.
    """
    rms_list = _read_wav_rms(tts_clip_path, fps)

    if not rms_list:
        logger.info("TTS clip rỗng → closed: %s", tts_clip_path.name)
        return [{"frame": 0, "state": "closed"}]

    cadence_frames = max(1, round(cadence_ms / 1000.0 * fps))

    # Tính state mỗi frame
    events: List[Dict[str, Any]] = []
    for frame_idx, rms in enumerate(rms_list):
        db = _rms_to_db(rms)
        is_silent = db < silence_db

        if is_silent:
            state = "closed"
        else:
            amplitude = _rms_normalized(rms_list[max(0, frame_idx - 2):frame_idx + 1], silence_db)
            state = _state_from_amplitude(amplitude, num_mouth_states)

        if not events or events[-1]["state"] != state:
            if mode == "hybrid" and state != "closed" and events:
                # Debounce: không chuyển non-closed state nếu chưa đủ cadence
                last_ev = events[-1]
                if last_ev["state"] != "closed" and (frame_idx - last_ev["frame"]) < cadence_frames:
                    continue
            events.append({"frame": frame_idx, "state": state})

    # Đảm bảo có ít nhất 1 event
    if not events:
        events.append({"frame": 0, "state": "closed"})

    # Merge silence ngắn (< min_silence_ms)
    if min_silence_ms > 0:
        min_frames = max(1, round(min_silence_ms / 1000.0 * fps))
        events = _merge_short_silence(events, min_frames)

    return events


def _merge_short_silence(
    events: List[Dict[str, Any]],
    min_frames: int,
) -> List[Dict[str, Any]]:
    """Merge khoảng 'closed' ngắn (< min_frames) với state lân cận.

    Ví dụ: closed(2 frame) ở giữa open → bỏ 2 closed đi, gộp open.
    Sau đó merge các state giống nhau liên tiếp.
    """
    if len(events) < 3:
        return events

    result: List[Dict[str, Any]] = []
    for i, ev in enumerate(events):
        if (
            ev["state"] == "closed"
            and 0 < i < len(events) - 1
            and events[i - 1]["state"] == events[i + 1]["state"]
        ):
            # Closed bị kẹp giữa 2 cùng state → kiểm tra duration
            dur = (
                (events[i + 1]["frame"] - ev["frame"])
                if i < len(events) - 1
                else min_frames + 1
            )
            if dur <= min_frames:
                # Bỏ event closed này
                continue
        result.append(ev)

    # Merge consecutive same states (sau khi xóa closed)
    if not result:
        return result
    merged: List[Dict[str, Any]] = [result[0]]
    for ev in result[1:]:
        if ev["state"] != merged[-1]["state"]:
            merged.append(ev)
        # Cùng state với event trước → bỏ qua, GIỮ NGUYÊN frame chuyển trạng thái
        # sớm nhất. KHÔNG set merged[-1]["frame"] = ev["frame"]: làm vậy sẽ dời
        # onset (vd "open") về sau khi hai đoạn cùng state quanh một 'closed' ngắn
        # (đã bị loại ở vòng trên) bị gộp lại — đây là gốc rễ lỗi miệng mở trễ ~1s.
    return merged


def build_mouth_events_for_segment(
    seg_start_frame: int,
    seg_end_frame: int,
    has_tts: bool,
    tts_clip_path: Optional[Path],
    fps: float,
    **opts: Any,
) -> Optional[List[Dict[str, Any]]]:
    """Tạo mouthEvents cho 1 segment.

    Args:
        seg_start_frame: startFrame của segment (global).
        seg_end_frame: endFrame của segment (global).
        has_tts: segment có TTS không.
        tts_clip_path: Path tới TTS clip (WAV), None nếu không có.
        fps: Frame rate.
        **opts: silence_db, min_silence_ms, cadence_ms, num_mouth_states.

    Returns:
        List[{frame: global_frame, state: mouth_state}] hoặc None (no TTS).
    """
    if not has_tts or not tts_clip_path or not Path(tts_clip_path).exists():
        return None  # Không TTS → miệng luôn closed (xử lý ở caller)

    events = analyze_tts_amplitude(Path(tts_clip_path), fps, **opts)
    # Cộng frame offset về global frame
    for ev in events:
        ev["frame"] = ev["frame"] + seg_start_frame
    return events


def build_mouth_events_for_segments(
    segments: List[Dict[str, Any]],
    fps: float,
    tts_clip_dir: Optional[Path] = None,
    **opts: Any,
) -> List[Dict[str, Any]]:
    """Build mouthEvents cho list segment dict.

    Args:
        segments: List các segment từ group_manifest (startFrame, endFrame, hasTts, ...).
        fps: Frame rate.
        tts_clip_dir: Thư mục chứa TTS clips (để resolve tts_clip_path tương đối).
        **opts: silence_db, min_silence_ms, cadence_ms, num_mouth_states.

    Returns:
        segments với trường mouthEvents đã điền (None nếu không TTS).
    """
    result: List[Dict[str, Any]] = []
    for seg in segments:
        tts_path: Optional[Path] = None
        raw_path = seg.get("tts_clip_path")
        if raw_path:
            p = Path(raw_path)
            if not p.is_absolute() and tts_clip_dir:
                p = tts_clip_dir / p
            if p.exists():
                tts_path = p

        mouth_events = build_mouth_events_for_segment(
            seg["startFrame"],
            seg["endFrame"],
            bool(seg.get("hasTts", False)),
            tts_path,
            fps,
            **opts,
        )
        seg_copy = dict(seg)
        seg_copy["mouthEvents"] = mouth_events
        result.append(seg_copy)

    return result
