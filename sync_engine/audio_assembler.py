import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

from sync_engine.models import TimelineSegment
from utils.media_utils import _build_atempo_filter

def compress_tts_clip(wav_path: str, audio_speed: float, output_path: str) -> None:
    if audio_speed <= 1.0:
        shutil.copy(wav_path, output_path)
        return
    filter_str = _build_atempo_filter(audio_speed)  # Reuse từ media_utils.py
    subprocess.run([
        "ffmpeg", "-y", "-i", wav_path,
        "-filter:a", filter_str,
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
        output_path,
    ], check=True, capture_output=True)

def extract_quoted_audio(
    video_path: str,
    orig_start_ms: float,
    orig_end_ms: float,
    output_path: str,
) -> None:
    """
    Sửa lỗi #8: Đọc thẳng từ video.mp4 theo timecode gốc.
    Không dùng pre-extracted file để tránh lệch timecode.
    """
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", f"{orig_start_ms/1000:.6f}",
        "-t",  f"{(orig_end_ms - orig_start_ms)/1000:.6f}",
        "-i",  video_path,
        "-vn", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
        output_path,
    ], check=True, capture_output=True)

def build_ambient_mask(
    timeline: List[TimelineSegment],
    total_ms: float,
) -> List[Tuple[float, float]]:
    """
    Trả về list khoảng (new_start, new_end) cho phép ambient phát.
    Ambient bị tắt trong khoảng new_start..new_end của mute segments.
    """
    mute_ranges = sorted(
        [(s.new_start, s.new_end) for s in timeline if s.block_type == "mute"]
    )
    ambient = []
    cursor = 0.0
    for ms, me in mute_ranges:
        if cursor < ms:
            ambient.append((cursor, ms))
        cursor = me
    if cursor < total_ms:
        ambient.append((cursor, total_ms))
    return ambient

def assemble_audio_track(
    timeline: List[TimelineSegment],
    video_path: str,
    ambient_path: Optional[str],
    output_path: str,
    tmp_dir: str,
    sample_rate: int = 48000,
) -> None:
    """
    Sửa lỗi #7: Không stretch BGM. Dùng ambient.mp3 tĩnh.
    Ambient phát liên tục ngoại trừ mute regions (quoted audio).
    Layer order (bottom → top): ambient → quoted → TTS
    """
    from pydub import AudioSegment

    if not timeline:
        AudioSegment.silent(duration=0).export(output_path, format="wav")
        return

    total_ms = int(timeline[-1].new_end)
    result   = AudioSegment.silent(duration=total_ms, frame_rate=sample_rate)

    # ── Layer 1: Ambient (tĩnh, không stretch, tắt ở mute) ──────────
    if ambient_path and Path(ambient_path).exists():
        ambient_src = AudioSegment.from_file(ambient_path)
        if len(ambient_src) > 0:
            # Loop đủ dài
            while len(ambient_src) < total_ms:
                ambient_src += ambient_src
            ambient_src = ambient_src[:total_ms]

            for amb_s, amb_e in build_ambient_mask(timeline, total_ms):
                chunk = ambient_src[int(amb_s):int(amb_e)]
                result = result.overlay(chunk, position=int(amb_s))

    # ── Layer 2: Quoted audio từ video.mp4 gốc ──────────────────────
    for seg in timeline:
        if seg.block_type != "mute":
            continue
        tmp_q = str(Path(tmp_dir) / f"quoted_{int(seg.orig_start)}.wav")
        extract_quoted_audio(video_path, seg.orig_start, seg.orig_end, tmp_q)
        if Path(tmp_q).exists():
            quoted_chunk = AudioSegment.from_file(tmp_q)
            result = result.overlay(quoted_chunk, position=int(seg.new_start))

    # ── Layer 3: TTS clips ───────────────────────────────────────────
    for seg in timeline:
        if seg.block_type != "tts" or not seg.tts_clip_path:
            continue
        if seg.audio_speed > 1.01:
            tmp_c = str(Path(tmp_dir) / f"compressed_{int(seg.new_start)}.wav")
            compress_tts_clip(seg.tts_clip_path, seg.audio_speed, tmp_c)
            if Path(tmp_c).exists():
                clip = AudioSegment.from_file(tmp_c)
                result = result.overlay(clip, position=int(seg.new_start))
        else:
            if Path(seg.tts_clip_path).exists():
                clip = AudioSegment.from_file(seg.tts_clip_path)
                result = result.overlay(clip, position=int(seg.new_start))

    result.set_frame_rate(sample_rate).set_channels(2).export(output_path, format="wav")
