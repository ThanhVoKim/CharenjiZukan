#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/media_utils.py — Media Speed Utilities Module

Module tái sử dụng chứa các hàm xử lý media speed.
Hỗ trợ cả slow down (speed < 1.0) và speed up (speed > 1.0).

Ví dụ:
    from utils.media_utils import (
        detect_media_type,
        scale_time_ms,
        stretch_audio_rubberband,
        change_video_speed,
        scale_srt_timestamps,
        scale_ass_timestamps,
    )
    
    # Scale timestamp
    new_ms = scale_time_ms(10000, 0.65)  # 15385 ms
    
    # Stretch audio
    stretch_audio_rubberband("input.wav", "output.wav", 0.65)
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger
from utils.srt_parser import parse_srt_file, segments_to_srt

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────

# Supported file extensions
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.webm', '.flv', '.wmv'}
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.wma'}
SRT_EXTENSIONS = {'.srt'}
ASS_EXTENSIONS = {'.ass'}

# Audio settings
AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 2


# ─────────────────────────────────────────────────────────────────────
# TYPE DETECTION
# ─────────────────────────────────────────────────────────────────────

def detect_media_type(path: str) -> str:
    """
    Nhận dạng loại file media dựa trên extension.
    
    Args:
        path: Đường dẫn file
        
    Returns:
        Loại file: 'video', 'audio', 'srt', 'ass', hoặc 'unknown'
    
    Example:
        >>> detect_media_type("video.mp4")
        'video'
        >>> detect_media_type("audio.wav")
        'audio'
        >>> detect_media_type("subtitle.srt")
        'srt'
    """
    ext = Path(path).suffix.lower()
    
    if ext in VIDEO_EXTENSIONS:
        return 'video'
    elif ext in AUDIO_EXTENSIONS:
        return 'audio'
    elif ext in SRT_EXTENSIONS:
        return 'srt'
    elif ext in ASS_EXTENSIONS:
        return 'ass'
    else:
        return 'unknown'


# ─────────────────────────────────────────────────────────────────────
# TIME SCALING
# ─────────────────────────────────────────────────────────────────────

def scale_time_ms(ms: int, speed: float) -> int:
    """
    Scale milliseconds theo speed factor.
    
    Args:
        ms: Thời gian gốc (milliseconds)
        speed: Hệ số tốc độ (> 0)
            - speed < 1.0: slow down → thời gian tăng
            - speed > 1.0: speed up → thời gian giảm
    
    Returns:
        Thời gian mới (milliseconds), đã làm tròn
    
    Example:
        >>> scale_time_ms(10000, 0.65)  # Slow down
        15385
        >>> scale_time_ms(10000, 1.5)   # Speed up
        6667
    """
    if speed <= 0:
        raise ValueError(f"speed phải > 0, got {speed}")
    
    new_ms = round(ms / speed)
    return max(1, new_ms)  # Đảm bảo >= 1ms


# ─────────────────────────────────────────────────────────────────────
# RUBBERBAND CHECK
# ─────────────────────────────────────────────────────────────────────

# Cache kết quả check
_RUBBERBAND_BINARY: Optional[str] = None
_PYRUBBERBAND_INSTALLED: Optional[bool] = None


def check_rubberband_available() -> Tuple[bool, str]:
    """
    Kiểm tra rubberband binary và pyrubberband library có sẵn không.
    
    Returns:
        Tuple (available, reason):
            - available: True nếu có thể dùng rubberband
            - reason: Lý do nếu không available
    """
    global _RUBBERBAND_BINARY, _PYRUBBERBAND_INSTALLED
    
    # Check binary (cache)
    if _RUBBERBAND_BINARY is None:
        _RUBBERBAND_BINARY = shutil.which("rubberband") or ""
    
    # Check library (cache)
    if _PYRUBBERBAND_INSTALLED is None:
        try:
            import pyrubberband  # noqa: F401
            _PYRUBBERBAND_INSTALLED = True
        except ImportError:
            _PYRUBBERBAND_INSTALLED = False
    
    if _RUBBERBAND_BINARY and _PYRUBBERBAND_INSTALLED:
        return True, f"rubberband binary + pyrubberband available"
    elif _RUBBERBAND_BINARY and not _PYRUBBERBAND_INSTALLED:
        return False, "rubberband binary có, nhưng pyrubberband lib chưa cài. Cài đặt: pip install pyrubberband"
    elif not _RUBBERBAND_BINARY and _PYRUBBERBAND_INSTALLED:
        return False, "pyrubberband lib có, nhưng rubberband binary chưa cài. Cài đặt: apt-get install rubberband-cli"
    else:
        return False, "Cả rubberband binary và pyrubberband lib đều chưa cài. Cài đặt: apt-get install rubberband-cli && pip install pyrubberband"


# ─────────────────────────────────────────────────────────────────────
# AUDIO STRETCHING
# ─────────────────────────────────────────────────────────────────────

def stretch_audio_rubberband(input_path: str, output_path: str, speed: float) -> bool:
    """
    Time-stretch audio bằng pyrubberband (giữ nguyên pitch).
    
    Args:
        input_path: Đường dẫn file audio input
        output_path: Đường dẫn file audio output
        speed: Hệ số tốc độ
            - speed < 1.0: slow down
            - speed > 1.0: speed up
    
    Returns:
        True nếu thành công
        
    Raises:
        RuntimeError: Nếu rubberband không available
    """
    available, reason = check_rubberband_available()
    if not available:
        raise RuntimeError(f"Cannot use rubberband: {reason}")
    
    import numpy as np
    import soundfile as sf
    import pyrubberband as pyrb
    
    logger.info(f"Stretching audio with rubberband: {input_path} (speed={speed})")
    
    # Read audio
    y, sr = sf.read(input_path, always_2d=True)
    current_duration_ms = int(len(y) / sr * 1000)
    
    # Calculate stretch rate for pyrubberband
    # pyrubberband.time_stretch(y, sr, rate):
    #   - rate < 1.0 → slow down (audio dài hơn)
    #   - rate > 1.0 → speed up (audio ngắn hơn)
    # speed = 0.65 → stretch_rate = 0.65 (audio dài hơn, chậm hơn)
    # speed = 1.5 → stretch_rate = 1.5 (audio ngắn hơn, nhanh hơn)
    stretch_rate = speed
    
    logger.debug(f"  Current duration: {current_duration_ms}ms, stretch_rate: {stretch_rate:.3f}")
    
    # Stretch
    y_out = pyrb.time_stretch(y, sr, stretch_rate)
    
    # Ensure stereo output if input was stereo
    if y_out.ndim == 1 and y.ndim == 2:
        y_out = np.column_stack((y_out, y_out))
    
    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, y_out, sr)
    
    new_duration_ms = int(len(y_out) / sr * 1000)
    logger.info(f"  Output duration: {new_duration_ms}ms (expected ~{round(current_duration_ms / speed)}ms)")
    
    return True


def _build_atempo_filter(speed: float) -> str:
    """
    Tạo chuỗi atempo filter cho FFmpeg.
    atempo chỉ chấp nhận [0.5, 2.0] → chain nhiều filter khi ngoài khoảng.
    
    Args:
        speed: Hệ số tốc độ
        
    Returns:
        FFmpeg filter string
    
    Example:
        >>> _build_atempo_filter(0.65)
        'atempo=0.650000'
        >>> _build_atempo_filter(0.3)
        'atempo=0.5,atempo=0.600000'
    """
    parts = []
    f = speed
    
    # Chain atempo filters
    while f > 2.0:
        parts.append("atempo=2.0")
        f /= 2.0
    while f < 0.5:
        parts.append("atempo=0.5")
        f *= 2.0
    
    parts.append(f"atempo={f:.6f}")
    return ",".join(parts)


def stretch_audio_atempo(input_path: str, output_path: str, speed: float) -> bool:
    """
    Time-stretch audio bằng FFmpeg atempo filter.
    
    Args:
        input_path: Đường dẫn file audio input
        output_path: Đường dẫn file audio output
        speed: Hệ số tốc độ
    
    Returns:
        True nếu thành công
        
    Raises:
        RuntimeError: Nếu FFmpeg không available hoặc lỗi
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg không có trong PATH. Cài đặt: apt-get install ffmpeg")
    
    logger.info(f"Stretching audio with FFmpeg atempo: {input_path} (speed={speed})")
    
    # Build filter
    filter_str = _build_atempo_filter(speed)
    logger.debug(f"  atempo filter: {filter_str}")
    
    # Determine output codec based on extension
    output_ext = Path(output_path).suffix.lower()
    codec_map = {
        '.wav': 'pcm_s16le',
        '.m4a': 'aac',
        '.mp4': 'aac',
        '.mp3': 'libmp3lame',
        '.flac': 'flac',
        '.ogg': 'libvorbis',
        '.aac': 'aac',
    }
    audio_codec = codec_map.get(output_ext, 'aac')
    
    # Build command
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter:a", filter_str,
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", str(AUDIO_CHANNELS),
        "-c:a", audio_codec,
        output_path,
    ]
    
    # Run
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        logger.info(f"  Output: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="ignore")[-500:]
        logger.error(f"FFmpeg error: {stderr}")
        raise RuntimeError(f"FFmpeg failed: {stderr}")
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg timeout")
        raise RuntimeError("FFmpeg timeout after 300s")


def stretch_audio(input_path: str, output_path: str, speed: float) -> bool:
    """
    Time-stretch audio (tự động chọn method tốt nhất).
    
    Priority:
        1. pyrubberband (giữ pitch, chất lượng cao)
        2. FFmpeg atempo (fallback)
    
    Args:
        input_path: Đường dẫn file audio input
        output_path: Đường dẫn file audio output
        speed: Hệ số tốc độ
    
    Returns:
        True nếu thành công
    """
    available, _ = check_rubberband_available()
    
    if available:
        try:
            return stretch_audio_rubberband(input_path, output_path, speed)
        except Exception as e:
            logger.warning(f"rubberband failed: {e}, falling back to atempo")
    
    return stretch_audio_atempo(input_path, output_path, speed)


# ─────────────────────────────────────────────────────────────────────
# VIDEO SPEED CHANGE
# ─────────────────────────────────────────────────────────────────────

def change_video_speed(
    input_path: str,
    output_path: str,
    speed: float,
    keep_audio: bool = True
) -> bool:
    """
    Thay đổi tốc độ video.
    
    Args:
        input_path: Đường dẫn file video input
        output_path: Đường dẫn file video output
        speed: Hệ số tốc độ
        keep_audio: Nếu True, giữ audio và stretch theo speed
    
    Returns:
        True nếu thành công
        
    Raises:
        RuntimeError: Nếu FFmpeg không available hoặc lỗi
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg không có trong PATH. Cài đặt: apt-get install ffmpeg")
    
    logger.info(f"Changing video speed: {input_path} (speed={speed}, keep_audio={keep_audio})")
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_video = Path(tmpdir) / "video_temp.mp4"
        tmp_audio = Path(tmpdir) / "audio_temp.wav"
        tmp_audio_stretched = Path(tmpdir) / "audio_stretched.wav"
        
        # Step 1: Change video speed (without audio)
        # setpts=PTS/speed → speed up if speed > 1, slow down if speed < 1
        video_filter = f"setpts=PTS/{speed}"
        
        cmd_video = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter:v", video_filter,
            "-an",  # No audio
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            str(tmp_video),
        ]
        
        logger.debug(f"  Running video filter: {video_filter}")
        try:
            subprocess.run(cmd_video, check=True, capture_output=True, timeout=600)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="ignore")[-500:]
            logger.error(f"FFmpeg video error: {stderr}")
            raise RuntimeError(f"FFmpeg video filter failed: {stderr}")
        
        # Step 2: Handle audio
        if keep_audio:
            # Extract audio
            cmd_extract = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-vn",
                "-c:a", "pcm_s16le",
                "-ar", str(AUDIO_SAMPLE_RATE),
                "-ac", str(AUDIO_CHANNELS),
                str(tmp_audio),
            ]
            
            try:
                subprocess.run(cmd_extract, check=True, capture_output=True, timeout=60)
            except subprocess.CalledProcessError as e:
                logger.warning(f"Could not extract audio: {e}")
                keep_audio = False
            
            if keep_audio and tmp_audio.exists():
                # Stretch audio
                try:
                    stretch_audio(str(tmp_audio), str(tmp_audio_stretched), speed)
                except Exception as e:
                    logger.warning(f"Could not stretch audio: {e}")
                    keep_audio = False
        
        # Step 3: Merge video + audio (or just copy video)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if keep_audio and tmp_audio_stretched.exists():
            cmd_merge = [
                "ffmpeg", "-y",
                "-i", str(tmp_video),
                "-i", str(tmp_audio_stretched),
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                output_path,
            ]
            
            logger.debug("  Merging video + stretched audio")
            try:
                subprocess.run(cmd_merge, check=True, capture_output=True, timeout=60)
            except subprocess.CalledProcessError as e:
                logger.warning(f"Could not merge audio, saving video only: {e}")
                # Fallback: copy video only
                import shutil as sh
                sh.copy(str(tmp_video), output_path)
        else:
            # No audio, just copy video
            import shutil as sh
            sh.copy(str(tmp_video), output_path)
    
    logger.info(f"  Output: {output_path}")
    return True


# ─────────────────────────────────────────────────────────────────────
# SRT TIMESTAMP SCALING
# ─────────────────────────────────────────────────────────────────────

def scale_srt_timestamps(input_path: str, output_path: str, speed: float) -> int:
    """
    Scale timestamps trong file SRT.
    
    Args:
        input_path: Đường dẫn file SRT input
        output_path: Đường dẫn file SRT output
        speed: Hệ số tốc độ
    
    Returns:
        Số segments đã xử lý
    """
    logger.info(f"Scaling SRT timestamps: {input_path} (speed={speed})")
    
    # Parse SRT
    segments = parse_srt_file(input_path)
    logger.debug(f"  Found {len(segments)} segments")
    
    # Scale timestamps
    for seg in segments:
        new_start = scale_time_ms(seg['start_time'], speed)
        new_end = scale_time_ms(seg['end_time'], speed)
        
        # Ensure end > start
        if new_end <= new_start:
            new_end = new_start + 1
        
        seg['start_time'] = new_start
        seg['end_time'] = new_end
    
    # Export
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    srt_content = segments_to_srt(segments)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)
    
    logger.info(f"  Output: {output_path} ({len(segments)} segments)")
    return len(segments)


# ─────────────────────────────────────────────────────────────────────
# ASS TIMESTAMP SCALING
# ─────────────────────────────────────────────────────────────────────

def parse_ass_timestamp_to_ms(ts: str) -> int:
    """
    Convert ASS timestamp trực tiếp sang milliseconds.
    ASS format: "H:MM:SS.cc" (centiseconds)
    
    Args:
        ts: ASS timestamp string (e.g., "0:01:24.23")
    
    Returns:
        Milliseconds
    
    Example:
        >>> parse_ass_timestamp_to_ms("0:01:24.23")
        84230
        >>> parse_ass_timestamp_to_ms("1:30:45.50")
        5445500
    """
    parts = ts.strip().split(':')
    
    if len(parts) != 3:
        raise ValueError(f"Invalid ASS timestamp format: {ts}")
    
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds_parts = parts[2].split('.')
    
    seconds = int(seconds_parts[0])
    centiseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
    
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + centiseconds * 10


def ms_to_ass_timestamp(ms: int) -> str:
    """
    Convert milliseconds sang ASS timestamp.
    ASS format: "H:MM:SS.cc" (centiseconds)
    
    Args:
        ms: Milliseconds
    
    Returns:
        ASS timestamp string
    
    Example:
        >>> ms_to_ass_timestamp(84230)
        '0:01:24.23'
        >>> ms_to_ass_timestamp(5445500)
        '1:30:45.50'
    """
    total_seconds = ms // 1000
    centiseconds = (ms % 1000) // 10
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def scale_ass_timestamps(input_path: str, output_path: str, speed: float) -> int:
    """
    Scale timestamps trong file ASS.
    Chỉ thay đổi Start/End trong các dòng Dialogue.
    
    Args:
        input_path: Đường dẫn file ASS input
        output_path: Đường dẫn file ASS output
        speed: Hệ số tốc độ
    
    Returns:
        Số dialogue lines đã xử lý
    """
    logger.info(f"Scaling ASS timestamps: {input_path} (speed={speed})")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    output_lines = []
    dialogue_count = 0
    
    for line in lines:
        if line.startswith('Dialogue:'):
            # Split thành tối đa 10 phần, phần cuối là Text
            # Format: Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
            parts = line.rstrip('\n').split(',', 9)
            
            if len(parts) >= 10:
                # Scale Start (index 1) và End (index 2)
                try:
                    start_ms = parse_ass_timestamp_to_ms(parts[1])
                    end_ms = parse_ass_timestamp_to_ms(parts[2])
                    
                    new_start_ms = scale_time_ms(start_ms, speed)
                    new_end_ms = scale_time_ms(end_ms, speed)
                    
                    # Ensure end > start
                    if new_end_ms <= new_start_ms:
                        new_end_ms = new_start_ms + 1
                    
                    parts[1] = ms_to_ass_timestamp(new_start_ms)
                    parts[2] = ms_to_ass_timestamp(new_end_ms)
                    
                    output_lines.append(','.join(parts) + '\n')
                    dialogue_count += 1
                except ValueError as e:
                    logger.warning(f"Could not parse timestamp in line: {line[:50]}... Error: {e}")
                    output_lines.append(line)
            else:
                output_lines.append(line)
        else:
            output_lines.append(line)
    
    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
    
    logger.info(f"  Output: {output_path} ({dialogue_count} dialogues)")
    return dialogue_count


# ─────────────────────────────────────────────────────────────────────
# OUTPUT PATH HELPER
# ─────────────────────────────────────────────────────────────────────

def get_default_output_path(input_path: str, speed: float) -> str:
    """
    Tạo tên output mặc định dựa trên speed.
    
    Args:
        input_path: Đường dẫn file input
        speed: Hệ số tốc độ
    
    Returns:
        Đường dẫn output mặc định
    
    Naming convention:
        - speed < 1.0 → *_slow.*
        - speed > 1.0 → *_fast.*
        - speed = 1.0 → *_copy.*
    
    Example:
        >>> get_default_output_path("video.mp4", 0.65)
        'video_slow.mp4'
        >>> get_default_output_path("audio.wav", 1.5)
        'audio_fast.wav'
    """
    path = Path(input_path)
    stem = path.stem
    suffix = path.suffix
    
    if speed < 1.0:
        suffix_type = "_slow"
    elif speed > 1.0:
        suffix_type = "_fast"
    else:
        suffix_type = "_copy"
    
    return str(path.parent / f"{stem}{suffix_type}{suffix}")


# ─────────────────────────────────────────────────────────────────────
# MAIN (for testing)
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test timestamp conversion
    print("Testing ASS timestamp conversion:")
    test_cases = [
        ("0:01:24.23", 84230),
        ("1:30:45.50", 5445500),
        ("0:00:01.00", 1000),
    ]
    
    for ts, expected_ms in test_cases:
        result = parse_ass_timestamp_to_ms(ts)
        back = ms_to_ass_timestamp(result)
        status = "✓" if result == expected_ms else "✗"
        print(f"  {status} parse_ass_timestamp_to_ms('{ts}') = {result} (expected: {expected_ms})")
        print(f"     ms_to_ass_timestamp({result}) = '{back}'")
    
    # Test scale_time_ms
    print("\nTesting scale_time_ms:")
    scale_tests = [
        (10000, 0.65, 15385),
        (10000, 1.5, 6667),
        (5000, 0.5, 10000),
    ]
    
    for ms, speed, expected in scale_tests:
        result = scale_time_ms(ms, speed)
        status = "✓" if result == expected else "✗"
        print(f"  {status} scale_time_ms({ms}, {speed}) = {result} (expected: {expected})")
    
    # Test detect_media_type
    print("\nTesting detect_media_type:")
    type_tests = [
        ("video.mp4", "video"),
        ("audio.wav", "audio"),
        ("subtitle.srt", "srt"),
        ("overlay.ass", "ass"),
        ("unknown.xyz", "unknown"),
    ]
    
    for path, expected in type_tests:
        result = detect_media_type(path)
        status = "✓" if result == expected else "✗"
        print(f"  {status} detect_media_type('{path}') = '{result}'")
    
    # Test get_default_output_path
    print("\nTesting get_default_output_path:")
    output_tests = [
        ("video.mp4", 0.65, "video_slow.mp4"),
        ("audio.wav", 1.5, "audio_fast.wav"),
        ("file.srt", 1.0, "file_copy.srt"),
    ]
    
    for input_p, speed, expected in output_tests:
        result = get_default_output_path(input_p, speed)
        status = "✓" if result == expected else "✗"
        print(f"  {status} get_default_output_path('{input_p}', {speed}) = '{result}'")