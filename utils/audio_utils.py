# -*- coding: utf-8 -*-
"""
audio_utils.py — Audio processing utilities

Các hàm xử lý audio dùng chung cho các CLI modules.
"""

from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# AUDIO LOADING
# ─────────────────────────────────────────────────────────────────────

def load_audio(file_path: str):
    """
    Load audio từ file audio hoặc video.
    Hỗ trợ nhiều format: mp3, wav, mp4, mkv, ...
    
    Args:
        file_path: Đường dẫn tới file audio/video
    
    Returns:
        AudioSegment object
    
    Raises:
        ImportError: Nếu pydub chưa được cài đặt
        Exception: Nếu không thể load file
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        logger.error("pydub chưa được cài đặt. Chạy: pip install pydub")
        raise
    
    logger.info(f"Loading audio from: {file_path}")
    audio = AudioSegment.from_file(file_path)
    logger.info(f"Audio loaded: {len(audio)}ms, {audio.channels} channels, {audio.frame_rate}Hz")
    return audio


# ─────────────────────────────────────────────────────────────────────
# AUDIO EXPORT
# ─────────────────────────────────────────────────────────────────────

def export_audio(audio, output_path: str, sample_rate: int = 16000, channels: int = 1):
    """
    Export audio ra file WAV.
    Mặc định: 16kHz mono, tối ưu cho WhisperX.
    
    Args:
        audio: AudioSegment object
        output_path: Đường dẫn file output
        sample_rate: Sample rate (mặc định: 16000Hz cho WhisperX)
        channels: Số kênh (mặc định: 1 = mono)
    """
    logger.info(f"Preparing audio for export: {sample_rate}Hz, {channels} channel(s)")
    
    # Convert to mono
    if audio.channels != channels:
        audio = audio.set_channels(channels)
        logger.debug(f"Converted to {channels} channel(s)")
    
    # Set sample rate
    if audio.frame_rate != sample_rate:
        audio = audio.set_frame_rate(sample_rate)
        logger.debug(f"Converted to {sample_rate}Hz")
    
    # Export
    logger.info(f"Exporting to: {output_path}")
    audio.export(output_path, format='wav')
    logger.info(f"Export complete: {output_path}")


# ─────────────────────────────────────────────────────────────────────
# SILENCE GENERATION
# ─────────────────────────────────────────────────────────────────────

def create_silence(duration_ms: int):
    """
    Tạo silence segment có độ dài xác định.
    
    Args:
        duration_ms: Độ dài silence tính bằng milliseconds
    
    Returns:
        AudioSegment containing silence
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        logger.error("pydub chưa được cài đặt. Chạy: pip install pydub")
        raise
    
    return AudioSegment.silent(duration=duration_ms)


# ─────────────────────────────────────────────────────────────────────
# DIRECT AUDIO EXTRACTION (FFMPEG)
# ─────────────────────────────────────────────────────────────────────

def extract_audio_direct(input_path: str, output_wav: str, sample_rate: int = 16000, channels: int = 1) -> float:
    """
    Trích xuất audio trực tiếp qua ffmpeg để tránh ngốn RAM khi dùng pydub cho file lớn.
    Mặc định: 16kHz mono WAV, tối ưu cho WhisperX.
    Trả về thời lượng audio tính bằng giây.
    
    Args:
        input_path: Đường dẫn file audio/video đầu vào
        output_wav: Đường dẫn file WAV đầu ra
        sample_rate: Sample rate (mặc định: 16000)
        channels: Số kênh (mặc định: 1)
        
    Returns:
        Thời lượng audio (giây)
    """
    import subprocess
    from pathlib import Path
    
    logger.info(f"Extracting audio direct: {Path(input_path).name} → {sample_rate}Hz {channels}ch WAV")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        output_wav,
    ]
    
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(
            f"FFmpeg thất bại:\n{r.stderr.decode(errors='ignore')[-400:]}"
        )
        
    return get_audio_duration_direct(output_wav)

def get_audio_duration_direct(wav_path: str) -> float:
    """
    Lấy thời lượng audio bằng ffprobe để không cần load file vào RAM.
    
    Args:
        wav_path: Đường dẫn file audio
        
    Returns:
        Thời lượng (giây)
    """
    import subprocess
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
            capture_output=True, text=True, timeout=30,
        )
        return float(r.stdout.strip())
    except Exception as e:
        logger.warning(f"Không thể lấy thời lượng bằng ffprobe: {e}")
        return 0.0
