#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mute_srt.py — CLI: Mute audio segments từ file mute.srt

Thay thế các đoạn audio được đánh dấu trong file mute.srt bằng silence.
Giữ nguyên độ dài audio - KHÔNG cắt bỏ đoạn audio.
Output mặc định là WAV 16kHz mono, tối ưu cho WhisperX.

Ví dụ nhanh:
    uv run mute_srt.py --input video.mp4 --mute mute.srt
    uv run mute_srt.py --input audio.mp3 --mute mute.srt --output muted.wav
"""

import sys
import argparse
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.srt_parser import parse_srt_file  # noqa: E402
from utils.logger import setup_logging, get_logger  # noqa: E402

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# AUDIO PROCESSING
# ─────────────────────────────────────────────────────────────────────

def load_audio(file_path: str):
    """
    Load audio từ file audio hoặc video.
    Hỗ trợ nhiều format: mp3, wav, mp4, mkv, ...
    
    Args:
        file_path: Đường dẫn tới file audio/video
    
    Returns:
        AudioSegment object
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


def apply_mute(audio, segments):
    """
    Thay thế các đoạn được đánh dấu bằng silence.
    Giữ nguyên độ dài audio.
    
    Args:
        audio: AudioSegment object
        segments: List of dict với keys 'start_time', 'end_time' (milliseconds)
    
    Returns:
        AudioSegment với các đoạn đã được mute
    """
    if not segments:
        logger.warning("Không có segment nào để mute")
        return audio
    
    logger.info(f"Applying mute to {len(segments)} segments")
    
    # Sort segments theo start_time descending để không bị lệch index khi mute
    sorted_segments = sorted(segments, key=lambda x: x['start_time'], reverse=True)
    
    result = audio
    total_muted = 0
    
    for seg in sorted_segments:
        start = seg['start_time']
        end = seg['end_time']
        duration = end - start
        
        if start < 0 or end > len(audio):
            logger.warning(f"Segment {seg.get('line', '?')} ngoài phạm vi audio: {start}-{end}ms (audio length: {len(audio)}ms)")
            continue
        
        # Tạo silence có cùng duration
        from pydub import AudioSegment
        silence = AudioSegment.silent(duration=duration)
        
        # Thay thế đoạn audio bằng silence
        result = result[:start] + silence + result[end:]
        total_muted += duration
        
        logger.debug(f"Muted segment {seg.get('line', '?')}: {start}ms - {end}ms ({duration}ms)")
    
    logger.info(f"Total muted: {total_muted}ms ({total_muted/1000:.2f}s)")
    return result


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
# CLI
# ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mute_srt",
        description="Mute audio segments từ file mute.srt. "
                    "Thay thế các đoạn được đánh dấu bằng silence, giữ nguyên độ dài audio. "
                    "Output mặc định là WAV 16kHz mono cho WhisperX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Cơ bản - output mặc định là <input>_muted.wav
  python mute_srt.py --input video.mp4 --mute mute.srt
  
  # Với output tùy chỉnh
  python mute_srt.py --input audio.mp3 --mute mute.srt --output muted_audio.wav

File mute.srt format:
  1
  00:01:24,233 --> 00:01:27,566
  [MUTE] Đoạn video gốc được trích dẫn
  
  2
  00:05:30,000 --> 00:05:45,500
  [MUTE] Đoạn ngôn ngữ thứ hai

Lưu ý: Text trong file mute.srt không quan trọng, chỉ cần timestamp.
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="FILE",
        help="File audio/video đầu vào (mp3, wav, mp4, mkv, ...)",
    )
    parser.add_argument(
        "--mute", "-m",
        required=True,
        metavar="FILE",
        help="File mute.srt chứa các đoạn cần mute",
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="File audio đầu ra (mặc định: <input>_muted.wav)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        metavar="HZ",
        help="Sample rate output (mặc định: 16000 cho WhisperX)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Hiển thị log chi tiết",
    )
    
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"File không tồn tại: {args.input}")
        sys.exit(1)
    
    # Validate mute file
    mute_path = Path(args.mute)
    if not mute_path.exists():
        logger.error(f"File mute.srt không tồn tại: {args.mute}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Default: <input_name>_muted.wav
        output_path = input_path.parent / f"{input_path.stem}_muted.wav"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Parse mute segments
    logger.info(f"Parsing mute file: {args.mute}")
    try:
        segments = parse_srt_file(str(mute_path))
    except Exception as e:
        logger.error(f"Lỗi parse file mute.srt: {e}")
        sys.exit(1)
    
    if not segments:
        logger.warning("Không có segment nào trong file mute.srt")
        logger.info("Tạo output file với audio gốc (không có thay đổi)")
    
    logger.info(f"Found {len(segments)} segments to mute")
    
    # Load audio
    try:
        audio = load_audio(str(input_path))
    except Exception as e:
        logger.error(f"Lỗi load audio: {e}")
        sys.exit(1)
    
    # Apply mute
    muted_audio = apply_mute(audio, segments)
    
    # Export
    try:
        export_audio(muted_audio, str(output_path), sample_rate=args.sample_rate)
    except Exception as e:
        logger.error(f"Lỗi export audio: {e}")
        sys.exit(1)
    
    # Summary
    print(f"\n✅ Hoàn thành!")
    print(f"   Input:  {args.input}")
    print(f"   Mute:   {args.mute}")
    print(f"   Output: {output_path}")
    print(f"   Segments muted: {len(segments)}")
    print(f"   Audio length: {len(muted_audio)/1000:.2f}s (unchanged)")


if __name__ == "__main__":
    main()