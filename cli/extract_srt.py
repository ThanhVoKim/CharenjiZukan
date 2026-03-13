#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_srt.py — CLI: Extract audio segments từ file mute.srt

Giữ lại CHỈ các đoạn audio được đánh dấu trong file mute.srt.
Các đoạn KHÔNG được đánh dấu sẽ được thay thế bằng silence.
Giữ nguyên độ dài audio - KHÔNG cắt bỏ đoạn audio.
Output mặc định là WAV 16kHz mono, tối ưu cho WhisperX.

Ví dụ nhanh:
    uv run cli/extract_srt.py --input video.mp4 --mute mute.srt
    uv run cli/extract_srt.py --input audio.mp3 --mute mute.srt --output extracted.wav

Lưu ý: Module này là NGƯỢC với mute_srt.py:
    - mute_srt.py: Mute các đoạn TRONG mute.srt → silence
    - extract_srt.py: Giữ lại CHỈ các đoạn TRONG mute.srt, các đoạn khác → silence
"""

import sys
import argparse
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.srt_parser import parse_srt_file  # noqa: E402
from utils.audio_utils import load_audio, export_audio, create_silence  # noqa: E402
from utils.logger import setup_logging, get_logger  # noqa: E402

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# AUDIO PROCESSING
# ─────────────────────────────────────────────────────────────────────

def apply_extract(audio, segments):
    """
    Giữ lại CHỈ các đoạn được đánh dấu trong segments.
    Các đoạn KHÔNG được đánh dấu sẽ được thay thế bằng silence.
    Giữ nguyên độ dài audio.
    
    Args:
        audio: AudioSegment object
        segments: List of dict với keys 'start_time', 'end_time' (milliseconds)
    
    Returns:
        AudioSegment với chỉ các đoạn được extract
    """
    if not segments:
        logger.warning("Không có segment nào để extract")
        # Trả về silence có độ dài bằng audio gốc
        return create_silence(len(audio))
    
    logger.info(f"Extracting {len(segments)} segments")
    
    # Tạo silence có độ dài bằng audio gốc
    result = create_silence(len(audio))
    
    # Sort segments theo start_time để xử lý đúng thứ tự
    sorted_segments = sorted(segments, key=lambda x: x['start_time'])
    
    total_extracted = 0
    
    for seg in sorted_segments:
        start = seg['start_time']
        end = seg['end_time']
        duration = end - start
        
        if start < 0 or end > len(audio):
            logger.warning(f"Segment {seg.get('line', '?')} ngoài phạm vi audio: {start}-{end}ms (audio length: {len(audio)}ms)")
            continue
        
        # Copy đoạn audio gốc vào vị trí tương ứng trong kết quả
        # Sử dụng overlay để ghi đè silence bằng audio gốc tại vị trí segment
        result = result.overlay(audio[start:end], position=start)
        total_extracted += duration
        
        logger.debug(f"Extracted segment {seg.get('line', '?')}: {start}ms - {end}ms ({duration}ms)")
    
    logger.info(f"Total extracted: {total_extracted}ms ({total_extracted/1000:.2f}s)")
    return result


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_srt",
        description="Extract audio segments từ file mute.srt. "
                    "Giữ lại CHỈ các đoạn được đánh dấu, các đoạn khác thành silence. "
                    "Giữ nguyên độ dài audio. "
                    "Output mặc định là WAV 16kHz mono cho WhisperX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Cơ bản - output mặc định là <input>_extracted.wav
  python extract_srt.py --input video.mp4 --mute mute.srt
  
  # Với output tùy chỉnh
  python extract_srt.py --input audio.mp3 --mute mute.srt --output extracted_audio.wav
  
  # Với sample rate tùy chỉnh
  python extract_srt.py --input video.mp4 --mute mute.srt --sample-rate 48000

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
        help="File mute.srt chứa các đoạn cần extract (giữ lại)",
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="File audio đầu ra (mặc định: <input>_extracted.wav)",
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
        # Default: <input_name>_extracted.wav
        output_path = input_path.parent / f"{input_path.stem}_extracted.wav"
    
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
        logger.info("Tạo output file với toàn bộ silence")
    
    logger.info(f"Found {len(segments)} segments to extract")
    
    # Load audio
    try:
        audio = load_audio(str(input_path))
    except Exception as e:
        logger.error(f"Lỗi load audio: {e}")
        sys.exit(1)
    
    # Apply extract
    extracted_audio = apply_extract(audio, segments)
    
    # Export
    try:
        export_audio(extracted_audio, str(output_path), sample_rate=args.sample_rate)
    except Exception as e:
        logger.error(f"Lỗi export audio: {e}")
        sys.exit(1)
    
    # Summary
    print(f"\n✅ Hoàn thành!")
    print(f"   Input:  {args.input}")
    print(f"   Mute:   {args.mute}")
    print(f"   Output: {output_path}")
    print(f"   Segments extracted: {len(segments)}")
    print(f"   Audio length: {len(extracted_audio)/1000:.2f}s (unchanged)")


if __name__ == "__main__":
    main()