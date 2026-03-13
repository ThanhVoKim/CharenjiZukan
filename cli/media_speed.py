#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli/media_speed.py — CLI: Thay đổi tốc độ media

Hỗ trợ cả slow down (speed < 1.0) và speed up (speed > 1.0) cho:
- Video (mp4, mkv, mov, ...)
- Audio (wav, mp3, m4a, ...)
- Subtitle SRT (.srt)
- Subtitle ASS (.ass)

Ví dụ:
    # Slow down 0.65x
    uv run cli/media_speed.py --input video.mp4 --speed 0.65
    uv run cli/media_speed.py -i audio.wav -s 0.65 -o audio_slow.wav
    
    # Speed up 1.5x
    uv run cli/media_speed.py --input video.mp4 --speed 1.5
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.media_utils import (
    detect_media_type,
    stretch_audio,
    change_video_speed,
    scale_srt_timestamps,
    scale_ass_timestamps,
    get_default_output_path,
    check_rubberband_available,
)
from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Tạo argument parser."""
    parser = argparse.ArgumentParser(
        prog="media_speed",
        description="Thay đổi tốc độ media (video, audio, SRT, ASS). "
                    "Hỗ trợ cả slow down (speed < 1.0) và speed up (speed > 1.0).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Slow down video 0.65x
  uv run cli/media_speed.py --input video.mp4 --speed 0.65
  
  # Speed up audio 1.5x
  uv run cli/media_speed.py -i audio.wav -s 1.5 -o audio_fast.wav
  
  # Scale SRT timestamps
  uv run cli/media_speed.py -i subtitle.srt -s 0.65
  
  # Scale ASS timestamps
  uv run cli/media_speed.py -i overlay.ass -s 0.65
  
  # Video without audio
  uv run cli/media_speed.py -i video.mp4 -s 0.65 --no-keep-audio

Output naming mặc định:
  - speed < 1.0 → *_slow.*
  - speed > 1.0 → *_fast.*
  - speed = 1.0 → *_copy.*
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="FILE",
        help="File input (video, audio, SRT, hoặc ASS)",
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="File output (mặc định: <input>_slow.* hoặc <input>_fast.*)",
    )
    
    parser.add_argument(
        "--speed", "-s",
        type=float,
        default=0.65,
        metavar="RATE",
        help="Hệ số tốc độ (mặc định: 0.65). "
             "speed < 1.0: slow down, speed > 1.0: speed up",
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["auto", "video", "audio", "srt", "ass"],
        default="auto",
        help="Loại file input (mặc định: auto-detect từ extension)",
    )
    
    parser.add_argument(
        "--no-keep-audio",
        action="store_true",
        help="Không giữ audio trong video output (chỉ áp dụng cho video)",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Bật log chi tiết",
    )
    
    return parser


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"File không tồn tại: {args.input}")
        sys.exit(1)
    
    # Validate speed
    if args.speed <= 0:
        logger.error(f"Speed phải > 0, got {args.speed}")
        sys.exit(1)
    
    # Detect type
    if args.type == "auto":
        media_type = detect_media_type(str(input_path))
        if media_type == "unknown":
            logger.error(f"Không thể nhận dạng loại file: {input_path.suffix}")
            logger.error("Sử dụng --type để chỉ định loại file")
            sys.exit(1)
    else:
        media_type = args.type
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = get_default_output_path(str(input_path), args.speed)
    
    # Print info
    print(f"\n{'='*55}")
    print(f"  🎬 Media Speed Changer")
    print(f"{'='*55}")
    print(f"  Input  : {args.input}")
    print(f"  Output : {output_path}")
    print(f"  Type   : {media_type}")
    print(f"  Speed  : {args.speed}")
    if media_type == "video":
        print(f"  Audio  : {'Keep' if not args.no_keep_audio else 'Drop'}")
    print(f"{'='*55}\n")
    
    # Check rubberband availability (info only)
    available, reason = check_rubberband_available()
    if media_type in ["video", "audio"]:
        if available:
            logger.info("Using rubberband for audio stretching (pitch-preserving)")
        else:
            logger.info(f"Using FFmpeg atempo for audio stretching ({reason})")
    
    # Process based on type
    try:
        if media_type == "video":
            keep_audio = not args.no_keep_audio
            change_video_speed(str(input_path), output_path, args.speed, keep_audio=keep_audio)
            
        elif media_type == "audio":
            stretch_audio(str(input_path), output_path, args.speed)
            
        elif media_type == "srt":
            count = scale_srt_timestamps(str(input_path), output_path, args.speed)
            print(f"✅ Scaled {count} SRT segments")
            
        elif media_type == "ass":
            count = scale_ass_timestamps(str(input_path), output_path, args.speed)
            print(f"✅ Scaled {count} ASS dialogues")
            
        else:
            logger.error(f"Unsupported media type: {media_type}")
            sys.exit(1)
        
        print(f"\n✅ Done! Output: {output_path}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()