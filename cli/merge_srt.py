#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli/merge_srt.py — CLI: Merge 2 file SRT theo timestamp

Merge subtitle_commentary.srt và subtitle_quoted.srt thành subtitle_merged.srt.
Sắp xếp theo timestamp và đánh số lại line.

Ví dụ:
    uv run cli/merge_srt.py --commentary subtitle_commentary.srt --quoted subtitle_quoted.srt
    uv run cli/merge_srt.py -c commentary.srt -q quoted.srt -o merged.srt
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.srt_parser import parse_srt_file, segments_to_srt
from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# MERGE LOGIC
# ─────────────────────────────────────────────────────────────────────

def check_overlap(segments: List[Dict]) -> List[Tuple[int, int, Dict, Dict]]:
    """
    Kiểm tra overlapping segments.
    
    Args:
        segments: List of segments (đã sort theo start_time)
    
    Returns:
        List of tuples: (index1, index2, segment1, segment2)
    """
    overlaps = []
    for i in range(len(segments) - 1):
        curr = segments[i]
        next_seg = segments[i + 1]
        
        # Check if current segment overlaps with next
        if curr['end_time'] > next_seg['start_time']:
            overlaps.append((i, i + 1, curr, next_seg))
    
    return overlaps


def merge_srt_segments(
    commentary_segments: List[Dict],
    quoted_segments: List[Dict],
    check_overlaps: bool = True
) -> List[Dict]:
    """
    Merge 2 list segments thành 1 list, sort theo timestamp.
    
    Args:
        commentary_segments: Segments từ subtitle_commentary.srt
        quoted_segments: Segments từ subtitle_quoted.srt
        check_overlaps: Nếu True, log warning khi có overlap
    
    Returns:
        Merged and sorted segments với line number đã được đánh lại
    """
    # Merge 2 lists
    merged = commentary_segments + quoted_segments
    
    if not merged:
        logger.warning("Cả 2 file đều rỗng")
        return []
    
    # Sort theo start_time
    merged.sort(key=lambda x: x['start_time'])
    
    # Check overlaps
    if check_overlaps:
        overlaps = check_overlap(merged)
        if overlaps:
            for i, j, seg1, seg2 in overlaps:
                logger.error(
                    f"❌ OVERLAP DETECTED: segment {i+1} ({seg1['start_time']}-{seg1['end_time']}ms) "
                    f"overlaps with segment {j+1} ({seg2['start_time']}-{seg2['end_time']}ms)"
                )
    
    # Đánh số lại line
    for i, seg in enumerate(merged, 1):
        seg['line'] = i
    
    return merged


def merge_srt_files(
    commentary_path: str,
    quoted_path: str,
    output_path: str,
    check_overlaps: bool = True
) -> int:
    """
    Merge 2 file SRT thành 1 file.
    
    Args:
        commentary_path: Đường dẫn file subtitle_commentary.srt
        quoted_path: Đường dẫn file subtitle_quoted.srt
        output_path: Đường dẫn file output
        check_overlaps: Nếu True, log warning khi có overlap
    
    Returns:
        Số segments trong file output
    
    Raises:
        FileNotFoundError: Nếu 1 trong 2 file không tồn tại
    """
    # Parse files
    logger.info(f"Loading commentary: {commentary_path}")
    commentary = parse_srt_file(commentary_path)
    logger.info(f"  Found {len(commentary)} segments")
    
    logger.info(f"Loading quoted: {quoted_path}")
    quoted = parse_srt_file(quoted_path)
    logger.info(f"  Found {len(quoted)} segments")
    
    # Merge
    merged = merge_srt_segments(commentary, quoted, check_overlaps)
    logger.info(f"Merged: {len(merged)} segments")
    
    # Export
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    srt_content = segments_to_srt(merged)
    with open(output, 'w', encoding='utf-8') as f:
        f.write(srt_content)
    
    logger.info(f"Output: {output_path}")
    return len(merged)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="merge_srt",
        description="Merge 2 file SRT theo timestamp. "
                    "Kết hợp subtitle_commentary.srt và subtitle_quoted.srt "
                    "thành subtitle_merged.srt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Cơ bản - output mặc định là subtitle_merged.srt
  python merge_srt.py --commentary subtitle_commentary.srt --quoted subtitle_quoted.srt
  
  # Với output tùy chỉnh
  python merge_srt.py -c commentary.srt -q quoted.srt -o merged.srt
  
  # Bỏ qua check overlap
  python merge_srt.py -c commentary.srt -q quoted.srt --no-check-overlap
"""
    )
    
    parser.add_argument(
        "-c", "--commentary",
        required=True,
        help="File SRT chứa subtitle cho phần bình luận (từ WhisperX)"
    )
    
    parser.add_argument(
        "-q", "--quoted",
        required=True,
        help="File SRT chứa subtitle cho phần video trích dẫn (tạo thủ công)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="subtitle_merged.srt",
        help="File SRT output (mặc định: subtitle_merged.srt)"
    )
    
    parser.add_argument(
        "--no-check-overlap",
        action="store_true",
        help="Bỏ qua kiểm tra overlapping segments"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    # Validate input files
    commentary = Path(args.commentary)
    quoted = Path(args.quoted)
    
    if not commentary.exists():
        logger.error(f"File không tồn tại: {commentary}")
        sys.exit(1)
    
    if not quoted.exists():
        logger.error(f"File không tồn tại: {quoted}")
        sys.exit(1)
    
    # Merge
    try:
        count = merge_srt_files(
            commentary_path=str(commentary),
            quoted_path=str(quoted),
            output_path=args.output,
            check_overlaps=not args.no_check_overlap
        )
        
        if count > 0:
            print(f"✅ Merged {count} segments → {args.output}")
        else:
            print("⚠️ No segments to merge")
        
    except Exception as e:
        logger.error(f"Lỗi khi merge: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()