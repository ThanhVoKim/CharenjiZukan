#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli/srt_to_ass.py — CLI: Convert SRT file to ASS format

Module to convert SRT subtitle files to ASS (Advanced SubStation Alpha) format
for overlaying notes on video.

Quick examples:
    uv run cli/srt_to_ass.py --input note_translated.srt --output note_overlay.ass
    uv run cli/srt_to_ass.py -i input.srt -t assets/sample.ass -o output.ass --max-chars 20
"""

import sys
import argparse
import io
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Add project root to path for imports
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.srt_parser import parse_srt_file
from utils.ass_utils import (
    parse_ass_file,
    write_ass_file,
    convert_srt_segments_to_ass_dialogues,
)
from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# DEFAULT VALUES
# ─────────────────────────────────────────────────────────────

DEFAULT_TEMPLATE = Path("assets/sample.ass")
DEFAULT_MAX_CHARS = 14
DEFAULT_STYLE = "NoteStyle"


# ─────────────────────────────────────────────────────────────
# TEMPLATE LOADING
# ─────────────────────────────────────────────────────────────

def load_template(template_path: str) -> dict:
    """
    Load ASS template file và trả về dict với các components.
    
    Args:
        template_path: Đường dẫn tới file .ass template
    
    Returns:
        Dictionary với các keys:
        - script_info: dict của [Script Info]
        - styles: list of style strings
        - events_format: format string của [Events]
    
    Raises:
        FileNotFoundError: Nếu template file không tồn tại
    """
    path = Path(template_path)
    if not path.exists():
        raise FileNotFoundError(f"Template file không tồn tại: {template_path}")
    
    logger.info(f"Loading template: {template_path}")
    return parse_ass_file(template_path)


def convert_srt_to_ass(
    srt_path: str,
    output_path: str,
    template_path: str = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    style: str = DEFAULT_STYLE,
) -> int:
    """
    Chuyển đổi file SRT sang ASS format.
    
    Args:
        srt_path: Đường dẫn file SRT input
        output_path: Đường dẫn file ASS output
        template_path: Đường dẫn file ASS template (mặc định: assets/sample.ass)
        max_chars: Số ký tự tối đa mỗi dòng (mặc định: 14)
        style: Tên style cho dialogue (mặc định: NoteStyle)
    
    Returns:
        Số dialogue lines đã tạo
    
    Raises:
        FileNotFoundError: Nếu input file hoặc template không tồn tại
    """
    # Load template
    if template_path is None:
        template_path = DEFAULT_TEMPLATE
    
    template = load_template(template_path)
    
    # Parse SRT
    logger.info(f"Parsing SRT: {srt_path}")
    segments = parse_srt_file(srt_path)
    logger.info(f"  Found {len(segments)} segments")
    
    # Convert to ASS dialogues
    dialogues = convert_srt_segments_to_ass_dialogues(
        segments,
        max_chars=max_chars,
        style=style,
    )
    
    # Write ASS file
    write_ass_file(
        output_path,
        script_info=template['script_info'],
        styles=template['styles'],
        events_format=template['events_format'],
        dialogues=dialogues,
    )
    
    logger.info(f"Converted {len(dialogues)} dialogues to {output_path}")
    return len(dialogues)


# ─────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog="srt_to_ass",
        description="Convert SRT file to ASS format for video note overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick examples:
  uv run cli/srt_to_ass.py --input note_translated.srt --output note_overlay.ass

With custom template:
  uv run cli/srt_to_ass.py -i input.srt -t custom.ass -o output.ass

With custom max chars:
  uv run cli/srt_to_ass.py -i input.srt --max-chars 20 -o output.ass
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="FILE",
        help="Input SRT file path (required)",
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="Output ASS file path (default: <input>.ass)",
    )
    
    parser.add_argument(
        "--template", "-t",
        default=str(DEFAULT_TEMPLATE),
        metavar="FILE",
        help=f"ASS template file path (default: {DEFAULT_TEMPLATE})",
    )
    
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        metavar="N",
        help=f"Max characters per line (default: {DEFAULT_MAX_CHARS})",
    )
    
    parser.add_argument(
        "--style",
        default=DEFAULT_STYLE,
        metavar="NAME",
        help=f"Dialogue style name (default: {DEFAULT_STYLE})",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    """Main entry point for CLI."""
    parser = build_parser()
    args = parser.parse_args()
    
    # Setup logging
    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"File not found: {args.input}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Default: same name as input but with .ass extension
        output_path = input_path.with_suffix(".ass")
    
    # Validate template
    template_path = Path(args.template)
    if not template_path.exists():
        logger.error(f"Template file not found: {args.template}")
        sys.exit(1)
    
    # Convert
    try:
        count = convert_srt_to_ass(
            srt_path=str(input_path),
            output_path=str(output_path),
            template_path=str(template_path),
            max_chars=args.max_chars,
            style=args.style,
        )
        print(f"[OK] Converted {count} dialogue lines")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
    except Exception as e:
        logger.error(f"Conversion error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()