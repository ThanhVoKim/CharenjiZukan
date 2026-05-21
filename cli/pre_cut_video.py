#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli/pre_cut_video.py — CLI: Pre-cut video by removing unwanted segments.

Removes segments specified in a remove SRT file from the source video,
producing a clean video for transcript/translate/sync pipeline.

Example:
    uv run cli/pre_cut_video.py --input source.mp4 --output clean.mp4 --remove-srt remove.srt
    uv run cli/pre_cut_video.py --input source.mp4 --output clean.mp4 --remove-srt remove.srt --method reencode-smooth
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import setup_logging, get_logger  # noqa: E402
from utils.video_cutter import run_pre_cut  # noqa: E402

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pre_cut_video",
        description=(
            "Pre-cut video: remove unwanted segments before transcript/sync. "
            "Input is a remove SRT file with timestamps of segments to delete."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default hybrid-copy (video stream copy, audio AAC encode)
  uv run cli/pre_cut_video.py --input source.mp4 --output clean.mp4 --remove-srt remove.srt

  # Re-encode with HEVC NVENC for smoother cuts
  uv run cli/pre_cut_video.py --input source.mp4 --output clean.mp4 --remove-srt remove.srt --method reencode-smooth

  # Keep temp files for debugging
  uv run cli/pre_cut_video.py --input source.mp4 --output clean.mp4 --remove-srt remove.srt --keep-tmp -v

Remove SRT format:
  1
  00:00:12,500 --> 00:00:18,000
  CUT intro mistake

  2
  00:03:10,000 --> 00:03:25,200
  CUT sponsor
        """,
    )

    parser.add_argument(
        "--input", "-i", required=True, metavar="FILE",
        help="Input video file.",
    )
    parser.add_argument(
        "--output", "-o", required=True, metavar="FILE",
        help="Output clean video file.",
    )
    parser.add_argument(
        "--remove-srt", "-r", required=True, metavar="FILE",
        help="SRT file with segments to remove (timestamps in source video timeline).",
    )
    parser.add_argument(
        "--manifest", default=None, metavar="FILE",
        help="Manifest output path. Default: <output>_cut_manifest.json.",
    )
    parser.add_argument(
        "--method", default="hybrid-copy",
        choices=["hybrid-copy", "reencode-smooth"],
        help="Cut method. hybrid-copy (default): video stream copy. reencode-smooth: HEVC NVENC re-encode.",
    )
    parser.add_argument(
        "--hevc-cq", type=int, default=28,
        help="CQ value for reencode-smooth (default: 28).",
    )
    parser.add_argument(
        "--maxrate-ratio", type=float, default=1.15,
        help="Multiply input bitrate by this for maxrate (default: 1.15).",
    )
    parser.add_argument(
        "--hevc-preset", default="p4",
        help="HEVC NVENC preset (default: p4).",
    )
    parser.add_argument(
        "--audio-bitrate", default="256k",
        help="AAC audio bitrate (default: 256k).",
    )
    parser.add_argument(
        "--audio-fade-ms", type=float, default=10,
        help="Audio fade duration in ms at part boundaries (default: 10).",
    )
    parser.add_argument(
        "--safe-margin-ms", type=float, default=100,
        help="Expand remove ranges by this margin in ms (default: 100).",
    )
    parser.add_argument(
        "--disable-audio-fade", action="store_true",
        help="Disable audio fade at part boundaries.",
    )
    parser.add_argument(
        "--keep-tmp", action="store_true",
        help="Keep temporary part files after concat for debugging.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging (DEBUG level).",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    srt_path = Path(args.remove_srt)
    if not srt_path.exists():
        logger.error("Remove SRT file not found: %s", args.remove_srt)
        sys.exit(1)

    try:
        result = run_pre_cut(
            input_path=str(input_path),
            output_path=args.output,
            remove_srt_path=str(srt_path),
            manifest_path=args.manifest,
            method=args.method,
            hevc_cq=args.hevc_cq,
            maxrate_ratio=args.maxrate_ratio,
            hevc_preset=args.hevc_preset,
            audio_bitrate=args.audio_bitrate,
            audio_fade_ms=args.audio_fade_ms,
            safe_margin_ms=args.safe_margin_ms,
            audio_fade_enabled=not args.disable_audio_fade,
            keep_tmp=args.keep_tmp,
        )
    except RuntimeError as e:
        logger.error("Pre-cut failed: %s", e)
        sys.exit(1)

    print(f"\nPre-cut complete!")
    print(f"  Input:    {args.input}")
    print(f"  Output:   {result.output_path}")
    print(f"  Manifest: {result.manifest_path}")
    print(f"  Method:   {args.method}")
    keep_count = len(result.manifest.get("keep_ranges", []))
    print(f"  Keep segments: {keep_count}")
    drift = result.manifest.get("duration_drift_ms", 0)
    print(f"  Duration drift: {drift:.0f}ms")
    if result.manifest.get("warnings"):
        print(f"  Warnings: {len(result.manifest['warnings'])}")
        for w in result.manifest["warnings"]:
            print(f"    - {w}")


if __name__ == "__main__":
    main()
