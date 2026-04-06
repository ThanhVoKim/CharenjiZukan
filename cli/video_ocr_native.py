#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Native Video Subtitle Extractor - CLI Entry Point

Trích xuất subtitle bằng Qwen3-VL Native Video mode.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import setup_logging, get_logger
from video_subtitle_extractor.box_manager import parse_boxes_file, OcrBox
from video_subtitle_extractor.native_video_extractor import NativeVideoSubtitleExtractor

logger = get_logger(__name__)


DEFAULT_CONFIG_PATH = str(PROJECT_ROOT / "config/native_video_ocr_config.yaml")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Trích xuất subtitle bằng Qwen3-VL Native Video mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4

  %(prog)s video.mp4 --model Qwen/Qwen3-VL-8B-Thinking --device cuda

  %(prog)s video.mp4 --frame-interval 6 --sample-fps 5.0 --batch-duration 60

  %(prog)s video.mp4 --warn-english --save-minify-txt
        """,
    )

    parser.add_argument("video", help="Đường dẫn file video cần xử lý")

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Đường dẫn file config YAML (mặc định: {DEFAULT_CONFIG_PATH})",
    )

    parser.add_argument(
        "--boxes-file",
        default=argparse.SUPPRESS,
        help="File cấu hình ROI boxes (mặc định lấy từ config)",
    )

    parser.add_argument(
        "--output-dir",
        default=argparse.SUPPRESS,
        help="Thư mục output (mặc định: cùng thư mục video)",
    )

    parser.add_argument(
        "--prompt-file",
        default=argparse.SUPPRESS,
        help="Đường dẫn prompt template file",
    )

    # Sampling
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=argparse.SUPPRESS,
        help="Lấy 1 frame mỗi N frame (mặc định: 6)",
    )
    parser.add_argument(
        "--batch-duration",
        type=float,
        default=argparse.SUPPRESS,
        help="Số giây mỗi batch xử lý (mặc định: 60.0)",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=argparse.SUPPRESS,
        help="sample_fps cho Native Video frame-list (mặc định: 5.0)",
    )

    # Model
    parser.add_argument(
        "--model",
        default=argparse.SUPPRESS,
        help="Model name (Qwen/Qwen3-VL-8B-Instruct hoặc Qwen/Qwen3-VL-8B-Thinking)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=argparse.SUPPRESS,
        help="Thiết bị chạy model (mặc định: cuda)",
    )
    parser.add_argument(
        "--hf-token",
        default=argparse.SUPPRESS,
        help="Hugging Face token (nếu cần)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=argparse.SUPPRESS,
        help="Số token tối đa model sinh ra (mặc định: 2048)",
    )
    parser.add_argument(
        "--total-pixels",
        type=int,
        default=argparse.SUPPRESS,
        help="Giới hạn total_pixels cho video input",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=argparse.SUPPRESS,
        help="Giới hạn min_pixels cho video input",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=argparse.SUPPRESS,
        help="Giới hạn max_frames cho mỗi batch",
    )

    # Optional outputs
    parser.add_argument(
        "--warn-english",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Lưu file [video]_subtitle_english_warnings.txt",
    )
    parser.add_argument(
        "--save-minify-txt",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Lưu file [video]_native_script.txt chỉ chứa text",
    )

    # Logging
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Tăng verbosity (-v: INFO, -vv: DEBUG)",
    )
    parser.add_argument("--quiet", action="store_true", help="Chỉ hiển thị errors")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load config từ YAML file"""
    try:
        import yaml

        path = Path(config_path)
        if not path.exists():
            logger.warning("Config file không tồn tại: %s. Dùng defaults.", config_path)
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        logger.warning("PyYAML chưa cài. Dùng defaults.")
        return {}
    except Exception as e:
        logger.warning("Không thể load config file: %s", e)
        return {}


def main():
    args = parse_args()

    # Setup logging
    if args.quiet:
        log_level = 40
    elif args.verbose >= 2:
        log_level = 10
    elif args.verbose >= 1:
        log_level = 20
    else:
        log_level = 20

    setup_logging(level=log_level)

    # Load config
    config = load_config(args.config)

    def get_param(cli_name: str, yaml_path: tuple, default_val):
        if hasattr(args, cli_name):
            return getattr(args, cli_name)

        val = config
        for key in yaml_path:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                val = None
                break

        if val is not None:
            return val
        return default_val

    # Boxes
    boxes_file = get_param("boxes_file", ("roi", "boxes_file"), str(PROJECT_ROOT / "assets/boxesOCR.txt"))
    boxes = parse_boxes_file(boxes_file)
    if not boxes:
        logger.warning("Không đọc được boxes từ %s. Dùng fallback mặc định.", boxes_file)
        boxes = [OcrBox(name="subtitle", x=370, y=984, w=1180, h=80)]

    # Extractor
    extractor = NativeVideoSubtitleExtractor(
        boxes=boxes,
        model_name=get_param("model", ("ocr", "model"), "Qwen/Qwen3-VL-8B-Instruct"),
        device=get_param("device", ("ocr", "device"), "cuda"),
        hf_token=get_param("hf_token", ("ocr", "hf_token"), None),
        frame_interval=get_param("frame_interval", ("video", "frame_interval"), 6),
        batch_duration=get_param("batch_duration", ("video", "batch_duration"), 60.0),
        sample_fps=get_param("sample_fps", ("video", "sample_fps"), 5.0),
        max_new_tokens=get_param("max_new_tokens", ("ocr", "max_new_tokens"), 2048),
        total_pixels=get_param("total_pixels", ("ocr", "total_pixels"), 20480 * 32 * 32),
        min_pixels=get_param("min_pixels", ("ocr", "min_pixels"), 64 * 32 * 32),
        max_frames=get_param("max_frames", ("ocr", "max_frames"), 2048),
        warn_english=get_param("warn_english", ("output", "warn_english"), False),
        save_minify_txt=get_param("save_minify_txt", ("output", "save_minify_txt"), False),
        prompt_file=get_param("prompt_file", ("prompt", "file"), str(PROJECT_ROOT / "prompts/native_video_ocr_prompt.txt")),
    )

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        sys.exit(1)

    output_dir = getattr(args, "output_dir", None)

    try:
        result = extractor.extract(str(video_path), output_dir=output_dir)

        print("\n" + "=" * 60)
        print("✅ Native Video OCR complete")
        print("=" * 60)
        print(f"Entries: {result.total_entries}")
        for key, value in result.output_paths.items():
            print(f"{key}: {value}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Error: %s", e)
        if args.verbose >= 2:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
