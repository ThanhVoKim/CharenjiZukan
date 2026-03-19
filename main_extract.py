#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video Subtitle Extractor - Entry Point

Trích xuất subtitle tiếng Trung từ video sử dụng DeepSeek-OCR-2
Hỗ trợ xuất nhiều file subtitle cho từng vùng box.

Sử dụng:
    python main_extract.py video.mp4
    python main_extract.py video.mp4 --boxes-file assets/boxesOCR.txt
    python main_extract.py video.mp4 --output-dir ./subtitles
    python main_extract.py video.mp4 --frame-interval 60

Examples:
    # Cơ bản với cấu hình box mặc định
    python main_extract.py my_video.mp4
    
    # Với các tùy chọn
    python main_extract.py my_video.mp4 --frame-interval 60 --boxes-file my_boxes.txt
    
    # Xử lý batch
    python main_extract.py --input-dir ./videos --output-dir ./subtitles
"""

import argparse
import sys
from pathlib import Path

# Thêm project root vào path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logging, get_logger
from video_subtitle_extractor import VideoSubtitleExtractor
from video_subtitle_extractor.box_manager import parse_boxes_file, OcrBox

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Trích xuất subtitle tiếng Trung từ video sử dụng DeepSeek-OCR-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4
      Trích xuất subtitle từ video.mp4

  %(prog)s video.mp4 --output-dir subtitles
      Trích xuất với output directory chỉ định

  %(prog)s video.mp4 --boxes-file assets/boxesOCR.txt
      Sử dụng file cấu hình box tùy chỉnh

  %(prog)s --input-dir ./videos --output-dir ./subtitles
      Xử lý tất cả video trong thư mục
        """
    )
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "video",
        nargs="?",
        help="Đường dẫn file video cần xử lý"
    )
    input_group.add_argument(
        "--input-dir",
        help="Thư mục chứa các file video cần xử lý (batch mode)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        help="Thư mục chứa các file output (mặc định: cùng thư mục video)"
    )
    
    # Box configuration
    parser.add_argument(
        "--boxes-file",
        default="assets/boxesOCR.txt",
        help="Đường dẫn file txt cấu hình các vùng box (mặc định: assets/boxesOCR.txt)"
    )
    
    # Frame processing
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=30,
        help="Số frame bỏ qua giữa mỗi lần xử lý (mặc định: 30, tức là mỗi 1 giây với video 30fps)"
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=30.0,
        help="Ngưỡng phát hiện chuyển cảnh (mặc định: 30.0)"
    )
    parser.add_argument(
        "--no-scene-detection",
        action="store_true",
        help="Tắt scene detection"
    )
    
    # Chinese filter
    parser.add_argument(
        "--min-chars",
        type=int,
        default=2,
        help="Số ký tự Trung tối thiểu (mặc định: 2)"
    )
    parser.add_argument(
        "--no-punctuation",
        action="store_true",
        help="Không giữ dấu câu tiếng Trung"
    )
    parser.add_argument(
        "--enable-chinese-filter",
        action="store_true",
        help="Bật tính năng lọc tiếng Trung (mặc định: nhận tất cả ngôn ngữ)"
    )
    
    # OCR settings
    parser.add_argument(
        "--hf-token",
        help="Hugging Face Token để tải model (nếu cần)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Thiết bị chạy OCR (mặc định: cuda)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size cho OCR (mặc định: 8)"
    )
    
    # Output format
    parser.add_argument(
        "--format",
        choices=["srt", "txt"],
        default="srt",
        help="Định dạng output (mặc định: srt)"
    )
    
    # Config file
    parser.add_argument(
        "--config",
        help="Đường dẫn file config YAML"
    )
    
    # Logging
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Tăng verbosity (-v: INFO, -vv: DEBUG)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Chỉ hiển thị errors"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load config từ YAML file"""
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except ImportError:
        logger.warning("PyYAML not installed. Using default config.")
        return {}
    except Exception as e:
        logger.warning(f"Cannot load config file: {e}")
        return {}


def main():
    """Main entry point"""
    args = parse_args()
    
    # Setup logging
    if args.quiet:
        log_level = 40  # ERROR
    elif args.verbose >= 2:
        log_level = 10  # DEBUG
    elif args.verbose >= 1:
        log_level = 20  # INFO
    else:
        log_level = 20  # INFO
    
    setup_logging(level=log_level)
    
    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Lấy thông tin box
    boxes = parse_boxes_file(args.boxes_file)
    if not boxes:
        logger.warning(f"Không tìm thấy cấu hình box hợp lệ trong {args.boxes_file}. Sử dụng box mặc định.")
        boxes = [OcrBox(name="subtitle", x=0, y=800, w=1920, h=280)] # Fallback
        
    # Create extractor
    extractor = VideoSubtitleExtractor(
        boxes=boxes,
        
        # Frame processing
        frame_interval=args.frame_interval,
        scene_threshold=args.scene_threshold if not args.no_scene_detection else 0,
        
        # Chinese filter
        keep_punctuation=not args.no_punctuation,
        min_char_count=args.min_chars,
        enable_chinese_filter=args.enable_chinese_filter,
        
        # OCR
        device=args.device,
        batch_size=args.batch_size,
        hf_token=args.hf_token,
        
        # Output
        output_format=args.format
    )
    
    # Progress callback
    def progress_callback(current, total, ocr_done):
        if total > 0:
            percent = (current / total) * 100
            print(f"\r   Progress: {percent:.1f}% ({ocr_done} box OCR calls)", end="", flush=True)
    
    try:
        if args.input_dir:
            # Batch mode
            logger.info(f"Batch mode: Processing videos in {args.input_dir}")
            results = extractor.extract_from_directory(
                args.input_dir,
                args.output_dir
            )
            
            print(f"\n{'='*60}")
            print(f"✅ Batch processing complete!")
            print(f"   Processed: {len(results)} videos")
            total_subs = sum(sum(counts.values()) for r in results for counts in [r.subtitles_count])
            print(f"   Total subtitles: {total_subs}")
            print(f"{'='*60}")
            
        else:
            # Single video mode
            if not args.video:
                logger.error("No video file specified!")
                sys.exit(1)
            
            if not Path(args.video).exists():
                logger.error(f"Video file not found: {args.video}")
                sys.exit(1)
            
            result = extractor.extract(
                args.video,
                args.output_dir,
                progress_callback=progress_callback
            )
            
            print()  # New line after progress
            
            total_extracted = sum(result.subtitles_count.values())
            if total_extracted > 0:
                print(f"\n🎉 Done! Extracted {total_extracted} subtitles across {len(result.subtitles_count)} boxes")
                for box_name, count in result.subtitles_count.items():
                    print(f"   - {box_name}: {count} entries -> {result.output_paths[box_name]}")
            else:
                print(f"\n⚠️ No subtitles found in video")
                
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()