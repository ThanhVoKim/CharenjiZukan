#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video Subtitle Extractor - Entry Point

Trích xuất subtitle tiếng Trung từ video sử dụng DeepSeek-OCR-2
Hỗ trợ xuất nhiều file subtitle cho từng vùng box.

Sử dụng:
    python cli/video_ocr.py video.mp4
    python cli/video_ocr.py video.mp4 --boxes-file assets/boxesOCR.txt
    python cli/video_ocr.py video.mp4 --output-dir ./subtitles
    python cli/video_ocr.py video.mp4 --frame-interval 60

Examples:
    # Cơ bản với cấu hình box mặc định
    python cli/video_ocr.py my_video.mp4
    
    # Với các tùy chọn
    python cli/video_ocr.py my_video.mp4 --frame-interval 60 --boxes-file my_boxes.txt
    
    # Xử lý batch
    python cli/video_ocr.py --input-dir ./videos --output-dir ./subtitles
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import setup_logging, get_logger
from video_subtitle_extractor import VideoSubtitleExtractor
from video_subtitle_extractor.box_manager import parse_boxes_file, OcrBox
from video_subtitle_extractor.text_isolator import TextIsolationConfig, parse_color_spec

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
        default=str(PROJECT_ROOT / "assets/boxesOCR.txt"),
        help="Đường dẫn file txt cấu hình các vùng box (mặc định: assets/boxesOCR.txt)"
    )
    
    # Frame processing
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=argparse.SUPPRESS,
        help="Số frame bỏ qua giữa mỗi lần xử lý (mặc định: 30, tức là mỗi 1 giây với video 30fps)"
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=argparse.SUPPRESS,
        help="Ngưỡng phần trăm thay đổi pixel để phát hiện chuyển cảnh (mặc định: 1.5)"
    )
    parser.add_argument(
        "--min-scene-frames",
        type=int,
        default=argparse.SUPPRESS,
        help="Số frame tối thiểu giữa 2 lần chuyển cảnh (mặc định: 3)"
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=argparse.SUPPRESS,
        help="Ngưỡng Hamming distance cho perceptual hash (mặc định: 4). Giá trị nhỏ hơn = nhạy hơn với thay đổi nhỏ."
    )
    parser.add_argument(
        "--noise-threshold",
        type=int,
        default=argparse.SUPPRESS,
        help="Ngưỡng loại bỏ nhiễu nén video khi so sánh pixel (mặc định: 25)"
    )
    parser.add_argument(
        "--no-scene-detection",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Tắt scene detection (set threshold=0)"
    )

    # CV pre-filtering
    parser.add_argument(
        "--cv-prefilter",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Bật tiền lọc OpenCV (Canny edge) để bỏ qua ROI không có dấu hiệu text"
    )
    parser.add_argument(
        "--cv-min-edge-density",
        type=float,
        default=argparse.SUPPRESS,
        help="Ngưỡng mật độ cạnh tối thiểu cho CV prefilter (mặc định: 0.03)"
    )
    parser.add_argument(
        "--cv-edge-low",
        type=int,
        default=argparse.SUPPRESS,
        help="Ngưỡng thấp Canny edge detector (mặc định: 50)"
    )
    parser.add_argument(
        "--cv-edge-high",
        type=int,
        default=argparse.SUPPRESS,
        help="Ngưỡng cao Canny edge detector (mặc định: 150)"
    )
    
    # Text isolation (lọc watermark/overlay mờ trước OCR)
    parser.add_argument(
        "--isolate-text",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Bật lọc watermark/overlay mờ (opacity masking) trước khi OCR"
    )
    parser.add_argument(
        "--isolate-config",
        default=argparse.SUPPRESS,
        help="File JSON config từ tools/calibrate_text_isolation.py (nạp ngưỡng đã dò)"
    )
    parser.add_argument(
        "--subtitle-colors",
        default=argparse.SUPPRESS,
        help='Màu phụ đề chỉ định: "white,#FFD700" hoặc "255,255,255"'
    )
    parser.add_argument(
        "--color-tolerance",
        type=int,
        default=argparse.SUPPRESS,
        help="Sai số khoảng cách màu Lab (mặc định: 40)"
    )
    parser.add_argument(
        "--subtitle-min-contrast",
        type=int,
        default=argparse.SUPPRESS,
        help="Ngưỡng tương phản (proxy opacity); component mờ hơn bị xóa (mặc định: 40)"
    )
    parser.add_argument(
        "--stroke-max-luminance",
        type=int,
        default=argparse.SUPPRESS,
        help="Ngưỡng độ sáng coi là viền tối (mặc định: 80)"
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=argparse.SUPPRESS,
        help="Diện tích tối thiểu giữ component — chỉ diệt nhiễu, đặt nhỏ (mặc định: 8)"
    )
    parser.add_argument(
        "--no-require-stroke",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Tắt kiểm tra viền tối (dùng khi phụ đề không có viền)"
    )

    # Chinese filter
    parser.add_argument(
        "--min-chars",
        type=int,
        default=argparse.SUPPRESS,
        help="Số ký tự tối thiểu để ghi nhận (mặc định: 2)"
    )
    parser.add_argument(
        "--strip-punctuation",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Bỏ MỌI dấu câu (Unicode P*) khỏi text OCR — input sạch cho punctuate-srt. "
             "Hoạt động độc lập, không cần bật chinese filter."
    )
    parser.add_argument(
        "--enable-chinese-filter",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Bật tính năng lọc tiếng Trung (mặc định: nhận tất cả ngôn ngữ)"
    )
    
    # OCR settings
    parser.add_argument(
        "--ocr-model",
        default=argparse.SUPPRESS,
        help="Model Hugging Face OCR. DeepSeek: 'deepseek-ai/DeepSeek-OCR-2'. Qwen3-VL nhanh: 'Qwen/Qwen3-VL-8B-Instruct'. Qwen3-VL chính xác: 'Qwen/Qwen3-VL-8B-Thinking'."
    )
    parser.add_argument(
        "--qwen-max-new-tokens",
        type=int,
        default=argparse.SUPPRESS,
        help="[Chỉ Qwen3-VL] Số token tối đa sinh ra (mặc định: 256)"
    )
    parser.add_argument(
        "--qwen-min-pixels",
        type=int,
        default=argparse.SUPPRESS,
        help="[Chỉ Qwen3-VL] Pixel blocks tối thiểu (mặc định: 256). Ảnh hưởng VRAM."
    )
    parser.add_argument(
        "--qwen-max-pixels",
        type=int,
        default=argparse.SUPPRESS,
        help="[Chỉ Qwen3-VL] Pixel blocks tối đa (mặc định: 1280). Ảnh hưởng VRAM."
    )
    parser.add_argument(
        "--hf-token",
        default=argparse.SUPPRESS,
        help="Hugging Face Token để tải model (nếu cần)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=argparse.SUPPRESS,
        help="Thiết bị chạy OCR (mặc định: cuda)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=argparse.SUPPRESS,
        help="Batch size cho OCR (mặc định: 4)"
    )
    
    # Output format
    parser.add_argument(
        "--format",
        choices=["srt", "txt"],
        default=argparse.SUPPRESS,
        help="Định dạng output (mặc định: srt)"
    )
    parser.add_argument(
        "--default-duration",
        type=float,
        default=argparse.SUPPRESS,
        help="Thời lượng mặc định cho mỗi subtitle (mặc định: 3.0s)"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=argparse.SUPPRESS,
        help="Thời lượng tối thiểu cho subtitle (mặc định: 1.0s)"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=argparse.SUPPRESS,
        help="Thời lượng tối đa cho subtitle (mặc định: 7.0s)"
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Tắt loại bỏ text trùng lặp liên tiếp"
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Tắt in timestamp (dành cho format txt)"
    )
    
    # Feature
    parser.add_argument(
        "--warn-english",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Xuất cảnh báo ra file txt nếu subtitle chứa ký tự tiếng Anh/số [a-zA-Z0-9]"
    )
    parser.add_argument(
        "--save-minify-txt",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Lưu file [video]_script.txt thuần văn bản (minify), mỗi câu 1 dòng"
    )
    parser.add_argument(
        "--min-subtitle-frames",
        type=int,
        dest="min_subtitle_frames",
        default=argparse.SUPPRESS,
        help="Ngưỡng raw frame tối thiểu; subtitle xuất hiện ít hơn giá trị này sẽ bị cảnh báo trong file _warnings.txt (mặc định: 15)"
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


def _coerce_colors(value):
    """Chuẩn hóa subtitle_colors về list[tuple[int,int,int]] từ str/list/tuple."""
    if not value:
        return []
    if isinstance(value, str):
        return parse_color_spec(value)
    colors = []
    for item in value:
        if isinstance(item, str):
            colors.extend(parse_color_spec(item))
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            colors.append((int(item[0]), int(item[1]), int(item[2])))
    return colors


def build_text_isolation_config(args, config: dict) -> TextIsolationConfig:
    """Dựng TextIsolationConfig theo thứ tự ưu tiên: CLI > YAML > JSON-calibrate > default.

    Bật khi có `--isolate-text`, hoặc YAML/JSON đặt `enabled: true`.
    """
    yaml_ti = config.get("text_isolation", {}) if isinstance(config, dict) else {}

    # Nạp JSON từ tools/calibrate_text_isolation.py nếu được chỉ định.
    json_vals: dict = {}
    iso_path = getattr(args, "isolate_config", None) or yaml_ti.get("config_file")
    if iso_path:
        try:
            with open(iso_path, "r", encoding="utf-8") as f:
                json_vals = json.load(f)
        except Exception as e:
            logger.warning(f"Không đọc được isolate-config {iso_path}: {e}")

    defaults = TextIsolationConfig()

    def resolve(cli_name, yaml_key, default):
        if hasattr(args, cli_name):
            return getattr(args, cli_name)
        if yaml_key in yaml_ti and yaml_ti[yaml_key] is not None:
            return yaml_ti[yaml_key]
        if yaml_key in json_vals and json_vals[yaml_key] is not None:
            return json_vals[yaml_key]
        return default

    enabled = (
        hasattr(args, "isolate_text")
        or bool(yaml_ti.get("enabled", False))
        or bool(json_vals.get("enabled", False))
    )

    colors_raw = resolve("subtitle_colors", "subtitle_colors", defaults.subtitle_colors)
    require_stroke = (
        False if hasattr(args, "no_require_stroke")
        else bool(resolve("require_stroke", "require_stroke", defaults.require_stroke))
    )

    return TextIsolationConfig(
        enabled=enabled,
        subtitle_colors=_coerce_colors(colors_raw),
        color_tolerance=int(resolve("color_tolerance", "color_tolerance", defaults.color_tolerance)),
        min_contrast=int(resolve("subtitle_min_contrast", "min_contrast", defaults.min_contrast)),
        stroke_max_luminance=int(resolve("stroke_max_luminance", "stroke_max_luminance", defaults.stroke_max_luminance)),
        stroke_search_px=int(resolve("stroke_search_px", "stroke_search_px", defaults.stroke_search_px)),
        min_component_area=int(resolve("min_component_area", "min_component_area", defaults.min_component_area)),
        require_stroke=require_stroke,
        bright_luminance=int(resolve("bright_luminance", "bright_luminance", defaults.bright_luminance)),
    )


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
    boxes = []
    
    # Precedence cho boxes_file: YAML > CLI override fallback default
    boxes_file = getattr(args, "boxes_file", str(PROJECT_ROOT / "assets/boxesOCR.txt"))
    if config.get("roi", {}).get("boxes_file"):
        boxes_file = config["roi"]["boxes_file"]
    
    boxes = parse_boxes_file(boxes_file)
    if not boxes:
        # Fallback to YAML inline boxes if file fails
        yaml_boxes = config.get("roi", {}).get("boxes", [])
        if yaml_boxes:
            logger.info("Sử dụng cấu hình inline boxes từ YAML config.")
            for b in yaml_boxes:
                boxes.append(OcrBox(
                    name=b.get("name", "box"),
                    x=b.get("x", 0),
                    y=b.get("y", 0),
                    w=b.get("w", 0),
                    h=b.get("h", 0)
                ))
                
    if not boxes:
        logger.warning(f"Không tìm thấy cấu hình box hợp lệ. Sử dụng box mặc định.")
        boxes = [OcrBox(name="subtitle", x=0, y=800, w=1920, h=280)] # Fallback
        
    # Helper để lấy config với thứ tự ưu tiên: CLI > YAML > Default
    def get_param(cli_name: str, yaml_path: tuple, default_val):
        # 1. Check CLI (nếu có và không phải SUPPRESS - đã xử lý ở parse_args bằng suppress)
        if hasattr(args, cli_name):
            return getattr(args, cli_name)
            
        # 2. Check YAML
        val = config
        for key in yaml_path:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                val = None
                break
                
        if val is not None:
            return val
            
        # 3. Default fallback
        return default_val

    # Create extractor
    extractor = VideoSubtitleExtractor(
        boxes=boxes,
        
        # Frame processing
        frame_interval=get_param("frame_interval", ("video", "frame_interval"), 30),
        scene_threshold=0 if hasattr(args, "no_scene_detection") else get_param("scene_threshold", ("scene_detection", "threshold"), 1.5),
        min_scene_frames=get_param("min_scene_frames", ("scene_detection", "min_scene_frames"), 3),
        phash_threshold=get_param("phash_threshold", ("scene_detection", "phash_threshold"), 4),
        noise_threshold=get_param("noise_threshold", ("scene_detection", "noise_threshold"), 25),

        # CV pre-filtering
        cv_prefilter=get_param("cv_prefilter", ("cv_prefilter", "enabled"), False),
        cv_min_edge_density=get_param("cv_min_edge_density", ("cv_prefilter", "min_edge_density"), 0.03),
        cv_edge_low=get_param("cv_edge_low", ("cv_prefilter", "edge_low_threshold"), 50),
        cv_edge_high=get_param("cv_edge_high", ("cv_prefilter", "edge_high_threshold"), 150),
        
        # Chinese filter
        min_char_count=get_param("min_chars", ("chinese_filter", "min_char_count"), 2),
        enable_chinese_filter=get_param("enable_chinese_filter", ("chinese_filter", "enabled"), False),

        # Punctuation
        strip_punctuation=get_param("strip_punctuation", ("output", "strip_punctuation"), False),
        
        # Text isolation (lọc watermark/overlay mờ trước OCR)
        text_isolation=build_text_isolation_config(args, config),

        # OCR
        ocr_model=get_param("ocr_model", ("ocr", "model"), "deepseek-ai/DeepSeek-OCR-2"),
        device=get_param("device", ("ocr", "device"), "cuda"),
        batch_size=get_param("batch_size", ("ocr", "batch_size"), 4),
        hf_token=get_param("hf_token", ("ocr", "hf_token"), None),
        qwen_max_new_tokens=get_param("qwen_max_new_tokens", ("ocr", "qwen_max_new_tokens"), 256),
        qwen_min_pixels=get_param("qwen_min_pixels", ("ocr", "qwen_min_pixels"), 256),
        qwen_max_pixels=get_param("qwen_max_pixels", ("ocr", "qwen_max_pixels"), 1280),
        
        # Output
        output_format=get_param("format", ("output", "format"), "srt"),
        default_subtitle_duration=get_param("default_duration", ("output", "default_duration"), 3.0),
        warn_english=get_param("warn_english", ("output", "warn_english"), False),
        save_minify_txt=get_param("save_minify_txt", ("output", "save_minify_txt"), False),
        min_subtitle_frames=get_param("min_subtitle_frames", ("output", "min_subtitle_frames"), 15),
    )
    
    # Thiết lập Writer parameters (những tham số này được sử dụng khi gọi write_srt/write_txt)
    # Lưu chúng vào extractor.metadata hoặc inject thẳng vào writers
    for writer in extractor.writers.values():
        writer.min_duration = get_param("min_duration", ("output", "min_duration"), 1.0)
        writer.max_duration = get_param("max_duration", ("output", "max_duration"), 7.0)
        
    # Lưu flag output formatter
    extractor.include_timestamp = not hasattr(args, "no_timestamp") if hasattr(args, "no_timestamp") else get_param("include_timestamp", ("output", "include_timestamp"), True)
    extractor.deduplicate_output = not hasattr(args, "no_deduplicate") if hasattr(args, "no_deduplicate") else get_param("deduplicate", ("output", "deduplicate"), True)
    
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