# -*- coding: utf-8 -*-
"""
Video Subtitle Extractor - Main class

Trích xuất subtitle tiếng Trung từ video sử dụng DeepSeek-OCR-2
Hỗ trợ Multi-Box OCR: Theo dõi và trích xuất nhiều vùng phụ đề/ghi chú độc lập.

Workflow:
1. Frame Sampling - Lấy mỗi N frame
2. Box Cropping & Scene Detection - Theo dõi chuyển cảnh từng box độc lập
3. Selective OCR - Chỉ gọi OCR cho box bị thay đổi
4. Chinese Filter - Lọc chỉ tiếng Trung
5. Deduplication - Loại bỏ trùng lặp
6. Output SRT - Xuất nhiều file subtitle theo từng box
"""

import os
import sys
import cv2
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import time

# Thêm project root vào path để import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

from .frame_processor import FrameProcessor
from .chinese_filter import ChineseFilter
from .subtitle_writer import SubtitleWriter, SubtitleEntry
from .deepseek_ocr import DeepSeekOCR
from .box_manager import OcrBox, BoxState

logger = get_logger(__name__)


@dataclass
class ExtractionResult:
    """Kết quả extraction"""
    video_path: str
    total_frames: int
    processed_frames: int
    ocr_frames: int
    subtitles_count: Dict[str, int]
    subtitles: Dict[str, List[SubtitleEntry]]
    output_paths: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


class VideoSubtitleExtractor:
    """
    Trích xuất subtitle từ video sử dụng DeepSeek-OCR-2 với hỗ trợ Multi-box
    
    Features:
    - Multi-box ROI: Định nghĩa nhiều vùng OCR, xuất file SRT riêng
    - Selective Scene Detection: Chỉ OCR box khi có thay đổi hình ảnh
    - Frame Sampling: Giảm số frame cần xử lý
    - Chinese Filter: Chỉ giữ tiếng Trung
    - Deduplication: Loại bỏ text trùng
    """
    
    def __init__(
        self,
        # Multi-box OCR
        boxes: List[OcrBox],
        
        # Frame processing
        frame_interval: int = 30,
        scene_threshold: float = 30.0,
        min_scene_frames: int = 10,
        
        # Chinese filter
        keep_punctuation: bool = True,
        min_char_count: int = 2,
        enable_chinese_filter: bool = False,
        
        # OCR settings
        ocr_model: str = "deepseek-ai/DeepSeek-OCR-2",
        device: str = "cuda",
        batch_size: int = 8,
        hf_token: Optional[str] = None,
        
        # Output settings
        output_format: str = "srt",
        default_subtitle_duration: float = 3.0,
    ):
        """Khởi tạo VideoSubtitleExtractor với danh sách các box"""
        
        if not boxes:
            raise ValueError("Phải cung cấp ít nhất 1 OcrBox")
            
        self.boxes = boxes
        self.box_states = {box.name: BoxState(box=box) for box in boxes}
        
        # Initialize components
        self.frame_processor = FrameProcessor(
            frame_interval=frame_interval,
            scene_threshold=scene_threshold
        )
        self.min_scene_frames = min_scene_frames
        
        self.chinese_filter = ChineseFilter(
            keep_punctuation=keep_punctuation,
            min_char_count=min_char_count
        )
        self.enable_chinese_filter = enable_chinese_filter
        
        self.writers = {
            box.name: SubtitleWriter(default_duration=default_subtitle_duration) 
            for box in boxes
        }
        
        # Settings
        self.ocr_model_name = ocr_model
        self.device = device
        self.batch_size = batch_size
        self.hf_token = hf_token
        self.output_format = output_format
        self.default_subtitle_duration = default_subtitle_duration
        self.frame_interval = frame_interval
        
        # OCR model (lazy load)
        self._ocr_model = None
        self._model_loaded = False
        
        logger.info(f"VideoSubtitleExtractor initialized with {len(boxes)} boxes:")
        for b in boxes:
            logger.info(f"  - Box '{b.name}': x={b.x}, y={b.y}, w={b.w}, h={b.h}")
        logger.info(f"  OCR Model: {ocr_model}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Frame interval: {frame_interval}")
    
    def load_ocr_model(self):
        """Load DeepSeek-OCR-2 model"""
        if self._model_loaded:
            return
        
        logger.info(f"🔄 Loading {self.ocr_model_name} on {self.device}...")
        try:
            self._ocr_model = DeepSeekOCR(
                model_name=self.ocr_model_name,
                device=self.device,
                hf_token=self.hf_token
            )
            self._model_loaded = True
            logger.info(f"✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def ocr_image(self, image) -> str:
        """OCR một image"""
        if not self._model_loaded:
            self.load_ocr_model()
        try:
            return self._ocr_model.recognize(image)
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
    
    def ocr_batch(self, images: List[np.ndarray]) -> List[str]:
        """OCR một list các images"""
        if not images:
            return []
        if not self._model_loaded:
            self.load_ocr_model()
            
        results = []
        for i, image in enumerate(images):
            text = self.ocr_image(image)
            results.append(text)
        return results
    
    def extract(
        self, 
        video_path: str, 
        output_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, int], None]] = None
    ) -> ExtractionResult:
        """
        Trích xuất subtitle từ video ra nhiều file
        
        Args:
            video_path: Đường dẫn file video
            output_dir: Thư mục chứa các file xuất ra (tự động tạo nếu None)
            progress_callback: Callback(current_frame, total_frames, ocr_done)
        """
        start_time = time.time()
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        # Determine output directory
        video_dir = Path(video_path).parent
        video_stem = Path(video_path).stem
        
        if output_dir is None:
            # Nếu không truyền output_dir, mặc định lưu chung thư mục với video
            out_dir = video_dir
        else:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info("="*60)
        logger.info("🎬 Video Multi-Box Subtitle Extraction")
        logger.info("="*60)
        logger.info(f"Input:  {video_path}")
        logger.info(f"Output Dir: {out_dir}")
        logger.info("="*60)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Load OCR sớm
        self.load_ocr_model()
        
        frame_number = 0
        processed_frames_count = 0
        ocr_total_calls = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 1. Frame sampling
            if not self.frame_processor.should_process_frame(frame_number):
                frame_number += 1
                continue
                
            timestamp = frame_number / fps
            processed_frames_count += 1
            
            ocr_needed_images = []
            ocr_needed_boxes = []
            
            # 2. Xử lý từng box độc lập
            for box_name, state in self.box_states.items():
                curr_roi = self.frame_processor.crop_roi(frame, state.box)
                
                is_changed = self.frame_processor.detect_scene_change_for_box(curr_roi, state.prev_roi)
                
                # Bỏ qua scene change nếu chưa đủ số frame tối thiểu kể từ lần cuối thay đổi
                frames_since_last = frame_number - state.last_scene_frame
                if is_changed and frames_since_last < self.min_scene_frames:
                    is_changed = False
                
                if not is_changed:
                    # Kéo dài thời gian của entry hiện tại nếu có
                    if state.entries:
                        state.entries[-1].end_time = timestamp
                else:
                    state.last_scene_frame = frame_number
                    ocr_needed_images.append(curr_roi)
                    ocr_needed_boxes.append(box_name)
                    
                state.prev_roi = curr_roi.copy()
            
            # 3. Selective OCR - Gọi batch OCR cho các vùng thay đổi
            if ocr_needed_images:
                results = self.ocr_batch(ocr_needed_images)
                ocr_total_calls += len(ocr_needed_images)
                
                for box_name, text in zip(ocr_needed_boxes, results):
                    state = self.box_states[box_name]
                    
                    if self.enable_chinese_filter:
                        processed_text = self.chinese_filter.filter_text(text)
                    else:
                        processed_text = text.strip() if text else ""
                        
                    if processed_text:
                        # Thêm entry mới
                        entry = SubtitleEntry(
                            index=len(state.entries) + 1,
                            start_time=timestamp,
                            end_time=timestamp + self.default_subtitle_duration,
                            text=processed_text
                        )
                        state.entries.append(entry)
            
            # Progress callback
            if progress_callback and processed_frames_count % 10 == 0:
                progress_callback(frame_number, total_frames, ocr_total_calls)
                
            frame_number += 1
            
        cap.release()
        
        # 4. Ghi file output
        logger.info(f"\n📝 Writing outputs...")
        output_paths = {}
        subtitles_count = {}
        subtitles_dict = {}
        
        ext = ".srt" if self.output_format == "srt" else ".txt"
        
        # Đọc các metadata output cấu hình thêm từ main
        include_timestamp = getattr(self, "include_timestamp", True)
        deduplicate_output = getattr(self, "deduplicate_output", True)
        
        for box_name, state in self.box_states.items():
            writer = self.writers[box_name]
            
            # Tên file: {video_name}_{box_name}.srt
            file_name = f"{video_stem}_{box_name}{ext}"
            out_path = out_dir / file_name
            
            if self.output_format == "srt":
                writer.write_srt(state.entries, str(out_path), deduplicate=deduplicate_output)
            else:
                writer.write_txt(state.entries, str(out_path), include_timestamp=include_timestamp, deduplicate=deduplicate_output)
                
            output_paths[box_name] = str(out_path)
            subtitles_count[box_name] = len(state.entries)
            subtitles_dict[box_name] = state.entries
            
        processing_time = time.time() - start_time
        
        # Reset state để chuẩn bị cho video tiếp theo (nếu có)
        for state in self.box_states.values():
            state.prev_roi = None
            state.last_scene_frame = 0
            state.entries = []
            
        # Summary
        logger.info("\n" + "="*60)
        logger.info("✅ Extraction Complete!")
        logger.info("="*60)
        logger.info(f"   Total frames:    {total_frames}")
        logger.info(f"   Processed:       {processed_frames_count}")
        logger.info(f"   OCR ROI calls:   {ocr_total_calls}")
        for box_name, count in subtitles_count.items():
            logger.info(f"   - Box '{box_name}': {count} subtitles -> {output_paths[box_name]}")
        logger.info(f"   Processing time: {processing_time:.2f}s")
        logger.info("="*60)
        
        return ExtractionResult(
            video_path=video_path,
            total_frames=total_frames,
            processed_frames=processed_frames_count,
            ocr_frames=ocr_total_calls, # Đổi thành số lượng calls cho các ROI
            subtitles_count=subtitles_count,
            subtitles=subtitles_dict,
            output_paths=output_paths,
            metadata={
                "ocr_model": self.ocr_model_name,
                "device": self.device,
                "output_format": self.output_format,
                "frame_interval": self.frame_interval
            },
            processing_time=processing_time
        )
        
    def extract_from_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        extensions: List[str] = None
    ) -> List[ExtractionResult]:
        """Trích xuất tất cả video trong thư mục"""
        if extensions is None:
            extensions = [".mp4", ".avi", ".mkv", ".mov", ".wmv"]
            
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
            
        output_path = Path(output_dir) if output_dir else input_path
        output_path.mkdir(parents=True, exist_ok=True)
        
        video_files = []
        for ext in extensions:
            video_files.extend(input_path.glob(f"*{ext}"))
            video_files.extend(input_path.glob(f"*{ext.upper()}"))
            
        if not video_files:
            logger.warning(f"No video files found in {input_dir}")
            return []
            
        results = []
        for i, video_file in enumerate(video_files, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {i}/{len(video_files)}: {video_file.name}")
            logger.info(f"{'='*60}")
            try:
                result = self.extract(str(video_file), str(output_path))
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {video_file}: {e}")
                
        return results

def extract_chinese_subtitles(
    video_path: str,
    boxes: List[OcrBox],
    output_dir: Optional[str] = None,
    **kwargs
) -> ExtractionResult:
    """Convenience function"""
    extractor = VideoSubtitleExtractor(boxes=boxes, **kwargs)
    return extractor.extract(video_path, output_dir)
