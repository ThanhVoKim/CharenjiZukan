# -*- coding: utf-8 -*-
"""
Video Subtitle Extractor - Main class

Trích xuất subtitle tiếng Trung từ video sử dụng DeepSeek-OCR-2

Workflow:
1. Frame Sampling - Lấy mỗi N frame
2. Scene Detection - Chỉ xử lý khi có chuyển cảnh
3. ROI Cropping - Chỉ OCR vùng subtitle
4. DeepSeek-OCR-2 - Nhận dạng text
5. Chinese Filter - Lọc chỉ tiếng Trung
6. Deduplication - Loại bỏ trùng lặp
7. Output SRT - Xuất file subtitle
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import time

# Thêm project root vào path để import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

from .frame_processor import FrameProcessor, FrameInfo
from .chinese_filter import ChineseFilter
from .subtitle_writer import SubtitleWriter, SubtitleEntry

logger = get_logger(__name__)


@dataclass
class ExtractionResult:
    """Kết quả extraction"""
    video_path: str
    total_frames: int
    processed_frames: int
    ocr_frames: int
    subtitles_count: int
    subtitles: List[SubtitleEntry]
    output_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


class VideoSubtitleExtractor:
    """
    Trích xuất subtitle từ video sử dụng DeepSeek-OCR-2
    
    Features:
    - Frame Sampling: Giảm số frame cần xử lý
    - Scene Detection: Chỉ OCR khi có thay đổi
    - ROI Cropping: Tối ưu vùng OCR
    - Chinese Filter: Chỉ giữ tiếng Trung
    - Deduplication: Loại bỏ text trùng
    
    Example:
        >>> extractor = VideoSubtitleExtractor(
        ...     frame_interval=30,
        ...     roi_y_start=0.85,
        ...     scene_threshold=30.0
        ... )
        >>> result = extractor.extract("video.mp4", "output.srt")
        >>> print(f"Extracted {result.subtitles_count} subtitles")
    """
    
    def __init__(
        self,
        # Frame processing
        frame_interval: int = 30,
        roi_y_start: float = 0.85,
        roi_y_end: float = 1.0,
        scene_threshold: float = 30.0,
        min_scene_frames: int = 10,
        
        # Chinese filter
        keep_punctuation: bool = True,
        min_char_count: int = 2,
        keep_numbers: bool = False,
        
        # OCR settings
        ocr_model: str = "deepseek-ocr-2",
        device: str = "cuda",
        batch_size: int = 8,
        
        # Output settings
        output_format: str = "srt",
        default_subtitle_duration: float = 3.0,
    ):
        """
        Khởi tạo VideoSubtitleExtractor
        
        Args:
            frame_interval: Số frame bỏ qua giữa mỗi lần xử lý (default: 30)
            roi_y_start: Vị trí bắt đầu ROI (0-1, default: 0.85)
            roi_y_end: Vị trí kết thúc ROI (0-1, default: 1.0)
            scene_threshold: Ngưỡng phát hiện chuyển cảnh (default: 30.0)
            min_scene_frames: Số frame tối thiểu giữa 2 scene (default: 10)
            keep_punctuation: Giữ dấu câu tiếng Trung (default: True)
            min_char_count: Số ký tự tối thiểu (default: 2)
            keep_numbers: Giữ số tiếng Trung (default: False)
            ocr_model: Tên model OCR (default: "deepseek-ocr-2")
            device: Thiết bị chạy OCR - "cuda" hoặc "cpu" (default: "cuda")
            batch_size: Batch size cho OCR (default: 8)
            output_format: Format output - "srt" hoặc "txt" (default: "srt")
            default_subtitle_duration: Thời lượng mặc định (seconds, default: 3.0)
        """
        # Initialize components
        self.frame_processor = FrameProcessor(
            frame_interval=frame_interval,
            roi_y_start=roi_y_start,
            roi_y_end=roi_y_end,
            scene_threshold=scene_threshold,
            min_scene_frames=min_scene_frames
        )
        
        self.chinese_filter = ChineseFilter(
            keep_punctuation=keep_punctuation,
            min_char_count=min_char_count,
            keep_numbers=keep_numbers
        )
        
        self.subtitle_writer = SubtitleWriter(
            default_duration=default_subtitle_duration
        )
        
        # Settings
        self.ocr_model_name = ocr_model
        self.device = device
        self.batch_size = batch_size
        self.output_format = output_format
        self.default_subtitle_duration = default_subtitle_duration
        
        # OCR model (lazy load)
        self._ocr_model = None
        self._model_loaded = False
        
        logger.info(f"VideoSubtitleExtractor initialized")
        logger.info(f"  OCR Model: {ocr_model}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Frame interval: {frame_interval}")
        logger.info(f"  ROI: {roi_y_start}-{roi_y_end}")
        logger.info(f"  Scene threshold: {scene_threshold}")
    
    def load_ocr_model(self):
        """
        Load DeepSeek-OCR-2 model
        
        Note: Cài đặt DeepSeek-OCR-2 trước khi sử dụng:
        pip install deepseek-ocr
        hoặc theo hướng dẫn từ GitHub
        """
        if self._model_loaded:
            return
        
        logger.info(f"🔄 Loading {self.ocr_model_name} on {self.device}...")
        
        try:
            # TODO: Implement actual model loading khi DeepSeek-OCR-2 available
            # Example implementation:
            # from deepseek_ocr import DeepSeekOCR
            # self._ocr_model = DeepSeekOCR(
            #     model_name=self.ocr_model_name,
            #     device=self.device
            # )
            
            # Placeholder - sử dụng mock cho testing
            logger.warning("⚠️ DeepSeek-OCR-2 model not implemented yet!")
            logger.warning("   Using mock OCR for testing purposes.")
            self._ocr_model = MockOCR()
            
            self._model_loaded = True
            logger.info(f"✅ Model loaded successfully")
            
        except ImportError as e:
            logger.error(f"❌ Cannot import DeepSeek-OCR-2: {e}")
            logger.error("   Please install: pip install deepseek-ocr")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def ocr_image(self, image) -> str:
        """
        OCR một image sử dụng DeepSeek-OCR-2
        
        Args:
            image: Image numpy array (BGR format)
            
        Returns:
            Text được nhận dạng
        """
        if not self._model_loaded:
            self.load_ocr_model()
        
        try:
            # TODO: Implement actual OCR call
            # text = self._ocr_model.recognize(image)
            text = self._ocr_model.recognize(image)
            return text
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
    
    def ocr_batch(self, images: List) -> List[str]:
        """
        OCR batch of images
        
        Args:
            images: List of image numpy arrays
            
        Returns:
            List of recognized texts
        """
        if not self._model_loaded:
            self.load_ocr_model()
        
        results = []
        for i, image in enumerate(images):
            text = self.ocr_image(image)
            results.append(text)
            
            if (i + 1) % 10 == 0:
                logger.debug(f"OCR progress: {i+1}/{len(images)}")
        
        return results
    
    def extract(
        self, 
        video_path: str, 
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, int], None]] = None
    ) -> ExtractionResult:
        """
        Trích xuất subtitle từ video
        
        Args:
            video_path: Đường dẫn file video
            output_path: Đường dẫn file output (tự động tạo nếu None)
            progress_callback: Callback cho progress updates
                              callback(current_frame, total_frames, ocr_done)
            
        Returns:
            ExtractionResult với thông tin kết quả
        """
        start_time = time.time()
        
        # Validate input
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Generate output path
        if output_path is None:
            base_name = os.path.splitext(video_path)[0]
            ext = ".srt" if self.output_format == "srt" else ".txt"
            output_path = f"{base_name}_chinese{ext}"
        
        logger.info("="*60)
        logger.info("🎬 Video Subtitle Extraction")
        logger.info("="*60)
        logger.info(f"Input:  {video_path}")
        logger.info(f"Output: {output_path}")
        logger.info("="*60)
        
        # Step 1: Extract frames
        logger.info("\n📹 Step 1: Extracting frames...")
        self.frame_processor.reset()
        
        def frame_progress(current, total, ocr_count):
            if progress_callback:
                progress_callback(current, total, ocr_count)
        
        frames, frame_metadata = self.frame_processor.extract_frames(
            video_path, 
            progress_callback=frame_progress
        )
        
        if not frames:
            logger.warning("No frames to process!")
            return ExtractionResult(
                video_path=video_path,
                total_frames=frame_metadata.get("total_frames", 0),
                processed_frames=0,
                ocr_frames=0,
                subtitles_count=0,
                subtitles=[],
                output_path=output_path,
                metadata=frame_metadata,
                processing_time=time.time() - start_time
            )
        
        # Step 2: OCR
        logger.info("\n🔍 Step 2: Running OCR...")
        self.load_ocr_model()
        
        subtitle_entries: List[SubtitleEntry] = []
        ocr_count = 0
        
        for i, frame_info in enumerate(frames):
            # OCR on ROI
            text = self.ocr_image(frame_info.roi_data)
            ocr_count += 1
            
            # Filter Chinese only
            chinese_text = self.chinese_filter.filter_text(text)
            
            if chinese_text:
                # Calculate end time (next frame or default duration)
                if i + 1 < len(frames):
                    end_time = frames[i + 1].timestamp
                else:
                    end_time = frame_info.timestamp + self.default_subtitle_duration
                
                entry = SubtitleEntry(
                    index=len(subtitle_entries) + 1,
                    start_time=frame_info.timestamp,
                    end_time=end_time,
                    text=chinese_text
                )
                subtitle_entries.append(entry)
            
            # Progress
            if (i + 1) % 10 == 0 or (i + 1) == len(frames):
                logger.info(f"   OCR progress: {i+1}/{len(frames)} frames, "
                           f"{len(subtitle_entries)} subtitles found")
                
                if progress_callback:
                    progress_callback(
                        frame_metadata.get("total_frames", 0),
                        frame_metadata.get("total_frames", 0),
                        i + 1
                    )
        
        # Step 3: Write output
        logger.info(f"\n📝 Step 3: Writing output...")
        if self.output_format == "srt":
            self.subtitle_writer.write_srt(subtitle_entries, output_path)
        else:
            self.subtitle_writer.write_txt(subtitle_entries, output_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Result
        result = ExtractionResult(
            video_path=video_path,
            total_frames=frame_metadata.get("total_frames", 0),
            processed_frames=len(frames),
            ocr_frames=ocr_count,
            subtitles_count=len(subtitle_entries),
            subtitles=subtitle_entries,
            output_path=output_path,
            metadata={
                **frame_metadata,
                "ocr_model": self.ocr_model_name,
                "device": self.device,
                "output_format": self.output_format,
            },
            processing_time=processing_time
        )
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("✅ Extraction Complete!")
        logger.info("="*60)
        logger.info(f"   Total frames:    {result.total_frames}")
        logger.info(f"   Processed:       {result.processed_frames}")
        logger.info(f"   OCR frames:      {result.ocr_frames}")
        logger.info(f"   Subtitles:       {result.subtitles_count}")
        logger.info(f"   Processing time: {processing_time:.2f}s")
        logger.info(f"   Output:          {output_path}")
        logger.info("="*60)
        
        return result
    
    def extract_from_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        extensions: List[str] = None
    ) -> List[ExtractionResult]:
        """
        Trích xuất subtitle từ tất cả video trong thư mục
        
        Args:
            input_dir: Thư mục chứa video
            output_dir: Thư mục output (mặc định = input_dir)
            extensions: List extension cần xử lý (default: [".mp4", ".avi", ".mkv"])
            
        Returns:
            List các ExtractionResult
        """
        if extensions is None:
            extensions = [".mp4", ".avi", ".mkv", ".mov", ".wmv"]
        
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        
        output_path = Path(output_dir) if output_dir else input_path
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all videos
        video_files = []
        for ext in extensions:
            video_files.extend(input_path.glob(f"*{ext}"))
            video_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not video_files:
            logger.warning(f"No video files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(video_files)} videos to process")
        
        results = []
        for i, video_file in enumerate(video_files, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {i}/{len(video_files)}: {video_file.name}")
            logger.info(f"{'='*60}")
            
            output_file = output_path / f"{video_file.stem}_chinese.srt"
            
            try:
                result = self.extract(str(video_file), str(output_file))
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {video_file}: {e}")
                continue
        
        return results


class MockOCR:
    """
    Mock OCR class cho testing khi DeepSeek-OCR-2 chưa available
    """
    
    def recognize(self, image) -> str:
        """
        Mock recognize - trả về text giả để test
        """
        # Trong thực tế, đây sẽ gọi DeepSeek-OCR-2
        return ""


# Convenience function
def extract_chinese_subtitles(
    video_path: str,
    output_path: Optional[str] = None,
    **kwargs
) -> ExtractionResult:
    """
    Convenience function để extract Chinese subtitles
    
    Args:
        video_path: Đường dẫn file video
        output_path: Đường dẫn file output
        **kwargs: Additional arguments cho VideoSubtitleExtractor
        
    Returns:
        ExtractionResult
    """
    extractor = VideoSubtitleExtractor(**kwargs)
    return extractor.extract(video_path, output_path)


# Test
if __name__ == "__main__":
    from utils.logger import setup_logging
    setup_logging(level=10)  # DEBUG
    
    # Test với mock
    extractor = VideoSubtitleExtractor(
        frame_interval=30,
        roi_y_start=0.85,
        scene_threshold=30.0,
        device="cpu"  # Use CPU for testing
    )
    
    print("\nVideoSubtitleExtractor initialized successfully!")
    print(f"OCR Model: {extractor.ocr_model_name}")
    print(f"Device: {extractor.device}")