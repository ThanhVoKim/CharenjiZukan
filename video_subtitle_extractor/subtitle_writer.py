# -*- coding: utf-8 -*-
"""
Subtitle Writer - Xuất subtitle ra file SRT/TXT

Hỗ trợ:
- Format SRT chuẩn
- Format TXT đơn giản
- Deduplication (loại bỏ trùng lặp)
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import timedelta
from pathlib import Path
import sys

# Thêm project root vào path để import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SubtitleEntry:
    """Một entry trong file subtitle"""
    index: int
    start_time: float  # seconds
    end_time: float    # seconds
    text: str
    
    def __post_init__(self):
        """Validate sau khi init"""
        if self.start_time < 0:
            self.start_time = 0
        if self.end_time < self.start_time:
            self.end_time = self.start_time + 1.0


class SubtitleWriter:
    """
    Ghi subtitle ra file SRT hoặc TXT
    
    Format SRT chuẩn:
    ```
    1
    00:00:01,000 --> 00:00:04,000
    这是第一句字幕
    
    2
    00:00:05,000 --> 00:00:08,000
    这是第二句字幕
    ```
    
    Attributes:
        min_duration: Thời lượng tối thiểu cho một subtitle (seconds)
        max_duration: Thời lượng tối đa cho một subtitle (seconds)
        default_duration: Thời lượng mặc định nếu không có end_time (seconds)
    """
    
    def __init__(
        self,
        min_duration: float = 1.0,
        max_duration: float = 7.0,
        default_duration: float = 3.0
    ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.default_duration = default_duration
        
        logger.info(f"SubtitleWriter initialized: "
                   f"duration={min_duration}-{max_duration}s")
    
    def format_timestamp(self, seconds: float) -> str:
        """
        Format timestamp sang định dạng SRT: HH:MM:SS,mmm
        
        Args:
            seconds: Thời gian tính bằng giây
            
        Returns:
            String format HH:MM:SS,mmm
        """
        if seconds < 0:
            seconds = 0
        
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int((seconds - int(seconds)) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def parse_timestamp(self, timestamp: str) -> float:
        """
        Parse timestamp từ string SRT format
        
        Args:
            timestamp: String format HH:MM:SS,mmm
            
        Returns:
            Thời gian tính bằng giây
        """
        try:
            # Format: HH:MM:SS,mmm
            parts = timestamp.strip().replace(',', '.').split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            
            return hours * 3600 + minutes * 60 + seconds
        except Exception as e:
            logger.error(f"Cannot parse timestamp '{timestamp}': {e}")
            return 0.0
    
    def deduplicate(self, entries: List[SubtitleEntry], similarity_threshold: float = 0.85) -> List[SubtitleEntry]:
        """
        Loại bỏ các entry trùng lặp liên tiếp
        
        Khi text trùng lặp hoặc tương đồng cao (để tránh nhiễu OCR), 
        sẽ merge thời gian của các entry liên tiếp.
        
        Args:
            entries: List các subtitle entry
            similarity_threshold: Ngưỡng tương đồng để gộp (mặc định 0.85)
            
        Returns:
            List đã loại bỏ trùng lặp
        """
        if not entries:
            return []
            
        import difflib
        import re
        
        deduped = [SubtitleEntry(
            index=1,
            start_time=entries[0].start_time,
            end_time=entries[0].end_time,
            text=entries[0].text
        )]
        
        for entry in entries[1:]:
            prev_text = deduped[-1].text
            curr_text = entry.text
            
            # Kiểm tra text giống nhau hoàn toàn
            is_same = (curr_text == prev_text)
            
            # Nếu không giống hoàn toàn, kiểm tra mức độ tương đồng (bỏ qua khoảng trắng và dấu câu)
            if not is_same and similarity_threshold < 1.0:
                clean_prev = re.sub(r'[^\w\s]', '', prev_text).replace(' ', '')
                clean_curr = re.sub(r'[^\w\s]', '', curr_text).replace(' ', '')
                
                # Chỉ kiểm tra nếu cả 2 chuỗi đều có ký tự chữ/số sau khi làm sạch
                if clean_prev and clean_curr:
                    ratio = difflib.SequenceMatcher(None, clean_prev, clean_curr).ratio()
                    if ratio >= similarity_threshold:
                        is_same = True
            
            if is_same:
                # Mở rộng thời gian của entry trước
                deduped[-1].end_time = entry.end_time
                # Nếu text mới dài hơn text cũ (nhiều thông tin hơn), ta có thể thay thế
                if len(curr_text) > len(prev_text):
                    deduped[-1].text = curr_text
            else:
                # Tạo entry mới
                deduped.append(SubtitleEntry(
                    index=len(deduped) + 1,
                    start_time=entry.start_time,
                    end_time=entry.end_time,
                    text=entry.text
                ))
        
        # Validate duration và chống overlapping (sửa lỗi timestamps lồng nhau)
        for i in range(len(deduped)):
            # Lấy start_time của entry tiếp theo làm giới hạn, nếu là entry cuối thì không giới hạn
            max_end_time = deduped[i+1].start_time if i < len(deduped) - 1 else float('inf')
            
            duration = deduped[i].end_time - deduped[i].start_time
            
            # Nếu duration quá ngắn, cố gắng kéo dài tới min_duration nhưng không vượt quá next entry
            if duration < self.min_duration:
                proposed_end = deduped[i].start_time + self.min_duration
                deduped[i].end_time = min(proposed_end, max_end_time)
            elif duration > self.max_duration:
                # Cắt bớt nếu quá dài
                deduped[i].end_time = deduped[i].start_time + self.max_duration
                
            # Đảm bảo không bao giờ overlap với entry tiếp theo
            if deduped[i].end_time > max_end_time:
                deduped[i].end_time = max_end_time
                
        logger.info(f"Deduplicated: {len(entries)} -> {len(deduped)} entries")
        return deduped
    
    def merge_close_entries(
        self, 
        entries: List[SubtitleEntry],
        max_gap: float = 0.5
    ) -> List[SubtitleEntry]:
        """
        Merge các entry có text giống nhau và khoảng cách thời gian nhỏ
        
        Args:
            entries: List các subtitle entry
            max_gap: Khoảng cách thời gian tối đa để merge (seconds)
            
        Returns:
            List đã merge
        """
        if len(entries) <= 1:
            return entries
        
        merged = [SubtitleEntry(
            index=1,
            start_time=entries[0].start_time,
            end_time=entries[0].end_time,
            text=entries[0].text
        )]
        
        for entry in entries[1:]:
            prev = merged[-1]
            gap = entry.start_time - prev.end_time
            
            if entry.text == prev.text and gap <= max_gap:
                # Merge
                prev.end_time = entry.end_time
            else:
                merged.append(SubtitleEntry(
                    index=len(merged) + 1,
                    start_time=entry.start_time,
                    end_time=entry.end_time,
                    text=entry.text
                ))
        
        return merged
    
    def write_srt(
        self, 
        entries: List[SubtitleEntry], 
        output_path: str,
        deduplicate: bool = True
    ) -> str:
        """
        Ghi file SRT
        
        Args:
            entries: List các subtitle entry
            output_path: Đường dẫn file output
            deduplicate: Có loại bỏ trùng lặp không
            
        Returns:
            Đường dẫn file đã ghi
        """
        if not entries:
            logger.warning("No entries to write")
            # Tạo file rỗng
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("")
            return output_path
        
        # Deduplicate nếu cần
        if deduplicate:
            entries = self.deduplicate(entries)
        
        # Tạo thư mục nếu cần
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, entry in enumerate(entries, 1):
                # Index
                f.write(f"{i}\n")
                
                # Timestamp
                start = self.format_timestamp(entry.start_time)
                end = self.format_timestamp(entry.end_time)
                f.write(f"{start} --> {end}\n")
                
                # Text
                f.write(f"{entry.text}\n")
                
                # Blank line
                f.write("\n")
        
        logger.info(f"📝 Written {len(entries)} subtitles to {output_path}")
        return output_path
    
    def write_txt(
        self, 
        entries: List[SubtitleEntry], 
        output_path: str,
        include_timestamp: bool = True,
        deduplicate: bool = True
    ) -> str:
        """
        Ghi file text đơn giản
        
        Format:
        [00:00:01] 这是第一句字幕
        [00:00:05] 这是第二句字幕
        
        Args:
            entries: List các subtitle entry
            output_path: Đường dẫn file output
            include_timestamp: Có bao gồm timestamp không
            deduplicate: Có loại bỏ trùng lặp không
            
        Returns:
            Đường dẫn file đã ghi
        """
        if not entries:
            logger.warning("No entries to write")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("")
            return output_path
        
        # Deduplicate nếu cần
        if deduplicate:
            entries = self.deduplicate(entries)
        
        # Tạo thư mục nếu cần
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                if include_timestamp:
                    timestamp = self.format_timestamp(entry.start_time)
                    f.write(f"[{timestamp}] {entry.text}\n")
                else:
                    f.write(f"{entry.text}\n")
        
        logger.info(f"📝 Written {len(entries)} lines to {output_path}")
        return output_path
    
    def read_srt(self, srt_path: str) -> List[SubtitleEntry]:
        """
        Đọc file SRT
        
        Args:
            srt_path: Đường dẫn file SRT
            
        Returns:
            List các SubtitleEntry
        """
        entries = []
        
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by blank lines
        blocks = content.strip().split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
            
            try:
                # Index
                index = int(lines[0])
                
                # Timestamp
                timestamp_line = lines[1]
                start_str, end_str = timestamp_line.split(' --> ')
                start_time = self.parse_timestamp(start_str)
                end_time = self.parse_timestamp(end_str)
                
                # Text (có thể nhiều dòng)
                text = '\n'.join(lines[2:])
                
                entries.append(SubtitleEntry(
                    index=index,
                    start_time=start_time,
                    end_time=end_time,
                    text=text
                ))
            except Exception as e:
                logger.warning(f"Cannot parse block: {block[:50]}... Error: {e}")
                continue
        
        logger.info(f"📖 Read {len(entries)} entries from {srt_path}")
        return entries
    
    def get_statistics(self, entries: List[SubtitleEntry]) -> dict:
        """
        Lấy thống kê về các subtitle entries
        
        Args:
            entries: List các subtitle entry
            
        Returns:
            Dict với các thống kê
        """
        if not entries:
            return {
                "total_entries": 0,
                "total_duration": 0,
                "total_chars": 0,
                "avg_duration": 0,
                "avg_chars": 0
            }
        
        total_duration = sum(e.end_time - e.start_time for e in entries)
        total_chars = sum(len(e.text) for e in entries)
        
        return {
            "total_entries": len(entries),
            "total_duration": total_duration,
            "total_chars": total_chars,
            "avg_duration": total_duration / len(entries),
            "avg_chars": total_chars / len(entries),
            "min_duration": min(e.end_time - e.start_time for e in entries),
            "max_duration": max(e.end_time - e.start_time for e in entries),
        }


# Test function
if __name__ == "__main__":
    from utils.logger import setup_logging
    setup_logging(level=10)  # DEBUG
    
    writer = SubtitleWriter()
    
    # Test entries
    entries = [
        SubtitleEntry(1, 0.0, 3.0, "你好世界"),
        SubtitleEntry(2, 3.5, 6.5, "这是测试"),
        SubtitleEntry(3, 7.0, 10.0, "你好世界"),  # Duplicate
        SubtitleEntry(4, 10.5, 13.5, "最后一行"),
    ]
    
    print("\n" + "="*60)
    print("Subtitle Writer Test")
    print("="*60)
    
    # Test timestamp formatting
    print("\nTimestamp formatting:")
    for secs in [0.0, 1.5, 65.123, 3661.999]:
        print(f"  {secs}s -> {writer.format_timestamp(secs)}")
    
    # Test deduplication
    print("\nDeduplication test:")
    deduped = writer.deduplicate(entries)
    for e in deduped:
        print(f"  [{writer.format_timestamp(e.start_time)}] {e.text}")
    
    # Test statistics
    stats = writer.get_statistics(entries)
    print(f"\nStatistics: {stats}")