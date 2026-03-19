# -*- coding: utf-8 -*-
"""
Chinese Filter - Lọc và xử lý text tiếng Trung

Loại bỏ các ký tự không phải tiếng Trung, chỉ giữ lại:
- Chinese characters (CJK Unified Ideographs)
- Chinese punctuation (tùy chọn)
"""

import re
from typing import List, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Thêm project root vào path để import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChineseText:
    """Kết quả text tiếng Trung sau khi lọc"""
    original: str       # Text gốc
    chinese_only: str   # Chỉ tiếng Trung
    char_count: int     # Số ký tự tiếng Trung
    has_chinese: bool   # Có tiếng Trung hay không
    removed_chars: int  # Số ký tự đã loại bỏ


class ChineseFilter:
    """
    Lọc chỉ giữ lại text tiếng Trung
    
    Unicode ranges cho Chinese:
    - \\u4e00-\\u9fff: CJK Unified Ideographs (phổ biến nhất)
    - \\u3400-\\u4dbf: CJK Extension A
    - \\U00020000-\\U0002a6df: CJK Extension B
    - \\u3000-\\u303f: CJK Symbols and Punctuation
    - \\uff00-\\uffef: Halfwidth and Fullwidth Forms
    
    Attributes:
        keep_punctuation: Có giữ dấu câu tiếng Trung không (default: True)
        min_char_count: Số ký tự tối thiểu để text hợp lệ (default: 2)
    """
    
    # Regex pattern cho Chinese characters (CJK Unified Ideographs)
    CHINESE_CHAR_PATTERN = re.compile(
        r'[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df]+'
    )
    
    # Pattern cho Chinese punctuation
    CHINESE_PUNCTUATION_PATTERN = re.compile(
        r'[\u3000-\u303f\uff00-\uffef，。！？、；：""''（）【】《》…—]'
    )
    
    def __init__(
        self,
        keep_punctuation: bool = True,
        min_char_count: int = 2
    ):
        self.keep_punctuation = keep_punctuation
        self.min_char_count = min_char_count
        
        logger.info(
            f"ChineseFilter initialized: punctuation={keep_punctuation}, "
            f"min_chars={min_char_count}"
        )
    
    def extract_chinese(self, text: str) -> ChineseText:
        """
        Trích xuất chỉ text tiếng Trung từ chuỗi hỗn hợp
        
        Args:
            text: Text đầu vào (có thể chứa tiếng Anh, số, v.v.)
            
        Returns:
            ChineseText với thông tin chi tiết
        """
        if not text or not text.strip():
            return ChineseText(
                original=text or "",
                chinese_only="",
                char_count=0,
                has_chinese=False,
                removed_chars=0
            )
        
        original_len = len(text)
        
        # Tìm tất cả Chinese characters
        chinese_matches = self.CHINESE_CHAR_PATTERN.findall(text)
        chinese_only = ''.join(chinese_matches)
        
        # Thêm punctuation nếu cần
        if self.keep_punctuation and chinese_only:
            punct_matches = self.CHINESE_PUNCTUATION_PATTERN.findall(text)
            chinese_only = chinese_only + ''.join(punct_matches)
        
        char_count = len(chinese_only)
        removed_chars = original_len - char_count
        
        return ChineseText(
            original=text,
            chinese_only=chinese_only,
            char_count=char_count,
            has_chinese=char_count > 0,
            removed_chars=removed_chars
        )
    
    def filter_text(self, text: str) -> Optional[str]:
        """
        Lọc text, trả về None nếu không đủ Chinese
        
        Args:
            text: Text đầu vào
            
        Returns:
            Text tiếng Trung hoặc None nếu không hợp lệ
        """
        result = self.extract_chinese(text)
        
        # Kiểm tra số ký tự tối thiểu
        if result.char_count < self.min_char_count:
            return None
        
        return result.chinese_only if result.has_chinese else None
    
    def filter_batch(self, texts: List[str]) -> List[str]:
        """
        Lọc batch of texts
        
        Args:
            texts: List các text đầu vào
            
        Returns:
            List các text tiếng Trung hợp lệ
        """
        filtered = []
        for text in texts:
            chinese = self.filter_text(text)
            if chinese:
                filtered.append(chinese)
        
        logger.info(f"Filtered {len(texts)} texts -> {len(filtered)} Chinese texts")
        return filtered
    
    def is_chinese_char(self, char: str) -> bool:
        """
        Kiểm tra một ký tự có phải tiếng Trung không
        
        Args:
            char: Ký tự cần kiểm tra
            
        Returns:
            True nếu là Chinese character
        """
        if len(char) != 1:
            return False
        
        code = ord(char)
        
        # CJK Unified Ideographs
        if 0x4e00 <= code <= 0x9fff:
            return True
        # CJK Extension A
        if 0x3400 <= code <= 0x4dbf:
            return True
        # CJK Extension B and beyond
        if 0x20000 <= code <= 0x2a6df:
            return True
        
        return False
    
    def get_chinese_ratio(self, text: str) -> float:
        """
        Tính tỷ lệ ký tự tiếng Trung trong text
        
        Args:
            text: Text cần tính
            
        Returns:
            Tỷ lệ từ 0.0 đến 1.0
        """
        if not text:
            return 0.0
        
        chinese_count = sum(1 for char in text if self.is_chinese_char(char))
        return chinese_count / len(text)
    
    def filter_by_ratio(
        self, 
        text: str, 
        min_ratio: float = 0.5
    ) -> Optional[str]:
        """
        Lọc text dựa trên tỷ lệ tiếng Trung
        
        Args:
            text: Text đầu vào
            min_ratio: Tỷ lệ tối thiểu (0.0 - 1.0)
            
        Returns:
            Text nếu tỷ lệ >= min_ratio, None nếu không
        """
        ratio = self.get_chinese_ratio(text)
        
        if ratio >= min_ratio:
            return self.filter_text(text)
        
        return None
    
    def clean_text(self, text: str) -> str:
        """
        Làm sạch text, chỉ giữ Chinese chars và punctuation
        
        Args:
            text: Text đầu vào
            
        Returns:
            Text đã làm sạch
        """
        result = self.extract_chinese(text)
        return result.chinese_only
    
    def extract_with_context(
        self, 
        text: str,
        context_chars: int = 5
    ) -> List[dict]:
        """
        Trích xuất Chinese text với context xung quanh
        
        Hữu ích khi cần debug hoặc xem text gốc xung quanh Chinese text
        
        Args:
            text: Text đầu vào
            context_chars: Số ký tự context mỗi bên
            
        Returns:
            List các dict với chinese_text và context
        """
        results = []
        
        for match in self.CHINESE_CHAR_PATTERN.finditer(text):
            start = match.start()
            end = match.end()
            
            # Lấy context
            context_start = max(0, start - context_chars)
            context_end = min(len(text), end + context_chars)
            
            results.append({
                "chinese_text": match.group(),
                "start": start,
                "end": end,
                "context": text[context_start:context_end]
            })
        
        return results


# Test function
if __name__ == "__main__":
    # Setup logging
    from utils.logger import setup_logging
    setup_logging(level=10)  # DEBUG
    
    filter = ChineseFilter(keep_punctuation=True, min_char_count=2)
    
    # Test cases
    test_texts = [
        "Hello 你好 World 世界",
        "这是一段中文文字",
        "English only text",
        "混合 text 中文 and English 123",
        "短",  # Quá ngắn
        "你好！今天天气很好。",
        "第1集 Episode One 第一集",
        "123456789",
        "你好，世界！Hello World!",
    ]
    
    print("\n" + "="*60)
    print("Chinese Filter Test")
    print("="*60)
    
    for text in test_texts:
        result = filter.extract_chinese(text)
        filtered = filter.filter_text(text)
        ratio = filter.get_chinese_ratio(text)
        
        print(f"\nOriginal: {text}")
        print(f"Chinese:  {result.chinese_only}")
        print(f"Count:    {result.char_count}")
        print(f"Ratio:    {ratio:.2%}")
        print(f"Filtered: {filtered}")
        print("-"*40)