#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/srt_parser.py — SRT Parser Module

Parse subtitle files (.srt) thành list of segments.
Được sử dụng chung bởi nhiều module trong project.

Ví dụ:
    from utils.srt_parser import parse_srt, parse_srt_file
    
    # Parse từ string
    segments = parse_srt(srt_content)
    
    # Parse từ file
    segments = parse_srt_file("video.srt")
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
import textwrap


def is_cjk(text: str) -> bool:
    """Kiểm tra xem chuỗi có chứa ký tự CJK (Chinese, Japanese, Korean) hay không."""
    return bool(re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))


def wrap_subtitle_text(text: str, max_chars: int) -> str:
    """
    Ngắt dòng đoạn text dựa trên số lượng ký tự tối đa.
    Hỗ trợ cả CJK (ngắt theo ký tự) và Alphabet (ngắt theo từ).
    
    Args:
        text: Nội dung subtitle
        max_chars: Số lượng ký tự tối đa trên mỗi dòng (0 = không ngắt)
        
    Returns:
        Chuỗi đã được ngắt dòng
    """
    if max_chars <= 0 or not text:
        return text
        
    text = text.replace('\n', ' ').strip()
    text = re.sub(r'\s+', ' ', text)
    
    if is_cjk(text):
        lines = []
        current_line = ""
        punctuations = set("。，、！？：；）】》”’.,!?:;)]}")
        
        for char in text:
            if len(current_line) >= max_chars:
                if char in punctuations or char == ' ':
                    current_line += char
                else:
                    lines.append(current_line.strip())
                    current_line = char
            else:
                current_line += char
                
        if current_line:
            lines.append(current_line.strip())
            
        return "\n".join(lines)
    else:
        lines = textwrap.wrap(text, width=max_chars, break_long_words=False)
        return "\n".join(lines)


def ts_to_ms(ts: str) -> int:
    """
    Chuyển đổi timestamp SRT sang milliseconds.
    
    Args:
        ts: Timestamp string (format: "HH:MM:SS,mmm" hoặc "HH:MM:SS.mmm")
    
    Returns:
        Số milliseconds từ đầu video
    
    Example:
        >>> ts_to_ms("00:01:24,233")
        84233
        >>> ts_to_ms("01:30:45.500")
        5445500
    """
    ts = ts.strip().replace(',', '.')
    parts = ts.split(':')
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return int((h * 3600 + m * 60 + s) * 1000)


def parse_srt(content: str, skip_empty_text: bool = True) -> List[Dict]:
    """
    Parse SRT content thành list of segments.
    
    Args:
        content: Nội dung file .srt (string)
        skip_empty_text: Nếu True, bỏ qua các segment có text rỗng
    
    Returns:
        List of dict với các keys:
        - line: Số thứ tự trong file SRT
        - start_time: Thời gian bắt đầu (milliseconds)
        - end_time: Thời gian kết thúc (milliseconds)
        - startraw: Timestamp bắt đầu gốc (string)
        - endraw: Timestamp kết thúc gốc (string)
        - time: Dòng timestamp đầy đủ (string)
        - text: Nội dung subtitle (đã loại HTML tags)
    
    Example:
        >>> srt_content = '''1
        ... 00:01:24,233 --> 00:01:27,566
        ... Hello world'''
        >>> segments = parse_srt(srt_content)
        >>> segments[0]['start_time']
        84233
        >>> segments[0]['text']
        'Hello world'
    """
    blocks = re.split(r'\n\s*\n', content.strip())
    result = []
    
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:  # Cần ít nhất index và timestamp
            continue
        
        # Parse index
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue
        
        # Parse timestamp
        if '-->' not in lines[1]:
            continue
        
        t_parts = lines[1].split('-->')
        try:
            start_ms = ts_to_ms(t_parts[0])
            end_ms = ts_to_ms(t_parts[1])
        except (ValueError, IndexError):
            continue
        
        # Parse text (có thể có nhiều dòng)
        text = "\n".join(lines[2:]).strip() if len(lines) > 2 else ""
        
        # Loại bỏ HTML tags
        text = re.sub(r'</?[a-zA-Z]+>', '', text, flags=re.I)
        text = re.sub(r'\n{2,}', '\n', text).strip()
        
        # Validate
        if end_ms <= start_ms:
            continue
        if skip_empty_text and not text:
            continue
        
        result.append({
            "line": index,
            "start_time": start_ms,
            "end_time": end_ms,
            "startraw": t_parts[0].strip(),
            "endraw": t_parts[1].strip(),
            "time": lines[1].strip(),
            "text": text,
        })
    
    return result


def parse_srt_file(file_path: str, encoding: str = 'utf-8') -> List[Dict]:
    """
    Parse SRT file từ đường dẫn.
    
    Args:
        file_path: Đường dẫn tới file .srt
        encoding: Encoding của file (mặc định: utf-8)
    
    Returns:
        List of segments (xem parse_srt)
    
    Raises:
        FileNotFoundError: File không tồn tại
        UnicodeDecodeError: Lỗi encoding
    
    Example:
        >>> segments = parse_srt_file("video.srt")
        >>> print(f"Có {len(segments)} segments")
        Có 150 segments
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File không tồn tại: {file_path}")
    
    with open(path, 'r', encoding=encoding) as f:
        content = f.read()
    
    return parse_srt(content)


def segments_to_srt(segments: List[Dict], include_text: bool = True) -> str:
    """
    Chuyển list of segments thành SRT format string.
    
    Args:
        segments: List of segment dicts (từ parse_srt)
        include_text: Có bao gồm text trong output không
    
    Returns:
        SRT format string
    
    Example:
        >>> segments = [{'line': 1, 'start_time': 0, 'end_time': 1000, 'text': 'Hello'}]
        >>> print(segments_to_srt(segments))
        1
        00:00:00,000 --> 00:00:01,000
        Hello
    """
    def ms_to_ts(ms: int) -> str:
        """Chuyển milliseconds sang timestamp SRT format."""
        h = ms // 3600000
        m = (ms % 3600000) // 60000
        s = (ms % 60000) // 1000
        ms_rem = ms % 1000
        return f"{h:02d}:{m:02d}:{s:02d},{ms_rem:03d}"
    
    lines = []
    for seg in segments:
        lines.append(str(seg['line']))
        start_ts = ms_to_ts(seg['start_time'])
        end_ts = ms_to_ts(seg['end_time'])
        lines.append(f"{start_ts} --> {end_ts}")
        if include_text and seg.get('text'):
            lines.append(seg['text'])
        lines.append("")  # Empty line between blocks
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test với sample SRT
    sample = """1
00:01:24,233 --> 00:01:27,566
Hello world

2
00:02:30,000 --> 00:02:35,500
This is a test
"""
    segments = parse_srt(sample)
    print(f"Parsed {len(segments)} segments:")
    for seg in segments:
        print(f"  Line {seg['line']}: {seg['start_time']}ms - {seg['end_time']}ms: {seg['text']}")