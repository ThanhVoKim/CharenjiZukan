#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/ass_utils.py — ASS (Advanced SubStation Alpha) Utilities Module

Module xử lý file ASS format cho việc overlay note lên video.
Hỗ trợ chuyển đổi SRT timestamp sang ASS timestamp, wrap text, và tạo dialogue lines.

Ví dụ:
    from utils.ass_utils import (
        srt_timestamp_to_ass,
        wrap_text,
        normalize_newlines,
        create_dialogue_line,
        parse_ass_file,
        write_ass_file,
    )
    
    # Convert timestamp
    ass_ts = srt_timestamp_to_ass("00:01:24,233")  # "0:01:24.23"
    
    # Wrap text
    wrapped = wrap_text("サムのギアリストサムの", max_chars=14)
    
    # Create dialogue line
    dialogue = create_dialogue_line("0:01:24.23", "0:01:27.57", "Hello\\NWorld")
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# TIMESTAMP CONVERSION
# ─────────────────────────────────────────────────────────────

def srt_timestamp_to_ass(timestamp: str) -> str:
    """
    Convert SRT timestamp to ASS timestamp.
    
    Args:
        timestamp: SRT timestamp string (format: "HH:MM:SS,mmm")
    
    Returns:
        ASS timestamp string (format: "H:MM:SS.cc")
    
    Example:
        >>> srt_timestamp_to_ass("00:01:24,233")
        '0:01:24.23'
        >>> srt_timestamp_to_ass("00:00:01,000")
        '0:00:01.00'
        >>> srt_timestamp_to_ass("01:30:45,500")
        '1:30:45.50'
    """
    # Split by ',' to separate time and milliseconds
    parts = timestamp.strip().replace(',', '.').split(':')
    
    if len(parts) != 3:
        raise ValueError(f"Invalid SRT timestamp format: {timestamp}")
    
    hours = parts[0]
    minutes = parts[1]
    seconds_parts = parts[2].split('.')
    
    if len(seconds_parts) != 2:
        raise ValueError(f"Invalid SRT timestamp format: {timestamp}")
    
    seconds = seconds_parts[0]
    milliseconds = seconds_parts[1]
    
    # Convert milliseconds to centiseconds (divide by 10, take first 2 digits)
    centiseconds = str(int(milliseconds) // 10).zfill(2)
    
    # Remove leading zero from hours if single digit
    hours_int = int(hours)
    
    return f"{hours_int}:{minutes}:{seconds}.{centiseconds}"


def ass_timestamp_to_srt(timestamp: str) -> str:
    """
    Convert ASS timestamp to SRT timestamp.
    
    Args:
        timestamp: ASS timestamp string (format: "H:MM:SS.cc" or "H:MM:SS.cc")
    
    Returns:
        SRT timestamp string (format: "HH:MM:SS,mmm")
    
    Example:
        >>> ass_timestamp_to_srt("0:01:24.23")
        '00:01:24,230'
        >>> ass_timestamp_to_srt("1:30:45.50")
        '01:30:45,500'
    """
    parts = timestamp.strip().split(':')
    
    if len(parts) != 3:
        raise ValueError(f"Invalid ASS timestamp format: {timestamp}")
    
    hours = parts[0]
    minutes = parts[1]
    seconds_parts = parts[2].split('.')
    
    if len(seconds_parts) != 2:
        raise ValueError(f"Invalid ASS timestamp format: {timestamp}")
    
    seconds = seconds_parts[0]
    centiseconds = seconds_parts[1]
    
    # Convert centiseconds to milliseconds (multiply by 10)
    milliseconds = str(int(centiseconds) * 10).zfill(3)
    
    # Format hours with leading zero
    hours_formatted = str(int(hours)).zfill(2)
    
    return f"{hours_formatted}:{minutes}:{seconds},{milliseconds}"


# ─────────────────────────────────────────────────────────────
# TEXT PROCESSING
# ─────────────────────────────────────────────────────────────

def normalize_newlines(text: str) -> str:
    """
    Chuẩn hóa xuống dòng trong text cho ASS format.
    
    Args:
        text: Text cần chuẩn hóa
    
    Returns:
        Text với newline được thay bằng \\N (ASS line break)
    
    Example:
        >>> normalize_newlines("Line1\\nLine2")
        'Line1\\NLine2'
        >>> normalize_newlines("Line1\\r\\nLine2")
        'Line1\\NLine2'
        >>> normalize_newlines("Line1\\NLine2")
        'Line1\\NLine2'
    """
    # Replace all newline variants with \N
    text = text.replace('\r\n', '\\N')
    text = text.replace('\n', '\\N')
    text = text.replace('\r', '\\N')
    return text


def wrap_text(text: str, max_chars: int = 14) -> str:
    """
    Ngắt dòng text nếu quá max_chars ký tự.
    
    Args:
        text: Text cần wrap (có thể đã chứa \\N)
        max_chars: Số ký tự tối đa mỗi dòng (mặc định: 14)
    
    Returns:
        Text đã được wrap với \\N
    
    Example:
        >>> wrap_text("サムのギアリストサムの", 14)
        'サムのギアリストサ\\Nムの'
        >>> wrap_text("Line1\\NLine2", 14)
        'Line1\\NLine2'
        >>> wrap_text("Short", 14)
        'Short'
    """
    # Split by existing \N
    lines = text.split('\\N')
    result_lines = []
    
    for line in lines:
        # If line is short enough, keep as is
        if len(line) <= max_chars:
            result_lines.append(line)
        else:
            # Wrap long lines
            while len(line) > max_chars:
                result_lines.append(line[:max_chars])
                line = line[max_chars:]
            if line:  # Add remaining part
                result_lines.append(line)
    
    return '\\N'.join(result_lines)


# ─────────────────────────────────────────────────────────────
# DIALOGUE LINE CREATION
# ─────────────────────────────────────────────────────────────

def create_dialogue_line(
    start: str,
    end: str,
    text: str,
    style: str = "NoteStyle",
    layer: int = 0,
    name: str = "",
    margin_l: int = 0,
    margin_r: int = 0,
    margin_v: int = 0,
    effect: str = "",
) -> str:
    """
    Tạo dòng Dialogue trong ASS format.
    
    Args:
        start: Thời gian bắt đầu (ASS format: "H:MM:SS.cc")
        end: Thời gian kết thúc (ASS format: "H:MM:SS.cc")
        text: Nội dung text (đã được normalize và wrap)
        style: Tên style (mặc định: "NoteStyle")
        layer: Layer number (mặc định: 0)
        name: Actor name (mặc định: "")
        margin_l: Left margin (mặc định: 0)
        margin_r: Right margin (mặc định: 0)
        margin_v: Vertical margin (mặc định: 0)
        effect: Effect name (mặc định: "")
    
    Returns:
        Dòng Dialogue hoàn chỉnh
    
    Example:
        >>> create_dialogue_line("0:01:24.23", "0:01:27.57", "Hello\\NWorld")
        'Dialogue: 0,0:01:24.23,0:01:27.57,NoteStyle,,0,0,0,,Hello\\NWorld'
    """
    # Format: Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
    return f"Dialogue: {layer},{start},{end},{style},{name},{margin_l},{margin_r},{margin_v},{effect},{text}"


def wrap_text_pixel(
    text: str,
    max_width_px: int,
    font,
    measure_fn=None
) -> List[str]:
    """
    Wrap text bằng độ rộng pixel.

    - Latin/ASCII words ưu tiên ngắt tại whitespace.
    - CJK ngắt theo từng ký tự.
    - Token dài hơn max_width_px sẽ hard-split bằng binary search theo pixel.
    - Existing ASS line break ``\\N`` được xử lý như paragraph boundary.
    """
    if max_width_px <= 0:
        raise ValueError("max_width_px must be > 0")

    def default_measure(s: str) -> int:
        if hasattr(font, "getlength"):
            return int(font.getlength(s))
        if hasattr(font, "getbbox"):
            bbox = font.getbbox(s)
            return int(bbox[2] - bbox[0])
        return font.getsize(s)[0]

    measure = measure_fn or default_measure

    def is_cjk_char(char: str) -> bool:
        return (
            "\u4e00" <= char <= "\u9fff"
            or "\u3400" <= char <= "\u4dbf"
            or "\u3040" <= char <= "\u30ff"
            or "\uac00" <= char <= "\ud7af"
            or "\uf900" <= char <= "\ufaff"
        )

    def hard_split(token: str) -> List[str]:
        pieces = []
        rest = token
        while rest:
            low, high = 1, len(rest)
            best = 1
            while low <= high:
                mid = (low + high) // 2
                if measure(rest[:mid]) <= max_width_px:
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1
            pieces.append(rest[:best])
            rest = rest[best:]
        return pieces

    def tokenize_paragraph(paragraph: str) -> List[str]:
        tokens: List[str] = []
        current = ""
        for char in paragraph:
            if char.isspace():
                if current:
                    tokens.append(current)
                    current = ""
                if not tokens or tokens[-1] != " ":
                    tokens.append(" ")
            elif is_cjk_char(char):
                if current:
                    tokens.append(current)
                    current = ""
                tokens.append(char)
            else:
                current += char
        if current:
            tokens.append(current)
        return tokens

    output_lines: List[str] = []
    for paragraph in text.split("\\N"):
        if paragraph == "":
            output_lines.append("")
            continue

        current_line = ""
        for token in tokenize_paragraph(paragraph):
            if token == " " and not current_line:
                continue

            candidate = current_line + token
            if measure(candidate) <= max_width_px:
                current_line = candidate
                continue

            if current_line.strip():
                output_lines.append(current_line.rstrip())
                current_line = ""

            token = token.lstrip()
            if not token:
                continue

            if measure(token) > max_width_px:
                pieces = hard_split(token)
                output_lines.extend(pieces[:-1])
                current_line = pieces[-1]
            else:
                current_line = token

        if current_line or not output_lines:
            output_lines.append(current_line.rstrip())

    return output_lines


def build_ass_drawing_rect(x: int, y: int, w: int, h: int) -> str:
    """Tạo ASS drawing string cho hình chữ nhật."""
    return rf"{{\pos({x},{y})\p1}}m 0 0 l {w} 0 l {w} {h} l 0 {h}{{\p0}}"

def ass_color_from_rgba(r: int, g: int, b: int, a: int = 255) -> str:
    """Chuyển đổi RGBA sang ASS color hex string (&HAABBGGRR)."""
    alpha_ass = 255 - a
    return f"&H{alpha_ass:02X}{b:02X}{g:02X}{r:02X}"


def parse_ass_file(file_path: str) -> Dict:
    """
    Parse file ASS thành dictionary.
    
    Args:
        file_path: Đường dẫn tới file .ass
    
    Returns:
        Dictionary với các keys:
        - script_info: dict của các cặp key-value trong [Script Info]
        - styles: list of style strings
        - events_format: format string của [Events]
        - dialogues: list of dialogue strings (không bao gồm Format line)
    
    Example:
        >>> ass_data = parse_ass_file("assets/sample.ass")
        >>> ass_data['script_info']['ScriptType']
        'v4.00+'
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File không tồn tại: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    result = {
        'script_info': {},
        'styles': [],
        'events_format': None,
        'dialogues': [],
    }
    
    current_section = None
    
    for line in content.splitlines():
        line = line.strip()
        
        # Detect section
        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1].lower()
            continue
        
        # Skip empty lines
        if not line:
            continue
        
        # Parse based on section
        if current_section == 'script info':
            # Parse key: value
            if ':' in line:
                key, value = line.split(':', 1)
                result['script_info'][key.strip()] = value.strip()
        
        elif current_section == 'v4+ styles':
            # Skip Format line, keep Style lines
            if line.startswith('Format:'):
                pass  # Ignore format line for styles
            elif line.startswith('Style:'):
                result['styles'].append(line)
        
        elif current_section == 'events':
            if line.startswith('Format:'):
                result['events_format'] = line
            elif line.startswith('Dialogue:'):
                result['dialogues'].append(line)
    
    logger.debug(f"Parsed ASS file: {file_path}")
    logger.debug(f"  Script Info: {len(result['script_info'])} entries")
    logger.debug(f"  Styles: {len(result['styles'])}")
    logger.debug(f"  Dialogues: {len(result['dialogues'])}")
    
    return result


def write_ass_file(
    output_path: str,
    script_info: Dict,
    styles: List[str],
    events_format: str,
    dialogues: List[str],
) -> None:
    """
    Ghi file ASS từ các components.
    
    Args:
        output_path: Đường dẫn file output
        script_info: Dict các cặp key-value cho [Script Info]
        styles: List các style strings
        events_format: Format string cho [Events]
        dialogues: List các dialogue strings
    
    Example:
        >>> write_ass_file(
        ...     "output.ass",
        ...     {"ScriptType": "v4.00+", "PlayResX": "1920"},
        ...     ["Style: NoteStyle,Noto Sans CJK JP,60,..."],
        ...     "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ...     ["Dialogue: 0,0:01:00.00,0:01:05.00,NoteStyle,,0,0,0,,Hello"]
        ... )
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    
    # [Script Info]
    lines.append("[Script Info]")
    for key, value in script_info.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    
    # [V4+ Styles]
    lines.append("[V4+ Styles]")
    lines.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding")
    for style in styles:
        lines.append(style)
    lines.append("")
    
    # [Events]
    lines.append("[Events]")
    lines.append(events_format)
    for dialogue in dialogues:
        lines.append(dialogue)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Written ASS file: {output_path}")
    logger.info(f"  Dialogues: {len(dialogues)}")


# ─────────────────────────────────────────────────────────────
# HIGH-LEVEL CONVERSION FUNCTIONS
# ─────────────────────────────────────────────────────────────

def extract_layout_key_from_srt_text(
    text: str,
    known_layout_keys: Optional[set],
    fallback_layout_key: str,
    mode: str = "warn"
) -> Tuple[str, str, Optional[str]]:
    """
    Extract layout key from the first SRT text line when the block has body text.

    Returns ``(layout_key, body_text, warning_message)``. A single-line block is
    always treated as body text, even if it equals a known layout key.
    """
    if mode not in {"warn", "strict", "off"}:
        raise ValueError(f"Invalid srt layout key mode: {mode}")

    if not text.strip() or mode == "off":
        return fallback_layout_key, text, None

    raw_lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    if len(raw_lines) <= 1:
        body = text.strip()
        if body.startswith('\\'):
            body = body[1:]
        return fallback_layout_key, body, None

    first_line = raw_lines[0].strip()

    if first_line.startswith('\\'):
        body_lines = [raw_lines[0].replace('\\', '', 1), *raw_lines[1:]]
        return fallback_layout_key, '\n'.join(body_lines).strip(), None

    is_valid_key = False
    if known_layout_keys is None:
        is_valid_key = mode != "strict"
    elif first_line in known_layout_keys:
        is_valid_key = True
    elif fallback_layout_key and first_line == fallback_layout_key:
        is_valid_key = True

    if is_valid_key:
        body = '\n'.join(raw_lines[1:]).strip()
        if not body:
            warning = f"Block specifies layout key '{first_line}' but has empty body."
            return first_line, "", warning
        return first_line, body, None

    if mode == "strict":
        raise ValueError(f"Unknown layout key '{first_line}'.")

    warning = f"Unknown layout key '{first_line}'. Falling back to '{fallback_layout_key}'."
    return fallback_layout_key, text.strip(), warning

def convert_srt_segments_to_ass_dialogues(
    segments: List[Dict],
    max_chars: int = 14,
    style: str = "NoteStyle",
    layout_key: str = "",
    mode: str = "warn",
    known_layout_keys: Optional[set] = None,
) -> List[str]:
    """
    Convert list of SRT segments thành list of ASS dialogue lines.
    
    Args:
        segments: List of segment dicts từ parse_srt_file()
        max_chars: Số ký tự tối đa mỗi dòng (mặc định: 14)
        style: Tên style cho dialogue (mặc định: "NoteStyle")
        layout_key: Fallback layout key (mặc định: "")
        mode: Layout key parsing mode ("warn", "strict", "off")
        known_layout_keys: Set chứa tên các layout hợp lệ
    
    Returns:
        List of ASS dialogue strings
    
    Example:
        >>> from utils.srt_parser import parse_srt_file
        >>> segments = parse_srt_file("video.srt")
        >>> dialogues = convert_srt_segments_to_ass_dialogues(segments)
    """
    dialogues = []
    
    for i, seg in enumerate(segments):
        # Get timestamp from segment (use raw timestamp if available)
        if 'startraw' in seg and 'endraw' in seg:
            start_ass = srt_timestamp_to_ass(seg['startraw'])
            end_ass = srt_timestamp_to_ass(seg['endraw'])
        else:
            # Convert from milliseconds
            start_ms = seg['start_time']
            end_ms = seg['end_time']
            # Convert ms to SRT timestamp first, then to ASS
            h = start_ms // 3600000
            m = (start_ms % 3600000) // 60000
            s = (start_ms % 60000) // 1000
            ms = start_ms % 1000
            start_srt = f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            
            h = end_ms // 3600000
            m = (end_ms % 3600000) // 60000
            s = (end_ms % 60000) // 1000
            ms = end_ms % 1000
            end_srt = f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            
            start_ass = srt_timestamp_to_ass(start_srt)
            end_ass = srt_timestamp_to_ass(end_srt)
        
        # Process text
        text = seg.get('text', '')
        extracted_layout_key, text, warning = extract_layout_key_from_srt_text(
            text=text,
            known_layout_keys=known_layout_keys,
            fallback_layout_key=layout_key,
            mode=mode
        )
        if warning:
            logger.warning(f"Segment {i+1}: {warning}")
            
        if not text.strip():
            logger.warning(f"Segment {i+1} has empty text after layout key extraction. Skipping.")
            continue
            
        text = normalize_newlines(text)
        if max_chars and max_chars > 0:
            text = wrap_text(text, max_chars)
        
        # Create dialogue line
        dialogue = create_dialogue_line(start_ass, end_ass, text, style, name=extracted_layout_key)
        dialogues.append(dialogue)
    
    logger.debug(f"Converted {len(dialogues)} segments to ASS dialogues")
    return dialogues


# ─────────────────────────────────────────────────────────────
# MAIN (for testing)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test timestamp conversion
    print("Testing timestamp conversion:")
    test_cases = [
        ("00:01:24,233", "0:01:24.23"),
        ("00:00:01,000", "0:00:01.00"),
        ("01:30:45,500", "1:30:45.50"),
    ]
    
    for srt_ts, expected in test_cases:
        result = srt_timestamp_to_ass(srt_ts)
        status = "✓" if result == expected else "✗"
        print(f"  {status} srt_timestamp_to_ass('{srt_ts}') = '{result}' (expected: '{expected}')")
    
    # Test text wrapping
    print("\nTesting text wrapping:")
    wrap_tests = [
        ("サムのギアリストサムの", 14, "サムのギアリストサ\\Nムの"),
        ("Line1\\NLine2", 14, "Line1\\NLine2"),
        ("Short", 14, "Short"),
    ]
    
    for text, max_chars, expected in wrap_tests:
        result = wrap_text(text, max_chars)
        status = "✓" if result == expected else "✗"
        print(f"  {status} wrap_text('{text}', {max_chars}) = '{result}'")
    
    # Test newline normalization
    print("\nTesting newline normalization:")
    newline_tests = [
        ("Line1\nLine2", "Line1\\NLine2"),
        ("Line1\r\nLine2", "Line1\\NLine2"),
        ("Line1\\NLine2", "Line1\\NLine2"),
    ]
    
    for text, expected in newline_tests:
        result = normalize_newlines(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} normalize_newlines('{repr(text)}') = '{result}'")
    
    # Test dialogue creation
    print("\nTesting dialogue creation:")
    dialogue = create_dialogue_line("0:01:24.23", "0:01:27.57", "Hello\\NWorld")
    print(f"  {dialogue}")