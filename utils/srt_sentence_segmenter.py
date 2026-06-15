#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/srt_sentence_segmenter.py — Gom block SRT liên tiếp thành câu hoàn chỉnh.

Thuật toán (v1):
    Chỉ cắt khi dấu ngắt câu rơi đúng cuối một block OCR. Gom tất cả block chưa
    kết thúc bằng dấu ngắt câu; khi gặp block kết thúc bằng dấu ngắt → đóng
    thành 1 block mới với:
        start = block đầu nhóm.start_time   (timestamp OCR thật, ms)
        end   = block cuối nhóm.end_time    (timestamp OCR thật, ms)
    Không nội suy, không model, không GPU.

NOTE (hoãn lại — chưa implement ở v1):
    Dấu ngắt câu nằm GIỮA block subtitle (vd "去学校。然后") → v1 mặc kệ, không cắt.
    Khi cần: nội suy timestamp theo tỉ lệ ký tự trong block (dùng start_time/end_time
    thật của block đó), không nội suy qua nhiều block.
"""

from typing import Optional

from utils.srt_parser import is_cjk
from utils.text_segmenter import (
    CLOSING_PUNCT,
    DEFAULT_GRAMMAR_SPLIT_CHARS,
    _should_split_after_period,
)

# Mở rộng CLOSING_PUNCT để bao gồm ngoặc góc CJK (」『』〕) và ASCII quote (")
# không có trong bộ gốc của text_segmenter.
_CLOSING_PUNCT = CLOSING_PUNCT | set('」』〕〗"')


def _ends_with_sentence_break(text: str, split_chars: set) -> bool:
    """True nếu text kết thúc bằng dấu ngắt câu thật.

    Bỏ qua ngoặc/nháy đóng ở đuôi trước khi kiểm tra, nên nhận được cả
    '他说。」' hay '"…"'. Với dấu chấm ASCII, ủy quyền cho
    _should_split_after_period để loại số thập phân (1.5) và abbreviation (e.g.).
    """
    stripped = text.rstrip()
    if not stripped:
        return False

    # Bỏ qua ngoặc/nháy đóng ở đuôi
    while stripped and stripped[-1] in _CLOSING_PUNCT:
        stripped = stripped[:-1]

    if not stripped:
        return False

    last_char = stripped[-1]
    if last_char not in split_chars:
        return False

    if last_char == ".":
        period_idx = len(stripped) - 1
        fake_tokens = [{"text": stripped}]
        return _should_split_after_period(fake_tokens, 0, period_idx)

    return True


def _join_sentence_text(texts: list) -> str:
    """Nối text nhiều block thành 1 câu.

    CJK: nối không khoảng trắng (bỏ whitespace trong từng block).
    Latin: collapse whitespace thành single space, nối bằng space.
    """
    combined = "".join(texts)
    cjk = is_cjk(combined)

    chunks = []
    for t in texts:
        t = "".join(t.split()) if cjk else " ".join(t.split())
        if t:
            chunks.append(t)

    sep = "" if cjk else " "
    return sep.join(chunks).strip()


def _make_block(group: list, idx: int) -> dict:
    text = _join_sentence_text([seg["text"] for seg in group])
    start_raw = group[0].get("startraw", "")
    end_raw = group[-1].get("endraw", "")
    return {
        "line": idx,
        "start_time": group[0]["start_time"],
        "end_time": group[-1]["end_time"],
        "startraw": start_raw,
        "endraw": end_raw,
        "time": f"{start_raw} --> {end_raw}",
        "text": text,
    }


def resegment_srt_by_sentence(
    segments: list,
    split_chars: Optional[set] = None,
) -> list:
    """Gom block SRT liên tiếp thành câu hoàn chỉnh theo dấu ngắt câu cuối block.

    Args:
        segments: List segment từ parse_srt (cần key: text, start_time, end_time).
        split_chars: Tập dấu ngắt câu. Mặc định: DEFAULT_GRAMMAR_SPLIT_CHARS
            (.!?:。！？：；, không gồm dấu phẩy).

    Returns:
        List segment mới: mỗi phần tử là 1 câu hoàn chỉnh, giữ nguyên key
        {line (1-based), start_time, end_time, startraw, endraw, time, text}.
    """
    if split_chars is None:
        split_chars = DEFAULT_GRAMMAR_SPLIT_CHARS

    result = []
    buffer = []

    for seg in segments:
        buffer.append(seg)
        if _ends_with_sentence_break(seg["text"], split_chars):
            result.append(_make_block(buffer, len(result) + 1))
            buffer = []

    if buffer:
        result.append(_make_block(buffer, len(result) + 1))

    return result
