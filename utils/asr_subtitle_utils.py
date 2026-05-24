#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/asr_subtitle_utils.py — Helper chung cho ASR/subtitle processing.

Module này chứa logic dùng chung giữa `cli/qwen3_asr.py` và
`sync_engine/forced_alignment_subtitle.py`, bao gồm:
- Merge dấu câu từ full transcript vào word timestamps
- Format timestamp SRT
- Segment word timestamps thành subtitle blocks (tuân thủ invariant max_chars)
- Ghi subtitle blocks ra file SRT

Không phụ thuộc vào `cli/` hay `sync_engine/`.
"""

import string
from typing import List, Dict, Any


# ── Bộ dấu câu (dùng trong merge_punctuation) ──────────────────────
CJK_PUNCT = set("，。！？；：\u201c\u201d\u2018\u2019（《》【】、")
ALL_PUNCT_SET = set(string.punctuation) | CJK_PUNCT
OPENING_PUNCT = set("\u201c\u2018（《【")
CLOSING_PUNCT = set("\u201d\u2019）》】")
BRACKET_PAIRS = {"（": "）", "《": "》", "【": "】", "\u201c": "\u201d", "\u2018": "\u2019"}


def format_srt_time(seconds: float) -> str:
    """Định dạng thời gian SRT từ giây.

    Example:
        >>> format_srt_time(3661.5)
        '01:01:01,500'
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def merge_punctuation(words, full_text: str) -> List[Dict]:
    """Gắn dấu câu từ full_text vào mảng words timestamp.

    Thu thập cả prefix (dấu mở ngoặc/quote đứng trước token) và suffix
    để tránh mất dấu câu ở đầu câu hoặc gắn nhầm dấu mở ngoặc vào token trước đó.

    Hỗ trợ cả object (có attribute .text, .start_time, .end_time)
    và dict (có key "text", "start_time", "end_time").

    Args:
        words: Danh sách các word object hoặc dict.
        full_text: Chuỗi text đầy đủ (đã biết trước vị trí dấu câu).

    Returns:
        Danh sách dict, mỗi dict có key "text", "start_time", "end_time".

    Edge cases handled:
      - Token rỗng (Case 36): không gây IndexError.
      - Token cuối cùng có hậu tố chữ (Case 43): vớt toàn bộ phần còn lại.
      - Token không khớp hoàn toàn (Case 33, 45): dùng Partial Match để tránh kẹt con trỏ.
    """
    merged_words = []
    text_idx = 0
    full_len = len(full_text)
    total_words = len(words)

    for i, word_obj in enumerate(words):
        # Hỗ trợ cả object (attribute) và dict (key)
        if isinstance(word_obj, dict):
            clean_word = word_obj.get("text", "")
            w_start = word_obj.get("start_time", 0.0)
            w_end = word_obj.get("end_time", 0.0)
        else:
            clean_word = word_obj.text
            w_start = word_obj.start_time
            w_end = word_obj.end_time

        prefix_chars = ""
        trailing_chars = ""
        word_len = len(clean_word)

        # Case 36: Token rỗng → giữ nguyên, không gây IndexError
        if word_len == 0:
            merged_words.append({
                "text": "",
                "start_time": w_start,
                "end_time": w_end
            })
            continue

        # Thu thập prefix: các ký tự không phải chữ/số trước khi gặp ký tự đầu tiên của token
        while text_idx < full_len and full_text[text_idx].lower() != clean_word[0].lower():
            prefix_chars += full_text[text_idx]
            text_idx += 1

        # Partial Match: đếm số ký tự khớp liên tiếp giữa token và full_text
        match_len = 0
        while (match_len < word_len and
               text_idx + match_len < full_len and
               full_text[text_idx + match_len].lower() == clean_word[match_len].lower()):
            match_len += 1
        text_idx += match_len

        # Thu thập trailing chars
        while text_idx < full_len:
            char = full_text[text_idx]
            if char.isalnum() and char not in ALL_PUNCT_SET:
                break
            # Dấu mở ngoặc thuộc về token tiếp theo
            if char in OPENING_PUNCT:
                break
            trailing_chars += char
            text_idx += 1

        # Case 43: Nếu là token cuối cùng và còn chữ/số chưa được lấy,
        # vớt toàn bộ phần còn lại của full_text để tránh mất chữ.
        if i == total_words - 1 and text_idx < full_len:
            remaining = full_text[text_idx:]
            # Chỉ vớt nếu phần còn lại toàn là chữ/số (không có dấu câu mở ngoặc)
            if remaining and all(c.isalnum() or c in ALL_PUNCT_SET for c in remaining):
                # Nếu ký tự đầu tiên của remaining là chữ/số thì gom vào trailing
                if remaining[0].isalnum():
                    trailing_chars += remaining
                    text_idx = full_len

        merged_words.append({
            "text": prefix_chars + clean_word + trailing_chars,
            "start_time": w_start,
            "end_time": w_end
        })
    return merged_words


def segment_words_to_subtitles(
    merged_words: List[Dict],
    max_chars: int = 42,
    min_chars: int = 0,
    split_on_comma: bool = True,
) -> List[List[Dict]]:
    """Segment word timestamps thành subtitle blocks.

    Invariant bắt buộc:
    1. Nếu max_chars > 0 và tổng số ký tự của cả câu/candidate <= max_chars,
       không được ngắt thành hai subtitle blocks.
    2. max_chars == 0 nghĩa là grammar-only: chia theo dấu câu nhưng không ép
       giới hạn độ dài.
    3. max_chars < 0 giữ chế độ legacy single-block cho caller muốn tắt
       segmentation hoàn toàn.
    4. min_chars == 0 nghĩa là không có giới hạn tối thiểu; subtitle ngắn vẫn hợp lệ.
    5. split_on_comma chỉ mở rộng điểm cắt sang dấu phẩy/yếu; mặc định vẫn ưu tiên
       dấu câu mạnh.

    Args:
        merged_words: Danh sách dict word (có key "text", "start_time", "end_time").
        max_chars: Số ký tự tối đa mỗi subtitle block; 0 = chỉ chia theo dấu câu,
            số âm = ghi một block duy nhất.
        min_chars: Số ký tự tối thiểu (0 = không giới hạn).
        split_on_comma: Cho phép dùng dấu phẩy làm điểm cắt khi segment.

    Returns:
        Danh sách các subtitle block, mỗi block là list of word dicts.
    """
    if not merged_words:
        return []

    # max_chars < 0: giữ chế độ legacy single-block cho caller muốn tắt segmentation hoàn toàn.
    if max_chars < 0:
        return [merged_words]

    # Gọi smart_segment thống nhất cho cả grammar-only (max_chars == 0) và min/max.
    from utils.text_segmenter import smart_segment

    if max_chars > 0:
        # Tính tổng số ký tự
        total_len = sum(len(w.get("text", "")) for w in merged_words)
        # Nếu tổng <= max_chars: không split, một block duy nhất
        if total_len <= max_chars:
            return [merged_words]

    blocks = smart_segment(
        merged_words,
        min_chars=min_chars,
        max_chars=max_chars,
        ideal_chars=max_chars,
        split_on_comma=split_on_comma,
    )

    if max_chars <= 0:
        return blocks

    # Post-processing: merge adjacent blocks where combined length <= max_chars
    # CHỈ merge khi block trước KHÔNG kết thúc bằng dấu kết câu mạnh
    # (nghĩa là nó bị split tại comma/weak punctuation, không phải sentence boundary)
    # Đảm bảo short sentences bị split tại comma được gộp lại
    # nếu tổng vẫn nằm trong giới hạn max_chars.
    _STRONG_END_CHARS = set(".!?。！？")

    if len(blocks) > 1:
        merged_blocks = [blocks[0][:]]  # copy first block
        for block in blocks[1:]:
            prev_text = "".join(w.get("text", "") for w in merged_blocks[-1]).rstrip()
            prev_len = len(prev_text)
            cur_len = sum(len(w.get("text", "")) for w in block)

            # Chỉ merge nếu block trước không kết thúc bằng dấu câu mạnh
            # VÀ tổng độ dài <= max_chars
            prev_ends_weak = not prev_text or prev_text[-1] not in _STRONG_END_CHARS
            if prev_ends_weak and prev_len + cur_len <= max_chars:
                merged_blocks[-1].extend(block)
            else:
                merged_blocks.append(block[:])  # copy block
        blocks = merged_blocks

    return blocks


def write_subtitle_srt(
    subtitle_blocks: List[List[Dict]],
    output_path: str,
    offset_seconds: float = 0.24,
) -> None:
    """Ghi subtitle blocks ra file SRT.

    Timestamp start/end của mỗi block lấy từ word đầu/cuối,
    sau đó cộng offset_seconds.

    Args:
        subtitle_blocks: Danh sách các subtitle block từ segment_words_to_subtitles().
        output_path: Đường dẫn file SRT output.
        offset_seconds: Offset cộng vào timestamp (giây).
    """
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        seq = 0  # SRT sequence number (1-based, chỉ tăng khi block thực sự được ghi)
        for block in subtitle_blocks:
            if not block:
                continue

            seq += 1
            raw_start = max(0.0, block[0].get("start_time", 0.0) + offset_seconds)
            raw_end = block[-1].get("end_time", 0.0) + offset_seconds

            start_time = format_srt_time(raw_start)
            end_time = format_srt_time(raw_end)
            text = "".join([w.get("text", "") for w in block]).strip()

            f.write(f"{seq}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")
