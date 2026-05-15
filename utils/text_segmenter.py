#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/text_segmenter.py — Thuật toán chia đoạn văn bản thông minh 2-phase.

Hỗ trợ đa mục đích: subtitle, title, paragraph, v.v.
Xử lý được cả Latin và CJK.

Thuật toán:
  Giai đoạn 1 — CHIA THEO NGỮ PHÁP (Base Blocks):
    Cắt toàn bộ văn bản DỰA VÀO DẤU CÂU.
    Không bảo vệ ngoặc bọc — cắt tại mọi dấu câu để tối đa hóa
    số điểm cắt theo ngữ pháp, hạn chế cắt cơ học ở Giai đoạn 2.

  Giai đoạn 2 — XỬ LÝ MIN/MAX (Post-processing):
    • < min  → gộp với block liền kề nếu không vượt max.
    • min–max → giữ nguyên.
    • > max   → chia đều thành N khúc (N = ceil(len / ideal)),
                cắt ở điểm gần target_len nhất, ưu tiên khoảng trắng
                hoặc khoảng lặng âm thanh lớn nhất.
"""

import math
import string
from typing import List, Dict, Any, Optional

# ── Bộ dấu câu ──────────────────────────────────────────────────────
CJK_PUNCT = set("，。！？；：“”‘’（）《》【】、")
ALL_PUNCT_SET = set(string.punctuation) | CJK_PUNCT
SPLIT_PUNCT_SET = set(".,!?:;。，！？：；、")
ELLIPSIS_PUNCT = "……"
OPENING_PUNCT = set("“‘（《【")
CLOSING_PUNCT = set("”’）》】")
BRACKET_PAIRS = {"（": "）", "《": "》", "【": "】", "“": "”", "‘": "’"}

# Dấu câu mặc định dùng để cắt ở Giai đoạn 1 (strong split, KHÔNG bao gồm dấu phẩy)
DEFAULT_GRAMMAR_SPLIT_CHARS = set(".!?:。！？：；")

# Dấu câu mở rộng — bao gồm cả dấu phẩy (weak split)
EXTENDED_GRAMMAR_SPLIT_CHARS = set(".,!?:;。，！？：；、")

# Dấu câu mạnh — dùng để tăng điểm khi chấm điểm cắt ở GĐ2
STRONG_SPLIT_CHARS = set(".!?。！？")

# Dấu phân cách số — không được coi là điểm cắt khi nằm giữa 2 chữ số
NUMERIC_SEPARATOR_CHARS = set(".,，．")

# Ký tự hậu tố đơn vị/tiền tệ/phần trăm thường dính trực tiếp sau số
NUMERIC_UNIT_SUFFIX_CHARS = set("%％‰°℃℉¥$€£₩₫円元")


def _block_text_len(block: List[Dict[str, Any]]) -> int:
    """Tổng số ký tự của một block."""
    return sum(len(t.get("text", "")) for t in block)


def _get_char_before(tokens: List[Dict[str, Any]], token_idx: int, char_idx: int) -> str:
    """Lấy ký tự ngay trước vị trí hiện tại trong chuỗi ghép từ tokens."""
    for i in range(token_idx, -1, -1):
        text = tokens[i].get("text", "")
        start = char_idx - 1 if i == token_idx else len(text) - 1
        for j in range(start, -1, -1):
            return text[j]
    return ""


def _get_char_after(tokens: List[Dict[str, Any]], token_idx: int, char_idx: int) -> str:
    """Lấy ký tự ngay sau vị trí hiện tại trong chuỗi ghép từ tokens."""
    for i in range(token_idx, len(tokens)):
        text = tokens[i].get("text", "")
        start = char_idx + 1 if i == token_idx else 0
        for j in range(start, len(text)):
            return text[j]
    return ""


def _is_numeric_separator_at(
    tokens: List[Dict[str, Any]],
    token_idx: int,
    char_idx: int,
) -> bool:
    """True nếu dấu chấm/phẩy tại vị trí đang xét là dấu phân cách trong số."""
    text = tokens[token_idx].get("text", "")
    if char_idx < 0 or char_idx >= len(text):
        return False

    char = text[char_idx]
    if char not in NUMERIC_SEPARATOR_CHARS:
        return False

    prev_char = _get_char_before(tokens, token_idx, char_idx)
    next_char = _get_char_after(tokens, token_idx, char_idx)
    return prev_char.isdigit() and next_char.isdigit()


def _is_numeric_unit_suffix_start(char: str) -> bool:
    """True nếu ký tự có thể là phần đầu của hậu tố đơn vị/tiền tệ dính với số."""
    return bool(char) and (char.isalpha() or char in NUMERIC_UNIT_SUFFIX_CHARS)


def _is_numeric_split_boundary(block: List[Dict[str, Any]], idx: int) -> bool:
    """True nếu cắt sau token idx sẽ tách rời số, dấu phân cách số hoặc đơn vị."""
    if idx < 0 or idx + 1 >= len(block):
        return False

    left_text = block[idx].get("text", "")
    right_text = block[idx + 1].get("text", "")
    if not left_text or not right_text:
        return False

    left_char = left_text[-1]
    right_char = right_text[0]
    if left_char.isspace() or right_char.isspace():
        return False

    if left_char in NUMERIC_SEPARATOR_CHARS and right_char.isdigit():
        prev_char = _get_char_before(block, idx, len(left_text) - 1)
        return prev_char.isdigit()

    if left_char.isdigit() and right_char in NUMERIC_SEPARATOR_CHARS:
        next_after_separator = _get_char_after(block, idx + 1, 0)
        return next_after_separator.isdigit()

    if left_char.isdigit() and _is_numeric_unit_suffix_start(right_char):
        return True

    return False


def _has_sentence_split_punct(
    tokens: List[Dict[str, Any]],
    token_idx: int,
    split_chars: set,
) -> bool:
    """True nếu token có dấu câu thật dùng được làm điểm cắt câu."""
    text = tokens[token_idx].get("text", "")
    for char_idx, char in enumerate(text):
        if char not in split_chars:
            continue
        if _is_numeric_separator_at(tokens, token_idx, char_idx):
            continue
        return True
    return False


def _split_by_grammar(
    tokens: List[Dict[str, Any]],
    split_on_comma: bool = False,
) -> List[List[Dict[str, Any]]]:
    """Giai đoạn 1: Cắt toàn bộ văn bản dựa vào dấu câu.

    Không bảo vệ ngoặc bọc — cắt tại dấu câu để tối đa hóa
    số điểm cắt theo ngữ pháp.

    Args:
        split_on_comma: Nếu True, dấu phẩy (`,`, `，`, `、`) cũng được
            dùng làm điểm cắt. Mặc định False để tránh cắt quá nhỏ.
    """
    split_chars = EXTENDED_GRAMMAR_SPLIT_CHARS if split_on_comma else DEFAULT_GRAMMAR_SPLIT_CHARS

    blocks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for token_idx, token in enumerate(tokens):
        current.append(token)

        # Cắt tại dấu câu thật (không quan tâm ngoặc bọc), nhưng bỏ qua
        # dấu chấm/phẩy thuộc số như 1.2, 3,14, 1,000,000.
        if _has_sentence_split_punct(tokens, token_idx, split_chars):
            blocks.append(current)
            current = []

    if current:
        blocks.append(current)

    return blocks


def _score_split_point(
    block: List[Dict[str, Any]],
    idx: int,
    target_len: float,
    current_len: int,
) -> float:
    """Chấm điểm một vị trí cắt (sau token tại idx).

    Điểm càng cao → vị trí càng được ưu tiên.
    """
    score = 0.0

    # 1. Độ lệch so với target_len (càng gần càng tốt)
    deviation = abs(current_len - target_len)
    score -= deviation * 1.5

    token = block[idx]
    text = token.get("text", "")

    # 2. Ưu tiên cắt sau khoảng trắng hoặc dấu câu mạnh.
    #    Nếu boundary đang nằm trong chuỗi số/đơn vị thì phạt nặng để
    #    tránh chia 1.2, 12.5kg, 3,000円 thành nhiều block.
    if _is_numeric_split_boundary(block, idx):
        score -= 1000.0

    last_char_idx = len(text) - 1
    ends_with_numeric_separator = bool(text) and _is_numeric_separator_at(block, idx, last_char_idx)

    if text.endswith(" "):
        score += 8.0
    if text.endswith("\t"):
        score += 8.0
    if not ends_with_numeric_separator and any(text.endswith(c) for c in STRONG_SPLIT_CHARS):
        score += 12.0
    if not ends_with_numeric_separator and any(text.endswith(c) for c in set(",，、;；")):
        score += 6.0

    # 3. Ưu tiên khoảng lặng âm thanh (nếu có timestamp)
    if idx + 1 < len(block):
        next_token = block[idx + 1]
        cur_end = token.get("end_time")
        next_start = next_token.get("start_time")
        if cur_end is not None and next_start is not None:
            pause = next_start - cur_end
            score += pause * 30.0  # weight cho pause

    return score


def _split_long_block(
    block: List[Dict[str, Any]],
    min_chars: int,
    max_chars: int,
    ideal_chars: int,
) -> List[List[Dict[str, Any]]]:
    """Giai đoạn 2 — Trường hợp 3: Block quá dài (> max).

    Tính N = ceil(len / ideal), target = len / N.
    Chia thành N khúc, ưu tiên cắt gần target_len nhất.
    """
    total_len = _block_text_len(block)
    if total_len == 0:
        return [block]

    N = math.ceil(total_len / ideal_chars)
    if N < 2:
        N = 2
    target_len = total_len / N

    result: List[List[Dict[str, Any]]] = []
    remaining = block[:]

    for _ in range(N - 1):
        if not remaining:
            break

        best_idx = -1
        best_score = -float("inf")
        current_len = 0

        for idx, token in enumerate(remaining):
            text = token.get("text", "")
            token_len = len(text)
            current_len += token_len

            # Chỉ xét khi đã đạt ít nhất min_chars
            if current_len < min_chars:
                continue

            # Nếu đã vượt max và vẫn chưa có candidate nào,
            # buộc phải cắt ở đây (fallback)
            if current_len > max_chars and best_idx == -1:
                best_idx = idx
                break

            if current_len > max_chars:
                break

            score = _score_split_point(remaining, idx, target_len, current_len)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx == -1:
            # Trường hợp fallback: cắt ngay token đầu tiên
            best_idx = 0

        result.append(remaining[: best_idx + 1])
        remaining = remaining[best_idx + 1 :]

    if remaining:
        result.append(remaining)

    return result


def _merge_short_blocks(
    blocks: List[List[Dict[str, Any]]],
    min_chars: int,
    max_chars: int,
) -> List[List[Dict[str, Any]]]:
    """Gộp các block < min_chars với block liền kề nếu không vượt max."""
    if not blocks:
        return blocks

    merged: List[List[Dict[str, Any]]] = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        block_len = _block_text_len(block)

        if block_len < min_chars:
            # Ưu tiên gộp với block TRƯỚC
            if merged:
                prev_len = _block_text_len(merged[-1])
                if prev_len + block_len <= max_chars:
                    merged[-1].extend(block)
                    i += 1
                    continue

            # Thử gộp với block SAU
            if i + 1 < len(blocks):
                next_len = _block_text_len(blocks[i + 1])
                if block_len + next_len <= max_chars:
                    # Gộp block hiện tại vào block sau, rồi xử lý block sau ở lần lặp tiếp
                    blocks[i + 1] = block + blocks[i + 1]
                    i += 1
                    continue

            # Không gộp được → giữ nguyên (chấp nhận ngoại lệ)
            merged.append(block)
        else:
            merged.append(block)

        i += 1

    return merged


def smart_segment(
    tokens: List[Dict[str, Any]],
    min_chars: int = 8,
    max_chars: int = 40,
    ideal_chars: Optional[int] = None,
    split_on_comma: bool = False,
) -> List[List[Dict[str, Any]]]:
    """Chia đoạn văn bản thông minh 2-phase.

    Args:
        tokens: Danh sách các token, mỗi token là dict bắt buộc có key "text".
                Các key khác như "start_time", "end_time" là optional và
                sẽ được preserve nguyên vẹn.
        min_chars: Độ dài tối thiểu của một block. Nếu == 0 thì tắt merge.
        max_chars: Độ dài tối đa của một block. Nếu == 0 thì tắt GĐ2
                   (chỉ trả về grammar blocks).
        ideal_chars: Độ dài lý tưởng để tính số khúc N khi block quá dài.
                     Mặc định = max_chars.
        split_on_comma: Nếu True, dấu phẩy (`,`, `，`, `、`) cũng được
            dùng làm điểm cắt ở Giai đoạn 1. Mặc định False.

    Returns:
        Danh sách các block, mỗi block là list token.
    """
    if ideal_chars is None:
        ideal_chars = max_chars

    if not tokens:
        return []

    # ── Giai đoạn 1: Chia theo ngữ pháp ─────────────────────────────
    grammar_blocks = _split_by_grammar(tokens, split_on_comma=split_on_comma)

    # Nếu max_chars == 0 → tắt GĐ2, trả về grammar blocks thuần
    if max_chars == 0:
        return grammar_blocks

    # ── Giai đoạn 2: Xử lý min/max ──────────────────────────────────
    final: List[List[Dict[str, Any]]] = []

    for block in grammar_blocks:
        block_len = _block_text_len(block)

        if min_chars > 0 and block_len < min_chars:
            # Thử gộp với block trước
            if final:
                prev_len = _block_text_len(final[-1])
                if prev_len + block_len <= max_chars:
                    final[-1].extend(block)
                    continue
            # Thử gộp với block sau (lookahead) — xử lý ở pass merge
            final.append(block)

        elif block_len <= max_chars:
            final.append(block)

        else:  # block_len > max_chars
            sub_blocks = _split_long_block(block, min_chars, max_chars, ideal_chars)
            final.extend(sub_blocks)

    # Pass gộp cuối: xử lý các block ngắn còn sót lại (chỉ khi min_chars > 0)
    if min_chars > 0:
        final = _merge_short_blocks(final, min_chars, max_chars)

    return final
