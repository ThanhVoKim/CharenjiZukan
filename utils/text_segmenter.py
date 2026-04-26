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

# Dấu câu dùng để cắt ở Giai đoạn 1 (tất cả dấu ngắt nhịp — strong + weak)
GRAMMAR_SPLIT_CHARS = set(".,!?:;。，！？：；、")

# Dấu câu mạnh — dùng để tăng điểm khi chấm điểm cắt ở GĐ2
STRONG_SPLIT_CHARS = set(".!?。！？")


def _block_text_len(block: List[Dict[str, Any]]) -> int:
    """Tổng số ký tự của một block."""
    return sum(len(t.get("text", "")) for t in block)


def _split_by_grammar(tokens: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Giai đoạn 1: Cắt toàn bộ văn bản dựa vào dấu câu.

    Không bảo vệ ngoặc bọc — cắt tại mọi dấu câu để tối đa hóa
    số điểm cắt theo ngữ pháp.
    """
    blocks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for token in tokens:
        text = token.get("text", "")
        current.append(token)

        # Cắt tại mọi dấu câu (không quan tâm ngoặc bọc)
        if any(c in text for c in GRAMMAR_SPLIT_CHARS):
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

    # 2. Ưu tiên cắt sau khoảng trắng hoặc dấu câu mạnh
    if text.endswith(" "):
        score += 8.0
    if text.endswith("\t"):
        score += 8.0
    if any(text.endswith(c) for c in STRONG_SPLIT_CHARS):
        score += 12.0
    if any(text.endswith(c) for c in set(",，、;；")):
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

    Returns:
        Danh sách các block, mỗi block là list token.
    """
    if ideal_chars is None:
        ideal_chars = max_chars

    if not tokens:
        return []

    # ── Giai đoạn 1: Chia theo ngữ pháp ─────────────────────────────
    grammar_blocks = _split_by_grammar(tokens)

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
