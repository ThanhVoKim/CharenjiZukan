#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/utils/test_text_segmenter.py
================================
Test thuật toán smart_segment 2-phase trong utils/text_segmenter.py.

Cấu trúc layers:
  Layer 1 — Unit Tests: Kiểm tra logic chia block, gộp block, chấm điểm cắt
            không cần AI model/GPU.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

import pytest

# ── Project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.text_segmenter import smart_segment, _split_by_grammar, _merge_short_blocks


def make_tokens(texts: List[str], times: List[tuple] = None) -> List[Dict[str, Any]]:
    """Tạo nhanh danh sách token từ text."""
    if times is None:
        times = [(i * 1.0, (i + 1) * 1.0) for i in range(len(texts))]
    return [
        {"text": t, "start_time": s, "end_time": e}
        for t, (s, e) in zip(texts, times)
    ]


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_TextSegmenterGrammar:
    """Test Giai đoạn 1: Chia theo ngữ pháp (KHÔNG bảo vệ ngoặc)."""

    # ── Các test cũ giữ nguyên behavior bằng split_on_comma=True ──

    def test_split_by_punctuation_with_comma(self):
        """Cắt tại mọi dấu câu (strong + weak split) khi split_on_comma=True."""
        tokens = make_tokens(["Hello", ", ", "world", ". ", "Next"])
        blocks = _split_by_grammar(tokens, split_on_comma=True)
        # Dấu phẩy cũng là điểm cắt
        assert len(blocks) == 3
        assert "".join(t["text"] for t in blocks[0]) == "Hello, "
        assert "".join(t["text"] for t in blocks[1]) == "world. "
        assert "".join(t["text"] for t in blocks[2]) == "Next"

    def test_no_protect_brackets_with_comma(self):
        """Cắt cả bên trong cặp ngoặc khi split_on_comma=True."""
        tokens = make_tokens(["这是", "（测试", "，文本）", "结束。"])
        blocks = _split_by_grammar(tokens, split_on_comma=True)
        # Dấu phẩy bên trong ngoặc cũng bị cắt
        assert len(blocks) == 2
        assert "".join(t["text"] for t in blocks[0]) == "这是（测试，文本）"
        assert "".join(t["text"] for t in blocks[1]) == "结束。"

    def test_no_protect_quotes_with_comma(self):
        """Cắt cả bên trong cặp quote khi split_on_comma=True."""
        tokens = make_tokens(["他说", "“你好", "，世界”", "结束。"])
        blocks = _split_by_grammar(tokens, split_on_comma=True)
        # Dấu phẩy bên trong quote cũng bị cắt
        assert len(blocks) == 2
        assert "".join(t["text"] for t in blocks[0]) == "他说“你好，世界”"
        assert "".join(t["text"] for t in blocks[1]) == "结束。"

    def test_split_after_closing_bracket_with_comma(self):
        """GĐ1 cắt tại mọi dấu câu (cả dấu phẩy) khi split_on_comma=True.

        Input: （测试），继续。
        Expected GĐ1: 2 blocks (cắt tại dấu phẩy và dấu chấm)
        """
        tokens = make_tokens(["（测试）", "，", "继续。"])
        blocks = _split_by_grammar(tokens, split_on_comma=True)
        assert len(blocks) == 2
        assert "".join(t["text"] for t in blocks[0]) == "（测试），"
        assert "".join(t["text"] for t in blocks[1]) == "继续。"

    # ── Các test mới cho split_on_comma=False (mặc định) ──

    def test_split_by_punctuation_no_comma(self):
        """Mặc định không cắt tại dấu phẩy — chỉ cắt tại strong split."""
        tokens = make_tokens(["Hello", ", ", "world", ". ", "Next"])
        blocks = _split_by_grammar(tokens, split_on_comma=False)
        # Dấu phẩy KHÔNG phải điểm cắt
        assert len(blocks) == 2
        assert "".join(t["text"] for t in blocks[0]) == "Hello, world. "
        assert "".join(t["text"] for t in blocks[1]) == "Next"

    def test_no_protect_brackets(self):
        """KHÔNG bảo vệ ngoặc — cắt tại dấu câu mạnh bên trong ngoặc."""
        tokens = make_tokens(["（测试）", "。", "继续"])
        blocks = _split_by_grammar(tokens, split_on_comma=False)
        # Dấu chấm bên trong/trước ngoặc vẫn cắt (Phương án 1)
        assert len(blocks) == 2
        assert "".join(t["text"] for t in blocks[0]) == "（测试）。"
        assert "".join(t["text"] for t in blocks[1]) == "继续"

    def test_no_protect_brackets_no_comma(self):
        """Dấu phẩy trong ngoặc không cắt khi split_on_comma=False."""
        tokens = make_tokens(["这是", "（测试", "，文本）", "结束。"])
        blocks = _split_by_grammar(tokens, split_on_comma=False)
        # Dấu phẩy bên trong ngoặc KHÔNG bị cắt
        assert len(blocks) == 1
        assert "".join(t["text"] for t in blocks[0]) == "这是（测试，文本）结束。"

    def test_no_protect_quotes_no_comma(self):
        """Dấu phẩy trong quote không cắt khi split_on_comma=False."""
        tokens = make_tokens(["他说", "“你好", "，世界”", "结束。"])
        blocks = _split_by_grammar(tokens, split_on_comma=False)
        # Dấu phẩy bên trong quote KHÔNG bị cắt
        assert len(blocks) == 1
        assert "".join(t["text"] for t in blocks[0]) == "他说“你好，世界”结束。"

    def test_split_after_closing_bracket_no_comma(self):
        """Dấu phẩy sau ngoặc không cắt khi split_on_comma=False.

        Input: （测试），继续。
        Expected GĐ1: 1 block (chỉ cắt tại dấu chấm)
        """
        tokens = make_tokens(["（测试）", "，", "继续。"])
        blocks = _split_by_grammar(tokens, split_on_comma=False)
        assert len(blocks) == 1
        assert "".join(t["text"] for t in blocks[0]) == "（测试），继续。"


class TestLayer1_TextSegmenterMinMax:
    """Test Giai đoạn 2: Xử lý min/max."""

    # ── Các test cũ giữ nguyên behavior bằng split_on_comma=True ──

    def test_ideal_length_no_split_with_comma(self):
        """Block trong khoảng min-max giữ nguyên (split_on_comma=True).

        GĐ1 cắt tại mọi dấu câu, nên "Hello, " và "world."
        thành 2 blocks riêng. GĐ2 giữ nguyên vì cả 2 đều trong [5, 20].
        """
        tokens = make_tokens(["Hello", ", ", "world", "."])
        result = smart_segment(tokens, min_chars=5, max_chars=20, split_on_comma=True)
        assert len(result) == 2
        assert "".join(t["text"] for t in result[0]) == "Hello, "
        assert "".join(t["text"] for t in result[1]) == "world."

    def test_preserve_timestamps_with_comma(self):
        """Timestamp được preserve nguyên vẹn (split_on_comma=True).

        GĐ1 cắt tại dấu phẩy → 2 blocks. Block 1 gồm token 0+1.
        """
        tokens = make_tokens(["Hello", ", ", "world"], [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)])
        result = smart_segment(tokens, min_chars=5, max_chars=20, split_on_comma=True)
        assert len(result) == 2
        assert result[0][0]["start_time"] == 0.0
        assert result[0][-1]["end_time"] == 2.0
        assert result[1][0]["start_time"] == 2.0
        assert result[1][-1]["end_time"] == 3.0

    def test_cjk_and_latin_mixed_with_comma(self):
        """Xử lý cả CJK và Latin (split_on_comma=True)."""
        tokens = make_tokens(["Hello", "，", "世界", "！", "Test", "."])
        result = smart_segment(tokens, min_chars=3, max_chars=15, split_on_comma=True)
        texts = ["".join(t["text"] for t in block) for block in result]
        assert all(len(t) <= 15 for t in texts)
        assert "".join(texts) == "Hello，世界！Test."

    # ── Các test mới cho split_on_comma=False (mặc định) ──

    def test_ideal_length_no_split_no_comma(self):
        """Mặc định không cắt tại dấu phẩy → 1 block duy nhất.

        GĐ1 chỉ cắt tại strong split (.!?。！？：；), nên "Hello, world."
        thành 1 block. GĐ2 giữ nguyên vì trong [5, 20].
        """
        tokens = make_tokens(["Hello", ", ", "world", "."])
        result = smart_segment(tokens, min_chars=5, max_chars=20, split_on_comma=False)
        assert len(result) == 1
        assert "".join(t["text"] for t in result[0]) == "Hello, world."

    def test_preserve_timestamps_no_comma(self):
        """Mặc định không cắt tại dấu phẩy → 1 block duy nhất.

        GĐ1 chỉ cắt tại strong split, nên "Hello, world" là 1 block.
        """
        tokens = make_tokens(["Hello", ", ", "world"], [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)])
        result = smart_segment(tokens, min_chars=5, max_chars=20, split_on_comma=False)
        assert len(result) == 1
        assert result[0][0]["start_time"] == 0.0
        assert result[0][-1]["end_time"] == 3.0

    def test_cjk_and_latin_mixed_no_comma(self):
        """Xử lý cả CJK và Latin (mặc định split_on_comma=False)."""
        tokens = make_tokens(["Hello", "，", "世界", "！", "Test", "."])
        result = smart_segment(tokens, min_chars=3, max_chars=15, split_on_comma=False)
        texts = ["".join(t["text"] for t in block) for block in result]
        assert all(len(t) <= 15 for t in texts)
        assert "".join(texts) == "Hello，世界！Test."

    # ── Các test chung (không phụ thuộc split_on_comma) ──

    def test_merge_short_block(self):
        """Block < min được gộp với block sau."""
        tokens = make_tokens(["Hi", ". ", "This is a longer sentence here."])
        result = smart_segment(tokens, min_chars=10, max_chars=40)
        texts = ["".join(t["text"] for t in block) for block in result]
        assert len(texts) == 1
        assert texts[0] == "Hi. This is a longer sentence here."

    def test_split_long_block_evenly(self):
        """Block > max được chia đều."""
        # 60 ký tự, max=20 → N=3, target=20
        tokens = make_tokens(["一二三四五六七八九"] * 6)  # 9*6=54 ký tự
        result = smart_segment(tokens, min_chars=5, max_chars=20)
        lengths = [sum(len(t["text"]) for t in block) for block in result]
        # Mỗi block không vượt max
        assert all(l <= 20 for l in lengths)
        # Tổng vẫn bằng 54
        assert sum(lengths) == 54

    def test_split_long_block_with_pauses(self):
        """Ưu tiên cắt tại khoảng lặng lớn."""
        texts = ["A", "B", "C", "D", "E", "F"]
        times = [
            (0.0, 1.0), (1.0, 2.0), (2.0, 3.0),
            (5.0, 6.0),  # pause 2.0s giữa C và D
            (6.0, 7.0), (7.0, 8.0),
        ]
        tokens = make_tokens(texts, times)
        result = smart_segment(tokens, min_chars=1, max_chars=4)
        # Với max=4, tổng 6 ký tự → N=2, target=3
        # Ưu tiên cắt sau C (index 2) vì pause lớn
        assert len(result) == 2
        assert "".join(t["text"] for t in result[0]) == "ABC"
        assert "".join(t["text"] for t in result[1]) == "DEF"

    def test_empty_input(self):
        """Input rỗng trả về list rỗng."""
        assert smart_segment([]) == []

    def test_single_token_exceeds_max(self):
        """Token đơn lẻ vượt max vẫn giữ nguyên (không ép cắt)."""
        long_str = "一二三四五六七八九十一二三四五六七八九十"
        tokens = make_tokens([long_str])
        result = smart_segment(tokens, min_chars=5, max_chars=15)
        assert len(result) == 1
        assert "".join(t["text"] for t in result[0]) == long_str
