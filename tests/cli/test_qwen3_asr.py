#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/cli/test_qwen3_asr.py
====================================
Test logic gom dấu câu (merge_punctuation) và cắt phụ đề (segment_subtitles)
trong cli/qwen3_asr.py (Qwen3-ASR Transformers Backend).

Cấu trúc layers:
  Layer 1 — Unit Tests: Kiểm tra regex, gom prefix/suffix, tách câu không cần AI model/GPU.
"""

import sys
from pathlib import Path
from typing import List

import pytest

# ── Project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cli.qwen3_asr import merge_punctuation, segment_subtitles


# Mock object for ASR result words
class MockWord:
    def __init__(self, text: str, start_time: float, end_time: float):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time


def to_mock_words(texts: List[str]) -> List[MockWord]:
    """Tạo nhanh danh sách timestamp giả định từ danh sách từ."""
    return [MockWord(t, i * 1.0, (i + 1) * 1.0) for i, t in enumerate(texts)]


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_Qwen3ASRPunctuation:
    """Test Layer 1: merge_punctuation and segment_subtitles logic."""

    def test_case_10_quote_without_colon(self):
        """Case 10: Quote không có dấu hai chấm trước đó.

        Input: 他说“你好！”
        Tokens: 他说 | 你好
        Expected merged:
          他说
          “你好！”
        """
        words = to_mock_words(["他说", "你好"])
        full_text = '他说“你好！”'
        merged = merge_punctuation(words, full_text)
        assert len(merged) == 2
        assert merged[0]["text"] == "他说"
        assert merged[1]["text"] == "“你好！”"

    def test_case_37_leading_quote(self):
        """Case 37: Dấu câu ở đầu full text.

        Input: “你好！”
        Tokens: 你好
        Expected merged: “你好！”
        """
        words = to_mock_words(["你好"])
        full_text = '“你好！”'
        merged = merge_punctuation(words, full_text)
        assert len(merged) == 1
        assert merged[0]["text"] == "“你好！”"

    def test_case_41_open_bracket_before_first_token(self):
        """Case 41: Dấu mở ngoặc trước token đầu tiên.

        Input: （测试）完成。
        Tokens: 测试 | 完成
        Expected merged:
          （测试）
          完成。
        """
        words = to_mock_words(["测试", "完成"])
        full_text = '（测试）完成。'
        merged = merge_punctuation(words, full_text)
        assert len(merged) == 2
        assert merged[0]["text"] == "（测试）"
        assert merged[1]["text"] == "完成。"

    def test_case_40_colon_and_quote(self):
        """Case 40: Dấu quote mở ngay sau dấu hai chấm.

        Input: 他说：“你好！”
        Tokens: 他说 | 你好
        Expected merged:
          他说：
          “你好！”
        """
        words = to_mock_words(["他说", "你好"])
        full_text = '他说：“你好！”'
        merged = merge_punctuation(words, full_text)
        assert len(merged) == 2
        assert merged[0]["text"] == "他说："
        assert merged[1]["text"] == "“你好！”"

        # Test segment_subtitles (split priority at colon)
        subs = segment_subtitles(merged, max_chars=15)
        assert len(subs) == 2
        assert "".join([w["text"] for w in subs[0]]) == "他说："
        assert "".join([w["text"] for w in subs[1]]) == "“你好！”"

    def test_case_13_comma_inside_brackets(self):
        """Case 13: Dấu phẩy nằm bên trong cặp ngoặc không gây cắt dòng.

        Input: 这是（测试，文本）结束。
        Tokens: 这是 | 测试 | 文本 | 结束
        """
        words = to_mock_words(["这是", "测试", "文本", "结束"])
        full_text = '这是（测试，文本）结束。'
        merged = merge_punctuation(words, full_text)

        subs = segment_subtitles(merged, max_chars=15)
        # Không được phép cắt giữa dấu ngoặc （ ）
        assert len(subs) == 1
        joined = "".join([w["text"] for w in subs[0]])
        assert joined == "这是（测试，文本）结束。"

    def test_case_47_comma_short_sentence(self):
        """Case 47: Cắt do dấu phẩy làm câu quá ngắn (Soft break).

        Input: 我，来了。
        Tokens: 我 | 来了
        Expected: Không cắt vì chuỗi tổng cộng chỉ có 5 ký tự.
        """
        words = to_mock_words(["我", "来了"])
        full_text = '我，来了。'
        merged = merge_punctuation(words, full_text)

        subs = segment_subtitles(merged, max_chars=15)
        assert len(subs) == 1
        joined = "".join([w["text"] for w in subs[0]])
        assert joined == "我，来了。"

    def test_case_29_unicode_ellipsis(self):
        """Case 29: Dấu ba chấm Unicode …… được hiểu là ngắt câu mềm.

        Input: 等等……走吧。
        Tokens: 等等 | 走吧
        """
        words = to_mock_words(["等等", "走吧"])
        full_text = '等等……走吧。'
        merged = merge_punctuation(words, full_text)
        assert merged[0]["text"] == "等等……"
        assert merged[1]["text"] == "走吧。"

        # Giả lập dòng đã dài, dấu ellipsis sẽ gây cắt
        subs = segment_subtitles(merged, max_chars=5)
        assert len(subs) == 2
        assert subs[0][0]["text"] == "等等……"
        assert subs[1][0]["text"] == "走吧。"

    def test_case_23_multiple_spaces(self):
        """Case 23: Nhiều khoảng trắng giữa token phải được bảo toàn.

        Input: hello   world.
        Tokens: hello | world
        """
        words = to_mock_words(["hello", "world"])
        full_text = 'hello   world.'
        merged = merge_punctuation(words, full_text)
        assert merged[0]["text"] == "hello   "
        assert merged[1]["text"] == "world."

    def test_case_19_long_single_token(self):
        """Case 19: Token đơn lẻ dài hơn max_chars.

        Input: 一二三四五六七八九十一二三四五六七八九十 (20 ký tự)
        Tokens: [toàn bộ chuỗi]
        Expected: Không ép cắt token đơn lẻ vì sẽ làm sai lệch timestamp.
        """
        long_str = "一二三四五六七八九十一二三四五六七八九十"
        words = to_mock_words([long_str])
        full_text = long_str
        merged = merge_punctuation(words, full_text)

        subs = segment_subtitles(merged, max_chars=15)
        assert len(subs) == 1
        assert subs[0][0]["text"] == long_str

    def test_scenario_2_long_quote(self):
        """Kịch bản 2: Đoạn trích dẫn rất dài (> max_chars).

        Input: 他说：“你好！他来了。你知道吗？今天还要去吗？”
        Tokens: 他说 | 你好 | 他来了 | 你知道吗 | 今天还要去吗
        """
        words = to_mock_words(["他说", "你好", "他来了", "你知道吗", "今天还要去吗"])
        full_text = '他说：“你好！他来了。你知道吗？今天还要去吗？”'
        merged = merge_punctuation(words, full_text)

        # 1. Kiểm tra Prefix/Suffix đã gắn đúng chưa
        assert merged[0]["text"] == "他说："
        assert merged[1]["text"] == "“你好！"
        assert merged[2]["text"] == "他来了。"
        assert merged[3]["text"] == "你知道吗？"
        assert merged[4]["text"] == "今天还要去吗？”"

        # 2. Kiểm tra cắt câu với max_chars = 15
        subs = segment_subtitles(merged, max_chars=15)

        # Đảm bảo:
        # - Sub 1 cắt ngay sau dấu hai chấm.
        # - Các Sub tiếp theo cắt theo độ dài, không dòng nào quá 18 ký tự
        assert len(subs) >= 2
        assert "".join([w["text"] for w in subs[0]]) == "他说："

        for s in subs:
            text = "".join([w["text"] for w in s]).strip()
            # Yêu cầu dòng cắt thông minh không được vượt quá xa ngưỡng 15
            assert len(text) <= 18

    def test_case_36_empty_token(self):
        """Case 36: Token text rỗng không gây IndexError.

        Input: 你好。
        Tokens: chuỗi rỗng
        Expected: Không crash, trả về token rỗng.
        """
        words = to_mock_words([""])
        full_text = '你好。'
        merged = merge_punctuation(words, full_text)
        assert len(merged) == 1
        assert merged[0]["text"] == ""

    def test_case_43_trailing_text_after_last_token(self):
        """Case 43: Full text có hậu tố chữ sau token cuối.

        Input: 你好世界
        Tokens: 你好
        Expected: Vớt toàn bộ phần còn lại để tránh mất chữ.
        """
        words = to_mock_words(["你好"])
        full_text = '你好世界'
        merged = merge_punctuation(words, full_text)
        assert len(merged) == 1
        assert merged[0]["text"] == "你好世界"

    def test_case_45_misaligned_token(self):
        """Case 45: Token lặp nhưng token đầu không khớp hoàn toàn.

        Input: abc abc.
        Tokens: abx | abc
        Expected: Không kẹt con trỏ, token thứ 2 vẫn nhận được dấu câu.
        """
        words = to_mock_words(["abx", "abc"])
        full_text = 'abc abc.'
        merged = merge_punctuation(words, full_text)
        # Token đầu không khớp → con trỏ vẫn tiến về phía trước
        # Token thứ 2 match 'abc' và nhận dấu '.'
        assert merged[1]["text"] == "abc."
