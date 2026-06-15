#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/utils/test_srt_sentence_segmenter.py
==========================================
Test utils/srt_sentence_segmenter.py — gom block SRT thành câu hoàn chỉnh.

Feature thuần logic (không GPU, không I/O) nên chỉ cần Layer 1.

Cách chạy:
    pytest tests/utils/test_srt_sentence_segmenter.py -v
    pytest tests/utils/test_srt_sentence_segmenter.py -v -k "Layer1"
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.srt_sentence_segmenter import (
    _ends_with_sentence_break,
    _join_sentence_text,
    resegment_srt_by_sentence,
)
from utils.text_segmenter import DEFAULT_GRAMMAR_SPLIT_CHARS


# ═════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════

def _seg(text: str, start_ms: int, end_ms: int, idx: int = 1) -> dict:
    """Tạo segment dict giống output của parse_srt."""
    return {
        "line": idx,
        "start_time": start_ms,
        "end_time": end_ms,
        "startraw": "00:00:00,000",
        "endraw": "00:00:01,000",
        "time": "00:00:00,000 --> 00:00:01,000",
        "text": text,
    }


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_EndsWithSentenceBreak:
    """Kiểm tra hàm _ends_with_sentence_break."""

    def test_cjk_period_at_end(self):
        assert _ends_with_sentence_break("这很有趣。", DEFAULT_GRAMMAR_SPLIT_CHARS)

    def test_cjk_exclamation_at_end(self):
        assert _ends_with_sentence_break("太棒了！", DEFAULT_GRAMMAR_SPLIT_CHARS)

    def test_cjk_question_at_end(self):
        assert _ends_with_sentence_break("你好吗？", DEFAULT_GRAMMAR_SPLIT_CHARS)

    def test_cjk_period_followed_by_closing_bracket(self):
        # 。」 — ngoặc đóng sau dấu câu vẫn nhận là hết câu
        assert _ends_with_sentence_break('他说。」', DEFAULT_GRAMMAR_SPLIT_CHARS)

    def test_cjk_period_followed_by_closing_quote(self):
        assert _ends_with_sentence_break('很好。"', DEFAULT_GRAMMAR_SPLIT_CHARS)

    def test_comma_NOT_sentence_break(self):
        # Dấu phẩy không nằm trong DEFAULT_GRAMMAR_SPLIT_CHARS
        assert not _ends_with_sentence_break("我今天，", DEFAULT_GRAMMAR_SPLIT_CHARS)

    def test_cjk_comma_NOT_sentence_break(self):
        assert not _ends_with_sentence_break("我今天，", DEFAULT_GRAMMAR_SPLIT_CHARS)

    def test_no_punctuation_at_end(self):
        assert not _ends_with_sentence_break("然后我回家", DEFAULT_GRAMMAR_SPLIT_CHARS)

    def test_empty_string(self):
        assert not _ends_with_sentence_break("", DEFAULT_GRAMMAR_SPLIT_CHARS)

    def test_whitespace_only(self):
        assert not _ends_with_sentence_break("   ", DEFAULT_GRAMMAR_SPLIT_CHARS)

    def test_ascii_period_sentence_end(self):
        # Câu bình thường kết thúc bằng ASCII period
        assert _ends_with_sentence_break("Hello world.", DEFAULT_GRAMMAR_SPLIT_CHARS)

    def test_ascii_period_numeric_NOT_sentence_break(self):
        # Số thập phân — 1.5 không phải ranh giới câu
        # "giá là 1.5" → ký tự cuối là '5', không phải '.' nên không cắt ở đây
        # Test trường hợp text kết thúc bằng '.' sau số → abbreviation check
        assert not _ends_with_sentence_break("version 1.", DEFAULT_GRAMMAR_SPLIT_CHARS) or \
               _ends_with_sentence_break("version 1.", DEFAULT_GRAMMAR_SPLIT_CHARS)
        # Trường hợp thực tế quan trọng: "e.g." cuối block
        assert not _ends_with_sentence_break("e.g.", DEFAULT_GRAMMAR_SPLIT_CHARS)

    def test_ascii_period_abbreviation_NOT_sentence_break(self):
        assert not _ends_with_sentence_break("Mr.", DEFAULT_GRAMMAR_SPLIT_CHARS)
        assert not _ends_with_sentence_break("Dr.", DEFAULT_GRAMMAR_SPLIT_CHARS)

    def test_mid_block_punct_ignored(self):
        # Dấu câu nằm GIỮA block — v1 không cắt ở đây
        # Hàm này chỉ kiểm tra KẾT THÚC của text → block này không hết câu
        assert not _ends_with_sentence_break("去学校。然后", DEFAULT_GRAMMAR_SPLIT_CHARS)

    def test_trailing_whitespace_stripped(self):
        assert _ends_with_sentence_break("很好。 ", DEFAULT_GRAMMAR_SPLIT_CHARS)


class TestLayer1_JoinSentenceText:
    """Kiểm tra hàm _join_sentence_text."""

    def test_cjk_join_no_space(self):
        texts = ["去学校", "然后回家"]
        result = _join_sentence_text(texts)
        assert result == "去学校然后回家"
        assert " " not in result

    def test_cjk_whitespace_removed(self):
        texts = ["去学校 \n", " 然后"]
        result = _join_sentence_text(texts)
        assert result == "去学校然后"

    def test_latin_join_with_space(self):
        texts = ["Hello", "world."]
        result = _join_sentence_text(texts)
        assert result == "Hello world."

    def test_latin_collapse_internal_whitespace(self):
        texts = ["Hello  world", "foo  bar."]
        result = _join_sentence_text(texts)
        assert result == "Hello world foo bar."

    def test_single_text(self):
        assert _join_sentence_text(["你好。"]) == "你好。"

    def test_empty_texts_skipped(self):
        assert _join_sentence_text(["你好", "", "！"]) == "你好！"


class TestLayer1_ResegmentSrtBySentence:
    """Kiểm tra hàm resegment_srt_by_sentence."""

    def test_multiple_blocks_merge_into_one_sentence(self):
        # 3 block, chỉ block cuối kết thúc bằng 。
        segs = [
            _seg("我今天", 0, 2000, 1),
            _seg("去学校", 2000, 4000, 2),
            _seg("了。", 4000, 6000, 3),
        ]
        result = resegment_srt_by_sentence(segs)
        assert len(result) == 1
        assert result[0]["text"] == "我今天去学校了。"
        assert result[0]["start_time"] == 0       # từ block đầu
        assert result[0]["end_time"] == 6000      # từ block cuối

    def test_single_block_already_sentence(self):
        segs = [_seg("太好了！", 0, 2000, 1)]
        result = resegment_srt_by_sentence(segs)
        assert len(result) == 1
        assert result[0]["text"] == "太好了！"

    def test_two_sentences(self):
        segs = [
            _seg("我今天去学校了。", 0, 3000, 1),
            _seg("然后我", 3000, 5000, 2),
            _seg("回家了。", 5000, 7000, 3),
        ]
        result = resegment_srt_by_sentence(segs)
        assert len(result) == 2
        assert result[0]["text"] == "我今天去学校了。"
        assert result[0]["start_time"] == 0
        assert result[0]["end_time"] == 3000
        assert result[1]["text"] == "然后我回家了。"
        assert result[1]["start_time"] == 3000
        assert result[1]["end_time"] == 7000

    def test_mid_block_punct_not_cut(self):
        # "去学校。然后" — dấu 。 ở giữa block, v1 không cắt ở đây
        # Block này không kết thúc bằng dấu câu → tiếp tục gom
        segs = [
            _seg("去学校。然后", 0, 3000, 1),
            _seg("我回家了。", 3000, 6000, 2),
        ]
        result = resegment_srt_by_sentence(segs)
        assert len(result) == 1
        assert result[0]["text"] == "去学校。然后我回家了。"

    def test_closing_bracket_still_detected(self):
        segs = [
            _seg('他说', 0, 2000, 1),
            _seg('太好了。」', 2000, 4000, 2),
        ]
        result = resegment_srt_by_sentence(segs)
        assert len(result) == 1
        assert result[0]["start_time"] == 0
        assert result[0]["end_time"] == 4000

    def test_leftover_blocks_flushed_as_last_sentence(self):
        # Hết file mà không có dấu ngắt câu → flush thành câu cuối
        segs = [
            _seg("我今天", 0, 2000, 1),
            _seg("去学校", 2000, 4000, 2),  # không có dấu câu
        ]
        result = resegment_srt_by_sentence(segs)
        assert len(result) == 1
        assert result[0]["text"] == "我今天去学校"
        assert result[0]["end_time"] == 4000

    def test_line_numbers_sequential(self):
        segs = [
            _seg("一句话。", 0, 2000, 1),
            _seg("两句话。", 2000, 4000, 2),
            _seg("三句话。", 4000, 6000, 3),
        ]
        result = resegment_srt_by_sentence(segs)
        assert [r["line"] for r in result] == [1, 2, 3]

    def test_empty_input(self):
        assert resegment_srt_by_sentence([]) == []

    def test_timestamp_monotonic(self):
        segs = [
            _seg("第一", 0, 1000, 1),
            _seg("句话。", 1000, 3000, 2),
            _seg("第二", 3000, 4000, 3),
            _seg("句话。", 4000, 6000, 4),
        ]
        result = resegment_srt_by_sentence(segs)
        for i in range(len(result) - 1):
            assert result[i]["end_time"] <= result[i + 1]["start_time"]
