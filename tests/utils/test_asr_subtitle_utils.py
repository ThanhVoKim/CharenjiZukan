#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/utils/test_asr_subtitle_utils.py
======================================
Test Layer 1: Unit tests cho utils/asr_subtitle_utils.py

Logic thuần Python, không cần GPU/FFmpeg/file media.

Cách chạy:
    pytest tests/utils/test_asr_subtitle_utils.py -v -k "Layer1"
"""

import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.asr_subtitle_utils import (
    format_srt_time,
    merge_punctuation,
    segment_words_to_subtitles,
    write_subtitle_srt,
)


# ═════════════════════════════════════════════════════════════════════
# Helper: tạo word object giả lập ForcedAlignItem
# ═════════════════════════════════════════════════════════════════════

class _FakeWord:
    """Giả lập object có attribute .text, .start_time, .end_time."""
    def __init__(self, text, start_time, end_time):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_FormatSrtTime:
    """Test format_srt_time."""

    def test_zero(self):
        assert format_srt_time(0.0) == "00:00:00,000"

    def test_simple_seconds(self):
        assert format_srt_time(1.5) == "00:00:01,500"

    def test_minutes(self):
        assert format_srt_time(61.5) == "00:01:01,500"

    def test_hours(self):
        assert format_srt_time(3661.5) == "01:01:01,500"

    def test_milliseconds(self):
        assert format_srt_time(0.123) == "00:00:00,123"


class TestLayer1_MergePunctuation:
    """Test merge_punctuation — kế thừa logic từ test_qwen3_asr.py."""

    def test_simple_cjk(self):
        words = [_FakeWord("\u4f60", 0.0, 0.5), _FakeWord("\u597d", 0.5, 1.0)]
        full_text = "\u4f60\u597d"
        result = merge_punctuation(words, full_text)
        assert len(result) == 2
        assert result[0]["text"] == "\u4f60"
        assert result[1]["text"] == "\u597d"

    def test_trailing_punctuation(self):
        words = [_FakeWord("\u4f60", 0.0, 0.5)]
        full_text = "\u4f60\u3002"
        result = merge_punctuation(words, full_text)
        assert len(result) == 1
        assert result[0]["text"] == "\u4f60\u3002"

    def test_prefix_opening_quote(self):
        words = [_FakeWord("\u4f60\u597d", 0.0, 1.0)]
        full_text = "\u201c\u4f60\u597d\uff01\u201d"
        result = merge_punctuation(words, full_text)
        assert len(result) == 1
        assert result[0]["text"] == "\u201c\u4f60\u597d\uff01\u201d"

    def test_empty_token(self):
        words = [_FakeWord("", 0.0, 0.5), _FakeWord("abc", 0.5, 1.0)]
        full_text = "abc"
        result = merge_punctuation(words, full_text)
        assert len(result) == 2
        assert result[0]["text"] == ""
        assert result[1]["text"] == "abc"

    def test_dict_input(self):
        """merge_punctuation cũng hỗ trợ dict input."""
        words = [{"text": "hello", "start_time": 0.0, "end_time": 0.5}]
        full_text = "hello."
        result = merge_punctuation(words, full_text)
        assert len(result) == 1
        assert result[0]["text"] == "hello."

    def test_preserves_timestamps(self):
        words = [_FakeWord("test", 1.0, 2.5)]
        full_text = "test"
        result = merge_punctuation(words, full_text)
        assert result[0]["start_time"] == 1.0
        assert result[0]["end_time"] == 2.5

    @pytest.mark.parametrize(
        ("full_text", "token_texts", "expected_text"),
        [
            (
                "47-round pan magazine",
                ["47round", "round", "pan", "magazine"],
                "47-round pan magazine",
            ),
            (
                "large-capacity magazines",
                ["largecapacity", "capacity", "magazines"],
                "large-capacity magazines",
            ),
            (
                "continuous-fire weapons",
                ["continuousfire", "fire", "weapons"],
                "continuous-fire weapons",
            ),
            (
                "pre–war rifle",
                ["prewar", "war", "rifle"],
                "pre–war rifle",
            ),
            (
                "pre—war rifle",
                ["prewar", "war", "rifle"],
                "pre—war rifle",
            ),
        ],
    )
    def test_normalized_dash_compound_does_not_duplicate_suffix(
        self,
        full_text,
        token_texts,
        expected_text,
    ):
        """Aligner bỏ dash trong compound words không được tạo text lặp."""
        words = [
            _FakeWord(text, idx * 0.5, (idx + 1) * 0.5)
            for idx, text in enumerate(token_texts)
        ]

        result = merge_punctuation(words, full_text)
        joined_text = "".join(item["text"] for item in result)

        assert joined_text == expected_text
        assert "round-round" not in joined_text
        assert "capacity-capacity" not in joined_text
        assert "fire-fire" not in joined_text

    def test_ascii_hyphen_between_separate_tokens_keeps_existing_behavior(self):
        """Token split thường vẫn giữ dash ở token trước như legacy behavior."""
        words = [_FakeWord("360", 0.0, 0.5), _FakeWord("degree", 0.5, 1.0)]
        full_text = "360-degree"

        result = merge_punctuation(words, full_text)

        assert [item["text"] for item in result] == ["360-", "degree"]
        assert "".join(item["text"] for item in result) == full_text

    def test_underscore_between_separate_tokens_is_not_treated_as_dash_compound(self):
        """Underscore không nằm trong nhóm dash/hyphen cần normalize."""
        words = [_FakeWord("large", 0.0, 0.5), _FakeWord("capacity", 0.5, 1.0)]
        full_text = "large_capacity"

        result = merge_punctuation(words, full_text)

        assert [item["text"] for item in result] == ["large_", "capacity"]
        assert "".join(item["text"] for item in result) == full_text


class TestLayer1_SegmentWordsToSubtitles:
    """Test segment_words_to_subtitles — tuân thủ invariant max_chars."""

    def _make_words(self, texts):
        """Tạo list word dict từ list text."""
        t = 0.0
        words = []
        for text in texts:
            words.append({"text": text, "start_time": t, "end_time": t + 0.5})
            t += 0.5
        return words

    def test_empty_input(self):
        assert segment_words_to_subtitles([]) == []

    def test_max_chars_zero_returns_grammar_blocks(self):
        """max_chars=0 chỉ tắt giới hạn độ dài, vẫn chia theo dấu câu."""
        words = self._make_words(["Hello.", "World?", "Again"])
        result = segment_words_to_subtitles(words, max_chars=0)
        texts = ["".join(w["text"] for w in block) for block in result]
        assert texts == ["Hello.", "World?", "Again"]

    def test_max_chars_zero_without_punctuation_returns_single_grammar_block(self):
        """Không có dấu câu thì grammar-only không cắt cơ học theo độ dài."""
        words = self._make_words(["a"] * 100)
        result = segment_words_to_subtitles(words, max_chars=0)
        assert len(result) == 1

    def test_max_chars_zero_forwards_segmentation_params(self, monkeypatch):
        """Wrapper không hardcode min_chars/split_on_comma khi max_chars=0."""
        calls = {}

        def fake_smart_segment(tokens, min_chars, max_chars, ideal_chars, split_on_comma):
            calls["tokens"] = tokens
            calls["min_chars"] = min_chars
            calls["max_chars"] = max_chars
            calls["ideal_chars"] = ideal_chars
            calls["split_on_comma"] = split_on_comma
            return [tokens]

        monkeypatch.setattr("utils.text_segmenter.smart_segment", fake_smart_segment)
        words = self._make_words(["Hello.", "World?"])
        result = segment_words_to_subtitles(
            words,
            max_chars=0,
            min_chars=7,
            split_on_comma=False,
        )

        assert result == [words]
        assert calls["tokens"] is words
        assert calls["min_chars"] == 7
        assert calls["max_chars"] == 0
        assert calls["ideal_chars"] == 0
        assert calls["split_on_comma"] is False

    def test_max_chars_negative_returns_single_block_for_legacy_disable(self):
        """max_chars âm giữ đường tắt single-block để không phá caller cũ."""
        words = self._make_words(["Hello.", "World?", "Again!"])
        result = segment_words_to_subtitles(words, max_chars=-1)
        assert len(result) == 1
        assert result[0] == words

    def test_total_under_max_chars_no_split(self):
        """Invariant: nếu tổng ký tự <= max_chars, không split."""
        words = self._make_words(["hello", " world"])
        result = segment_words_to_subtitles(words, max_chars=42)
        assert len(result) == 1, f"Expected 1 block but got {len(result)}"

    def test_total_exactly_max_chars_no_split(self):
        """Invariant: tổng == max_chars cũng không split."""
        words = self._make_words(["a"] * 42)
        result = segment_words_to_subtitles(words, max_chars=42)
        assert len(result) == 1

    def test_total_over_max_chars_does_split(self):
        """Khi vượt max_chars, phải split."""
        words = self._make_words(["a"] * 100)
        result = segment_words_to_subtitles(words, max_chars=42)
        assert len(result) > 1

    def test_min_chars_zero_allows_short_blocks(self):
        """min_chars=0 không ép gộp subtitle ngắn."""
        words = self._make_words(["a", "b", "c"])
        # max_chars rất nhỏ để ép split
        result = segment_words_to_subtitles(words, max_chars=1, min_chars=0)
        # Mỗi word là 1 ký tự, max_chars=1, nên mỗi block có 1 word
        assert all(len(block) >= 1 for block in result)

    def test_sentence_shorter_than_max_not_split(self):
        """Câu ngắn hơn max_chars không bị ngắt dù có dấu phẩy."""
        words = [
            {"text": "Hello, world!", "start_time": 0.0, "end_time": 1.0},
        ]
        result = segment_words_to_subtitles(words, max_chars=42, split_on_comma=True)
        assert len(result) == 1, "Câu ngắn hơn max_chars không được split"


class TestLayer1_WriteSubtitleSrt:
    """Test write_subtitle_srt."""

    def test_writes_valid_srt(self, tmp_path):
        blocks = [
            [
                {"text": "Hello", "start_time": 0.0, "end_time": 1.0},
                {"text": " world", "start_time": 1.0, "end_time": 2.0},
            ],
            [
                {"text": "Bye", "start_time": 3.0, "end_time": 4.0},
            ],
        ]
        output = str(tmp_path / "test.srt")
        write_subtitle_srt(blocks, output, offset_seconds=0.0)

        content = Path(output).read_text(encoding="utf-8")
        assert "1\n" in content
        assert "00:00:00,000 --> 00:00:02,000" in content
        assert "Hello world" in content
        assert "2\n" in content
        assert "00:00:03,000 --> 00:00:04,000" in content
        assert "Bye" in content

    def test_offset_applied(self, tmp_path):
        blocks = [
            [
                {"text": "Test", "start_time": 1.0, "end_time": 2.0},
            ],
        ]
        output = str(tmp_path / "test_offset.srt")
        write_subtitle_srt(blocks, output, offset_seconds=0.24)

        content = Path(output).read_text(encoding="utf-8")
        # start = 1.0 + 0.24 = 1.24
        assert "00:00:01,240" in content
        # end = 2.0 + 0.24 = 2.24
        assert "00:00:02,240" in content

    def test_empty_blocks_skipped(self, tmp_path):
        blocks = [[], [{"text": "OK", "start_time": 0.0, "end_time": 1.0}]]
        output = str(tmp_path / "test_empty.srt")
        write_subtitle_srt(blocks, output, offset_seconds=0.0)

        content = Path(output).read_text(encoding="utf-8")
        # Block rỗng bị skip, block OK thành entry số 1
        assert "1\n" in content
        assert "OK" in content
