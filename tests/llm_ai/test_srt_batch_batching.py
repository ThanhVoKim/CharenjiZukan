#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/llm_ai/test_srt_batch_batching.py
=======================================
Test cho llm_ai/srt_batch/batching.py — parser numbered-line + map theo line id.

Định dạng I/O batch (sau refactor): mỗi block một dòng `"N. text"`, N = `line`
(chỉ số SRT toàn cục). `merge_translated_batch` parse output LLM theo SỐ và map vào
batch gốc theo line id (không theo vị trí) → chống lệch hàng.

Cấu trúc layers:
  Layer 1 — Unit: parse_numbered_lines + merge_translated_batch (thuần Python, không I/O).

Cách chạy:
    pytest tests/llm_ai/test_srt_batch_batching.py -v -k "Layer1"
"""

import sys
from pathlib import Path

import pytest

# ── Project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm_ai.srt_batch.batching import (  # noqa: E402
    BatchIntegrityError,
    merge_translated_batch,
    parse_numbered_lines,
)


# ═════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════

def _batch(*lines: int) -> list[dict]:
    """Dựng original_batch tối thiểu với line id cho trước (text/time giả)."""
    return [
        {"line": n, "time": "00:00:00,000 --> 00:00:01,000", "text": f"src{n}"}
        for n in lines
    ]


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT: parse_numbered_lines
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_ParseNumberedLines:
    def test_basic_parse(self):
        assert parse_numbered_lines("1. hello\n2. world") == {1: "hello", 2: "world"}

    def test_strips_whitespace_and_optional_space_after_dot(self):
        # Khoảng trắng đầu dòng + không có space sau dấu chấm vẫn parse được.
        assert parse_numbered_lines("  1.hello \n2.  world ") == {1: "hello", 2: "world"}

    def test_non_consecutive_ids(self):
        # Line id là chỉ số SRT toàn cục — không nhất thiết bắt đầu từ 1.
        assert parse_numbered_lines("17. a\n18. b") == {17: "a", 18: "b"}

    def test_orphan_line_appended_to_current_entry(self):
        # Dòng không có prefix số → nối vào entry đang mở (text bị LLM xuống dòng).
        parsed = parse_numbered_lines("1. dòng một\nnối tiếp\n2. dòng hai")
        assert parsed == {1: "dòng một nối tiếp", 2: "dòng hai"}

    def test_blank_lines_ignored(self):
        assert parse_numbered_lines("1. a\n\n\n2. b\n") == {1: "a", 2: "b"}

    def test_digit_inside_text_not_treated_as_prefix(self):
        # "50万" giữa câu không phải prefix id → không phá map.
        assert parse_numbered_lines("1. 賞金は50万ドルです") == {1: "賞金は50万ドルです"}

    def test_duplicate_id_raises(self):
        with pytest.raises(BatchIntegrityError):
            parse_numbered_lines("1. a\n1. b")


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT: merge_translated_batch (map theo line id)
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_MergeTranslatedBatch:
    def test_maps_text_by_line_id(self):
        batch = _batch(1, 2)
        merged = merge_translated_batch("1. dịch một\n2. dịch hai", batch)
        assert [b["text"] for b in merged] == ["dịch một", "dịch hai"]

    def test_preserves_timestamp_and_line(self):
        batch = _batch(5, 6)
        merged = merge_translated_batch("5. a\n6. b", batch)
        assert merged[0]["line"] == 5
        assert merged[0]["time"] == "00:00:00,000 --> 00:00:01,000"

    def test_does_not_mutate_original_batch(self):
        batch = _batch(1, 2)
        merge_translated_batch("1. x\n2. y", batch)
        assert [b["text"] for b in batch] == ["src1", "src2"]

    def test_order_independent_mapping(self):
        # Output đảo thứ tự dòng vẫn map đúng theo SỐ, không theo vị trí.
        batch = _batch(1, 2, 3)
        merged = merge_translated_batch("3. ba\n1. một\n2. hai", batch)
        assert [b["text"] for b in merged] == ["một", "hai", "ba"]

    def test_missing_id_raises(self):
        batch = _batch(1, 2)
        with pytest.raises(BatchIntegrityError, match="thiếu"):
            merge_translated_batch("1. chỉ một", batch)

    def test_extra_id_raises(self):
        batch = _batch(1, 2)
        with pytest.raises(BatchIntegrityError, match="thừa"):
            merge_translated_batch("1. a\n2. b\n3. thừa", batch)

    def test_non_consecutive_batch_ids(self):
        # Batch giữa file (line 17,18) — map theo đúng id, không reset về 1.
        batch = _batch(17, 18)
        merged = merge_translated_batch("17. a\n18. b", batch)
        assert [b["text"] for b in merged] == ["a", "b"]
