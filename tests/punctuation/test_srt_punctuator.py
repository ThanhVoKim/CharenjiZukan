#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/punctuation/test_srt_punctuator.py
========================================
Test cho punctuation/srt_punctuator.py — phục hồi dấu câu SRT bằng LLM (batch).

Cấu trúc layers:
  Layer 1 — Unit: validator _content_signature, flatten_srt_to_text.
  Layer 2 — Component: restore_punctuation_srt với provider MOCK (không gọi API).

Cách chạy:
    pytest tests/punctuation/test_srt_punctuator.py -v -k "Layer1"
    pytest tests/punctuation/test_srt_punctuator.py -v -k "Layer2"
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from punctuation.srt_punctuator import (
    _content_signature,
    flatten_srt_to_text,
    restore_punctuation_srt,
)

PROMPT_FILE = str(PROJECT_ROOT / "prompts/llm_tasks/punctuation_restoration.txt")


# ═════════════════════════════════════════════════════════════════════
# Mock provider
# ═════════════════════════════════════════════════════════════════════

class _MockProvider:
    """Provider giả: trả về response cố định, đếm số lần call."""

    def __init__(self, response: str, name: str = "mock"):
        self.name = name
        self._response = response
        self.calls = 0
        self._retry_attempts = 2
        self._retry_wait_seconds = 0

    def set_global_context(self, context: str) -> bool:
        return False  # không cache → context gửi inline

    def call(self, prompt: str) -> str:
        self.calls += 1
        return self._response


def _write_srt(tmp_path: Path, blocks: list[tuple[str, str, str]], name="in.srt") -> Path:
    body = "\n\n".join(f"{i+1}\n{ts}\n{txt}" for i, (ts, txt, _) in enumerate(blocks))
    p = tmp_path / name
    p.write_text(body + "\n", encoding="utf-8")
    return p


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_ContentSignature:
    def test_adding_punctuation_preserves_signature_cjk(self):
        assert _content_signature("你好世界") == _content_signature("你好，世界。")

    def test_adding_punctuation_preserves_signature_latin(self):
        assert _content_signature("I went to") == _content_signature("I went, to.")

    def test_changed_char_differs(self):
        assert _content_signature("你好世界") != _content_signature("你好星界")

    def test_whitespace_ignored(self):
        assert _content_signature("a b c") == _content_signature("abc")


class TestLayer1_Flatten:
    def test_cjk_joined_without_space(self, tmp_path):
        srt = "1\n00:00:01,000 --> 00:00:02,000\n你好世界，\n\n2\n00:00:03,000 --> 00:00:04,000\n这是测试。\n"
        i = tmp_path / "a.srt"; i.write_text(srt, encoding="utf-8")
        o = tmp_path / "a.txt"
        out = flatten_srt_to_text(str(i), str(o))
        assert out == "你好世界，这是测试。"
        assert "\n" not in o.read_text(encoding="utf-8")

    def test_latin_joined_with_space(self, tmp_path):
        srt = "1\n00:00:01,000 --> 00:00:02,000\nI went\n\n2\n00:00:03,000 --> 00:00:04,000\nto the store.\n"
        i = tmp_path / "a.srt"; i.write_text(srt, encoding="utf-8")
        o = tmp_path / "a.txt"
        out = flatten_srt_to_text(str(i), str(o))
        assert out == "I went to the store."


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — COMPONENT (mock provider, no API)
# ═════════════════════════════════════════════════════════════════════

class TestLayer2_RestorePunctuation:
    _TS1 = "00:00:01,000 --> 00:00:02,000"
    _TS2 = "00:00:03,000 --> 00:00:04,000"

    def test_success_adds_punctuation(self, tmp_path):
        src = _write_srt(tmp_path, [(self._TS1, "你好世界", ""), (self._TS2, "这是测试", "")])
        response = (
            "<PUNCT_TEXT>\n"
            f"1\n{self._TS1}\n你好世界，\n\n"
            f"2\n{self._TS2}\n这是测试。\n"
            "</PUNCT_TEXT>"
        )
        provider = _MockProvider(response)
        out = tmp_path / "out_punct.srt"

        stats = restore_punctuation_srt(
            input_srt=str(src), output_srt=str(out),
            prompt_file=PROMPT_FILE, provider=provider,
            language="Chinese", batch_size=30, use_full_context=False,
        )
        assert stats["success"] == 1
        assert stats["failed"] == 0
        assert stats["reverted_blocks"] == 0
        content = out.read_text(encoding="utf-8")
        assert "你好世界，" in content
        assert "这是测试。" in content

    def test_changed_char_reverts_to_original(self, tmp_path):
        """LLM đổi chữ (世→星) → validator chặn, retry hết → giữ nguyên text gốc."""
        src = _write_srt(tmp_path, [(self._TS1, "你好世界", ""), (self._TS2, "这是测试", "")])
        response = (
            "<PUNCT_TEXT>\n"
            f"1\n{self._TS1}\n你好星界，\n\n"   # 世 → 星 (đổi chữ — không hợp lệ)
            f"2\n{self._TS2}\n这是测试。\n"
            "</PUNCT_TEXT>"
        )
        provider = _MockProvider(response)
        out = tmp_path / "out_punct.srt"

        stats = restore_punctuation_srt(
            input_srt=str(src), output_srt=str(out),
            prompt_file=PROMPT_FILE, provider=provider,
            language="Chinese", batch_size=30, use_full_context=False,
        )
        assert stats["failed"] == 1
        assert stats["reverted_blocks"] == 2
        assert provider.calls == provider._retry_attempts  # đã retry
        content = out.read_text(encoding="utf-8")
        assert "你好世界" in content       # giữ nguyên text gốc
        assert "你好星界" not in content    # KHÔNG nhận text bị đổi

    def test_block_count_mismatch_reverts(self, tmp_path):
        """LLM trả thiếu block → BatchIntegrityError → revert."""
        src = _write_srt(tmp_path, [(self._TS1, "你好世界", ""), (self._TS2, "这是测试", "")])
        response = f"<PUNCT_TEXT>\n1\n{self._TS1}\n你好世界。\n</PUNCT_TEXT>"  # chỉ 1 block
        provider = _MockProvider(response)
        out = tmp_path / "out_punct.srt"

        stats = restore_punctuation_srt(
            input_srt=str(src), output_srt=str(out),
            prompt_file=PROMPT_FILE, provider=provider,
            language="Chinese", batch_size=30, use_full_context=False,
        )
        assert stats["failed"] == 1
        assert stats["reverted_blocks"] == 2
