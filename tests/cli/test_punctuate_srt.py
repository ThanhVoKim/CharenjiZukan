#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/cli/test_punctuate_srt.py
===============================
Test cho cli/punctuate_srt.py — CLI standalone phục hồi dấu câu SRT (LLM, batch).

Cấu trúc layers:
  Layer 1 — Unit: argument parser, default task-config, resolve_cli_tasks ánh xạ
            output `_punct.srt`, validate đuôi .srt; validator _content_signature;
            flatten_srt_to_text.
  Layer 2 — Component: restore_punctuation_srt với provider MOCK (không gọi API);
            anti-hallucination (đổi chữ / sai block count → giữ gốc).
  Layer 3 — Pipeline: chạy main() với provider MOCK (monkeypatch create_task_provider),
            không gọi API thật; xác nhận sinh _punct.srt + _flat.txt, batch fail giữ gốc.

Cách chạy:
    pytest tests/cli/test_punctuate_srt.py -v -k "Layer1"
    pytest tests/cli/test_punctuate_srt.py -v -k "Layer2"
    pytest tests/cli/test_punctuate_srt.py -v -k "Layer3"
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cli.punctuate_srt as punctuate_srt


# ═════════════════════════════════════════════════════════════════════
# Mock provider — trả response cố định, đếm số lần call
# ═════════════════════════════════════════════════════════════════════

class _MockProvider:
    def __init__(self, response: str, name: str = "mock"):
        self.name = name
        self._response = response
        self.calls = 0
        self._retry_attempts = 2
        self._retry_wait_seconds = 0

    def set_global_context(self, context: str) -> bool:
        return False

    def call(self, prompt: str) -> str:
        self.calls += 1
        return self._response


_TS1 = "00:00:01,000 --> 00:00:02,000"
_TS2 = "00:00:03,000 --> 00:00:04,000"

PROMPT_FILE = str(PROJECT_ROOT / "prompts/llm_tasks/punctuation_restoration.txt")


def _write_srt(path: Path, blocks: list[tuple[str, str]]) -> Path:
    body = "\n\n".join(f"{i+1}\n{ts}\n{txt}" for i, (ts, txt) in enumerate(blocks))
    path.write_text(body + "\n", encoding="utf-8")
    return path


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT: core helpers (gộp từ tests/punctuation/ cũ)
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_ContentSignature:
    def test_adding_punctuation_preserves_signature_cjk(self):
        assert punctuate_srt._content_signature("你好世界") == punctuate_srt._content_signature("你好，世界。")

    def test_adding_punctuation_preserves_signature_latin(self):
        assert punctuate_srt._content_signature("I went to") == punctuate_srt._content_signature("I went, to.")

    def test_changed_char_differs(self):
        assert punctuate_srt._content_signature("你好世界") != punctuate_srt._content_signature("你好星界")

    def test_whitespace_ignored(self):
        assert punctuate_srt._content_signature("a b c") == punctuate_srt._content_signature("abc")


class TestLayer1_Flatten:
    def test_cjk_joined_without_space(self, tmp_path):
        srt = "1\n00:00:01,000 --> 00:00:02,000\n你好世界，\n\n2\n00:00:03,000 --> 00:00:04,000\n这是测试。\n"
        i = tmp_path / "a.srt"; i.write_text(srt, encoding="utf-8")
        o = tmp_path / "a.txt"
        out = punctuate_srt.flatten_srt_to_text(str(i), str(o))
        assert out == "你好世界，这是测试。"
        assert "\n" not in o.read_text(encoding="utf-8")

    def test_latin_joined_with_space(self, tmp_path):
        srt = "1\n00:00:01,000 --> 00:00:02,000\nI went\n\n2\n00:00:03,000 --> 00:00:04,000\nto the store.\n"
        i = tmp_path / "a.srt"; i.write_text(srt, encoding="utf-8")
        o = tmp_path / "a.txt"
        out = punctuate_srt.flatten_srt_to_text(str(i), str(o))
        assert out == "I went to the store."


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — COMPONENT: restore_punctuation_srt (mock provider, no API)
# ═════════════════════════════════════════════════════════════════════

class TestLayer2_RestorePunctuation:
    def test_success_adds_punctuation(self, tmp_path):
        src = _write_srt(tmp_path / "in.srt", [(_TS1, "你好世界"), (_TS2, "这是测试")])
        response = (
            "<PUNCT_TEXT>\n"
            "1. 你好世界，\n"
            "2. 这是测试。\n"
            "</PUNCT_TEXT>"
        )
        provider = _MockProvider(response)
        out = tmp_path / "out_punct.srt"

        stats = punctuate_srt.restore_punctuation_srt(
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
        src = _write_srt(tmp_path / "in.srt", [(_TS1, "你好世界"), (_TS2, "这是测试")])
        response = (
            "<PUNCT_TEXT>\n"
            "1. 你好星界，\n"   # 世 → 星 (đổi chữ — không hợp lệ)
            "2. 这是测试。\n"
            "</PUNCT_TEXT>"
        )
        provider = _MockProvider(response)
        out = tmp_path / "out_punct.srt"

        stats = punctuate_srt.restore_punctuation_srt(
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
        """LLM trả thiếu line (thiếu id) → BatchIntegrityError → revert."""
        src = _write_srt(tmp_path / "in.srt", [(_TS1, "你好世界"), (_TS2, "这是测试")])
        response = "<PUNCT_TEXT>\n1. 你好世界。\n</PUNCT_TEXT>"  # chỉ 1 line (thiếu id 2)
        provider = _MockProvider(response)
        out = tmp_path / "out_punct.srt"

        stats = punctuate_srt.restore_punctuation_srt(
            input_srt=str(src), output_srt=str(out),
            prompt_file=PROMPT_FILE, provider=provider,
            language="Chinese", batch_size=30, use_full_context=False,
        )
        assert stats["failed"] == 1
        assert stats["reverted_blocks"] == 2


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT: CLI arg parser
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_ArgParser:
    def test_default_task_config_points_to_punctuation_yaml(self):
        parser = punctuate_srt.build_parser()
        args = parser.parse_args(["--input", "a.srt"])
        assert args.task_config.endswith("punctuation_restoration.yaml")

    def test_flatten_default_on(self):
        parser = punctuate_srt.build_parser()
        args = parser.parse_args(["--input", "a.srt"])
        assert args.flatten is True
        args2 = parser.parse_args(["--input", "a.srt", "--no-flatten"])
        assert args2.flatten is False

    def test_context_tristate_default_none(self):
        parser = punctuate_srt.build_parser()
        assert parser.parse_args(["-i", "a.srt"]).context is None
        assert parser.parse_args(["-i", "a.srt", "--no-context"]).context is False
        assert parser.parse_args(["-i", "a.srt", "--context"]).context is True

    def test_resolve_by_priority(self):
        # CLI thắng task config; task config thắng default.
        assert punctuate_srt._resolve_by_priority(50, {"batch_size": 30}, ["batch_size"], 30) == 50
        assert punctuate_srt._resolve_by_priority(None, {"batch_size": 30}, ["batch", "batch_size"], 99) == 30
        assert punctuate_srt._resolve_by_priority(None, {}, ["batch_size"], 99) == 99

    def test_output_maps_to_punct_srt(self, tmp_path):
        from utils.task_utils import resolve_cli_tasks
        src = _write_srt(tmp_path / "ocr.srt", [(_TS1, "你好世界")])
        tasks = resolve_cli_tasks(
            task_file=None, input_file=str(src), output_path=None, default_ext="_punct.srt",
        )
        assert tasks[0]["output"].endswith("ocr_punct.srt")


# ═════════════════════════════════════════════════════════════════════
# LAYER 3 — PIPELINE (mock provider, no API)
# ═════════════════════════════════════════════════════════════════════

class TestLayer3_Pipeline:
    def _run_main(self, monkeypatch, argv: list[str], provider: _MockProvider):
        monkeypatch.setattr(punctuate_srt, "create_task_provider", lambda args, cfg: provider)
        monkeypatch.setattr(sys, "argv", ["punctuate-srt"] + argv)
        with pytest.raises(SystemExit) as exc:
            punctuate_srt.main()
        return exc.value.code

    def test_success_writes_punct_and_flat(self, tmp_path, monkeypatch):
        src = _write_srt(tmp_path / "ocr.srt", [(_TS1, "你好世界"), (_TS2, "这是测试")])
        response = (
            "<PUNCT_TEXT>\n"
            "1. 你好世界，\n"
            "2. 这是测试。\n"
            "</PUNCT_TEXT>"
        )
        provider = _MockProvider(response)
        out = tmp_path / "ocr_punct.srt"

        code = self._run_main(
            monkeypatch,
            ["--input", str(src), "--output", str(out), "--no-context", "--lang", "Chinese"],
            provider,
        )
        assert code == 0
        assert out.exists()
        flat = tmp_path / "ocr_punct_flat.txt"
        assert flat.exists()
        assert flat.read_text(encoding="utf-8") == "你好世界，这是测试。"
        assert "你好世界，" in out.read_text(encoding="utf-8")

    def test_no_flatten_skips_flat_txt(self, tmp_path, monkeypatch):
        src = _write_srt(tmp_path / "ocr.srt", [(_TS1, "你好世界")])
        response = "<PUNCT_TEXT>\n1. 你好世界。\n</PUNCT_TEXT>"
        provider = _MockProvider(response)
        out = tmp_path / "ocr_punct.srt"

        code = self._run_main(
            monkeypatch,
            ["--input", str(src), "--output", str(out), "--no-context", "--no-flatten"],
            provider,
        )
        assert code == 0
        assert out.exists()
        assert not (tmp_path / "ocr_punct_flat.txt").exists()

    def test_changed_char_reverts_to_original(self, tmp_path, monkeypatch):
        src = _write_srt(tmp_path / "ocr.srt", [(_TS1, "你好世界"), (_TS2, "这是测试")])
        response = (
            "<PUNCT_TEXT>\n"
            "1. 你好星界，\n"   # 世 → 星 (đổi chữ)
            "2. 这是测试。\n"
            "</PUNCT_TEXT>"
        )
        provider = _MockProvider(response)
        out = tmp_path / "ocr_punct.srt"

        code = self._run_main(
            monkeypatch,
            ["--input", str(src), "--output", str(out), "--no-context"],
            provider,
        )
        # Toàn bộ batch revert → vẫn coi là 1 task hoàn tất (exit 0), nhưng giữ text gốc.
        assert code == 0
        content = out.read_text(encoding="utf-8")
        assert "你好世界" in content
        assert "你好星界" not in content

    def test_non_srt_input_errors(self, tmp_path, monkeypatch):
        bad = tmp_path / "ocr.txt"
        bad.write_text("không phải srt", encoding="utf-8")
        provider = _MockProvider("<PUNCT_TEXT></PUNCT_TEXT>")
        # parser.error → SystemExit(2)
        code = self._run_main(monkeypatch, ["--input", str(bad)], provider)
        assert code == 2
