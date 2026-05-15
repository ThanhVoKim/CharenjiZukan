#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/llm_ai/test_generic_text_task.py
======================================
Test generic LLM task feature.

Cấu trúc layers:
  Layer 1 — Unit Tests          (prompt template, response parser)
  Layer 2 — Component Tests     (generic task with mocked provider)
  Layer 3 — Integration         (không cần trong file này)
  Layer 4 — Real API Tests      (không cần trong file này)

Cách chạy:
    pytest tests/llm_ai/test_generic_text_task.py -v -k "Layer1"
    pytest tests/llm_ai/test_generic_text_task.py -v -k "Layer2"
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_ai.base import BaseLLMProvider
from llm_ai.tasks.generic_text_task import GenericTextTaskConfig, run_generic_text_task
from llm_ai.tasks.prompt_template import render_single_input_prompt
from llm_ai.tasks.response_parser import parse_task_response


class FakeProvider(BaseLLMProvider):
    @property
    def name(self) -> str:
        return "FakeLLM"

    def call(self, message: str) -> str:
        return f"Generated from: {message}"


class TestLayer1_PromptTemplate:
    """Test render prompt bằng placeholder literal."""

    def test_render_single_input_prompt_replaces_placeholder(self):
        rendered = render_single_input_prompt(
            "Header\n[Video Content]",
            "raw video text",
            "[Video Content]",
        )
        assert rendered == "Header\nraw video text"

    def test_render_single_input_prompt_strict_missing_placeholder_raises(self):
        with pytest.raises(ValueError, match="Không tìm thấy placeholder"):
            render_single_input_prompt("Header only", "content", "[Video Content]")


class TestLayer1_ResponseParser:
    """Test parser raw/tag/json cho generic LLM task."""

    def test_parse_raw_strips_think_block(self):
        raw = "<think>hidden</think>\nVisible"
        assert parse_task_response(raw, "raw") == "Visible"

    def test_parse_tag_extracts_content(self):
        raw = "<META>Title\nDescription</META>"
        assert parse_task_response(raw, {"type": "tag", "tag": "META"}) == "Title\nDescription"

    def test_parse_json_pretty_prints(self):
        raw = '```json\n{"title":"Demo","tags":["a","b"]}\n```'
        parsed = parse_task_response(raw, "json")
        assert '"title": "Demo"' in parsed
        assert '"tags": [' in parsed


class TestLayer2_GenericTextTask:
    """Test generic task end-to-end với mocked provider, không gọi API thật."""

    def test_run_generic_text_task_writes_output(self, tmp_path: Path):
        prompt_file = tmp_path / "prompt.txt"
        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "output.md"

        prompt_file.write_text("Prompt:\n[Video Content]", encoding="utf-8")
        input_file.write_text("Sample content", encoding="utf-8")

        stats = run_generic_text_task(
            input_file=str(input_file),
            output_file=str(output_file),
            provider=FakeProvider(),
            task_config=GenericTextTaskConfig(
                task_name="seo_metadata",
                prompt_file=str(prompt_file),
                input_placeholder="[Video Content]",
                output_parser="raw",
            ),
        )

        assert output_file.exists()
        assert "Prompt:" in output_file.read_text(encoding="utf-8")
        assert stats["task_name"] == "seo_metadata"
        assert stats["output_chars"] > 0
