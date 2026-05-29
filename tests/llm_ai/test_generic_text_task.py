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
from llm_ai.openai_compat import CapabilityNotEnabledError
from llm_ai.provider_chain import FallbackLLMProvider, ProviderChainError
from llm_ai.retry import calculate_linear_retry_wait_seconds
from llm_ai.tasks.generic_text_task import GenericTextTaskConfig, run_generic_text_task
from llm_ai.tasks.prompt_template import render_single_input_prompt
from llm_ai.tasks.response_parser import parse_task_response


class FakeProvider(BaseLLMProvider):
    @property
    def name(self) -> str:
        return "FakeLLM"

    def call(self, message: str) -> str:
        return f"Generated from: {message}"


class FailingProvider(BaseLLMProvider):
    @property
    def name(self) -> str:
        return "FailingLLM"

    def call(self, message: str) -> str:
        raise RuntimeError("primary failed")


class CapabilityFailingProvider(BaseLLMProvider):
    @property
    def name(self) -> str:
        return "CapabilityFailingLLM"

    def call(self, message: str) -> str:
        raise CapabilityNotEnabledError(
            "capability disabled",
            profile_name="unit_profile",
            feature="reasoning_effort",
            api_mode="chat_completions",
        )


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


class TestLayer1_RetryAndProviderChain:
    """Test retry wait tuyến tính và fallback provider wrapper."""

    def test_calculate_linear_retry_wait_seconds(self):
        assert calculate_linear_retry_wait_seconds(10, 1) == 10
        assert calculate_linear_retry_wait_seconds(10, 2) == 20
        assert calculate_linear_retry_wait_seconds(10, 3) == 30

    def test_fallback_provider_uses_next_provider_after_failure(self):
        provider = FallbackLLMProvider(
            [FailingProvider(), FakeProvider()],
            names=["primary", "fallback"],
        )

        assert provider.call("hello") == "Generated from: hello"
        assert provider.active_provider_name == "FakeLLM"

    def test_fallback_provider_raises_chain_error_when_all_fail(self):
        provider = FallbackLLMProvider(
            [FailingProvider(), FailingProvider()],
            names=["primary", "fallback"],
        )

        with pytest.raises(ProviderChainError, match="Tất cả provider") as exc_info:
            provider.call("hello")

        assert len(exc_info.value.failures) == 2

    def test_fallback_provider_does_not_fallback_on_capability_error(self):
        provider = FallbackLLMProvider(
            [CapabilityFailingProvider(), FakeProvider()],
            names=["primary", "fallback"],
        )

        with pytest.raises(CapabilityNotEnabledError, match="capability disabled"):
            provider.call("hello")

        assert provider.active_provider_name == "CapabilityFailingLLM"


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
