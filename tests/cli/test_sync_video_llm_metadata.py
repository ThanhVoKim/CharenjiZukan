#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/cli/test_sync_video_llm_metadata.py
========================================
Test Layer 1 cho helper LLM metadata trong sync_engine/llm_metadata.py
và helper write_segments_to_flat_text trong utils/srt_parser.py.

Cấu trúc layers:
  Layer 1 — Unit Tests          (raw text writer, path policy, override, fail policy)

Cách chạy từng layer:
    pytest tests/cli/test_sync_video_llm_metadata.py -v -k "Layer1"
"""

import sys
from pathlib import Path

import pytest

# ── Project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sync_engine.llm_metadata import (  # noqa: E402
    apply_llm_metadata_override,
    execute_llm_metadata_task,
    resolve_llm_metadata_paths,
    run_llm_metadata_task,
)
from utils.srt_parser import write_segments_to_flat_text  # noqa: E402


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_LLMMetadataHelpers:
    def test_write_segments_to_flat_text_is_flat_raw_text(self, tmp_path: Path):
        segments = [
            {"line": 1, "text": "  First line\ncontinues   "},
            {"line": 2, "text": ""},
            {"line": 3, "text": "Second    block"},
        ]
        output_path = tmp_path / "test_input.txt"

        result = write_segments_to_flat_text(segments, str(output_path))

        assert result == output_path
        assert output_path.read_text(encoding="utf-8") == "First line continues Second block"

    def test_directory_policy_slash_uses_input_video_parent(self, tmp_path: Path):
        video_path = tmp_path / "content" / "a" / "b.mp4"
        cfg = {
            "input": {
                "write_debug_input": True,
                "debug_input_filename_template": "{video_stem}_metadata_input.txt",
            },
            "output": {
                "directory_policy": "/",
                "filename_template": "{video_stem}_metadata.md",
            },
        }

        output_path, debug_input_path = resolve_llm_metadata_paths(
            cfg,
            str(video_path),
            output_name="custom_output_name",
        )

        assert output_path == tmp_path / "content" / "a" / "b_metadata.md"
        assert debug_input_path == tmp_path / "content" / "a" / "b_metadata_input.txt"

    def test_llm_metadata_override_deep_merges_task_config(self):
        render_config = {
            "resolution": {"width": 1920},
            "llm_metadata": {
                "enabled": True,
                "input": {"write_debug_input": True},
                "output": {
                    "directory_policy": "/",
                    "filename_template": "{video_stem}_metadata.md",
                },
            },
        }

        merged = apply_llm_metadata_override(
            render_config,
            {
                "input": {"write_debug_input": False},
                "output": {"filename_template": "{output_name}_seo.md"},
            },
        )

        assert merged["resolution"] == {"width": 1920}
        assert merged["llm_metadata"]["enabled"] is True
        assert merged["llm_metadata"]["input"]["write_debug_input"] is False
        assert merged["llm_metadata"]["output"]["directory_policy"] == "/"
        assert merged["llm_metadata"]["output"]["filename_template"] == "{output_name}_seo.md"

    def test_boolean_llm_metadata_override_toggles_enabled(self):
        merged = apply_llm_metadata_override(
            {"llm_metadata": {"enabled": True, "fail_policy": "warn"}},
            False,
        )

        assert merged["llm_metadata"]["enabled"] is False
        assert merged["llm_metadata"]["fail_policy"] == "warn"

    def test_execute_writes_debug_input_and_metadata_output(self, monkeypatch, tmp_path: Path):
        class FakeProvider:
            name = "fake-provider"

        def fake_create_task_provider(args, task_cfg):
            assert args.provider is None
            assert task_cfg["prompt_file"].endswith("prompt.txt")
            return FakeProvider()

        def fake_load_task_config(config_path: str):
            assert config_path.endswith("metadata.yaml")
            return {
                "task_name": "fake_metadata",
                "prompt_file": str(tmp_path / "prompt.txt"),
                "input_placeholder": "[Video Content]",
                "default_ext": "_metadata.md",
                "output_parser": "raw",
            }

        def fake_run_generic_text_task(*, input_file, output_file, provider, task_config, prompt_file=None, input_placeholder=None):
            input_text = Path(input_file).read_text(encoding="utf-8")
            assert input_text == "Alpha Beta Gamma"
            assert provider.name == "fake-provider"
            Path(output_file).write_text("metadata output", encoding="utf-8")
            return {"output": output_file, "output_chars": len("metadata output")}

        monkeypatch.setattr("llm_ai.task_runner.create_task_provider", fake_create_task_provider)
        monkeypatch.setattr("llm_ai.tasks.generic_text_task.load_task_config", fake_load_task_config)
        monkeypatch.setattr("llm_ai.tasks.generic_text_task.run_generic_text_task", fake_run_generic_text_task)

        video_path = tmp_path / "content" / "a" / "b.mp4"
        input_text_path = tmp_path / "content" / "a" / "b_metadata_input.txt"
        input_text_path.parent.mkdir(parents=True, exist_ok=True)
        input_text_path.write_text("Alpha Beta Gamma", encoding="utf-8")

        stats = execute_llm_metadata_task(
            metadata_cfg={
                "task_config": str(tmp_path / "metadata.yaml"),
                "input": {
                    "write_debug_input": True,
                    "debug_input_filename_template": "{video_stem}_metadata_input.txt",
                },
                "output": {
                    "directory_policy": "/",
                    "filename_template": "{video_stem}_metadata.md",
                },
            },
            input_text_path=str(input_text_path),
            video_path=str(video_path),
            output_name="ignored_for_video_stem_template",
        )

        assert stats["output_chars"] == len("metadata output")
        assert (tmp_path / "content" / "a" / "b_metadata.md").read_text(encoding="utf-8") == "metadata output"

    def test_warn_fail_policy_swallows_metadata_errors(self, monkeypatch, tmp_path: Path):
        def boom(**kwargs):
            raise RuntimeError("mock llm failure")

        monkeypatch.setattr("sync_engine.llm_metadata.execute_llm_metadata_task", boom)

        input_text_path = tmp_path / "input.txt"
        input_text_path.write_text("raw subtitle text", encoding="utf-8")

        result = run_llm_metadata_task(
            input_text_path=str(input_text_path),
            render_config={"llm_metadata": {"enabled": True, "fail_policy": "warn"}},
            video_path=str(tmp_path / "video.mp4"),
            output_name="video",
        )

        assert result is None

    def test_raise_fail_policy_propagates_metadata_errors(self, monkeypatch, tmp_path: Path):
        def boom(**kwargs):
            raise RuntimeError("mock llm failure")

        monkeypatch.setattr("sync_engine.llm_metadata.execute_llm_metadata_task", boom)

        input_text_path = tmp_path / "input.txt"
        input_text_path.write_text("raw subtitle text", encoding="utf-8")

        with pytest.raises(RuntimeError, match="mock llm failure"):
            run_llm_metadata_task(
                input_text_path=str(input_text_path),
                render_config={"llm_metadata": {"enabled": True, "fail_policy": "raise"}},
                video_path=str(tmp_path / "video.mp4"),
                output_name="video",
            )
