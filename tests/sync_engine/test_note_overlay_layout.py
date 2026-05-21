#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/sync_engine/test_note_overlay_layout.py
=========================================
Tests for dynamic ASS note overlay layout.

Cấu trúc layers:
  Layer 1 — Unit Tests (wrap, config, geometry, emitter)
  Layer 2 — Component Tests (ASS expansion, SRT → final ASS contract)
  Layer 3 — Pipeline Integration (mocked heavy pipeline)

Cách chạy từng layer:
    pytest tests/sync_engine/test_note_overlay_layout.py -v -k "Layer1"
    pytest tests/sync_engine/test_note_overlay_layout.py -v -k "Layer2"
    pytest tests/sync_engine/test_note_overlay_layout.py -v -k "Layer3"
"""

import argparse
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cli.srt_to_ass import convert_srt_to_ass
from sync_engine.models import TimelineSegment
from sync_engine.note_overlay_layout import (
    _build_background_dialogue,
    _build_text_dialogue,
    _compute_box_geometry,
    _load_layout_config,
    _resolve_layout,
    expand_note_overlay_ass,
)
from sync_engine.timestamp_remapper import recalculate_ass
from utils.ass_utils import wrap_text_pixel


class MockFont:
    def __init__(self, char_width: int = 10):
        self.char_width = char_width

    def getlength(self, text: str) -> float:
        return len(text) * self.char_width


@pytest.fixture()
def dynamic_render_config() -> dict:
    return {
        "resolution": {"width": 1920, "height": 1080},
        "note_overlay": {
            "enabled": True,
            "mode": "dynamic_ass_box",
            "default_layout": "top_left",
            "font": {
                "fontname": "Arial",
                "font_path": None,
                "font_size": 42,
                "bold": False,
                "line_spacing": 1.25,
                "primary_color": "&H00FFFFFF",
            },
            "layouts": {
                "top_left": {
                    "anchor": "top_left",
                    "margin_x": 80,
                    "margin_y": 100,
                    "width": 680,
                    "height": 260,
                    "padding_left": 32,
                    "padding_right": 32,
                    "padding_top": 28,
                    "padding_bottom": 36,
                    "height_safety_margin": 10,
                    "background_color": "&HCC000000",
                },
                "bottom_right": {
                    "anchor": "bottom_right",
                    "margin_x": 80,
                    "margin_y": 180,
                    "width": 720,
                    "height": 300,
                    "padding_left": 32,
                    "padding_right": 32,
                    "padding_top": 28,
                    "padding_bottom": 40,
                    "height_safety_margin": 10,
                    "background_color": "&HCC000000",
                },
                "center_panel": {
                    "anchor": "center",
                    "width": 900,
                    "height": 360,
                    "padding_left": 40,
                    "padding_right": 40,
                    "padding_top": 32,
                    "padding_bottom": 40,
                    "height_safety_margin": 10,
                    "background_color": "&HCC000000",
                    "font_size": 44,
                },
            },
        },
    }


@pytest.fixture()
def sample_note_ass_path(tmp_path: Path) -> Path:
    content = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: NoteStyle,Arial,42,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:03.00,0:00:10.00,NoteStyle,top_left,0,0,0,,Quick note text.
Dialogue: 0,0:00:14.00,0:00:22.00,NoteStyle,bottom_right,0,0,0,,Longer text that should wrap onto multiple lines so we can verify min-height growth with a narrow layout.
Dialogue: 0,0:00:25.00,0:00:34.00,NoteStyle,unknown_key,0,0,0,,Fallback test.
"""
    path = tmp_path / "note.ass"
    path.write_text(content, encoding="utf-8")
    return path


def _dialogue_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("Dialogue:")]


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# ═════════════════════════════════════════════════════════════════════


class TestLayer1_WrapTextPixel:
    def test_short_text_no_wrap(self):
        assert wrap_text_pixel("short", 100, MockFont()) == ["short"]

    def test_english_break_at_whitespace(self):
        result = wrap_text_pixel("hello beautiful world", 100, MockFont())
        assert result == ["hello", "beautiful", "world"]
        assert all(" " not in line[:1] for line in result)

    def test_cjk_break_per_char(self):
        result = wrap_text_pixel("日本語字幕", 20, MockFont())
        assert result == ["日本", "語字", "幕"]

    def test_token_longer_than_width_hard_split(self):
        result = wrap_text_pixel("supercalifragilisticexpialidocious", 50, MockFont())
        assert len(result) > 3
        assert all(len(piece) <= 5 for piece in result)

    def test_preserve_existing_newlines(self):
        result = wrap_text_pixel("Line1\\NLine2", 200, MockFont())
        assert result == ["Line1", "Line2"]


class TestLayer1_ResolveLayout:
    def test_known_key_returns_preset(self, dynamic_render_config):
        cfg = _load_layout_config(dynamic_render_config)
        key, preset, used_fallback = _resolve_layout("bottom_right", cfg)
        assert key == "bottom_right"
        assert preset["anchor"] == "bottom_right"
        assert used_fallback is False

    def test_unknown_key_falls_back_default(self, dynamic_render_config):
        cfg = _load_layout_config(dynamic_render_config)
        key, preset, used_fallback = _resolve_layout("unknown", cfg)
        assert key == "top_left"
        assert preset["anchor"] == "top_left"
        assert used_fallback is True

    def test_empty_name_uses_default(self, dynamic_render_config):
        cfg = _load_layout_config(dynamic_render_config)
        key, _preset, used_fallback = _resolve_layout("", cfg)
        assert key == "top_left"
        assert used_fallback is True

    def test_missing_layouts_uses_hardcoded_default(self):
        cfg = _load_layout_config({"note_overlay": {"enabled": True, "layouts": {}}})
        assert cfg["default_layout"] == "top_left"
        assert cfg["layouts"]["top_left"]["width"] == 640

    def test_invalid_anchor_raises_valueerror(self, dynamic_render_config):
        dynamic_render_config["note_overlay"]["layouts"]["bad"] = {"anchor": "diagonal"}
        with pytest.raises(ValueError, match="Invalid anchor"):
            _load_layout_config(dynamic_render_config)

    def test_invalid_mode_raises_valueerror(self):
        with pytest.raises(ValueError, match="Invalid note_overlay.mode"):
            _load_layout_config({"note_overlay": {"enabled": True, "mode": "bad_mode"}})


class TestLayer1_ComputeBoxGeometry:
    def test_top_left_min_height_preserved(self, dynamic_render_config):
        preset = _load_layout_config(dynamic_render_config)["layouts"]["top_left"]
        assert _compute_box_geometry(preset, text_height=40, video_w=1920, video_h=1080) == (80, 100, 680, 260)

    def test_bottom_anchor_grows_upward(self, dynamic_render_config):
        preset = _load_layout_config(dynamic_render_config)["layouts"]["bottom_right"]
        x, y, w, h = _compute_box_geometry(preset, text_height=500, video_w=1920, video_h=1080)
        assert (x, w) == (1120, 720)
        assert h == 578
        assert y == 322

    def test_center_anchor_grows_both_directions(self, dynamic_render_config):
        preset = _load_layout_config(dynamic_render_config)["layouts"]["center_panel"]
        x, y, w, h = _compute_box_geometry(preset, text_height=500, video_w=1920, video_h=1080)
        assert (x, w) == (510, 900)
        assert h == 582
        assert y == 249

    def test_absolute_position_overrides_anchor(self, dynamic_render_config):
        preset = _load_layout_config(dynamic_render_config)["layouts"]["top_left"]
        preset = {**preset, "x": 10, "y": 20, "anchor": "bottom_right"}
        assert _compute_box_geometry(preset, text_height=40, video_w=1920, video_h=1080) == (10, 20, 680, 260)

    def test_clamp_to_video_bounds(self, dynamic_render_config):
        preset = _load_layout_config(dynamic_render_config)["layouts"]["top_left"]
        preset = {**preset, "x": 2000, "y": 2000}
        assert _compute_box_geometry(preset, text_height=40, video_w=1920, video_h=1080) == (1240, 820, 680, 260)


class TestLayer1_AssDrawingEmitter:
    def test_background_dialogue_has_correct_drawing_path_and_color(self):
        line = _build_background_dialogue("0", "0:00:01.00", "0:00:02.00", "top_left", 80, 100, 680, 260, "&HCC112233")
        assert "Dialogue: 0,0:00:01.00,0:00:02.00,NoteBox,top_left" in line
        assert "\\pos(80,100)" in line
        assert "\\1c&H112233&" in line
        assert "\\alpha&HCC&" in line
        assert "m 0 0 l 680 0 l 680 260 l 0 260" in line

    def test_text_dialogue_position_inside_box_and_escapes_braces(self):
        line = _build_text_dialogue(
            "1",
            "0:00:01.00",
            "0:00:02.00",
            "top_left",
            112,
            128,
            ["Value {x}"],
            {"font_size": 44, "text_color": "&H80010203"},
            42,
            "&H00FFFFFF",
        )
        assert "Dialogue: 1,0:00:01.00,0:00:02.00,NoteText,top_left" in line
        assert "\\pos(112,128)" in line
        assert "\\fs44" in line
        assert "\\c&H010203&" in line
        assert "\\alpha&H80&" in line
        assert r"\{x\}" in line


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — COMPONENT TESTS
# ═════════════════════════════════════════════════════════════════════


class TestLayer2_ExpandNoteOverlayASS:
    def test_output_has_doubled_dialogues_and_layer_order(self, sample_note_ass_path, dynamic_render_config, tmp_path):
        output_path = tmp_path / "note_overlay.ass"
        stats = expand_note_overlay_ass(str(sample_note_ass_path), str(output_path), dynamic_render_config, 1920, 1080, PROJECT_ROOT)

        dialogues = _dialogue_lines(output_path)
        assert stats["n_dialogues_in"] == 3
        assert stats["n_dialogues_out"] == 6
        assert len(dialogues) == 6
        for index in range(0, len(dialogues), 2):
            assert dialogues[index].startswith("Dialogue: 0,")
            assert ",NoteBox," in dialogues[index]
            assert dialogues[index + 1].startswith("Dialogue: 1,")
            assert ",NoteText," in dialogues[index + 1]

    def test_unknown_key_uses_default_layout_and_stats(self, sample_note_ass_path, dynamic_render_config, tmp_path):
        output_path = tmp_path / "note_overlay.ass"
        stats = expand_note_overlay_ass(str(sample_note_ass_path), str(output_path), dynamic_render_config, 1920, 1080, PROJECT_ROOT)
        dialogues = _dialogue_lines(output_path)
        fallback_pair = [line for line in dialogues if "Fallback test" in line or "unknown_key" in line]

        assert stats["fallback_count"] == 1
        assert stats["unknown_layout_keys"] == ["unknown_key"]
        assert any(",NoteText,top_left," in line and "Fallback test" in line for line in fallback_pair)

    def test_min_height_preserved_when_short(self, sample_note_ass_path, dynamic_render_config, tmp_path):
        output_path = tmp_path / "note_overlay.ass"
        expand_note_overlay_ass(str(sample_note_ass_path), str(output_path), dynamic_render_config, 1920, 1080, PROJECT_ROOT)
        first_bg = _dialogue_lines(output_path)[0]
        assert "m 0 0 l 680 0 l 680 260 l 0 260" in first_bg

    def test_min_height_growth(self, sample_note_ass_path, dynamic_render_config, tmp_path):
        dynamic_render_config["note_overlay"]["layouts"]["bottom_right"].update(
            {"width": 120, "height": 40, "padding_left": 10, "padding_right": 10, "padding_top": 10, "padding_bottom": 10, "font_size": 20, "line_spacing": 1.0}
        )
        output_path = tmp_path / "note_overlay.ass"
        expand_note_overlay_ass(str(sample_note_ass_path), str(output_path), dynamic_render_config, 1920, 1080, PROJECT_ROOT)
        bottom_bg = [line for line in _dialogue_lines(output_path) if ",NoteBox,bottom_right," in line][0]
        assert "l 120 40 l 0 40" not in bottom_bg

    def test_playres_matches_video_and_styles_exist(self, sample_note_ass_path, dynamic_render_config, tmp_path):
        output_path = tmp_path / "note_overlay.ass"
        expand_note_overlay_ass(str(sample_note_ass_path), str(output_path), dynamic_render_config, 1280, 720, PROJECT_ROOT)
        text = output_path.read_text(encoding="utf-8")
        assert "PlayResX: 1280" in text
        assert "PlayResY: 720" in text
        assert "Style: NoteBox" in text
        assert "Style: NoteText" in text

    def test_disabled_layout_writes_empty_output(self, sample_note_ass_path, dynamic_render_config, tmp_path):
        dynamic_render_config["note_overlay"]["enabled"] = False
        output_path = tmp_path / "disabled.ass"
        stats = expand_note_overlay_ass(str(sample_note_ass_path), str(output_path), dynamic_render_config, 1920, 1080, PROJECT_ROOT)
        assert stats == {"n_dialogues_in": 0, "n_dialogues_out": 0, "unknown_layout_keys": [], "fallback_count": 0}
        assert "Dialogue:" not in output_path.read_text(encoding="utf-8")


class TestLayer2_SrtToFinalAssContract:
    def test_srt_first_line_layout_to_final_ass_positions_and_text(self, dynamic_render_config, tmp_path):
        srt_path = tmp_path / "note.srt"
        srt_path.write_text(
            "1\n00:00:00,000 --> 00:00:01,166\ntop_left\nblock text 1\n\n"
            "2\n00:00:02,000 --> 00:00:04,000\nbottom_right\nblock text 2\n",
            encoding="utf-8",
        )
        note_input = tmp_path / "note_input.ass"
        count = convert_srt_to_ass(
            str(srt_path),
            str(note_input),
            max_chars=0,
            layout_key="top_left",
            known_layout_keys={"top_left", "bottom_right", "center_panel"},
        )
        input_text = note_input.read_text(encoding="utf-8")
        assert count == 2
        assert "NoteStyle,top_left,0,0,0,,block text 1" in input_text
        assert "NoteStyle,bottom_right,0,0,0,,block text 2" in input_text
        assert "top_left\\Nblock text" not in input_text

        timeline = [TimelineSegment(0.0, 4000.0, 0.0, 4000.0, 1.0, 1.0, 4000.0, "gap", None, 0.0)]
        note_synced = tmp_path / "note_synced.ass"
        recalculate_ass(str(note_input), timeline, str(note_synced), max_chars_per_line=0, fps_float=30.0, apply_text_wrap=False)

        final_ass = tmp_path / "note_overlay.ass"
        expand_note_overlay_ass(str(note_synced), str(final_ass), dynamic_render_config, 1920, 1080, PROJECT_ROOT)
        final_text = final_ass.read_text(encoding="utf-8")
        dialogues = _dialogue_lines(final_ass)

        assert len(dialogues) == 4
        assert "Style: NoteBox" in final_text
        assert "Style: NoteText" in final_text
        assert "block text 1" in final_text
        assert "block text 2" in final_text
        assert "{\\pos(80,100)" in final_text
        assert "{\\pos(1120,600)" in final_text
        assert "0:00:00.00,0:00:01.16" in final_text
        assert "0:00:02.00,0:00:04.00" in final_text

    def test_srt_without_layout_key_uses_fallback_in_final_ass(self, dynamic_render_config, tmp_path):
        srt_path = tmp_path / "note.srt"
        srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nplain note\n", encoding="utf-8")
        note_input = tmp_path / "note_input.ass"
        convert_srt_to_ass(str(srt_path), str(note_input), max_chars=0, layout_key="top_left", known_layout_keys={"top_left"})
        assert "NoteStyle,top_left,0,0,0,,plain note" in note_input.read_text(encoding="utf-8")

    def test_srt_strict_mode_rejects_unknown_layout_key_before_final_ass(self, tmp_path):
        srt_path = tmp_path / "note.srt"
        srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nunknown\nbody\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown layout key"):
            convert_srt_to_ass(
                str(srt_path),
                str(tmp_path / "note_input.ass"),
                max_chars=0,
                layout_key="top_left",
                mode="strict",
                known_layout_keys={"top_left"},
            )


# ═════════════════════════════════════════════════════════════════════
# LAYER 3 — PIPELINE INTEGRATION
# ═════════════════════════════════════════════════════════════════════


class TestLayer3_SyncPipelineNoteOverlay:
    def test_full_pipeline_generates_dynamic_overlay_with_mocks(self, dynamic_render_config, tmp_path, monkeypatch):
        import cli.sync_video as sync_video_cli
        import sync_engine.analyzer as analyzer_module

        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake-video")
        subtitle_path = tmp_path / "subtitle.srt"
        subtitle_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
        note_path = tmp_path / "note.ass"
        note_path.write_text(
            "[Script Info]\nScriptType: v4.00+\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: NoteStyle,Arial,42,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 0,0:00:00.00,0:00:01.00,NoteStyle,top_left,0,0,0,,Pipeline note\n",
            encoding="utf-8",
        )
        render_config_path = tmp_path / "render_config.json"
        render_config = {
            **dynamic_render_config,
            "watermark_img": {"enabled": False},
            "watermark_text": {"enabled": False},
            "black_strip": {"enabled": False},
            "subtitles": {"enabled": False, "burn_hardsub": False},
            "audio_mix": {"ambient_volume": 0.0, "bgm_volume": 0.0},
            "audio_separator": {"extract_bgm": False, "extract_vocals": False},
            "video_encoding": {
                "codec": "hevc_nvenc",
                "preset": "p4",
                "tune": "hq",
                "quality": ["-cq", "28"],
            },
            "forced_alignment_subtitle": {"enabled": False},
            "llm_metadata": {"enabled": False},
        }
        render_config_path.write_text(json.dumps(render_config), encoding="utf-8")

        class FakeCompletedProcess:
            def __init__(self, stdout: str):
                self.stdout = stdout

        def fake_run(cmd, *args, **kwargs):
            joined = " ".join(str(part) for part in cmd)
            if "stream=r_frame_rate" in joined:
                return FakeCompletedProcess("30/1\n")
            return FakeCompletedProcess("4.0\n")

        class FakeEdgeTTSEngine:
            def __init__(self, *args, **kwargs):
                self.queue_tts = kwargs.get("queue_tts", [])

            def run(self):
                for item in self.queue_tts:
                    Path(item["filename"]).write_bytes(b"fake-wav")
                return {"ok": len(self.queue_tts), "err": 0}

        class FakeVoicevoxEngine:
            def __init__(self, *args, **kwargs):
                pass

        def fake_process_video_chunks_parallel(*args, **kwargs):
            out = tmp_path / "stretched.mp4"
            out.write_bytes(b"fake-stretched")
            return str(out), [4000.0]

        timeline = [TimelineSegment(0.0, 4000.0, 0.0, 4000.0, 1.0, 1.0, 4000.0, "gap", None, 0.0)]

        fake_edgetts_module = ModuleType("tts.edgetts")
        fake_edgetts_module.EdgeTTSEngine = FakeEdgeTTSEngine
        fake_voicevox_module = ModuleType("tts.voicevox")
        fake_voicevox_module.VoicevoxTTSEngine = FakeVoicevoxEngine
        fake_voicevox_nemo_module = ModuleType("tts.voicevox_nemo")
        fake_voicevox_nemo_module.VoicevoxNemoTTSEngine = FakeVoicevoxEngine
        fake_qwen_module = ModuleType("tts.qwen")
        fake_qwen_module.QwenTTSEngine = FakeVoicevoxEngine
        monkeypatch.setitem(sys.modules, "tts.edgetts", fake_edgetts_module)
        monkeypatch.setitem(sys.modules, "tts.voicevox", fake_voicevox_module)
        monkeypatch.setitem(sys.modules, "tts.voicevox_nemo", fake_voicevox_nemo_module)
        monkeypatch.setitem(sys.modules, "tts.qwen", fake_qwen_module)
        monkeypatch.setattr(sync_video_cli.subprocess, "run", fake_run)
        monkeypatch.setattr(sync_video_cli, "classify_and_compute_slots", lambda *args, **kwargs: [SimpleNamespace(tts_duration=1000, slot_duration=4000, hard_limit_ms=None)])
        monkeypatch.setattr(sync_video_cli, "compute_speeds", lambda *args, **kwargs: (1.0, 1.0, 4000.0))
        monkeypatch.setattr(sync_video_cli, "build_timeline_map", lambda *args, **kwargs: timeline)
        monkeypatch.setattr(analyzer_module, "recalculate_timeline_from_actual_durations", lambda *args, **kwargs: timeline)
        monkeypatch.setattr(sync_video_cli, "process_video_chunks_parallel", fake_process_video_chunks_parallel)
        monkeypatch.setattr(sync_video_cli, "assemble_audio_track", lambda *args, **kwargs: Path(kwargs["output_path"]).write_bytes(b"fake-audio"))

        output_dir = tmp_path / "out"
        args = argparse.Namespace(
            video=str(video_path),
            subtitle=str(subtitle_path),
            tts_provider="edge",
            tts_voice="test-voice",
            tts_config=str(PROJECT_ROOT / "config" / "tts_config.yaml"),
            mute=None,
            note_overlay_ass=str(note_path),
            ambient=None,
            render_config=str(render_config_path),
            slow_cap=0.5,
            output_dir=str(output_dir),
            output_name="video_synced",
            no_hardsub=True,
            keep_tmp=False,
            workers=1,
            batch_size=100,
            no_gpu=False,
            subtitle_max_chars=0,
            llm_metadata_override=None,
        )

        sync_video_cli.run_sync_pipeline(args)

        note_overlay = output_dir / "video_synced_note_overlay.ass"
        assert note_overlay.exists()
        text = note_overlay.read_text(encoding="utf-8")
        assert "Style: NoteBox" in text
        assert "Style: NoteText" in text
        assert "Pipeline note" in text
