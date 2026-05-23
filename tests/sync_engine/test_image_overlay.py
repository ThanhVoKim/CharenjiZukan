#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/sync_engine/test_image_overlay.py
======================================

Layer 1: Unit tests cho domain logic image overlay SRT/PNG.
Layer 2: Component tests mock renderer FFmpeg command/filter graph.

Không commit media thật; mọi asset PNG/SRT/ASS/audio/video path đều tạo runtime
trong tmp_path theo docs/testing-guide.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ── Project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sync_engine.image_overlay import (  # noqa: E402
    ImageOverlayEvent,
    get_unique_image_overlay_assets,
    load_image_overlay_events,
    normalize_image_overlay_key,
    remap_image_overlay_events,
    render_intermediate_overlay_track,
    resolve_image_overlay_path,
    write_image_overlay_debug_srt,
)
from sync_engine.models import TimelineSegment  # noqa: E402
from utils.srt_parser import parse_srt_file  # noqa: E402


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — DOMAIN LOGIC
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_ImageOverlayDomain:
    def test_normalize_key_uses_first_line_and_rejects_unsafe_values(self):
        assert normalize_image_overlay_key(" frame_001 \nignored note") == "frame_001"
        assert normalize_image_overlay_key("FRAME_A", file_ext="png") == "FRAME_A"

        invalid_cases = [
            "",
            "   ",
            "../secret",
            "folder/frame",
            "folder\\frame",
            "frame.png",
            "FRAME.PNG",
        ]
        for text in invalid_cases:
            with pytest.raises(ValueError):
                normalize_image_overlay_key(text)

    def test_resolve_image_overlay_path_warn_skips_missing_and_raise_stops(self, tmp_path: Path):
        overlay_dir = tmp_path / "overlays"
        overlay_dir.mkdir()
        asset = overlay_dir / "frame_001.png"
        asset.write_bytes(b"fake png bytes")

        assert resolve_image_overlay_path("frame_001", overlay_dir) == asset
        assert resolve_image_overlay_path("missing", overlay_dir, missing_policy="warn") is None

        with pytest.raises(FileNotFoundError, match="Không tìm thấy image overlay asset"):
            resolve_image_overlay_path("missing", overlay_dir, missing_policy="raise")

    def test_load_events_resolves_assets_skips_missing_and_preserves_order(self, tmp_path: Path):
        overlay_dir = tmp_path / "png"
        overlay_dir.mkdir()
        (overlay_dir / "alpha.png").write_bytes(b"alpha")
        (overlay_dir / "beta.png").write_bytes(b"beta")

        srt_path = tmp_path / "image_overlay.srt"
        srt_path.write_text(
            """1
00:00:00,000 --> 00:00:01,200
alpha

2
00:00:01,200 --> 00:00:02,000
missing

3
00:00:02,000 --> 00:00:03,000
beta
extra line ignored
""",
            encoding="utf-8",
        )

        events = load_image_overlay_events(srt_path, overlay_dir, missing_policy="warn")

        assert [event.key for event in events] == ["alpha", "beta"]
        assert [event.start_time for event in events] == [0.0, 2000.0]
        assert [event.end_time for event in events] == [1200.0, 3000.0]
        assert Path(events[0].image_path).name == "alpha.png"
        assert Path(events[1].image_path).name == "beta.png"

    def test_get_unique_assets_deduplicates_by_resolved_path(self, tmp_path: Path):
        image_a = tmp_path / "a.png"
        image_b = tmp_path / "b.png"
        image_a.write_bytes(b"a")
        image_b.write_bytes(b"b")
        events = [
            ImageOverlayEvent("a", str(image_a), 0, 500, 1),
            ImageOverlayEvent("a_again", str(image_a), 500, 1000, 2),
            ImageOverlayEvent("b", str(image_b), 1000, 1500, 3),
        ]

        assets = get_unique_image_overlay_assets(events)

        assert [asset.key for asset in assets] == ["a", "b"]
        assert [Path(asset.image_path).name for asset in assets] == ["a.png", "b.png"]

    def test_remap_events_uses_timeline_mapping_and_min_duration(self, tmp_path: Path):
        image = tmp_path / "frame.png"
        image.write_bytes(b"frame")
        events = [
            ImageOverlayEvent("frame", str(image), 250.0, 750.0, 1),
            ImageOverlayEvent("frame", str(image), 1000.0, 1000.0, 2),
        ]
        timeline = [
            TimelineSegment(
                orig_start=0.0,
                orig_end=1000.0,
                new_start=0.0,
                new_end=2000.0,
                video_speed=0.5,
                audio_speed=1.0,
                new_chunk_dur=2000.0,
                block_type="tts",
                tts_clip_path=None,
                tts_duration=1000.0,
            )
        ]

        remapped = remap_image_overlay_events(events, timeline, fps_float=1000.0, min_duration_ms=100.0)

        assert remapped[0].start_time == 500.0
        assert remapped[0].end_time == 1500.0
        assert remapped[1].start_time == 2000.0
        assert remapped[1].end_time == 2100.0

    def test_write_debug_srt_uses_remapped_timestamp_and_key_text(self, tmp_path: Path):
        image = tmp_path / "frame.png"
        image.write_bytes(b"frame")
        events = [ImageOverlayEvent("frame", str(image), 500.0, 1500.0, 7)]
        output_path = tmp_path / "debug" / "image_overlay_synced.srt"

        result = write_image_overlay_debug_srt(events, output_path)

        assert result == output_path
        parsed = parse_srt_file(str(output_path))
        assert len(parsed) == 1
        assert parsed[0]["line"] == 1
        assert parsed[0]["start_time"] == 500
        assert parsed[0]["end_time"] == 1500
        assert parsed[0]["text"] == "frame"

    def test_intermediate_overlay_track_is_future_stub(self):
        with pytest.raises(NotImplementedError, match="reserved for a future phase"):
            render_intermediate_overlay_track()


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — RENDERER COMMAND/FILTER GRAPH
# ═════════════════════════════════════════════════════════════════════

class _FakeProcess:
    def __init__(self):
        self.stderr = []
        self.stdout = []
        self.returncode = 0

    def wait(self):
        return 0


def _patch_renderer_process(monkeypatch, captured: dict):
    import sync_engine.renderer as renderer

    monkeypatch.setattr(renderer, "detect_hevc_nvenc", lambda: True)
    monkeypatch.setattr(renderer, "_get_video_duration", lambda _path: 0.0)

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return _FakeProcess()

    monkeypatch.setattr(renderer.subprocess, "Popen", fake_popen)
    return renderer


def _touch(path: Path, data: bytes = b"x") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


class TestLayer2_ImageOverlayRenderer:
    def test_direct_filter_complex_deduplicates_png_splits_reuse_and_keeps_layer_order(self, tmp_path: Path, monkeypatch):
        renderer = _patch_renderer_process(monkeypatch, captured := {})

        video = _touch(tmp_path / "video.mp4")
        audio = _touch(tmp_path / "audio.wav")
        subtitle = _touch(tmp_path / "subtitle.srt", b"1\n00:00:00,000 --> 00:00:01,000\nHello\n")
        note_ass = _touch(tmp_path / "note.ass", b"[Script Info]\n")
        image = _touch(tmp_path / "png" / "frame.png", b"png")
        strip = _touch(tmp_path / "strip.png", b"strip")
        watermark = _touch(tmp_path / "wm.png", b"wm")
        output = str(tmp_path / "out.mp4")

        render_config = {
            "resolution": {"bypass_scale": False, "width": 1280, "height": 720},
            "image_overlay": {
                "enabled": True,
                "render_strategy": "direct",
                "fit": "stretch_to_output",
                "opacity": 0.5,
                "x": "0",
                "y": "0",
            },
            "note_overlay": {"enabled": True},
            "black_strip": {
                "enabled": True,
                "path": strip,
                "scale_width": "1280",
                "scale_height": "80",
                "x": "0",
                "y": "640",
            },
            "watermark_img": {"enabled": True, "path": watermark, "x": "10", "y": "20"},
            "watermark_text": {"enabled": True, "text": "WM", "fontsize": 24, "x": "30", "y": "40"},
            "subtitles": {"enabled": True, "burn_hardsub": True, "style": {"FontSize": "24"}},
        }
        events = [
            ImageOverlayEvent("frame", image, 0.0, 1000.0, 1),
            ImageOverlayEvent("frame", image, 1000.0, 2000.0, 2),
        ]

        renderer.render_final_video(
            stretched_video=video,
            mixed_audio=audio,
            subtitle_synced_srt=subtitle,
            output_path=output,
            note_overlay_synced_ass=note_ass,
            render_config=render_config,
            image_overlay_events=events,
        )

        cmd = captured["cmd"]
        assert "-filter_complex" in cmd
        assert "-filter_complex_script" not in cmd
        assert cmd.count(image.replace("\\", "/")) == 1
        assert "hevc_nvenc" in cmd
        assert "-map" in cmd

        graph = cmd[cmd.index("-filter_complex") + 1]
        assert "[2:v]scale=1280:720,format=rgba,colorchannelmixer=aa=0.500[img_0_base]" in graph
        assert "[img_0_base]split=2[img_0_0][img_0_1]" in graph
        assert "enable='between(t,0.000,1.000)'" in graph
        assert "enable='between(t,1.000,2.000)'" in graph

        image_overlay_pos = graph.index("enable='between(t,0.000,1.000)'")
        note_pos = graph.index("ass='")
        strip_pos = graph.index("[bg_scaled]overlay")
        watermark_img_pos = graph.index("overlay=x=10:y=20")
        watermark_text_pos = graph.index("drawtext=")
        subtitle_pos = graph.index("subtitles='")
        assert image_overlay_pos < note_pos < strip_pos < watermark_img_pos < watermark_text_pos < subtitle_pos

    def test_script_strategy_writes_filter_complex_script_and_keeps_cmd_short(self, tmp_path: Path, monkeypatch):
        renderer = _patch_renderer_process(monkeypatch, captured := {})

        video = _touch(tmp_path / "video.mp4")
        audio = _touch(tmp_path / "audio.wav")
        subtitle = _touch(tmp_path / "subtitle.srt", b"1\n00:00:00,000 --> 00:00:01,000\nHello\n")
        image = _touch(tmp_path / "png" / "frame.png", b"png")
        output = str(tmp_path / "out.mp4")

        renderer.render_final_video(
            stretched_video=video,
            mixed_audio=audio,
            subtitle_synced_srt=subtitle,
            output_path=output,
            render_config={
                "resolution": {"bypass_scale": False, "width": 640, "height": 360},
                "image_overlay": {"enabled": True, "render_strategy": "script"},
                "subtitles": {"enabled": False},
            },
            image_overlay_events=[ImageOverlayEvent("frame", image, 0.0, 1000.0, 1)],
            filter_complex_script_dir=str(tmp_path / "scripts"),
            keep_filter_complex_script=True,
        )

        cmd = captured["cmd"]
        assert "-filter_complex_script" in cmd
        assert "-filter_complex" not in cmd
        script_path = Path(cmd[cmd.index("-filter_complex_script") + 1])
        assert script_path.exists()
        script_text = script_path.read_text(encoding="utf-8")
        assert "scale=640:360" in script_text
        assert "overlay=x=0:y=0:shortest=1:enable='between(t,0.000,1.000)'" in script_text

    def test_auto_strategy_switches_to_script_when_event_count_exceeds_threshold(self, tmp_path: Path, monkeypatch):
        renderer = _patch_renderer_process(monkeypatch, captured := {})

        video = _touch(tmp_path / "video.mp4")
        audio = _touch(tmp_path / "audio.wav")
        subtitle = _touch(tmp_path / "subtitle.srt", b"1\n00:00:00,000 --> 00:00:01,000\nHello\n")
        image = _touch(tmp_path / "png" / "frame.png", b"png")
        output = str(tmp_path / "out.mp4")
        events = [
            ImageOverlayEvent("frame", image, idx * 100.0, idx * 100.0 + 50.0, idx + 1)
            for idx in range(2)
        ]

        renderer.render_final_video(
            stretched_video=video,
            mixed_audio=audio,
            subtitle_synced_srt=subtitle,
            output_path=output,
            render_config={
                "resolution": {"bypass_scale": False, "width": 640, "height": 360},
                "image_overlay": {
                    "enabled": True,
                    "render_strategy": "auto",
                    "direct_overlay_max_events": 1,
                    "command_line_max_chars": 999999,
                },
                "subtitles": {"enabled": False},
            },
            image_overlay_events=events,
            filter_complex_script_dir=str(tmp_path / "scripts"),
            keep_filter_complex_script=True,
        )

        cmd = captured["cmd"]
        assert "-filter_complex_script" in cmd
        assert Path(cmd[cmd.index("-filter_complex_script") + 1]).exists()

    def test_intermediate_strategy_is_explicitly_not_implemented(self, tmp_path: Path, monkeypatch):
        renderer = _patch_renderer_process(monkeypatch, captured := {})

        video = _touch(tmp_path / "video.mp4")
        audio = _touch(tmp_path / "audio.wav")
        subtitle = _touch(tmp_path / "subtitle.srt", b"1\n00:00:00,000 --> 00:00:01,000\nHello\n")
        image = _touch(tmp_path / "png" / "frame.png", b"png")

        with pytest.raises(NotImplementedError, match="intermediate"):
            renderer.render_final_video(
                stretched_video=video,
                mixed_audio=audio,
                subtitle_synced_srt=subtitle,
                output_path=str(tmp_path / "out.mp4"),
                render_config={
                    "resolution": {"bypass_scale": False, "width": 640, "height": 360},
                    "image_overlay": {"enabled": True, "render_strategy": "intermediate"},
                    "subtitles": {"enabled": False},
                },
                image_overlay_events=[ImageOverlayEvent("frame", image, 0.0, 1000.0, 1)],
            )
