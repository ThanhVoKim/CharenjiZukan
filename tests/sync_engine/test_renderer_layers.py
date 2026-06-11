#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/sync_engine/test_renderer_layers.py
=========================================

Layer 2: Component tests cho renderer final-render layer ordering, watermark
width scaling, và skip_layers (black_strip nung ở stretch).

Mock subprocess.Popen để bắt FFmpeg command; không chạy FFmpeg thật.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


class _FakeProcess:
    returncode = 0

    def __init__(self):
        self.stderr = iter(())

    def wait(self):
        return 0


def _patch_renderer(monkeypatch, captured: dict):
    import sync_engine.renderer as renderer

    monkeypatch.setattr(renderer, "detect_hevc_nvenc", lambda: True)
    monkeypatch.setattr(renderer, "_get_video_duration", lambda _p: 0.0)

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return _FakeProcess()

    monkeypatch.setattr(renderer.subprocess, "Popen", fake_popen)
    return renderer


def _touch(path: Path, data: bytes = b"x") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def _base_inputs(tmp_path: Path):
    return {
        "video": _touch(tmp_path / "video.mp4"),
        "audio": _touch(tmp_path / "audio.wav"),
        "subtitle": _touch(tmp_path / "sub.srt", b"1\n00:00:00,000 --> 00:00:01,000\nHi\n"),
    }


class TestLayer2_RendererWatermarkWidth:
    def test_watermark_width_scales_with_aspect(self, tmp_path: Path, monkeypatch):
        renderer = _patch_renderer(monkeypatch, captured := {})
        ins = _base_inputs(tmp_path)
        wm = _touch(tmp_path / "wm.png", b"wm")
        cfg = {
            "resolution": {"bypass_scale": False, "width": 1280, "height": 720},
            "watermark_img": {"enabled": True, "path": wm, "width": 300, "x": "10", "y": "20"},
        }
        renderer.render_final_video(
            stretched_video=ins["video"], mixed_audio=ins["audio"],
            subtitle_synced_srt=ins["subtitle"], output_path=str(tmp_path / "o.mp4"),
            render_config=cfg,
        )
        graph = captured["cmd"][captured["cmd"].index("-filter_complex") + 1]
        # width > 0 → scale theo width, height auto (-1) giữ aspect
        assert "scale=300:-1[wm_scaled]" in graph
        assert "[wm_scaled]overlay=x=10:y=20[v_wm_img]" in graph

    def test_watermark_width_null_keeps_original(self, tmp_path: Path, monkeypatch):
        renderer = _patch_renderer(monkeypatch, captured := {})
        ins = _base_inputs(tmp_path)
        wm = _touch(tmp_path / "wm.png", b"wm")
        cfg = {
            "resolution": {"bypass_scale": False, "width": 1280, "height": 720},
            "watermark_img": {"enabled": True, "path": wm, "width": None, "x": "10", "y": "20"},
        }
        renderer.render_final_video(
            stretched_video=ins["video"], mixed_audio=ins["audio"],
            subtitle_synced_srt=ins["subtitle"], output_path=str(tmp_path / "o.mp4"),
            render_config=cfg,
        )
        graph = captured["cmd"][captured["cmd"].index("-filter_complex") + 1]
        assert "wm_scaled" not in graph
        assert "overlay=x=10:y=20[v_wm_img]" in graph


class TestLayer2_RendererSkipLayers:
    def test_skip_black_strip_omits_strip_overlay(self, tmp_path: Path, monkeypatch):
        renderer = _patch_renderer(monkeypatch, captured := {})
        ins = _base_inputs(tmp_path)
        strip = _touch(tmp_path / "strip.png", b"s")
        cfg = {
            "resolution": {"bypass_scale": False, "width": 1280, "height": 720},
            "black_strip": {
                "enabled": True, "path": strip,
                "scale_width": "1280", "scale_height": "80", "x": "0", "y": "640",
            },
            "subtitles": {"enabled": True, "burn_hardsub": True, "style": {"FontSize": "24"}},
        }
        renderer.render_final_video(
            stretched_video=ins["video"], mixed_audio=ins["audio"],
            subtitle_synced_srt=ins["subtitle"], output_path=str(tmp_path / "o.mp4"),
            render_config=cfg, skip_layers={"black_strip"},
        )
        graph = captured["cmd"][captured["cmd"].index("-filter_complex") + 1]
        assert "bg_scaled" not in graph
        assert "v_strip" not in graph
        # strip không được nạp làm input loop
        assert strip.replace("\\", "/") not in captured["cmd"]

    def test_without_skip_strip_is_rendered(self, tmp_path: Path, monkeypatch):
        renderer = _patch_renderer(monkeypatch, captured := {})
        ins = _base_inputs(tmp_path)
        strip = _touch(tmp_path / "strip.png", b"s")
        cfg = {
            "resolution": {"bypass_scale": False, "width": 1280, "height": 720},
            "black_strip": {
                "enabled": True, "path": strip,
                "scale_width": "1280", "scale_height": "80", "x": "0", "y": "640",
            },
        }
        renderer.render_final_video(
            stretched_video=ins["video"], mixed_audio=ins["audio"],
            subtitle_synced_srt=ins["subtitle"], output_path=str(tmp_path / "o.mp4"),
            render_config=cfg,
        )
        graph = captured["cmd"][captured["cmd"].index("-filter_complex") + 1]
        assert "[bg_scaled]overlay=x=0:y=640:shortest=1[v_strip]" in graph


class TestLayer2_RendererLayerOrder:
    def test_layer_order_controls_sequence(self, tmp_path: Path, monkeypatch):
        renderer = _patch_renderer(monkeypatch, captured := {})
        ins = _base_inputs(tmp_path)
        wm = _touch(tmp_path / "wm.png", b"wm")
        # Đảo subtitles xuống trước watermark_img để verify layer_order điều khiển thứ tự
        cfg = {
            "resolution": {"bypass_scale": False, "width": 1280, "height": 720},
            "watermark_img": {"enabled": True, "path": wm, "x": "10", "y": "20"},
            "subtitles": {"enabled": True, "burn_hardsub": True, "style": {"FontSize": "24"}},
            "layer_order": ["subtitles", "watermark_img"],
        }
        renderer.render_final_video(
            stretched_video=ins["video"], mixed_audio=ins["audio"],
            subtitle_synced_srt=ins["subtitle"], output_path=str(tmp_path / "o.mp4"),
            render_config=cfg,
        )
        graph = captured["cmd"][captured["cmd"].index("-filter_complex") + 1]
        assert graph.index("subtitles='") < graph.index("overlay=x=10:y=20")

    def test_unknown_layer_is_ignored(self, tmp_path: Path, monkeypatch):
        renderer = _patch_renderer(monkeypatch, captured := {})
        ins = _base_inputs(tmp_path)
        cfg = {
            "resolution": {"bypass_scale": False, "width": 1280, "height": 720},
            "subtitles": {"enabled": True, "burn_hardsub": True, "style": {"FontSize": "24"}},
            "layer_order": ["bogus_layer", "subtitles"],
        }
        renderer.render_final_video(
            stretched_video=ins["video"], mixed_audio=ins["audio"],
            subtitle_synced_srt=ins["subtitle"], output_path=str(tmp_path / "o.mp4"),
            render_config=cfg,
        )
        graph = captured["cmd"][captured["cmd"].index("-filter_complex") + 1]
        assert "subtitles='" in graph
