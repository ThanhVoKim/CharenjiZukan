#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/sync_engine/test_tuber_prerender.py
==========================================
Test cho sync_engine/tuber_prerender.py.

Cấu trúc layers:
  Layer 1 — Unit: affine math, character box, track frame index
  Layer 2 — Component: warp synthetic sprite, pre-render 1 frame
  Layer 3 — Integration: pre-render toàn bộ + FFmpeg composite

Cách chạy:
    pytest tests/sync_engine/test_tuber_prerender.py -v -k "Layer1"
    pytest tests/sync_engine/test_tuber_prerender.py -v -k "Layer2"
    pytest tests/sync_engine/test_tuber_prerender.py -v -k "Layer3"
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sync_engine.tuber_prerender import (
    compute_affine,
    compute_character_box,
    compute_track_frame_index,
    apply_mouth_calibration,
    get_prerender_frame,
    _invert_affine,
    _lookup_mouth_state,
)


# ══════════════════════════════════════════════════════════════════════════
# LAYER 1 — Unit tests
# ══════════════════════════════════════════════════════════════════════════

class TestLayer1_AffineMath:
    """Unit: compute_affine, _invert_affine."""

    def test_identity_affine(self):
        """Identity mapping should return identity matrix."""
        m = compute_affine(
            (0, 0), (1, 0), (0, 1),
            (0, 0), (1, 0), (0, 1),
        )
        assert m is not None
        a, b, c, d, e, f = m
        # identity: a=1, d=1, others=0
        assert abs(a - 1) < 1e-6
        assert abs(b) < 1e-6
        assert abs(c) < 1e-6
        assert abs(d - 1) < 1e-6
        assert abs(e) < 1e-6
        assert abs(f) < 1e-6

    def test_scale_affine(self):
        """2x scale: map (0,0),(1,0),(0,1) → (0,0),(2,0),(0,2)."""
        m = compute_affine(
            (0, 0), (1, 0), (0, 1),
            (0, 0), (2, 0), (0, 2),
        )
        assert m is not None
        a, b, c, d, e, f = m
        assert abs(a - 2) < 1e-6  # scale x
        assert abs(d - 2) < 1e-6  # scale y
        assert abs(b) < 1e-6
        assert abs(c) < 1e-6

    def test_translation_affine(self):
        """Translation: map (0,0),(1,0),(0,1) → (10,10),(11,10),(10,11)."""
        m = compute_affine(
            (0, 0), (1, 0), (0, 1),
            (10, 10), (11, 10), (10, 11),
        )
        assert m is not None
        a, b, c, d, e, f = m
        assert abs(a - 1) < 1e-6
        assert abs(d - 1) < 1e-6
        assert abs(e - 10) < 1e-6
        assert abs(f - 10) < 1e-6

    def test_degenerate_triangle(self):
        """Colinear source → denominator = 0 → return None."""
        m = compute_affine(
            (0, 0), (1, 0), (2, 0),  # colinear
            (0, 0), (1, 0), (0, 1),
        )
        assert m is None

    def test_invert_affine(self):
        """Invert identity → identity (PIL format: src_x = a*dst_x + b*dst_y + c)."""
        inv = _invert_affine(1, 0, 0, 1, 0, 0)
        assert inv is not None
        pil_a, pil_b, pil_c, pil_d, pil_e, pil_f = inv
        # PIL: src_x = pil_a*dst_x + pil_b*dst_y + pil_c = 1*dst_x + 0 + 0 = dst_x
        assert abs(pil_a - 1) < 1e-6
        assert abs(pil_b) < 1e-6
        assert abs(pil_c) < 1e-6
        # PIL: src_y = pil_d*dst_x + pil_e*dst_y + pil_f = 0 + 1*dst_y + 0 = dst_y
        assert abs(pil_d) < 1e-6
        assert abs(pil_e - 1) < 1e-6
        assert abs(pil_f) < 1e-6

    def test_invert_scale(self):
        """Inverse of 2x scale → 0.5x scale."""
        inv = _invert_affine(2, 0, 0, 2, 0, 0)
        assert inv is not None
        pil_a, pil_b, pil_c, pil_d, pil_e, pil_f = inv
        assert abs(pil_a - 0.5) < 1e-6  # src_x = 0.5*dst_x
        assert abs(pil_b) < 1e-6
        assert abs(pil_c) < 1e-6
        assert abs(pil_d) < 1e-6  # src_y = 0.5*dst_y
        assert abs(pil_e - 0.5) < 1e-6
        assert abs(pil_f) < 1e-6

    def test_invert_translate(self):
        """Inverse of translate (10,20) → translate (-10,-20)."""
        inv = _invert_affine(1, 0, 0, 1, 10, 20)
        assert inv is not None
        pil_a, pil_b, pil_c, pil_d, pil_e, pil_f = inv
        # pil_c/ pil_f are translation components in PIL inverse format
        assert abs(pil_c - (-10)) < 1e-6  # src_x = dst_x + 0 + (-10)
        assert abs(pil_f - (-20)) < 1e-6  # src_y = 0 + dst_y + (-20)


class TestLayer1_CharacterBox:
    """Unit: compute_character_box."""

    def test_width_priority(self):
        box = compute_character_box(
            {"width": 400, "height": 999},
            1920, 1080, 1920 / 1080,
        )
        assert box["compWidth"] == 400
        assert box["compHeight"] == 225  # 400 / (16/9)

    def test_height_only(self):
        box = compute_character_box(
            {"height": 540, "left": 0, "top": 0},
            1920, 1080, 1920 / 1080,
        )
        assert box["compHeight"] == 540
        assert box["compWidth"] == 960  # 540 * (16/9)

    def test_width_ratio(self):
        box = compute_character_box(
            {"widthRatio": 0.5},
            1920, 1080, 1920 / 1080,
        )
        assert box["compWidth"] == 960
        assert box["compOffsetX"] == 1152  # 0.6 * 1920

    def test_default_fallback(self):
        box = compute_character_box(
            {},
            1920, 1080, 16 / 9,
        )
        assert box["compHeight"] == 648  # 0.6 * 1080
        assert abs(box["compWidth"] - 1152) < 0.01  # 648 * 16/9

    def test_left_top_override(self):
        box = compute_character_box(
            {"width": 512, "left": 100, "top": 200},
            1920, 1080, 1920 / 1080,
        )
        assert box["compOffsetX"] == 100
        assert box["compOffsetY"] == 200


class TestLayer1_TrackFrameIndex:
    """Unit: compute_track_frame_index."""

    def test_frame_zero(self):
        idx = compute_track_frame_index(0, 30, 30, 170)
        assert idx == 0

    def test_frame_within_loop(self):
        idx = compute_track_frame_index(50, 30, 30, 170)
        assert idx == 50

    def test_frame_past_loop_wraps(self):
        idx = compute_track_frame_index(170, 30, 30, 170)
        assert idx == 0  # wrap around

    def test_custom_loop_frames(self):
        idx = compute_track_frame_index(200, 30, 30, 170, loop_frames=170)
        # 200 % 170 = 30 → track_idx = floor(30/30 * 30) % 170 = 30
        assert idx == 30

    def test_different_fps_mismatch(self):
        # composition 29.97fps, track 30fps
        idx = compute_track_frame_index(100, 29.97, 30, 170)
        assert 0 <= idx < 170

    def test_single_frame_track(self):
        idx = compute_track_frame_index(9999, 30, 30, 1)
        assert idx == 0


class TestLayer1_ApplyCalibration:
    """Unit: apply_mouth_calibration."""

    def test_no_calibration_returns_identical(self):
        quad = [(100, 200), (200, 200), (200, 300), (100, 300)]
        result = apply_mouth_calibration(quad, None)
        assert result == quad

    def test_calibration_not_applied(self):
        quad = [(100, 200), (200, 200), (200, 300), (100, 300)]
        track = {"calibrationApplied": False}
        result = apply_mouth_calibration(quad, track)
        assert result == quad

    def test_calibration_applied_returns_quad(self):
        quad = [(100, 200), (200, 200), (200, 300), (100, 300)]
        track = {
            "calibrationApplied": True,
            "calibration": {"offset": [10, 10], "scale": 1, "rotation": 0},
        }
        result = apply_mouth_calibration(quad, track)
        assert len(result) == 4
        # scale=1, rotation=0 → just offset
        assert abs(result[0][0] - 110) < 0.01


class TestLayer1_GetPrerenderFrame:
    """Unit: get_prerender_frame."""

    def test_frame_path(self):
        p = get_prerender_frame(0, "closed", Path("/prerendered"), {"framePattern": "frame-{trackIdx:03d}_{state}.png"})
        assert p == Path("/prerendered") / "frame-000_closed.png"

    def test_last_frame(self):
        p = get_prerender_frame(169, "open", Path("/prerendered"), {"framePattern": "frame-{trackIdx:03d}_{state}.png"})
        assert p == Path("/prerendered") / "frame-169_open.png"


class TestLayer1_LookupMouthState:
    """Unit: _lookup_mouth_state."""

    def test_lookup_returns_closed_for_empty(self):
        state = _lookup_mouth_state(50, {})
        assert state == "closed"

    def test_lookup_segment_events(self):
        events_map = {
            "0": [
                {"frame": 0, "state": "closed"},
                {"frame": 10, "state": "open"},
                {"frame": 30, "state": "closed"},
            ],
        }
        at_5 = _lookup_mouth_state(5, events_map)
        assert at_5 == "closed"
        at_15 = _lookup_mouth_state(15, events_map)
        assert at_15 == "open"
        at_50 = _lookup_mouth_state(50, events_map)
        assert at_50 == "closed"


# ══════════════════════════════════════════════════════════════════════════
# LAYER 2 — Component tests (cần PIL)
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not os.environ.get("PRERENDER_E2E"),
                    reason="Set PRERENDER_E2E=1 để test PIL warp thật")
class TestLayer2_PILWarp:
    """Component: warp_sprite_to_quad với synthetic data."""

    def test_warp_identity_quad(self):
        """Warp sprite onto same-size quad → no distortion."""
        from PIL import Image
        from sync_engine.tuber_prerender import warp_sprite_to_quad

        # Create test sprite
        sprite = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        body = Image.new("RGBA", (200, 200), (0, 255, 0, 255))
        quad = [(0, 0), (100, 0), (100, 100), (0, 100)]

        canvas = warp_sprite_to_quad(body, sprite, quad, canvas_size=(200, 200))
        assert canvas is not None
        assert canvas.size == (200, 200)

    def test_warp_produces_valid_rgba(self):
        from PIL import Image
        from sync_engine.tuber_prerender import warp_sprite_to_quad

        sprite = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        body = Image.new("RGBA", (200, 200), (0, 255, 0, 255))
        quad = [(10, 10), (90, 10), (90, 90), (10, 90)]

        canvas = warp_sprite_to_quad(body, sprite, quad, canvas_size=(200, 200))
        assert canvas.mode == "RGBA"


# ══════════════════════════════════════════════════════════════════════════
# LAYER 3 — Integration tests (cần đầy đủ asset)
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not os.environ.get("PRERENDER_E2E"),
                    reason="Set PRERENDER_E2E=1 để test pre-render thật")
class TestLayer3_PrerenderIntegration:
    """Integration: pre-render thật với nike_loop_fix asset."""

    ASSET_DIR = PROJECT_ROOT / "assets" / "pngtuber" / "nike_loop_fix"

    def test_prerender_one_frame(self):
        from sync_engine.tuber_prerender import prerender_character, warp_sprite_to_quad
        from PIL import Image

        body_dir = self.ASSET_DIR / "body-transparent"
        mouth_dir = self.ASSET_DIR / "mouth"
        track_path = self.ASSET_DIR / "mouth_track.json"

        if not body_dir.exists():
            pytest.skip("body-transparent chưa có (chạy extract_body_transparent trước)")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = prerender_character(
                body_dir, mouth_dir, track_path,
                ["closed", "half", "open"],
                out,
                character_box={"compWidth": 512, "compHeight": 288,
                                "compOffsetX": 1280, "compOffsetY": 360},
            )
            assert manifest["outputCount"] > 0
            # Kiểm tra 1 output file
            assert (out / "frame-000_closed.png").exists()
            assert (out / "frame-000_open.png").exists()

    def test_prerender_manifest_structure(self):
        from sync_engine.tuber_prerender import prerender_character

        body_dir = self.ASSET_DIR / "body-transparent"
        mouth_dir = self.ASSET_DIR / "mouth"
        track_path = self.ASSET_DIR / "mouth_track.json"

        if not body_dir.exists():
            pytest.skip("body-transparent chưa có (chạy extract_body_transparent trước)")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = prerender_character(
                body_dir, mouth_dir, track_path,
                ["closed", "half"],
                out,
            )
            assert manifest["mouthStates"] == ["closed", "half"]
            assert manifest["outputWidth"] > 0
            assert manifest["outputHeight"] > 0
            assert manifest["framePattern"] == "frame-{trackIdx:03d}_{state}.png"

    def test_build_prerender_frame_sequence(self):
        from sync_engine.tuber_prerender import (
            prerender_character, build_prerender_frame_sequence,
        )

        body_dir = self.ASSET_DIR / "body-transparent"
        mouth_dir = self.ASSET_DIR / "mouth"
        track_path = self.ASSET_DIR / "mouth_track.json"

        if not body_dir.exists():
            pytest.skip("body-transparent chưa có (chạy extract_body_transparent trước)")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            pm = prerender_character(body_dir, mouth_dir, track_path, ["closed", "half", "open"], out)

            frames = build_prerender_frame_sequence(
                0, 50, 30, 30, 170, out, pm, {"0": [{"frame": 0, "state": "half"}]},
            )
            assert len(frames) == 50
            for fp in frames:
                assert fp.exists(), f"Thiếu frame: {fp}"
