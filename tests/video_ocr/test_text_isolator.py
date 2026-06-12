#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/video_ocr/test_text_isolator.py
======================================
Test cho video_subtitle_extractor/text_isolator.py — lọc watermark/overlay mờ
khỏi phụ đề bằng opacity/color masking (thuần OpenCV, không model, không GPU).

Cấu trúc layers:
  Layer 1 — Unit: parse_color_spec, TextIsolationConfig (thuần Python).
  Layer 2 — Component: isolate_subtitle_text trên ảnh tổng hợp (cần cv2/numpy).

Nguyên lý kiểm thử: watermark có opacity < 70% → khi blend với nền tối, vừa giảm
tương phản (gradient) vừa lệch màu → bị xóa; phụ đề đặc (α≈1) đúng màu → được giữ.

Cách chạy:
    pytest tests/video_ocr/test_text_isolator.py -v -k "Layer1"
    pytest tests/video_ocr/test_text_isolator.py -v -k "Layer2"
"""

import sys
from pathlib import Path
from typing import List, Tuple

import pytest

# ── Project root ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Lazy imports (skip nếu thiếu dependency nặng) ───────────────────
# Package video_subtitle_extractor.__init__ kéo theo cv2 (qua extractor),
# nên importorskip cv2 trước khi import bất kỳ thứ gì từ package.
cv2 = pytest.importorskip("cv2", reason="opencv-python chưa cài: pip install opencv-python")
np = pytest.importorskip("numpy", reason="numpy chưa cài")

from video_subtitle_extractor.text_isolator import (  # noqa: E402
    TextIsolationConfig,
    isolate_subtitle_text,
    parse_color_spec,
)


# ═════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════

BG = 20  # nền tối (giống synthetic video trong dự án)


def _blend(color_bgr: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int]:
    """Trộn màu với nền tối theo opacity α: hiển_thị = α·color + (1−α)·BG."""
    return tuple(int(round(alpha * c + (1 - alpha) * BG)) for c in color_bgr)


def _make_image(
    glyphs: List[Tuple[int, int, int, int, Tuple[int, int, int], float]],
    h: int = 80,
    w: int = 400,
) -> np.ndarray:
    """Tạo ảnh BGR nền tối với các 'glyph' hình chữ nhật.

    Mỗi glyph: (x, y, gw, gh, color_bgr, alpha). alpha=1.0 = đặc (phụ đề),
    alpha<0.7 = mờ (watermark).
    """
    img = np.full((h, w, 3), BG, dtype=np.uint8)
    for x, y, gw, gh, color, alpha in glyphs:
        img[y:y + gh, x:x + gw] = _blend(color, alpha)
    return img


def _filled_pixels(clean: np.ndarray, box: Tuple[int, int, int, int]) -> int:
    """Số pixel khác đen (đã giữ lại) trong vùng box (x, y, w, h)."""
    x, y, w, h = box
    region = clean[y:y + h, x:x + w]
    return int(np.count_nonzero(region.any(axis=2)))


WHITE = (255, 255, 255)
YELLOW_BGR = (0, 215, 255)  # vàng #FFD700 ở BGR
RED_BGR = (0, 0, 255)


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_ParseColorSpec:
    """parse_color_spec: tên / hex / bộ ba RGB / nhiều màu."""

    def test_name_and_hex(self):
        assert parse_color_spec("white,#FFD700") == [(255, 255, 255), (255, 215, 0)]

    def test_single_rgb_triple_with_commas(self):
        assert parse_color_spec("255,255,255") == [(255, 255, 255)]

    def test_semicolon_with_rgb_triple(self):
        assert parse_color_spec("white;255:215:0") == [(255, 255, 255), (255, 215, 0)]

    def test_named_and_short_hex(self):
        assert parse_color_spec("gold") == [(255, 215, 0)]
        assert parse_color_spec("#FFF") == [(255, 255, 255)]

    def test_empty_and_invalid(self):
        assert parse_color_spec("") == []
        assert parse_color_spec("nonsense,white") == [(255, 255, 255)]


class TestLayer1_Config:
    """TextIsolationConfig: mặc định an toàn (tắt)."""

    def test_default_disabled(self):
        cfg = TextIsolationConfig()
        assert cfg.enabled is False
        assert cfg.subtitle_colors == []
        assert cfg.min_contrast == 40


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — COMPONENT TESTS (synthetic images, no GPU/model)
# ═════════════════════════════════════════════════════════════════════

class TestLayer2_IsolateSubtitleText:
    """isolate_subtitle_text: giữ phụ đề đặc, xóa watermark mờ."""

    def test_disabled_returns_unchanged(self):
        img = _make_image([(10, 20, 40, 40, WHITE, 1.0)])
        cfg = TextIsolationConfig(enabled=False)
        out = isolate_subtitle_text(img, cfg)
        assert out is img  # không đụng tới khi tắt

    def test_keeps_opaque_white_text(self):
        img = _make_image([(20, 20, 60, 40, WHITE, 1.0)])
        cfg = TextIsolationConfig(
            enabled=True, subtitle_colors=[(255, 255, 255)],
            color_tolerance=50, min_contrast=150, require_stroke=False,
        )
        out = isolate_subtitle_text(img, cfg)
        assert _filled_pixels(out, (20, 20, 60, 40)) > 100

    def test_removes_faded_watermark(self):
        # Watermark trắng opacity 0.5 → mờ → phải bị xóa.
        img = _make_image([(20, 20, 60, 40, WHITE, 0.5)])
        cfg = TextIsolationConfig(
            enabled=True, subtitle_colors=[(255, 255, 255)],
            color_tolerance=50, min_contrast=150, require_stroke=False,
        )
        out = isolate_subtitle_text(img, cfg)
        assert _filled_pixels(out, (20, 20, 60, 40)) == 0

    def test_keeps_colored_subtitle_matching_spec(self):
        img = _make_image([(20, 20, 60, 40, YELLOW_BGR, 1.0)])
        cfg = TextIsolationConfig(
            enabled=True, subtitle_colors=[(255, 215, 0)],  # vàng RGB
            color_tolerance=50, min_contrast=120, require_stroke=False,
        )
        out = isolate_subtitle_text(img, cfg)
        assert _filled_pixels(out, (20, 20, 60, 40)) > 100

    def test_removes_color_not_in_spec(self):
        # Chữ đỏ đặc nhưng spec chỉ định màu vàng → color gate loại.
        img = _make_image([(20, 20, 60, 40, RED_BGR, 1.0)])
        cfg = TextIsolationConfig(
            enabled=True, subtitle_colors=[(255, 215, 0)],
            color_tolerance=40, min_contrast=100, require_stroke=False,
        )
        out = isolate_subtitle_text(img, cfg)
        assert _filled_pixels(out, (20, 20, 60, 40)) == 0

    def test_overlap_keeps_subtitle_drops_watermark(self):
        # Cùng một ảnh: phụ đề trắng đặc (trái) + watermark trắng mờ (phải).
        img = _make_image([
            (20, 20, 60, 40, WHITE, 1.0),    # phụ đề đặc
            (300, 20, 60, 40, WHITE, 0.5),   # watermark mờ
        ])
        cfg = TextIsolationConfig(
            enabled=True, subtitle_colors=[(255, 255, 255)],
            color_tolerance=50, min_contrast=150, require_stroke=False,
        )
        out = isolate_subtitle_text(img, cfg)
        assert _filled_pixels(out, (20, 20, 60, 40)) > 100      # phụ đề giữ
        assert _filled_pixels(out, (300, 20, 60, 40)) == 0      # watermark xóa

    def test_min_component_area_removes_speckle(self):
        # Đốm 2x2 (area=4) phải bị loại với min_component_area=20.
        img = _make_image([(20, 20, 2, 2, WHITE, 1.0)])
        cfg = TextIsolationConfig(
            enabled=True, subtitle_colors=[(255, 255, 255)],
            color_tolerance=50, min_contrast=100, min_component_area=20,
            require_stroke=False,
        )
        out = isolate_subtitle_text(img, cfg)
        assert _filled_pixels(out, (0, 0, 400, 80)) == 0

    def test_empty_roi_returns_unchanged(self):
        cfg = TextIsolationConfig(enabled=True)
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        out = isolate_subtitle_text(empty, cfg)
        assert out.size == 0
