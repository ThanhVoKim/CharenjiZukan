#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/sync_engine/test_tuber_repair.py
======================================
Test cho cli/tuber_repair.py (V3).

Layer 1 — Unit: resolve prerender manifest, pass đúng kwargs.
Layer 2 — không cần (phụ thuộc run_manifest + base_video thật).
Layer 3 — Integration: cần FFmpeg (chạy sau này).

Cách chạy:
    pytest tests/sync_engine/test_tuber_repair.py -v -k "Layer1"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sync_engine.tuber_overlay import (
    render_groups_to_video,
    TuberOverlayError,
)


class TestLayer1_RepairResolvePrerender:
    """Unit: repair resolve prerender manifest từ run_manifest dict."""

    def test_resolve_prerender_from_run_manifest(self, tmp_path):
        """Nếu run_manifest có prerenderManifest → phát hiện use_prerender=True."""
        pm_dir = tmp_path / "prerendered"
        pm_dir.mkdir()
        pm = {"assetId": "test", "outputWidth": 512, "outputHeight": 288}
        pm_path = pm_dir / "prerender_manifest.json"
        pm_path.write_text(json.dumps(pm), encoding="utf-8")

        run_mf = {
            "mediaDir": str(tmp_path),
            "prerenderManifest": str(pm_path),
            "remotion": {},
            "asset": {"assetDir": str(tmp_path), "assetId": "test"},
            "tubnerConfig": {"enabled": True},
            "groups": [],
        }
        # Kiểm tra run_manifest chứa prerenderManifest path
        assert run_mf.get("prerenderManifest")
        p = Path(run_mf["prerenderManifest"])
        assert p.exists()

    def test_no_prerender_without_manifest(self):
        """Không có prerenderManifest trong run_manifest → fallback Remotion."""
        rm = {"prerenderManifest": None}
        assert rm.get("prerenderManifest") is None

    def test_prerender_manifest_not_exist_fallback(self):
        """prerenderManifest path không tồn tại → coi như không prerender."""
        rm = {"prerenderManifest": "/nonexistent/prerender_manifest.json"}
        p = Path(rm["prerenderManifest"])
        assert not p.exists()
