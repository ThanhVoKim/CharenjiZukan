#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/utils/test_ffmpeg_probe.py
=================================
Tests for shared FFmpeg runtime capability probes.

Cấu trúc layers:
  Layer 1 — Unit Tests          (monkeypatch subprocess/shutil, no real FFmpeg)

Cách chạy:
    pytest tests/utils/test_ffmpeg_probe.py -v -k "Layer1"
"""

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import ffmpeg_probe
from utils.ffmpeg_probe import detect_hevc_nvenc, get_hevc_nvenc_unavailable_reason


@pytest.fixture(autouse=True)
def reset_probe_cache():
    ffmpeg_probe.reset_hevc_nvenc_probe_cache()
    yield
    ffmpeg_probe.reset_hevc_nvenc_probe_cache()


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_FFmpegProbe:
    def test_detect_hevc_nvenc_rejects_encoder_list_false_positive(self, monkeypatch):
        """`ffmpeg -encoders` có hevc_nvenc nhưng dummy encode fail thì phải trả False."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if "-encoders" in cmd:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=" V....D hevc_nvenc NVIDIA NVENC hevc encoder\n",
                    stderr="",
                )
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="[hevc_nvenc] Cannot load libcuda.so.1",
            )

        monkeypatch.setattr(ffmpeg_probe.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(ffmpeg_probe.subprocess, "run", fake_run)

        assert detect_hevc_nvenc(force_refresh=True) is False
        assert "libcuda.so.1" in get_hevc_nvenc_unavailable_reason()
        assert len(calls) == 2
        assert "-encoders" in calls[0]
        assert "-frames:v" in calls[1]

    def test_detect_hevc_nvenc_success_is_cached(self, monkeypatch):
        """Probe thành công được cache để không encode dummy nhiều lần trong cùng process."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if "-encoders" in cmd:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=" V....D hevc_nvenc NVIDIA NVENC hevc encoder\n",
                    stderr="",
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(ffmpeg_probe.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(ffmpeg_probe.subprocess, "run", fake_run)

        assert detect_hevc_nvenc(force_refresh=True) is True
        assert detect_hevc_nvenc() is True
        assert len(calls) == 2

    def test_detect_hevc_nvenc_reports_missing_ffmpeg_without_subprocess(self, monkeypatch):
        """Không có ffmpeg trong PATH thì fail-fast và không gọi subprocess."""
        monkeypatch.setattr(ffmpeg_probe.shutil, "which", lambda name: None)

        def fail_run(*args, **kwargs):
            pytest.fail("subprocess.run không nên được gọi khi thiếu ffmpeg")

        monkeypatch.setattr(ffmpeg_probe.subprocess, "run", fail_run)

        assert detect_hevc_nvenc(force_refresh=True) is False
        assert "Không tìm thấy ffmpeg" in get_hevc_nvenc_unavailable_reason()
