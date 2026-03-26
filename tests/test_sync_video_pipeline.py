#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_sync_video_pipeline.py
===========================
Test Layer 3: Pipeline Integration
run_sync_pipeline từ cli/sync_video.py với dữ liệu tổng hợp.

Cấu trúc layers:
  Layer 3 — Pipeline Integration

Cách chạy từng layer:
    pytest tests/test_sync_video_pipeline.py -v -k "Layer3"
"""

import sys
import shutil
import argparse
import subprocess
from pathlib import Path

import pytest

# ── Project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

cv2 = pytest.importorskip("cv2", reason="pip install opencv-python")
pydub = pytest.importorskip("pydub", reason="pip install pydub")
import numpy as np
from pydub import AudioSegment

from cli.sync_video import run_sync_pipeline


# ═════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ═════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory) -> Path:
    """Video 3 giây."""
    tmp_dir = tmp_path_factory.mktemp("video")
    path = tmp_dir / "synthetic_test.mp4"
    W, H, FPS = 640, 360, 30
    total_frames = int(3 * FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (W, H))

    for frame_no in range(total_frames):
        frame = np.full((H, W, 3), 100, dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return path

def _make_synthetic_inputs(tmp_dir: Path, tts_duration_ms: int):
    """Tạo SRT + 1 TTS clip với độ dài tùy chỉnh."""
    srt_path = tmp_dir / "sub.srt"
    srt_path.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nHello\n",
        encoding="utf-8"
    )

    tts_dir = tmp_dir / "tts"
    tts_dir.mkdir()
    tts_path = tts_dir / "dubb-0.wav"

    silence = AudioSegment.silent(duration=tts_duration_ms, frame_rate=48000)
    silence.set_channels(2).export(str(tts_path), format="wav")

    return {
        "srt": str(srt_path),
        "tts_dir": str(tts_dir)
    }


@pytest.fixture(scope="module")
def synthetic_inputs_borrow_gap(tmp_path_factory):
    """Case 1: TTS 2s, vừa đúng slot 1->3s nhờ mượn gap (không slow)."""
    tmp_dir = tmp_path_factory.mktemp("inputs_borrow_gap")
    return _make_synthetic_inputs(tmp_dir, tts_duration_ms=2000)


@pytest.fixture(scope="module")
def synthetic_inputs_force_slowdown(tmp_path_factory):
    """Case 2: TTS 4s, vượt slot 1->3s nên bắt buộc slow video."""
    tmp_dir = tmp_path_factory.mktemp("inputs_force_slowdown")
    return _make_synthetic_inputs(tmp_dir, tts_duration_ms=4000)


# ═════════════════════════════════════════════════════════════════════
# LAYER 3 — PIPELINE INTEGRATION
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg không có trong PATH")
class TestLayer3_SyncVideoPipeline:
    def _run_pipeline(self, synthetic_video, synthetic_inputs, output_dir: Path, output_name: str) -> Path:
        args = argparse.Namespace(
            video=str(synthetic_video),
            subtitle=synthetic_inputs["srt"],
            tts_dir=synthetic_inputs["tts_dir"],
            mute=None,
            note_overlay_png=None,
            note_overlay_ass=None,
            black_bg=None,
            ambient=None,
            slow_cap=0.5,
            output_dir=str(output_dir),
            output_name=output_name,
            no_hardsub=False,
            workers=2,
            no_gpu=True,  # CPU mode for CI
            subtitle_fontname="Arial",
            subtitle_fontsize=22,
            subtitle_color="&H00FFFFFF",
            subtitle_margin_v=6,
            note_max_chars=15,
        )

        run_sync_pipeline(args)

        final_video = output_dir / f"{output_name}.mp4"
        tts_srt = output_dir / f"{output_name}_tts_synced.srt"
        full_srt = output_dir / f"{output_name}_synced.srt"

        assert final_video.exists()
        assert tts_srt.exists()
        assert full_srt.exists()

        return final_video

    def _probe_duration(self, video_path: Path) -> float:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())

    def test_run_sync_pipeline_case1_borrow_gap(self, synthetic_video, synthetic_inputs_borrow_gap, tmp_path):
        """Case 1 (plan): TTS vừa trong slot mở rộng 1->3s, không cần slow video."""
        output_dir = tmp_path / "output_case1"
        output_dir.mkdir()

        final_video = self._run_pipeline(
            synthetic_video,
            synthetic_inputs_borrow_gap,
            output_dir,
            output_name="video_synced_case1",
        )

        dur = self._probe_duration(final_video)
        assert 2.8 <= dur <= 3.3

    def test_run_sync_pipeline_case2_force_slowdown(self, synthetic_video, synthetic_inputs_force_slowdown, tmp_path):
        """Case 2 (plan): TTS vượt slot 1->3s, buộc pipeline slow video."""
        output_dir = tmp_path / "output_case2"
        output_dir.mkdir()

        final_video = self._run_pipeline(
            synthetic_video,
            synthetic_inputs_force_slowdown,
            output_dir,
            output_name="video_synced_case2",
        )

        dur = self._probe_duration(final_video)
        assert 4.5 <= dur <= 5.5
