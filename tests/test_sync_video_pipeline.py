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

@pytest.fixture(scope="module")
def synthetic_inputs(tmp_path_factory):
    """Tạo SRT, TTS clips."""
    tmp_dir = tmp_path_factory.mktemp("inputs")
    
    # subtitle.srt (1 đoạn, 1->2s)
    srt_path = tmp_dir / "sub.srt"
    srt_path.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nHello\n",
        encoding="utf-8"
    )
    
    # tts_dir
    tts_dir = tmp_dir / "tts"
    tts_dir.mkdir()
    tts_path = tts_dir / "dubb-0.wav"
    
    # Tạo wav dài 2s (cần slow video)
    silence = AudioSegment.silent(duration=2000, frame_rate=48000)
    silence.set_channels(2).export(str(tts_path), format="wav")
    
    return {
        "srt": str(srt_path),
        "tts_dir": str(tts_dir)
    }


# ═════════════════════════════════════════════════════════════════════
# LAYER 3 — PIPELINE INTEGRATION
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg không có trong PATH")
class TestLayer3_SyncVideoPipeline:
    def test_run_sync_pipeline(self, synthetic_video, synthetic_inputs, tmp_path):
        """Test the end-to-end pipeline with synthetic data."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

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
            output_name="video_synced",
            no_hardsub=False,
            workers=2,
            no_gpu=True,  # CPU mode for CI
            subtitle_fontname="Arial",
            subtitle_fontsize=22,
            subtitle_color="&H00FFFFFF",
            subtitle_margin_v=6,
            note_max_chars=15
        )
        
        # Sẽ sinh ra: video_synced.mp4, _tts_synced.srt, _synced.srt
        run_sync_pipeline(args)
        
        final_video = output_dir / "video_synced.mp4"
        tts_srt = output_dir / "video_synced_tts_synced.srt"
        full_srt = output_dir / "video_synced_synced.srt"
        
        assert final_video.exists()
        assert tts_srt.exists()
        assert full_srt.exists()
        
        # Video gốc 3s. Sub 1-2s (slot 1s). TTS dài 2s. 
        # Gap1: 0-1s
        # Sub1: 1-3s (kéo giãn 0.5x, từ 1s thành 2s)
        # Tail: 2-3s gốc -> 3-4s
        # => Tổng 4s
        
        # Check output duration
        import subprocess
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", str(final_video)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        dur = float(result.stdout.strip())
        
        assert 3.5 <= dur <= 4.5
