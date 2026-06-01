#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_video_processor.py
===========================
Test Component Phase 2: Split and stretch video chunks.
query_keyframes, snap_to_nearest_keyframe, build_ffmpeg_chunk_cmd, process_video_chunks_parallel.

Cấu trúc layers:
  Layer 1 — Unit Tests          (Test build FFmpeg command)
  Layer 2 — Component Tests     (Test với synthetic video)

Cách chạy từng layer:
    pytest tests/test_video_processor.py -v -k "Layer1"
    pytest tests/test_video_processor.py -v -k "Layer2"
"""

import sys
import shutil
from pathlib import Path

import pytest

# ── Project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sync_engine.models import TimelineSegment
from sync_engine.video_processor import (
    snap_to_nearest_keyframe,
    build_ffmpeg_chunk_cmd,
    process_video_chunks_parallel,
)
from utils.ffmpeg_probe import detect_hevc_nvenc

# ═════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ═════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def synthetic_video_path(tmp_path_factory) -> Path:
    """Video ngắn 2 giây, 30fps."""
    cv2 = pytest.importorskip("cv2", reason="pip install opencv-python")
    import numpy as np

    tmp_dir = tmp_path_factory.mktemp("video")
    path = tmp_dir / "synthetic_test.mp4"
    W, H, FPS = 640, 360, 30
    total_frames = int(2 * FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (W, H))

    for frame_no in range(total_frames):
        frame = np.full((H, W, 3), 100, dtype=np.uint8)
        writer.write(frame)

    writer.release()
    assert path.exists()
    return path


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_VideoProcessorUnit:
    def test_snap_to_nearest_keyframe(self):
        kfs = [0.0, 2000.0, 4000.0, 6000.0]
        assert snap_to_nearest_keyframe(100.0, kfs) == 0.0
        assert snap_to_nearest_keyframe(1900.0, kfs) == 2000.0
        assert snap_to_nearest_keyframe(5000.0, kfs) == 4000.0  # Equal distance, min returns first? min(..., key) will return 4000 or 6000
        
        # When empty
        assert snap_to_nearest_keyframe(500.0, []) == 500.0

    def test_build_ffmpeg_chunk_cmd(self):
        # 1.0 speed: vẫn filter frame-accurate và encode HEVC NVENC cố định.
        cmd = build_ffmpeg_chunk_cmd(
            input_path="input.mp4",
            output_path="output.mp4",
            start_ms=1500.0,
            input_duration_ms=2500.0,
            video_speed=1.0,
            fps_str="30/1",
            fps_float=30.0,
            use_gpu=False,
        )
        assert "-c:v" in cmd
        assert cmd[cmd.index("-c:v") + 1] == "hevc_nvenc"
        assert cmd[cmd.index("-preset") + 1] == "p4"
        assert cmd[cmd.index("-tune") + 1] == "hq"
        assert cmd[cmd.index("-cq") + 1] == "28"
        assert "copy" not in cmd
        assert "libx264" not in cmd
        assert "-ss" in cmd
        assert cmd[cmd.index("-ss") + 1] == "0.000000"
        assert "-filter:v" in cmd
        filter_val = cmd[cmd.index("-filter:v") + 1]
        assert "trim=start=1.483333:duration=2.483333" in filter_val
        assert "setpts=1.000000*PTS" in filter_val
        assert "trim=end_frame=75" in filter_val

        # < 1.0 speed
        cmd2 = build_ffmpeg_chunk_cmd(
            input_path="input.mp4",
            output_path="output.mp4",
            start_ms=1000.0,
            input_duration_ms=2000.0,
            video_speed=0.5,
            fps_str="30/1",
            fps_float=30.0,
            use_gpu=False,
        )
        assert "-filter:v" in cmd2
        filter_val = cmd2[cmd2.index("-filter:v") + 1]
        assert "setpts" in filter_val
        assert "setpts=2.000000*PTS" in filter_val
        assert "hevc_nvenc" in cmd2
        assert "libx264" not in cmd2
        assert "h264_nvenc" not in cmd2

# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — COMPONENT TESTS
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg không có trong PATH")
class TestLayer2_VideoProcessorIntegration:
    def test_process_video_chunks_parallel(self, synthetic_video_path, tmp_path):
        """Test stretch video 2s thành 2 chunk: 1s giữ nguyên, 1s slow 0.5x => tổng 3s"""
        if not detect_hevc_nvenc():
            pytest.skip("hevc_nvenc không khả dụng; sync_video render video bắt buộc dùng HEVC NVENC")
        timeline = [
            TimelineSegment(
                orig_start=0.0, orig_end=1000.0,
                new_start=0.0, new_end=1000.0,
                video_speed=1.0, audio_speed=1.0, new_chunk_dur=1000.0,
                block_type="gap", tts_clip_path=None, tts_duration=0
            ),
            TimelineSegment(
                orig_start=1000.0, orig_end=2000.0,
                new_start=1000.0, new_end=3000.0,
                video_speed=0.5, audio_speed=1.0, new_chunk_dur=2000.0,
                block_type="tts", tts_clip_path=None, tts_duration=0
            )
        ]
        
        output_dir = tmp_path / "video_chunks"
        out_vid, actual_durations = process_video_chunks_parallel(
            video_path=str(synthetic_video_path),
            timeline=timeline,
            output_dir=str(output_dir),
            max_workers=2,
            use_gpu=False,  # Tham số tương thích cũ; HEVC NVENC vẫn được ép trong runtime.
        )
        assert actual_durations == pytest.approx([1000.0, 2000.0], abs=0.01)
        
        out_vid_path = Path(out_vid)
        assert out_vid_path.exists()
        assert out_vid_path.stat().st_size > 0
        
        # Check duration
        import subprocess
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", str(out_vid_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        dur = float(result.stdout.strip())
        
        # Due to 30fps and rounding, 3.0s might be slightly off like 2.9 or 3.1
        assert 2.5 <= dur <= 3.5
