#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_concat_demuxer.py
===========================
Kiểm tra Filter Complex Batching pipeline (build_ffmpeg_batch_cmd + process_video_chunks_parallel)
có gây ra lỗi desync video hay không.

Kiến trúc mới (2026-04-13): Thay vì cắt vật lý từng chunk rồi concat demuxer,
ta dùng FFmpeg Filter Complex để dựng Timeline ảo, gom nhóm (batching) để tránh tràn RAM.

Cấu trúc layers:
  Layer 1 — Unit Tests          (Test build FFmpeg batch command generation)
  Layer 2 — Component Tests     (Test với synthetic video, 10-50 chunks, batch_size=10)
  Layer 3 — Real Video Tests    (Test với video thật, 1000-3000 chunks)

Cách chạy từng layer:
    pytest tests/test_concat_demuxer.py -v -k "Layer1"
    pytest tests/test_concat_demuxer.py -v -k "Layer2"
    pytest tests/test_concat_demuxer.py -v -k "Layer3" --video-path="D:/videos/my_test.mp4"
"""

import sys
import shutil
import json
import math
import random
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

import pytest

# ── Project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Thêm logger theo chuẩn
logger = logging.getLogger(__name__)

# ── Lazy imports ─────────────────────────────────────────────────────
cv2 = pytest.importorskip("cv2", reason="pip install opencv-python")
import numpy as np

from sync_engine.models import TimelineSegment
from sync_engine.video_processor import (
    build_ffmpeg_batch_cmd,
    process_video_chunks_parallel,
    _concat_chunks,
)


# ═════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════

def _probe_duration(video_path: str) -> float:
    """Đo duration video bằng ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def _probe_frame_count(video_path: str) -> int:
    """Đếm tổng số frame video bằng ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(result.stdout.strip())


def _probe_all_pts(video_path: str) -> List[float]:
    """Lấy danh sách PTS của tất cả frame."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=pts_time",
        "-of", "csv=p=0", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def _probe_stream_info(video_path: str) -> dict:
    """Lấy thông tin stream video."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=time_base,codec_name,width,height,r_frame_rate",
        "-of", "json", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    if "streams" in info and len(info["streams"]) > 0:
        return info["streams"][0]
    return {}


def _write_desync_report(report: dict, output_dir: Path) -> Path:
    """Ghi report chi tiết ra JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    test_name = report.get("test_name", "concat_demuxer")

    report_path = output_dir / f"{test_name}_{timestamp}.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    return report_path


def _generate_chunks(
    source_dur_ms: float,
    fps: float,
    num_chunks: int,
    speed_range: Tuple[float, float],
) -> List[TimelineSegment]:
    """Tạo random timeline segments."""
    segments = []
    chunk_dur = source_dur_ms / num_chunks

    for i in range(num_chunks):
        speed = random.uniform(speed_range[0], speed_range[1])
        start = i * chunk_dur
        end = (i + 1) * chunk_dur
        if i == num_chunks - 1:
            end = source_dur_ms  # Đảm bảo segment cuối tới đúng end

        dur = end - start
        new_dur = dur / speed

        seg = TimelineSegment(
            orig_start=start,
            orig_end=end,
            new_start=0.0,  # Không quan trọng cho test này
            new_end=0.0,   # Không quan trọng cho test này
            video_speed=speed,
            audio_speed=1.0,
            new_chunk_dur=new_dur,
            block_type="gap",
            tts_clip_path=None,
            tts_duration=0.0,
        )
        segments.append(seg)
    return segments


def _compute_expected_batch_duration(
    segments: List[TimelineSegment],
    fps_float: float,
) -> float:
    """
    Tính toán expected duration (ms) của một batch segments
    dựa trên công thức giống hệt process_video_chunks_parallel.
    """
    total_ms = 0.0
    for seg in segments:
        duration_frames = round(((seg.orig_end - seg.orig_start) / 1000.0) * fps_float)
        duration_s = duration_frames / fps_float
        stretched_duration_s = duration_s / seg.video_speed
        expected_output_frames = math.floor(stretched_duration_s * fps_float) + 1
        actual_dur = (expected_output_frames / fps_float) * 1000.0
        total_ms += actual_dur
    return total_ms


# ═════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ═════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def synthetic_video_path(tmp_path_factory) -> Path:
    """Video 10 giây, 30fps."""
    tmp_dir = tmp_path_factory.mktemp("video")
    path = tmp_dir / "synthetic_test.mp4"
    W, H, FPS = 640, 360, 30
    total_frames = int(10 * FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (W, H))

    for frame_no in range(total_frames):
        # Background xám, vẽ số frame để thay đổi nội dung (nếu cần visual check)
        frame = np.full((H, W, 3), 100, dtype=np.uint8)
        cv2.putText(frame, f"Frame {frame_no}", (50, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        writer.write(frame)

    writer.release()
    return path


@pytest.fixture(scope="module")
def report_dir() -> Path:
    """Thư mục lưu report."""
    return PROJECT_ROOT / "tests" / "test_reports"


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS (build_ffmpeg_batch_cmd command generation)
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_FilterComplexBatchUnit:
    """Test thuần Python: kiểm tra lệnh FFmpeg được build đúng."""

    def test_single_segment_1x_speed(self):
        """1 segment, speed=1.0 → filter chain không stretch."""
        seg = TimelineSegment(
            orig_start=0.0, orig_end=1000.0,
            new_start=0.0, new_end=1000.0,
            video_speed=1.0, audio_speed=1.0, new_chunk_dur=1000.0,
            block_type="gap", tts_clip_path=None, tts_duration=0.0,
        )
        cmd = build_ffmpeg_batch_cmd(
            input_path="input.mp4",
            output_path="output.mp4",
            segments=[seg],
            fps_str="30/1",
            fps_float=30.0,
            use_gpu=False,
        )
        assert "-filter_complex" in cmd
        filter_val = cmd[cmd.index("-filter_complex") + 1]
        # Phải có trim, setpts, fps, concat
        assert "trim=start=" in filter_val
        assert "setpts=PTS-STARTPTS" in filter_val
        assert "setpts=1.000000*PTS" in filter_val  # speed=1.0 → factor=1.0
        assert "fps=30/1" in filter_val
        assert "concat=n=1:v=1:a=0[outv]" in filter_val
        assert "-map" in cmd
        assert "[outv]" in cmd
        assert "libx264" in cmd

    def test_single_segment_slow_speed(self):
        """1 segment, speed=0.5 → setpts factor=2.0."""
        seg = TimelineSegment(
            orig_start=1000.0, orig_end=3000.0,
            new_start=0.0, new_end=4000.0,
            video_speed=0.5, audio_speed=1.0, new_chunk_dur=4000.0,
            block_type="tts", tts_clip_path=None, tts_duration=0.0,
        )
        cmd = build_ffmpeg_batch_cmd(
            input_path="input.mp4",
            output_path="output.mp4",
            segments=[seg],
            fps_str="30/1",
            fps_float=30.0,
            use_gpu=False,
        )
        filter_val = cmd[cmd.index("-filter_complex") + 1]
        assert "setpts=2.000000*PTS" in filter_val  # 1/0.5 = 2.0
        assert "trim=start=1.000000:duration=2.000000" in filter_val

    def test_multiple_segments_concat_labels(self):
        """3 segments → phải có [v0], [v1], [v2] và concat=n=3."""
        segments = [
            TimelineSegment(
                orig_start=i * 1000.0, orig_end=(i + 1) * 1000.0,
                new_start=0.0, new_end=0.0,
                video_speed=1.0, audio_speed=1.0, new_chunk_dur=1000.0,
                block_type="gap", tts_clip_path=None, tts_duration=0.0,
            )
            for i in range(3)
        ]
        cmd = build_ffmpeg_batch_cmd(
            input_path="input.mp4",
            output_path="output.mp4",
            segments=segments,
            fps_str="30/1",
            fps_float=30.0,
            use_gpu=False,
        )
        filter_val = cmd[cmd.index("-filter_complex") + 1]
        assert "[v0]" in filter_val
        assert "[v1]" in filter_val
        assert "[v2]" in filter_val
        assert "concat=n=3:v=1:a=0[outv]" in filter_val

    def test_gpu_encoder_selection(self):
        """use_gpu=True → h264_nvenc, preset=p5."""
        seg = TimelineSegment(
            orig_start=0.0, orig_end=1000.0,
            new_start=0.0, new_end=1000.0,
            video_speed=1.0, audio_speed=1.0, new_chunk_dur=1000.0,
            block_type="gap", tts_clip_path=None, tts_duration=0.0,
        )
        cmd = build_ffmpeg_batch_cmd(
            input_path="input.mp4",
            output_path="output.mp4",
            segments=[seg],
            fps_str="30/1",
            fps_float=30.0,
            use_gpu=True,
        )
        assert "h264_nvenc" in cmd
        assert "p5" in cmd
        assert "-cq" in cmd

    def test_expected_duration_formula(self):
        """Kiểm tra công thức tính expected duration khớp với code trong video_processor."""
        fps = 30.0
        # Segment: 1000ms gốc, speed=0.5 → stretched 2000ms
        seg = TimelineSegment(
            orig_start=0.0, orig_end=1000.0,
            new_start=0.0, new_end=2000.0,
            video_speed=0.5, audio_speed=1.0, new_chunk_dur=2000.0,
            block_type="gap", tts_clip_path=None, tts_duration=0.0,
        )
        # Công thức từ process_video_chunks_parallel:
        duration_frames = round(((seg.orig_end - seg.orig_start) / 1000.0) * fps)
        assert duration_frames == 30  # 1000ms / 33.33ms = 30 frames

        duration_s = duration_frames / fps
        assert duration_s == 1.0

        stretched_duration_s = duration_s / seg.video_speed
        assert stretched_duration_s == 2.0

        expected_output_frames = math.floor(stretched_duration_s * fps) + 1
        assert expected_output_frames == 61  # floor(60) + 1

        actual_dur_ms = (expected_output_frames / fps) * 1000.0
        assert actual_dur_ms == pytest.approx(2033.333, abs=0.01)


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — COMPONENT TESTS (Synthetic Video, 10-50 Chunks)
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg không có trong PATH")
class TestLayer2_FilterComplexBatchSynthetic:
    """Test Filter Complex Batching với video tổng hợp 10 giây."""

    @pytest.fixture(scope="class")
    def setup_batch_concat(self, synthetic_video_path, tmp_path_factory, concat_workers):
        """Fixture chạy chung 1 lần xử lý batch cho tất cả test L2."""
        tmp_dir = tmp_path_factory.mktemp("l2_batch")
        fps = 30.0
        fps_str = "30/1"
        dur_s = _probe_duration(str(synthetic_video_path))
        num_chunks = random.randint(10, 50)
        batch_size = 10  # Gom 10 segments/batch

        timeline = _generate_chunks(dur_s * 1000, fps, num_chunks, (0.5, 1.0))

        # 1. Process qua Filter Complex Batching
        output_dir = tmp_dir / "batches"
        output_video, actual_durations = process_video_chunks_parallel(
            video_path=str(synthetic_video_path),
            timeline=timeline,
            output_dir=str(output_dir),
            max_workers=concat_workers,
            use_gpu=False,
            fps_str=fps_str,
            fps_float=fps,
            batch_size=batch_size,
        )

        return {
            "source": str(synthetic_video_path),
            "final": output_video,
            "timeline": timeline,
            "actual_durations": actual_durations,
            "fps": fps,
            "fps_str": fps_str,
            "num_chunks": num_chunks,
            "batch_size": batch_size,
            "batch_dir": output_dir,
        }

    def test_duration_total(self, setup_batch_concat):
        """Test 1: Tổng duration ≈ Σ expected durations (Tolerance: 2 frames per batch boundary)."""
        data = setup_batch_concat

        expected_dur_ms = _compute_expected_batch_duration(data["timeline"], data["fps"])
        actual_dur_ms = _probe_duration(data["final"]) * 1000

        delta_ms = abs(actual_dur_ms - expected_dur_ms)
        # Tolerance: mỗi batch boundary có thể lệch 1-2 frames
        frame_dur_ms = 1000.0 / data["fps"]
        tolerance_ms = 3 * frame_dur_ms  # Tối đa 3 frame lệch trên toàn bộ video

        logger.info(
            f"[Duration Total] Expected: {expected_dur_ms:.1f}ms, "
            f"Actual: {actual_dur_ms:.1f}ms, Delta: {delta_ms:.1f}ms, "
            f"Tolerance: {tolerance_ms:.1f}ms ({num_batches} batches)"
        )
        assert delta_ms <= tolerance_ms, (
            f"Duration drift {delta_ms:.1f}ms > {tolerance_ms:.1f}ms "
            f"({num_batches} batch boundaries)"
        )

    def test_frame_count(self, setup_batch_concat):
        """Test 2: Tổng frames ≈ Σ expected frames (Tolerance: 2 frames per batch boundary)."""
        data = setup_batch_concat

        expected_frames = 0
        for seg in data["timeline"]:
            duration_frames = round(((seg.orig_end - seg.orig_start) / 1000.0) * data["fps"])
            duration_s = duration_frames / data["fps"]
            stretched_duration_s = duration_s / seg.video_speed
            expected_output_frames = math.floor(stretched_duration_s * data["fps"]) + 1
            expected_frames += expected_output_frames

        actual_frames = _probe_frame_count(data["final"])

        tolerance_frames = 3  # Tối đa 3 frame lệch trên toàn bộ video

        delta_frames = abs(actual_frames - expected_frames)
        logger.info(
            f"[Frame Count] Expected: {expected_frames}, Actual: {actual_frames}, "
            f"Delta: {delta_frames}, Tolerance: {tolerance_frames}"
        )
        assert delta_frames <= tolerance_frames, (
            f"Frame count drift {delta_frames} > {tolerance_frames}"
        )

    def test_pts_monotonic(self, setup_batch_concat):
        """Test 3: PTS tăng dần nghiêm ngặt (0 violations)."""
        data = setup_batch_concat
        all_pts = _probe_all_pts(data["final"])

        violations = []
        for i in range(1, len(all_pts)):
            if all_pts[i] <= all_pts[i - 1]:
                violations.append((i, all_pts[i - 1], all_pts[i]))

        logger.info(f"[PTS Monotonic] Violations: {len(violations)}")
        assert len(violations) == 0, f"Found {len(violations)} non-monotonic PTS violations"

    def test_frame_delta_uniformity(self, setup_batch_concat):
        """Test 4: Khoảng cách frame đồng đều (Tolerance: 50% delta)."""
        data = setup_batch_concat
        fps = data["fps"]
        expected_delta = 1.0 / fps
        tolerance = expected_delta * 0.5

        all_pts = _probe_all_pts(data["final"])

        anomalies = []
        max_anomaly = 0
        for i in range(1, len(all_pts)):
            delta = all_pts[i] - all_pts[i - 1]
            diff = abs(delta - expected_delta)
            if diff > tolerance:
                anomalies.append((i, delta))
                if diff > max_anomaly:
                    max_anomaly = diff

        logger.info(
            f"[Frame Delta] Anomalies: {len(anomalies)}, Max Diff: {max_anomaly * 1000:.1f}ms"
        )
        assert len(anomalies) == 0, f"Found {len(anomalies)} frame delta anomalies"

    def test_batch_duration_accuracy(self, setup_batch_concat):
        """Test 5: Mỗi batch file có duration khớp với dự đoán (Tolerance: 2 frames)."""
        data = setup_batch_concat
        fps = data["fps"]
        batch_size = data["batch_size"]
        timeline = data["timeline"]
        batch_dir = data["batch_dir"]

        # Tính số batch
        num_batches = math.ceil(len(timeline) / batch_size)

        frame_dur_ms = 1000.0 / fps
        tolerance_ms = 3 * frame_dur_ms  # Tối đa 3 frame lệch trên toàn bộ video

        max_delta_ms = 0.0
        worst_batch = -1

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(timeline))
            batch_segments = timeline[start:end]

            # Expected duration cho batch này
            expected_ms = _compute_expected_batch_duration(batch_segments, fps)

            # Actual duration từ file batch
            batch_file = batch_dir / f"batch_{batch_idx:04d}.mp4"
            if not batch_file.exists():
                continue

            actual_ms = _probe_duration(str(batch_file)) * 1000
            delta_ms = abs(actual_ms - expected_ms)

            if delta_ms > max_delta_ms:
                max_delta_ms = delta_ms
                worst_batch = batch_idx

            assert delta_ms <= tolerance_ms, (
                f"Batch {batch_idx} duration drift {delta_ms:.1f}ms > "
                f"{tolerance_ms:.1f}ms (expected={expected_ms:.1f}ms, "
                f"actual={actual_ms:.1f}ms)"
            )

        logger.info(
            f"[Batch Duration] Max drift: {max_delta_ms:.1f}ms "
            f"at batch {worst_batch} (tolerance: {tolerance_ms:.1f}ms)"
        )

    def test_actual_durations_vs_ffprobe(self, setup_batch_concat):
        """Test 6: Σ actual_durations (từ code) ≈ ffprobe duration của final video.

        Kiểm chứng tính nhất quán: giá trị actual_durations mà process_video_chunks_parallel
        trả về phải phản ánh đúng độ dài video thực tế đo bằng ffprobe.
        Nếu code tính sai công thức nhưng FFmpeg vẫn xuất đúng, test này sẽ FAIL → phát hiện bug.
        """
        data = setup_batch_concat
        actual_durations = data["actual_durations"]
        timeline = data["timeline"]

        # Σ actual_durations từ code (ms)
        sum_actual_durations_ms = sum(actual_durations)

        # ffprobe duration của final video (ms)
        ffprobe_dur_ms = _probe_duration(data["final"]) * 1000

        # Tolerance: mỗi batch boundary có thể lệch 1-2 frames do concat demuxer
        # num_batches = math.ceil(len(timeline) / data["batch_size"])
        frame_dur_ms = 1000.0 / data["fps"]
        tolerance_ms = 3 * frame_dur_ms

        delta_ms = abs(sum_actual_durations_ms - ffprobe_dur_ms)

        logger.info(
            f"[Actual Durations vs FFprobe] "
            f"Σ actual_durations={sum_actual_durations_ms:.1f}ms, "
            f"ffprobe={ffprobe_dur_ms:.1f}ms, "
            f"delta={delta_ms:.1f}ms, tolerance={tolerance_ms:.1f}ms"
        )

        assert len(actual_durations) == len(timeline), (
            f"actual_durations length {len(actual_durations)} != "
            f"timeline length {len(timeline)}"
        )
        assert delta_ms <= tolerance_ms, (
            f"Σ actual_durations ({sum_actual_durations_ms:.1f}ms) "
            f"khớp ffprobe ({ffprobe_dur_ms:.1f}ms) "
            f"vượt tolerance {tolerance_ms:.1f}ms "
            f"(delta={delta_ms:.1f}ms)"
        )


# ═════════════════════════════════════════════════════════════════════
# LAYER 3 — REAL VIDEO FULL ANALYSIS (1000-3000 Chunks)
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg không có trong PATH")
class TestLayer3_FilterComplexBatchRealVideo:
    """Test Filter Complex Batching với video thật."""

    def test_full_desync_analysis(self, real_video_path, concat_workers, report_dir, tmp_path):
        """Test 7: Full analysis với video thật, tổng hợp TẤT CẢ kiểm tra và ghi report."""

        video_path_str = str(real_video_path)
        fps = 30.0
        fps_str = "30/1"
        dur_s = _probe_duration(video_path_str)
        num_chunks = random.randint(1000, 3000)
        batch_size = 100

        timeline = _generate_chunks(dur_s * 1000, fps, num_chunks, (0.5, 1.0))

        # --- THỰC THI ---
        output_dir = tmp_path / "batches"
        output_video, actual_durations = process_video_chunks_parallel(
            video_path=video_path_str,
            timeline=timeline,
            output_dir=str(output_dir),
            max_workers=concat_workers,
            use_gpu=False,
            fps_str=fps_str,
            fps_float=fps,
            batch_size=batch_size,
        )

        # --- THU THẬP SỐ LIỆU ---
        expected_dur_ms = _compute_expected_batch_duration(timeline, fps)
        actual_dur_ms = _probe_duration(output_video) * 1000
        dur_delta_ms = abs(actual_dur_ms - expected_dur_ms)

        # Tính số frame expected
        expected_frames = 0
        for seg in timeline:
            duration_frames = round(((seg.orig_end - seg.orig_start) / 1000.0) * fps)
            duration_s = duration_frames / fps
            stretched_duration_s = duration_s / seg.video_speed
            expected_output_frames = math.floor(stretched_duration_s * fps) + 1
            expected_frames += expected_output_frames

        # Probe thông tin (có thể chậm với video dài)
        actual_frames = _probe_frame_count(output_video)
        all_pts = _probe_all_pts(output_video)

        # Check monotonic
        monotonic_violations = sum(
            1 for i in range(1, len(all_pts)) if all_pts[i] <= all_pts[i - 1]
        )

        # Check frame delta anomalies
        expected_delta = 1.0 / fps
        delta_anomalies = [
            i for i in range(1, len(all_pts))
            if abs((all_pts[i] - all_pts[i - 1]) - expected_delta) > expected_delta * 0.5
        ]

        # Check batch duration accuracy (sample 10 batches)
        num_batches = math.ceil(num_chunks / batch_size)
        batch_dir = Path(output_dir)
        frame_dur_ms = 1000.0 / fps
        batch_tolerance_ms = 3 * frame_dur_ms  # Tối đa 3 frame lệch trên toàn bộ video

        batch_drifts = []
        sample_indices = list(range(0, min(num_batches, 10)))
        for batch_idx in sample_indices:
            start = batch_idx * batch_size
            end = min(start + batch_size, len(timeline))
            batch_segments = timeline[start:end]
            expected_batch_ms = _compute_expected_batch_duration(batch_segments, fps)

            batch_file = batch_dir / f"batch_{batch_idx:04d}.mp4"
            if batch_file.exists():
                actual_batch_ms = _probe_duration(str(batch_file)) * 1000
                drift = abs(actual_batch_ms - expected_batch_ms)
                batch_drifts.append({
                    "batch_idx": batch_idx,
                    "expected_ms": expected_batch_ms,
                    "actual_ms": actual_batch_ms,
                    "delta_ms": drift,
                    "pass": drift <= batch_tolerance_ms,
                })

        max_batch_drift = max((d["delta_ms"] for d in batch_drifts), default=0.0)
        all_batches_pass = all(d["pass"] for d in batch_drifts) if batch_drifts else True

        # --- GHI REPORT ---
        # Tolerance: tối đa 3 frame lệch trên toàn bộ video
        dur_tolerance_ms = 3 * frame_dur_ms
        frame_tolerance = 3

        frame_delta = abs(actual_frames - expected_frames)
        all_pass = (
            dur_delta_ms <= dur_tolerance_ms
            and frame_delta <= frame_tolerance
            and monotonic_violations == 0
            and len(delta_anomalies) == 0
            and all_batches_pass
        )

        report = {
            "test_name": "filter_complex_batch_full_analysis",
            "timestamp": datetime.now().isoformat(),
            "source_video": video_path_str,
            "params": {
                "chunk_count": num_chunks,
                "batch_size": batch_size,
                "num_batches": num_batches,
                "source_duration_s": dur_s,
                "source_fps": fps,
            },
            "checks": {
                "duration": {
                    "expected_s": expected_dur_ms / 1000.0,
                    "actual_s": actual_dur_ms / 1000.0,
                    "delta_ms": dur_delta_ms,
                    "tolerance_ms": dur_tolerance_ms,
                    "pass": dur_delta_ms <= dur_tolerance_ms,
                },
                "frame_count": {
                    "expected": expected_frames,
                    "actual": actual_frames,
                    "delta": frame_delta,
                    "tolerance": frame_tolerance,
                    "pass": frame_delta <= frame_tolerance,
                },
                "pts_monotonic": {
                    "violations": monotonic_violations,
                    "pass": monotonic_violations == 0,
                },
                "frame_delta": {
                    "anomalies": len(delta_anomalies),
                    "pass": len(delta_anomalies) == 0,
                },
                "batch_duration_accuracy": {
                    "sampled_batches": len(batch_drifts),
                    "max_drift_ms": max_batch_drift,
                    "tolerance_per_batch_ms": batch_tolerance_ms,
                    "all_pass": all_batches_pass,
                    "details": batch_drifts,
                    "pass": all_batches_pass,
                },
            },
            "verdict": "PASS" if all_pass else "FAIL",
            "diagnosis": (
                f"Duration drift {dur_delta_ms:.1f}ms, "
                f"Frame drift {frame_delta}, "
                f"Max batch drift {max_batch_drift:.1f}ms"
            ),
        }

        report_path = _write_desync_report(report, report_dir)
        print(f"\n📊 Full Desync Report: {report_path}")

        # --- ASSERT CUỐI CÙNG ---
        assert dur_delta_ms <= dur_tolerance_ms, (
            f"Duration drift {dur_delta_ms:.1f}ms > {dur_tolerance_ms}ms tolerance. "
            f"Xem chi tiết tại: {report_path}"
        )
        assert all_pass, (
            f"Một số kiểm tra desync đã thất bại. Xem chi tiết tại: {report_path}"
        )
