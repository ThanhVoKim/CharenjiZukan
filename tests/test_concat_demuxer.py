#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_concat_demuxer.py
===========================
Kiểm tra xem Concat Demuxer (-f concat -c copy) có gây ra lỗi desync video hay không.

Cấu trúc layers:
  Layer 2 — Component Tests (Test với synthetic video, random 10-50 chunks)
  Layer 3 — Real Video Tests (Test với video thật, random 1000-3000 chunks)

Cách chạy từng layer:
    pytest tests/test_concat_demuxer.py -v -k "Layer2"
    pytest tests/test_concat_demuxer.py -v -k "Layer3" --video-path="D:/videos/my_test.mp4"
"""

import sys
import shutil
import json
import random
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from sync_engine.video_processor import build_ffmpeg_chunk_cmd, _concat_chunks


# ═════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════

def _probe_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

def _probe_frame_count(video_path: str) -> int:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(result.stdout.strip())

def _probe_all_pts(video_path: str) -> List[float]:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=pts_time",
        "-of", "csv=p=0", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]

def _probe_stream_info(video_path: str) -> dict:
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
    
    # Extract test name to make unique file if possible
    test_name = report.get("test_name", "concat_demuxer")
    
    report_path = output_dir / f"{test_name}_{timestamp}.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    return report_path

def _generate_chunks(source_dur_ms: float, fps: float, num_chunks: int, speed_range: Tuple[float, float]) -> List[TimelineSegment]:
    """Tạo random timeline segments."""
    segments = []
    chunk_dur = source_dur_ms / num_chunks
    
    for i in range(num_chunks):
        speed = random.uniform(speed_range[0], speed_range[1])
        start = i * chunk_dur
        end = (i + 1) * chunk_dur
        if i == num_chunks - 1:
            end = source_dur_ms # Đảm bảo segment cuối tới đúng end
            
        dur = end - start
        new_dur = dur / speed
        
        seg = TimelineSegment(
            orig_start=start,
            orig_end=end,
            new_start=0.0, # Không quan trọng cho test này
            new_end=0.0,   # Không quan trọng cho test này
            video_speed=speed,
            audio_speed=1.0,
            new_chunk_dur=new_dur,
            block_type="gap",
            tts_clip_path=None,
            tts_duration=0.0
        )
        segments.append(seg)
    return segments

def _process_chunks_parallel(
    video_path: str,
    timeline: List[TimelineSegment],
    output_dir: Path,
    fps_float: float = 30.0,
    max_workers: int = 4,
) -> List[str]:
    """Cắt và stretch video chunks chạy song song."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def _run_single(args: tuple):
        idx, seg, out_path = args
        cmd = build_ffmpeg_chunk_cmd(
            input_path=video_path,
            output_path=str(out_path),
            start_ms=seg.orig_start,
            input_duration_ms=seg.orig_end - seg.orig_start,
            video_speed=seg.video_speed,
            fps_str=f"{fps_float}/1",
            fps_float=fps_float,
            use_gpu=False,
        )
        subprocess.run(cmd, check=True, capture_output=True)
        return idx, str(out_path)

    tasks = [
        (i, seg, output_dir / f"chunk_{i:04d}.mp4")
        for i, seg in enumerate(timeline)
    ]

    results = {}
    total_tasks = len(tasks)
    
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    logger.info(f"Đang xử lý {total_tasks} chunks bất đồng bộ ({max_workers} workers)...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_single, t): t[0] for t in tasks}
        
        future_iter = as_completed(futures)
        if has_tqdm:
            future_iter = tqdm(future_iter, total=total_tasks, desc="Processing chunks", unit="chunk")
            
        completed = 0
        for f in future_iter:
            idx, path = f.result()
            results[idx] = path
            completed += 1
            if not has_tqdm and completed % max(1, total_tasks // 10) == 0:
                logger.info(f"  Tiến độ: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%)")

    # Trả về theo đúng thứ tự timeline gốc
    return [results[i] for i in range(total_tasks)]


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
# LAYER 2 — SYNTHETIC VIDEO (10-50 Chunks)
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg không có trong PATH")
class TestLayer2_ConcatDemuxerSynthetic:
    
    @pytest.fixture(scope="class")
    def setup_synthetic_concat(self, synthetic_video_path, tmp_path_factory):
        """Fixture chạy chung 1 lần cắt ghép cho tất cả test L2 để tiết kiệm thời gian."""
        tmp_dir = tmp_path_factory.mktemp("l2_concat")
        fps = 30.0
        dur_s = _probe_duration(str(synthetic_video_path))
        num_chunks = random.randint(10, 50)
        
        timeline = _generate_chunks(dur_s * 1000, fps, num_chunks, (0.5, 1.0))
        
        # 1. Process chunks
        chunk_paths = _process_chunks_parallel(str(synthetic_video_path), timeline, tmp_dir, fps, max_workers=4)
        
        # 2. Concat
        final_video = str(tmp_dir / "final.mp4")
        _concat_chunks(chunk_paths, final_video)
        
        return {
            "source": str(synthetic_video_path),
            "final": final_video,
            "timeline": timeline,
            "chunk_paths": chunk_paths,
            "fps": fps,
            "num_chunks": num_chunks
        }
        
    def test_duration_total(self, setup_synthetic_concat):
        """Test 1: Tổng duration = Σ chunk durations (Tolerance: 500ms)"""
        data = setup_synthetic_concat
        
        expected_dur_ms = sum(seg.new_chunk_dur for seg in data["timeline"])
        actual_dur_ms = _probe_duration(data["final"]) * 1000
        
        delta_ms = abs(actual_dur_ms - expected_dur_ms)
        logger.info(f"[Duration Total] Expected: {expected_dur_ms:.1f}ms, Actual: {actual_dur_ms:.1f}ms, Delta: {delta_ms:.1f}ms")
        assert delta_ms <= 500, f"Duration drift {delta_ms:.1f}ms > 500ms tolerance"

    def test_frame_count(self, setup_synthetic_concat):
        """Test 2: Tổng frames = Σ expected frames (Tolerance: 0 frame)"""
        data = setup_synthetic_concat
        
        expected_frames = 0
        for seg in data["timeline"]:
            expected_frames += round((seg.new_chunk_dur / 1000.0) * data["fps"])
            
        actual_frames = _probe_frame_count(data["final"])
        logger.info(f"[Frame Count] Expected: {expected_frames}, Actual: {actual_frames}")
        assert actual_frames == expected_frames, f"Frame count mismatch: {actual_frames} != {expected_frames}"

    def test_pts_boundary(self, setup_synthetic_concat):
        """Test 3: PTS tại mỗi chunk boundary khớp expected (Tolerance: 1 frame = 33.3ms)"""
        data = setup_synthetic_concat
        fps = data["fps"]
        frame_dur_s = 1.0 / fps
        
        all_pts = _probe_all_pts(data["final"])
        
        cumulative_frames = 0
        max_delta = 0
        worst_chunk = -1
        
        for i, seg in enumerate(data["timeline"]):
            frames_in_chunk = round((seg.new_chunk_dur / 1000.0) * fps)
            cumulative_frames += frames_in_chunk
            
            # Tránh out of bounds nếu index vượt số PTS (sẽ bắt ở test khác)
            if cumulative_frames - 1 < len(all_pts):
                expected_pts = cumulative_frames / fps
                actual_pts = all_pts[cumulative_frames - 1]
                
                delta_s = abs(actual_pts - expected_pts)
                if delta_s > max_delta:
                    max_delta = delta_s
                    worst_chunk = i
                assert delta_s < frame_dur_s, f"Chunk {i} boundary PTS drift {delta_s*1000:.1f}ms >= {frame_dur_s*1000:.1f}ms"
        logger.info(f"[PTS Boundary] Max drift: {max_delta*1000:.1f}ms at chunk {worst_chunk}")

    def test_pts_monotonic(self, setup_synthetic_concat):
        """Test 4: PTS tăng dần nghiêm ngặt (0 violations)"""
        data = setup_synthetic_concat
        all_pts = _probe_all_pts(data["final"])
        
        violations = []
        for i in range(1, len(all_pts)):
            if all_pts[i] <= all_pts[i-1]:
                violations.append((i, all_pts[i-1], all_pts[i]))
                
        logger.info(f"[PTS Monotonic] Violations: {len(violations)}")
        assert len(violations) == 0, f"Found {len(violations)} non-monotonic PTS violations"

    def test_timebase_consistency(self, setup_synthetic_concat):
        """Test 5: Tất cả chunks có cùng timebase"""
        data = setup_synthetic_concat
        
        first_info = _probe_stream_info(data["chunk_paths"][0])
        first_timebase = first_info.get("time_base")
        
        logger.info(f"[Timebase Consistency] Target: {first_timebase}")
        for p in data["chunk_paths"][1:]:
            info = _probe_stream_info(p)
            assert info.get("time_base") == first_timebase, f"Timebase inconsistency: {info.get('time_base')} != {first_timebase}"

    def test_frame_delta_uniformity(self, setup_synthetic_concat):
        """Test 6: Khoảng cách frame đồng đều (Tolerance: 50% delta)"""
        data = setup_synthetic_concat
        fps = data["fps"]
        expected_delta = 1.0 / fps
        tolerance = expected_delta * 0.5
        
        all_pts = _probe_all_pts(data["final"])
        
        anomalies = []
        max_anomaly = 0
        for i in range(1, len(all_pts)):
            delta = all_pts[i] - all_pts[i-1]
            diff = abs(delta - expected_delta)
            if diff > tolerance:
                anomalies.append((i, delta))
                if diff > max_anomaly: max_anomaly = diff
                
        logger.info(f"[Frame Delta] Anomalies: {len(anomalies)}, Max Diff: {max_anomaly*1000:.1f}ms")
        assert len(anomalies) == 0, f"Found {len(anomalies)} frame delta anomalies"


# ═════════════════════════════════════════════════════════════════════
# LAYER 3 — REAL VIDEO FULL ANALYSIS (1000-3000 Chunks)
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg không có trong PATH")
class TestLayer3_ConcatDemuxerRealVideo:
    
    def test_full_desync_analysis(self, real_video_path, concat_workers, report_dir, tmp_path):
        """Test 7: Full analysis với video thật, tổng hợp TẤT CẢ kiểm tra và ghi report."""
        
        video_path_str = str(real_video_path)
        fps = 30.0 # Giả định normalize về 30fps
        dur_s = _probe_duration(video_path_str)
        num_chunks = random.randint(1000, 3000)
        
        timeline = _generate_chunks(dur_s * 1000, fps, num_chunks, (0.5, 1.0))
        
        # --- THỰC THI ---
        chunk_paths = _process_chunks_parallel(video_path_str, timeline, tmp_path, fps, max_workers=concat_workers)
        final_video = str(tmp_path / "final.mp4")
        _concat_chunks(chunk_paths, final_video)
        
        # --- THU THẬP SỐ LIỆU ---
        expected_dur_ms = sum(seg.new_chunk_dur for seg in timeline)
        actual_dur_ms = _probe_duration(final_video) * 1000
        dur_delta_ms = abs(actual_dur_ms - expected_dur_ms)
        
        # Tính số frame expected
        expected_frames = sum(round((seg.new_chunk_dur / 1000.0) * fps) for seg in timeline)
        
        # Probe thông tin (có thể chậm với video dài)
        actual_frames = _probe_frame_count(final_video)
        all_pts = _probe_all_pts(final_video)
        
        # Tính PTS boundaries
        max_pts_delta_s = 0.0
        worst_chunk = -1
        cumulative_frames = 0
        
        # Sample để tránh check quá lâu: check mỗi 10 chunks hoặc 200 chunks cuối
        for i, seg in enumerate(timeline):
            frames_in_chunk = round((seg.new_chunk_dur / 1000.0) * fps)
            cumulative_frames += frames_in_chunk
            
            if cumulative_frames - 1 < len(all_pts):
                expected_pts = cumulative_frames / fps
                actual_pts = all_pts[cumulative_frames - 1]
                delta_s = abs(actual_pts - expected_pts)
                if delta_s > max_pts_delta_s:
                    max_pts_delta_s = delta_s
                    worst_chunk = i
                    
        # Check monotonic
        monotonic_violations = sum(1 for i in range(1, len(all_pts)) if all_pts[i] <= all_pts[i-1])
        
        # Check timebase (chỉ sample 10 chunks đầu để nhanh)
        sample_paths = chunk_paths[:10]
        first_tb = _probe_stream_info(sample_paths[0]).get("time_base")
        tb_consistent = all(_probe_stream_info(p).get("time_base") == first_tb for p in sample_paths[1:])
        
        # Check frame delta anomalies
        expected_delta = 1.0 / fps
        delta_anomalies = [i for i in range(1, len(all_pts)) if abs((all_pts[i] - all_pts[i-1]) - expected_delta) > expected_delta * 0.5]
        
        # --- GHI REPORT ---
        dur_tolerance_ms = 500.0 + (num_chunks * 1.0) # Nới tolerance theo số chunk
        pts_tolerance_s = 1.0 / fps
        
        all_pass = (
            dur_delta_ms <= dur_tolerance_ms and
            actual_frames == expected_frames and
            max_pts_delta_s < pts_tolerance_s and
            monotonic_violations == 0 and
            tb_consistent and
            len(delta_anomalies) == 0
        )
        
        report = {
            "test_name": "concat_demuxer_full_analysis",
            "timestamp": datetime.now().isoformat(),
            "source_video": video_path_str,
            "params": {
                "chunk_count": num_chunks,
                "source_duration_s": dur_s,
                "source_fps": fps,
            },
            "checks": {
                "duration": {
                    "expected_s": expected_dur_ms / 1000.0,
                    "actual_s": actual_dur_ms / 1000.0,
                    "delta_ms": dur_delta_ms,
                    "tolerance_ms": dur_tolerance_ms,
                    "pass": dur_delta_ms <= dur_tolerance_ms
                },
                "frame_count": {
                    "expected": expected_frames,
                    "actual": actual_frames,
                    "pass": actual_frames == expected_frames
                },
                "pts_boundary": {
                    "max_delta_ms": max_pts_delta_s * 1000.0,
                    "worst_chunk_index": worst_chunk,
                    "tolerance_ms": pts_tolerance_s * 1000.0,
                    "pass": max_pts_delta_s < pts_tolerance_s
                },
                "pts_monotonic": {
                    "violations": monotonic_violations,
                    "pass": monotonic_violations == 0
                },
                "timebase": {
                    "consistent": tb_consistent,
                    "timebase": first_tb,
                    "pass": tb_consistent
                },
                "frame_delta": {
                    "anomalies": len(delta_anomalies),
                    "pass": len(delta_anomalies) == 0
                }
            },
            "verdict": "PASS" if all_pass else "FAIL",
            "diagnosis": f"Duration drift {dur_delta_ms:.1f}ms, Max PTS drift {max_pts_delta_s*1000:.1f}ms"
        }
        
        report_path = _write_desync_report(report, report_dir)
        print(f"\n📊 Full Desync Report: {report_path}")
        
        # --- ASSERT CUỐI CÙNG ---
        # Chạy assert duration đầu tiên, vì đây là lỗi user nghi ngờ nhất
        assert dur_delta_ms <= dur_tolerance_ms, f"Duration drift {dur_delta_ms:.1f}ms > {dur_tolerance_ms}ms tolerance. Xem chi tiết tại: {report_path}"
        assert all_pass, f"Một số kiểm tra desync đã thất bại. Xem chi tiết tại: {report_path}"