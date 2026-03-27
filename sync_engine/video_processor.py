import subprocess
from pathlib import Path
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from sync_engine.models import TimelineSegment

logger = logging.getLogger(__name__)

def query_keyframes(video_path: str) -> List[float]:
    """Lấy PTS (ms) của tất cả keyframe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v",
        "-show_entries", "frame=pkt_pts_time,key_frame",
        "-of", "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    kfs = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) == 2 and parts[1] == "1":
            try:
                kfs.append(float(parts[0]) * 1000)
            except ValueError:
                pass
    return sorted(kfs)

def snap_to_nearest_keyframe(ts_ms: float, keyframes: List[float]) -> float:
    if not keyframes:
        return ts_ms
    return min(keyframes, key=lambda k: abs(k - ts_ms))

def build_ffmpeg_chunk_cmd(
    input_path: str,
    output_path: str,
    start_ms: float,
    input_duration_ms: float,
    video_speed: float,
    use_gpu: bool = True,
) -> List[str]:
    """
    THAY ĐỔI:
    - LUÔN dùng setpts=(1/speed)*(PTS-STARTPTS) kể cả speed=1.0 để reset PTS.
    - -ss đặt SAU -i (slow seek) để seek chính xác, tránh frame thừa đầu chunk.
    - Không dùng -c:v copy vì không thể reset PTS.
    """
    start_s    = start_ms          / 1000.0
    duration_s = input_duration_ms / 1000.0
    pts_factor = 1.0 / video_speed  # speed=1.0 → pts_factor=1.0

    encoder = "h264_nvenc" if use_gpu else "libx264"
    preset  = "p4"         if use_gpu else "fast"
    quality = ["-cq", "23"] if use_gpu else ["-crf", "23"]

    return [
        "ffmpeg", "-y",
        "-i", input_path,
        # -ss SAU -i: slow seek, decode-from-keyframe nhưng exact timestamp
        "-ss", f"{start_s:.6f}",
        "-t",  f"{duration_s:.6f}",
        "-filter:v", f"setpts={pts_factor:.6f}*(PTS-STARTPTS)",
        "-an",
        "-c:v", encoder,
        "-preset", preset,
        *quality,
        output_path,
    ]

def _run_chunk(args: tuple) -> Tuple[int, str, str]:
    """Worker: chạy 1 FFmpeg command, trả về (index, output_path, error)."""
    idx, cmd, out_path = args
    logger.debug(f"Running chunk {idx} with command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        logger.debug(f"Chunk {idx} output: {result.stdout.decode(errors='ignore')}")
        return idx, out_path, ""
    except subprocess.TimeoutExpired:
        err_msg = f"Chunk {idx} timeout sau 600 giây."
        logger.error(err_msg)
        return idx, out_path, err_msg
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode(errors="ignore") if e.stderr else str(e)
        logger.error(f"Chunk {idx} failed with error: {err_msg}")
        return idx, out_path, err_msg[-2000:]

def process_video_chunks_parallel(
    video_path: str,
    timeline: List[TimelineSegment],
    output_dir: str,
    max_workers: int = 4,
    use_gpu: bool = True,
) -> str:
    """Split + stretch + concat. Returns path to video_stretched.mp4."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    chunk_tasks = []
    for i, seg in enumerate(timeline):
        out = str(Path(output_dir) / f"chunk_{i:04d}.mp4")
        cmd = build_ffmpeg_chunk_cmd(
            video_path, out,
            seg.orig_start,
            seg.orig_end - seg.orig_start,   # input duration
            seg.video_speed,
            use_gpu,
        )
        chunk_tasks.append((i, cmd, out))

    if not chunk_tasks:
        raise RuntimeError("Timeline rỗng: không có segment nào để tạo chunk video.")

    results: Dict[int, str] = {}
    failed_chunks: Dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_chunk, t): t[0] for t in chunk_tasks}
        for f in as_completed(futures):
            idx, out_path, err = f.result()
            if err:
                failed_chunks[idx] = err
                logger.error(f"Chunk {idx} lỗi: {err}")
                continue

            out_file = Path(out_path)
            if not out_file.exists() or out_file.stat().st_size <= 0:
                err_msg = f"Chunk output không hợp lệ: {out_path}"
                failed_chunks[idx] = err_msg
                logger.error(f"Chunk {idx} lỗi: {err_msg}")
                continue

            results[idx] = out_path

    if failed_chunks:
        failed_summary = "; ".join(
            f"#{idx}: {msg.replace(chr(10), ' ')[:200]}"
            for idx, msg in sorted(failed_chunks.items())
        )
        raise RuntimeError(
            f"{len(failed_chunks)}/{len(chunk_tasks)} chunk bị lỗi, hủy concat. Chi tiết: {failed_summary}"
        )

    if len(results) != len(chunk_tasks):
        missing = sorted(set(range(len(chunk_tasks))) - set(results.keys()))
        raise RuntimeError(f"Thiếu chunk output cho các index: {missing}. Hủy concat.")

    ordered = [results[i] for i in range(len(chunk_tasks))]
    if not ordered:
        raise RuntimeError("Không có chunk hợp lệ để concat.")

    output_video = str(Path(output_dir) / "video_stretched.mp4")
    _concat_chunks(ordered, output_video)
    return output_video

def _concat_chunks(chunk_paths: List[str], output_path: str) -> None:
    """
    THAY ĐỔI: Dùng filter_complex concat thay vì concat demuxer.
    filter_complex concat tự xử lý PTS discontinuity giữa các chunk.
    """
    if not chunk_paths:
        raise RuntimeError("Không có chunk nào để concat")

    # Validate tất cả chunks tồn tại
    for p in chunk_paths:
        if not Path(p).exists() or Path(p).stat().st_size == 0:
            raise RuntimeError(f"Chunk không hợp lệ hoặc rỗng: {p}")

    n = len(chunk_paths)
    input_args = []
    for p in chunk_paths:
        input_args += ["-i", str(Path(p).resolve())]

    # Build filter: [0:v][1:v][2:v]...concat=n=N:v=1[outv]
    stream_labels = "".join(f"[{i}:v]" for i in range(n))
    filter_str    = f"{stream_labels}concat=n={n}:v=1:a=0[outv]"

    try:
        subprocess.run([
            "ffmpeg", "-y",
            *input_args,
            "-filter_complex", filter_str,
            "-map", "[outv]",
            "-an",
            "-c:v", "libx264",  # Re-encode để đảm bảo stream clean cho render phase 5
            "-preset", "fast",
            "-crf", "23",
            output_path,
        ], check=True, capture_output=True, text=True)
        logger.info(f"Concat {n} chunks → {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error("FFmpeg concat failed (returncode=%s)", e.returncode)
        if e.stdout:
            logger.error("FFmpeg concat stdout:\n%s", e.stdout[-8000:])
        if e.stderr:
            logger.error("FFmpeg concat stderr:\n%s", e.stderr[-8000:])
        raise
