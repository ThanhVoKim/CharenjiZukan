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
    fps_str: str,
    fps_float: float,
    use_gpu: bool = True,
) -> List[str]:
    """
    Hybrid Seek (2-pass seek) cho cắt video chính xác frame-by-frame.

    Cơ chế:
    - Pass 1 (Input Seek): -ss TRƯỚC -i, lùi 2 giây so với mốc cắt thực tế
      → FFmpeg nhảy nhanh đến keyframe gần nhất, tránh decode từ đầu file.
    - Pass 2 (Output Seek): -ss SAU -i, với offset chính xác
      → FFmpeg decode từ keyframe đó và chỉ giữ frame từ offset trở đi,
      đảm bảo PTS bắt đầu chính xác từ 0, không có frame thừa.

    Kết hợp với:
    - Frame-accurate rounding (snap to frame boundaries)
    - CFR (Constant Frame Rate) qua filter fps=
    - setpts để stretch video theo tốc độ mong muốn
    """
    # Bước 1: Tính toán chính xác số frame để làm tròn điểm cắt
    start_frame = round((start_ms / 1000.0) * fps_float)
    duration_frames = round((input_duration_ms / 1000.0) * fps_float)

    # Bước 2: Quy đổi ngược lại thành thời gian thập phân siêu chuẩn xác
    start_s = start_frame / fps_float
    duration_s = duration_frames / fps_float

    # Bước 3: Hybrid Seek — lùi 2 giây để tìm keyframe, rồi offset chính xác
    rough_start_s = max(0.0, start_s - 2.0)
    exact_offset_s = start_s - rough_start_s

    pts_factor = 1.0 / video_speed

    encoder = "h264_nvenc" if use_gpu else "libx264"
    preset  = "p5"         if use_gpu else "fast"
    quality = ["-cq", "23"] if use_gpu else ["-crf", "23"]

    filter_chain = ",".join([
        f"trim=start={exact_offset_s:.6f}:duration={duration_s:.6f}",
        "setpts=PTS-STARTPTS",  # Đặt lại PTS về 0 ngay sau khi cắt
        f"setpts={pts_factor:.6f}*PTS", # Stretch video
        f"fps={fps_str}:round=1:eof_action=pass" # Đảm bảo constant frame rate, KHÔNG sinh frame thừa
    ])

    return [
        "ffmpeg", "-y",
        # Bước 1 (Fast Seek): Nhảy đến vị trí an toàn trước điểm cần cắt
        "-ss", f"{rough_start_s:.6f}",
        "-i", input_path,
        # Bước 2 (Accurate Trimming & Stretching) thông qua filter thay vì Output Seek
        "-filter:v", filter_chain,
        "-an",
        "-c:v", encoder,
        "-preset", preset,
        *quality,
        "-video_track_timescale", "90000",
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

def _probe_chunk_duration(chunk_path: str, fps_float: float) -> float:
    """Đo duration thực tế của chunk và snap về frame-aligned."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        chunk_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    raw_dur_s = float(result.stdout.strip())
    # Snap về frame-aligned để đồng nhất với video
    dur_frames = round(raw_dur_s * fps_float)
    return (dur_frames / fps_float) * 1000.0  # ms

def process_video_chunks_parallel(
    video_path: str,
    timeline: List[TimelineSegment],
    output_dir: str,
    max_workers: int = 4,
    use_gpu: bool = True,
    fps_str: str = "30/1",
    fps_float: float = 30.0,
) -> Tuple[str, List[float]]:
    """Split + stretch + concat. Returns (path to video_stretched.mp4, list of actual durations in ms)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    chunk_tasks = []
    for i, seg in enumerate(timeline):
        out = str(Path(output_dir) / f"chunk_{i:04d}.mp4")
        cmd = build_ffmpeg_chunk_cmd(
            video_path, out,
            seg.orig_start,
            seg.orig_end - seg.orig_start,   # input duration
            seg.video_speed,
            fps_str,
            fps_float,
            use_gpu,
        )
        chunk_tasks.append((i, cmd, out))

    if not chunk_tasks:
        raise RuntimeError("Timeline rỗng: không có segment nào để tạo chunk video.")

    results: Dict[int, str] = {}
    failed_chunks: Dict[int, str] = {}

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_chunk, t): t[0] for t in chunk_tasks}
        
        future_iter = as_completed(futures)
        if has_tqdm:
            future_iter = tqdm(future_iter, total=len(chunk_tasks), desc="Processing chunks", unit="chunk")
            
        for f in future_iter:
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
    
    actual_durations = []
    for i, chunk_path in enumerate(ordered):
        actual_dur = _probe_chunk_duration(chunk_path, fps_float)
        actual_durations.append(actual_dur)
        logger.debug(f"Chunk {i}: expected={timeline[i].new_chunk_dur:.3f}ms, actual={actual_dur:.3f}ms")
        
    return output_video, actual_durations

def _concat_chunks(chunk_paths: List[str], output_path: str) -> None:
    """
    THAY ĐỔI TỐI ƯU HÓA: Dùng concat demuxer (-c copy) thay vì filter_complex concat.
    Vì tất cả các chunk đã được ép chung 1 chuẩn FPS và Timebase ở bước trước,
    nên có thể copy nguyên luồng dữ liệu mà không cần render lại toàn bộ video.
    Giúp rút ngắn thời gian nối từ hàng chục phút xuống 1-2 giây.
    """
    if not chunk_paths:
        raise RuntimeError("Không có chunk nào để concat")

    # Validate tất cả chunks tồn tại
    for p in chunk_paths:
        if not Path(p).exists() or Path(p).stat().st_size == 0:
            raise RuntimeError(f"Chunk không hợp lệ hoặc rỗng: {p}")

    list_file = Path(output_path).with_suffix('.txt')
    try:
        with open(list_file, "w", encoding="utf-8") as f:
            for p in chunk_paths:
                # Dùng as_posix() để tránh lỗi backslash trên Windows
                f.write(f"file '{Path(p).resolve().as_posix()}'\n")

        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c:v", "copy",
            "-an",
            output_path,
        ], check=True, capture_output=True, text=True)
        logger.info(f"Concat {len(chunk_paths)} chunks siêu tốc → {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error("FFmpeg concat demuxer failed (returncode=%s)", e.returncode)
        if e.stdout:
            logger.error("FFmpeg concat stdout:\n%s", e.stdout[-8000:])
        if e.stderr:
            logger.error("FFmpeg concat stderr:\n%s", e.stderr[-8000:])
        raise
    finally:
        # Xóa file text tạm
        if list_file.exists():
            list_file.unlink(missing_ok=True)

