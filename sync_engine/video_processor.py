import math
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
    new_duration_s = duration_s * pts_factor  # Độ dài output sau stretch

    encoder = "h264_nvenc" if use_gpu else "libx264"
    preset  = "p5"         if use_gpu else "fast"
    quality = ["-cq", "23"] if use_gpu else ["-crf", "23"]

    filter_chain = ",".join([
        f"trim=start={exact_offset_s:.6f}:duration={duration_s:.6f}",
        "setpts=PTS-STARTPTS", # Đặt lại PTS về 0 ngay sau khi cắt
        f"setpts={pts_factor:.6f}*PTS", # Stretch video
        f"fps={fps_str}:eof_action=pass", # Đảm bảo constant frame rate, không sinh thêm frame khi EOF
        f"trim=duration={new_duration_s:.6f}",  # ← Chốt chặn frame count
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

def _run_batch(args: tuple) -> Tuple[int, str, str]:
    """Worker: chạy 1 lệnh FFmpeg xử lý 1 batch chunk, trả về (batch_idx, output_path, error)."""
    idx, cmd, out_path = args
    logger.debug(f"Running batch {idx} with command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, timeout=1200)
        logger.debug(f"Batch {idx} output: {result.stdout.decode(errors='ignore')}")
        return idx, out_path, ""
    except subprocess.TimeoutExpired:
        err_msg = f"Batch {idx} timeout sau 1200 giây."
        logger.error(err_msg)
        return idx, out_path, err_msg
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode(errors="ignore") if e.stderr else str(e)
        logger.error(f"Batch {idx} failed with error: {err_msg}")
        return idx, out_path, err_msg[-2000:]

def build_ffmpeg_batch_cmd(
    input_path: str,
    output_path: str,
    segments: List[TimelineSegment],
    fps_str: str,
    fps_float: float,
    use_gpu: bool = True,
) -> List[str]:
    """
    Xây dựng lệnh Filter Complex để xử lý nối tiếp nhiều segment trong 1 mẻ (Batching).
    
    Sử dụng Hybrid Seek (Fast Seek) để tránh FFmpeg phải decode từ đầu file:
    - Tìm min(start_s) của tất cả segments trong batch
    - Lùi 2 giây để tạo rough_start_s (đảm bảo keyframe an toàn)
    - Chèn -ss {rough_start_s} trước -i → FFmpeg nhảy thẳng đến vị trí batch
    - Trong filter, trừ rough_start_s để có exact_start_s (PTS đã được reset về 0)
    
    Điều này đảm bảo TẤT CẢ các batch đều chạy với tốc độ như nhau (không bị
    chậm dần theo cấp số nhân như khi không có Fast Seek).
    """
    if not segments:
        raise RuntimeError("Danh sách segments rỗng, không thể build batch command.")

    # ── Hybrid Seek: tính rough_start_s cho toàn bộ batch ──
    # Tìm start sớm nhất trong batch
    min_start_s = min(
        round((seg.orig_start / 1000.0) * fps_float) / fps_float
        for seg in segments
    )
    # Lùi 2 giây để đảm bảo keyframe an toàn (giống build_ffmpeg_chunk_cmd)
    rough_start_s = max(0.0, min_start_s - 2.0)

    filter_parts = []
    stream_labels = []

    for i, seg in enumerate(segments):
        start_frame = round((seg.orig_start / 1000.0) * fps_float)
        duration_frames = round(((seg.orig_end - seg.orig_start) / 1000.0) * fps_float)

        # start_s tuyệt đối trong video gốc
        abs_start_s = start_frame / fps_float
        duration_s = duration_frames / fps_float

        # exact_start_s: vị trí tương đối so với rough_start_s
        # Vì -ss trước -i reset PTS về 0 tại điểm seek,
        # nên trim chỉ cần dùng offset từ điểm seek
        exact_start_s = abs_start_s - rough_start_s

        pts_factor = 1.0 / seg.video_speed
        stretched_duration_s = duration_s / seg.video_speed

        # Độ dài output sau khi giãn — KHÔNG cộng thêm frame dư
        expected_duration_s = stretched_duration_s

        # Chuỗi filter cho 1 segment:
        # Cắt đúng thời gian (từ offset) -> Reset PTS -> Giãn PTS -> Nắn fps -> Chốt chặn đuôi trim
        chain = (
            f"[0:v]trim=start={exact_start_s:.6f}:duration={duration_s:.6f},"
            f"setpts=PTS-STARTPTS,"
            f"setpts={pts_factor:.6f}*PTS,"
            f"fps={fps_str}:eof_action=pass,"
            # Chốt chặn trim lần 2: Đảm bảo luồng không bao giờ dư 1 miligiây nào
            f"trim=duration={expected_duration_s:.6f}[v{i}]"
        )
        filter_parts.append(chain)
        stream_labels.append(f"[v{i}]")

    # Gom các luồng lại bằng filter concat
    concat_inputs = "".join(stream_labels)
    filter_parts.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=0[outv]")

    filter_complex = ";".join(filter_parts)

    encoder = "h264_nvenc" if use_gpu else "libx264"
    preset  = "p5"         if use_gpu else "fast"
    quality = ["-cq", "23"] if use_gpu else ["-crf", "23"]

    cmd = [
        "ffmpeg", "-y",
        # Hybrid Seek: nhảy thẳng đến vị trí của batch (Fast Seek)
        "-ss", f"{rough_start_s:.6f}",
        "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-an",
        "-c:v", encoder,
        "-preset", preset,
        *quality,
        "-video_track_timescale", "90000",
        output_path,
    ]

    logger.debug(
        f"Batch command: rough_start={rough_start_s:.3f}s, "
        f"min_abs_start={min_start_s:.3f}s, "
        f"segments={len(segments)}, seek_offset={rough_start_s:.3f}s"
    )

    return cmd

def process_video_chunks_parallel(
    video_path: str,
    timeline: List[TimelineSegment],
    output_dir: str,
    max_workers: int = 4,
    use_gpu: bool = True,
    fps_str: str = "30/1",
    fps_float: float = 30.0,
    batch_size: int = 100,
) -> Tuple[str, List[float]]:
    """Split + stretch + concat sử dụng Filter Complex Batching. Returns (path to video_stretched.mp4, list of actual durations in ms)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not timeline:
        raise RuntimeError("Timeline rỗng: không có segment nào để tạo chunk video.")

    # 1. Gom nhóm timeline thành các Batch
    batches = [timeline[i:i + batch_size] for i in range(0, len(timeline), batch_size)]
    
    batch_tasks = []
    for i, batch_segs in enumerate(batches):
        out = str(Path(output_dir) / f"batch_{i:04d}.mp4")
        cmd = build_ffmpeg_batch_cmd(
            video_path, out,
            batch_segs,
            fps_str, fps_float, use_gpu
        )
        batch_tasks.append((i, cmd, out))

    results: Dict[int, str] = {}
    failed_batches: Dict[int, str] = {}

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    logger.info(f"Bắt đầu xử lý {len(batches)} batches (mỗi batch {batch_size} chunks) song song...")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_batch, t): t[0] for t in batch_tasks}
        
        future_iter = as_completed(futures)
        if has_tqdm:
            future_iter = tqdm(future_iter, total=len(batch_tasks), desc="Processing batches", unit="batch")
            
        for f in future_iter:
            idx, out_path, err = f.result()
            if err:
                failed_batches[idx] = err
                logger.error(f"Batch {idx} lỗi: {err}")
                continue

            out_file = Path(out_path)
            if not out_file.exists() or out_file.stat().st_size <= 0:
                err_msg = f"Batch output không hợp lệ: {out_path}"
                failed_batches[idx] = err_msg
                logger.error(f"Batch {idx} lỗi: {err_msg}")
                continue

            results[idx] = out_path

    if failed_batches:
        failed_summary = "; ".join(
            f"#{idx}: {msg.replace(chr(10), ' ')[:200]}"
            for idx, msg in sorted(failed_batches.items())
        )
        raise RuntimeError(
            f"{len(failed_batches)}/{len(batch_tasks)} batch bị lỗi, hủy concat. Chi tiết: {failed_summary}"
        )

    if len(results) != len(batch_tasks):
        missing = sorted(set(range(len(batch_tasks))) - set(results.keys()))
        raise RuntimeError(f"Thiếu batch output cho các index: {missing}. Hủy concat.")

    ordered_batches = [results[i] for i in range(len(batch_tasks))]
    
    # 2. Concat các Batch lại thành file cuối
    output_video = str(Path(output_dir) / "video_stretched.mp4")
    _concat_chunks(ordered_batches, output_video)
    
    # 3. Tính toán actual_durations dựa trên công thức chuẩn (KHÔNG +1 frame dư)
    # Công thức: stretched_duration = (orig_duration / video_speed)
    # Điều này đảm bảo actual_durations khớp chính xác với video output,
    # tránh Audio/Subtitle bị lệch (delayed) so với hình ảnh.
    actual_durations = []
    for seg in timeline:
        duration_frames = round(((seg.orig_end - seg.orig_start) / 1000.0) * fps_float)
        duration_s = duration_frames / fps_float
        stretched_duration_s = duration_s / seg.video_speed
        actual_dur = stretched_duration_s * 1000.0
        actual_durations.append(actual_dur)
        
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

