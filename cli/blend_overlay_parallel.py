#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli/blend_overlay_parallel.py — Phủ một video "blend" (scratch/dust/noise) lên video
gốc bằng blend mode, render SONG SONG nhiều tiến trình FFmpeg để tăng tốc.

Bài toán
--------
Lệnh 1-pass (1 tiến trình ffmpeg, filter_complex đơn luồng) phải xử lý tuần tự cả
tiếng video → rất chậm. Đồng thời `-stream_loop -1` trên input blend khiến luồng
blend VÔ HẠN; nếu cơ chế `shortest` không nhả EOF đúng → render mãi không dừng.

Cách giải
---------
1. Probe fps gốc (r_frame_rate, GIỮ NGUYÊN dạng phân số → CFR chuẩn xác) + W×H gốc.
2. Quy mọi mốc cắt về SỐ FRAME NGUYÊN (không bao giờ cắt ở giây thập phân lẻ),
   chia video thành đúng `--workers` đoạn theo bội số NGUYÊN của độ dài blend (L):
   - Mọi điểm nối nội bộ rơi đúng k·L  → khi mỗi đoạn khởi động blend lại từ 0,
     pha texture tự khớp như 1-pass (liền mạch qua mối nối).
   - Phần dư lẻ (vd 0.5·L) DỒN VÀO ĐOẠN CUỐI — sau nó không còn mối nối nên vô hại.
3. Mỗi đoạn render CHỈ video (`-an`), chốt cứng `-frames:v N` → không phụ thuộc EOF
   của blend loop → KHÔNG còn nguy cơ render vô hạn; output dài đúng số frame gốc.
4. Concat các đoạn bằng concat demuxer `-c:v copy` (frame-exact, vài giây) và ghép
   AUDIO GỐC nguyên vẹn `-c:a copy` trong cùng một bước → audio zero-drift, độ dài
   output = đúng độ dài video gốc.

Video gốc GIỮ NGUYÊN độ phân giải (full width × full height); blend được scale-crop
cho khớp W×H gốc. `blend.mp4` chỉ là lớp pixel cosmetic (không mang timeline, không
audio) nên cắt frame KHÔNG ảnh hưởng đồng bộ — đồng bộ chỉ phụ thuộc cách cắt main.

Ví dụ
------
    uv run cli/blend_overlay_parallel.py \
        --video /content/Project/1/youtube.mp4 \
        --blend "/content/Scratch And Dust Screen.mp4" \
        --output /content/Project/1/youtube__flipped.mp4 \
        --workers 4

    # Batch nhiều video qua task-file JSON: [{"input": "...", "output": "..."}]
    uv run cli/blend_overlay_parallel.py \
        --blend "/content/Scratch And Dust Screen.mp4" \
        --task-file tasks.json --workers 4
"""

import argparse
import json
import logging
import sys
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.ffmpeg_probe import (  # noqa: E402
    HEVC_NVENC_VIDEO_ARGS,
    detect_hevc_nvenc,
    get_hevc_nvenc_unavailable_reason,
)

logger = logging.getLogger(__name__)

# Clone đuôi đoạn cuối trước khi chốt -frames:v, phòng main hụt vài frame ở EOF
# (rounding duration*fps) → đoạn cuối vẫn đủ số frame lý thuyết. Đoạn không-cuối
# bị -frames:v cắt bỏ phần clone ngay, không encode thừa.
_TAIL_PAD_SECONDS = 2.0


# ──────────────────────────────────────────────────────────────────────────────
# Probe helpers
# ──────────────────────────────────────────────────────────────────────────────
def _ffprobe(args: List[str]) -> str:
    return subprocess.run(
        ["ffprobe", "-v", "error", *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def probe_fps(video_path: str) -> Tuple[str, float]:
    """Trả về (fps_str, fps_float). Giữ fps_str dạng phân số để ép CFR chuẩn xác."""
    fps_str = _ffprobe([
        "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path,
    ])
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps_float = float(num) / float(den)
    else:
        fps_float = float(fps_str)
        fps_str = f"{fps_str}/1"
    if fps_float <= 0:
        raise RuntimeError(f"FPS không hợp lệ ({fps_str}) cho {video_path}")
    return fps_str, fps_float


def probe_dimensions(video_path: str) -> Tuple[int, int]:
    """Trả về (width, height) của luồng video đầu tiên."""
    out = _ffprobe([
        "-select_streams", "v:0", "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0", video_path,
    ])
    w_str, h_str = out.split("x")
    w, h = int(w_str), int(h_str)
    if w <= 0 or h <= 0:
        raise RuntimeError(f"Kích thước video không hợp lệ ({out}) cho {video_path}")
    return w, h


def probe_duration(video_path: str) -> float:
    """Độ dài (giây) lấy từ format=duration."""
    return float(_ffprobe([
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path,
    ]))


# ──────────────────────────────────────────────────────────────────────────────
# Segment planning
# ──────────────────────────────────────────────────────────────────────────────
def plan_segments(
    total_frames: int, blend_frames: int, workers: int,
) -> List[Tuple[int, int]]:
    """Chia [0, total_frames) thành các đoạn (start_frame, num_frames).

    Mọi điểm nối nội bộ là bội số NGUYÊN của blend_frames (L) → blend liền mạch.
    Phần dư lẻ dồn vào đoạn cuối. Nếu blend dài hơn video hoặc workers<=1 → 1 đoạn.
    """
    if blend_frames <= 0 or total_frames <= blend_frames or workers <= 1:
        return [(0, total_frames)]

    integer_loops = total_frames // blend_frames
    remainder = total_frames - integer_loops * blend_frames

    # Không đủ số loop để chia cho workers → giảm số đoạn xuống integer_loops
    n_seg = min(workers, integer_loops)
    base = integer_loops // n_seg
    extra = integer_loops % n_seg
    loops_per = [base + (1 if i < extra else 0) for i in range(n_seg)]  # vd [3,3,2,2]

    segs: List[Tuple[int, int]] = []
    cur = 0
    for i, lp in enumerate(loops_per):
        n = lp * blend_frames
        if i == n_seg - 1:
            n += remainder  # dư lẻ (vd 0.5·L) vào đoạn cuối — sau nó không còn mối nối
        segs.append((cur, n))
        cur += n
    return segs


# ──────────────────────────────────────────────────────────────────────────────
# FFmpeg command builders
# ──────────────────────────────────────────────────────────────────────────────
def build_segment_cmd(
    video_path: str,
    blend_path: str,
    output_path: str,
    start_frame: int,
    num_frames: int,
    is_last: bool,
    fps_str: str,
    fps_float: float,
    width: int,
    height: int,
    blend_mode: str,
    opacity: float,
) -> List[str]:
    """Lệnh render 1 đoạn: cắt main theo frame (giữ nguyên W×H) + phủ blend + chốt frame."""
    # Hybrid seek: nhảy nhanh tới gần điểm cắt rồi trim chính xác trong filter.
    start_s = start_frame / fps_float
    rough_start_s = max(0.0, start_s - 2.0)
    start_off = start_s - rough_start_s
    # Cắt sớm hơn nửa frame để không hụt frame đầu; -frames:v sẽ chốt đúng số frame.
    safe_start = max(0.0, start_off - 0.5 / fps_float)
    safe_dur = (num_frames + 1) / fps_float

    # Main giữ NGUYÊN độ phân giải gốc, chỉ chuẩn hoá pixel format cho blend.
    main_chain = (
        f"[0:v]trim=start={safe_start:.6f}:duration={safe_dur:.6f},"
        f"setpts=PTS-STARTPTS,"
        f"fps={fps_str}:eof_action=pass,"
        f"setsar=1,format=gbrp"
    )
    if is_last:
        # Clone đuôi phòng main hụt frame ở EOF; -frames:v cắt lại đúng N.
        main_chain += f",tpad=stop_mode=clone:stop_duration={_TAIL_PAD_SECONDS:.6f}"
    main_chain += "[main]"

    # Blend scale-crop cho khớp đúng W×H video gốc.
    blend_chain = (
        f"[1:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1,format=gbrp[blend]"
    )

    filter_complex = ";".join([
        main_chain,
        blend_chain,
        f"[main][blend]blend=all_mode='{blend_mode}':all_opacity={opacity}:shortest=1[bl]",
        "[bl]format=yuv420p[v]",
    ])

    return [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        # Main: fast seek tới gần điểm cắt
        "-ss", f"{rough_start_s:.6f}", "-i", _posix(video_path),
        # Blend: luôn khởi động lại từ 0 cho từng đoạn (loop vô hạn, shortest chặn)
        "-stream_loop", "-1", "-i", _posix(blend_path),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-an",
        # Chốt cứng số frame → không phụ thuộc EOF blend → KHÔNG render vô hạn
        "-frames:v", str(num_frames),
        *HEVC_NVENC_VIDEO_ARGS,
        "-video_track_timescale", "90000",
        _posix(output_path),
    ]


def _posix(p) -> str:
    return str(p).replace("\\", "/")


def _run_segment(task: Tuple[int, List[str], str]) -> Tuple[int, str, str]:
    idx, cmd, out_path = task
    logger.debug("Segment %d: %s", idx, " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return idx, out_path, f"Segment {idx} timeout sau 3600s"
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="ignore") if e.stderr else str(e)
        return idx, out_path, err[-2000:]
    if not Path(out_path).exists() or Path(out_path).stat().st_size <= 0:
        return idx, out_path, f"Segment {idx} output rỗng: {out_path}"
    return idx, out_path, ""


def concat_and_mux(
    segment_paths: List[str], video_path: str, output_path: str,
) -> None:
    """Concat video (copy) + ghép audio gốc (copy) trong 1 bước.

    Audio không bao giờ bị encode lại → zero drift; độ dài = đúng video gốc.
    """
    list_file = Path(output_path).with_suffix(".concat.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for p in segment_paths:
            f.write(f"file '{Path(p).resolve().as_posix()}'\n")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-i", _posix(video_path),
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "copy", "-c:a", "copy",
            _posix(output_path),
        ], check=True, capture_output=True, text=True)
    finally:
        list_file.unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Task resolution
# ──────────────────────────────────────────────────────────────────────────────
def resolve_tasks(args: argparse.Namespace) -> List[Tuple[str, str]]:
    """Trả về danh sách (input_video, output_video) từ --task-file hoặc --video/--output."""
    if args.task_file:
        data = json.loads(Path(args.task_file).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("task-file phải là JSON array các object {input, output}.")
        tasks: List[Tuple[str, str]] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict) or "input" not in item or "output" not in item:
                raise ValueError(f"task-file phần tử #{i} phải có khoá 'input' và 'output'.")
            tasks.append((str(item["input"]), str(item["output"])))
        if not tasks:
            raise ValueError("task-file rỗng: không có task nào.")
        return tasks

    if not args.video or not args.output:
        raise ValueError("Cần --video và --output, hoặc dùng --task-file.")
    return [(args.video, args.output)]


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────────
def process_one(
    video_path: str, blend_path: str, output_path: str, args: argparse.Namespace,
) -> int:
    """Xử lý 1 video: chia đoạn, render song song, concat + mux audio gốc."""
    for p in (video_path, blend_path):
        if not Path(p).exists():
            logger.error("Không tìm thấy file: %s", p)
            return 2

    fps_str, fps_float = probe_fps(video_path)
    width, height = probe_dimensions(video_path)
    total_frames = round(probe_duration(video_path) * fps_float)
    blend_frames = round(probe_duration(blend_path) * fps_float)
    logger.info(
        "%s | %dx%d @ %s (%.3f) | total_frames=%d | blend_frames=%d (~%.2f loops)",
        Path(video_path).name, width, height, fps_str, fps_float,
        total_frames, blend_frames,
        total_frames / blend_frames if blend_frames else 0,
    )

    segments = plan_segments(total_frames, blend_frames, args.workers)
    logger.info(
        "Chia %d đoạn (workers=%d): %s",
        len(segments), args.workers,
        ", ".join(f"[{s}..{s+n}) {n}f" for s, n in segments),
    )

    # Thư mục tạm: tmp/blend_<uid timestamp> ở project root
    tmp_dir = PROJECT_ROOT / "tmp" / f"blend_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    tasks: List[Tuple[int, List[str], str]] = []
    for i, (start_frame, num_frames) in enumerate(segments):
        out = str(tmp_dir / f"seg_{i:03d}.mp4")
        cmd = build_segment_cmd(
            video_path, blend_path, out, start_frame, num_frames,
            is_last=(i == len(segments) - 1),
            fps_str=fps_str, fps_float=fps_float,
            width=width, height=height,
            blend_mode=args.mode, opacity=args.opacity,
        )
        tasks.append((i, cmd, out))

    # In ra câu lệnh từng đoạn để theo dõi / chạy lại thủ công khi cần.
    for i, cmd, _ in tasks:
        logger.info("Segment %d:\n%s\n", i, " ".join(cmd))

    results: dict = {}
    failed: dict = {}
    logger.info("Render %d đoạn song song (max_workers=%d)...", len(tasks), args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_segment, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            idx, out_path, err = fut.result()
            if err:
                failed[idx] = err
                logger.error("Segment %d lỗi: %s", idx, err)
            else:
                results[idx] = out_path
                logger.info("Segment %d xong → %s", idx, out_path)

    try:
        if failed:
            logger.error("%d/%d đoạn lỗi, hủy: %s", len(failed), len(tasks), output_path)
            return 1

        ordered = [results[i] for i in range(len(tasks))]
        logger.info("Concat %d đoạn + ghép audio gốc → %s", len(ordered), output_path)
        concat_and_mux(ordered, video_path, output_path)
        logger.info("✅ Hoàn tất: %s", output_path)
        return 0
    finally:
        if not args.keep_tmp:
            for p in tmp_dir.glob("*"):
                p.unlink(missing_ok=True)
            try:
                tmp_dir.rmdir()
            except OSError:
                pass


def run(args: argparse.Namespace) -> int:
    if not detect_hevc_nvenc():
        reason = get_hevc_nvenc_unavailable_reason()
        logger.error("hevc_nvenc không khả dụng: %s", reason or "(không rõ)")
        return 2

    if not Path(args.blend).exists():
        logger.error("Không tìm thấy blend: %s", args.blend)
        return 2

    tasks = resolve_tasks(args)
    logger.info("Tổng %d video cần xử lý.", len(tasks))

    rc = 0
    for n, (video_path, output_path) in enumerate(tasks, 1):
        logger.info("─── [%d/%d] %s → %s ───", n, len(tasks), video_path, output_path)
        rc_one = process_one(video_path, args.blend, output_path, args)
        if rc_one != 0:
            rc = rc_one
            logger.error("Video #%d lỗi (rc=%d): %s", n, rc_one, video_path)
    return rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="blend_overlay_parallel",
        description="Phủ blend video (scratch/dust) lên video gốc, render song song.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video", help="Video gốc (mang timeline + audio). Bỏ qua nếu dùng --task-file.")
    p.add_argument("--output", help="File output. Bỏ qua nếu dùng --task-file.")
    p.add_argument("--task-file", "-t", metavar="JSON",
                   help="JSON array [{\"input\": \"...\", \"output\": \"...\"}] để xử lý nhiều video.")
    p.add_argument("--blend", required=True, help="Video blend phủ lên (loop, cosmetic).")
    p.add_argument("--workers", type=int, default=4, help="Số đoạn song song (mặc định 4).")
    p.add_argument("--mode", default="subtract", help="blend all_mode (mặc định subtract).")
    p.add_argument("--opacity", type=float, default=0.9, help="blend all_opacity (mặc định 0.9).")
    p.add_argument("--keep-tmp", action="store_true", help="Giữ thư mục đoạn tạm tmp/blend_*.")
    p.add_argument("-v", "--verbose", action="store_true", help="Log chi tiết (DEBUG).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return run(args)
    except (ValueError, json.JSONDecodeError) as e:
        logger.error("Tham số/Task-file lỗi: %s", e)
        return 2
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="ignore") if isinstance(e.stderr, bytes) else (e.stderr or "")
        logger.error("FFmpeg lỗi: %s", err[-2000:])
        return 1
    except Exception as e:  # noqa: BLE001
        logger.error("Lỗi: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
