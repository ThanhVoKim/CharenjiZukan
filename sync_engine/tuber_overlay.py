"""
sync_engine/tuber_overlay.py
============================
Orchestration tuber overlay (Phase I render, N composite, O validate, P concat,
R retry, T cleanup). Pipeline thuần Python/PIL/FFmpeg: character frames được
pre-render sẵn (tuber_prerender.py), runtime chỉ composite lên video stretched.

Dùng chung sync_video (all-in) và tuber_repair (late repair).

Flow một job:
  load prerender_manifest.json (bake sẵn nếu thiếu/stale)
  với mỗi group: composite overlay (direct pipe RGBA hoặc png_sequence) lên
    video_stretched → validate → cleanup overlay_frames
  retry group fail tới retryAttempts; hết → fallback (render_without_tuber)
  concat group video_with_tuber.mp4 → media/video_stretched_with_tuber.mp4
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from sync_engine.tuber_artifacts import cleanup_overlay_frames
from sync_engine import tuber_status as st
from sync_engine.tuber_status import compute_group_input_hash
from sync_engine.tuber_manifest import compute_character_box
from utils.ffmpeg_probe import HEVC_NVENC_VIDEO_ARGS as _HEVC_NVENC_VIDEO_ARGS

logger = logging.getLogger("sync_video")

class TuberOverlayError(RuntimeError):
    """Lỗi không thể phục hồi của tuber flow → caller fallback render_without_tuber."""


# ════════════════════════════════════════════════════════════════════
# COMPOSITE (Phase N) + VALIDATE (Phase O)
# ════════════════════════════════════════════════════════════════════

def _detect_frame_pattern(overlay_dir: Path) -> str:
    """Dò pattern frame_%0Nd.png từ file thật (zero-pad theo số frame)."""
    frames = sorted(overlay_dir.glob("frame_*.png"))
    if not frames:
        raise TuberOverlayError(f"Không có overlay frame trong {overlay_dir}")
    # width = số chữ số của phần numeric (frame_00.png → 2)
    stem = frames[0].stem  # frame_00
    num_part = stem.split("_", 1)[1]
    width = len(num_part)
    return f"frame_%0{width}d.png"


def composite_group_from_stretched(
    stretched_video: Path,
    overlay_dir: Path,
    output: Path,
    fps_str: str,
    fps_float: float,
    *,
    render_start_frame: int,
    render_duration_frames: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> Path:
    """Composite overlay lên stretched video (seek-by-frame) → video_with_tuber.

    Gộp 2 encode (stretch + overlay) thành 1 encode (seek trim + overlay).
    Seek trực tiếp vào video_stretched.mp4 (đã promote) bằng hybrid fast-seek,
    trim-by-frame trong filter, rồi overlay PNG sequence.
    """
    start_s = render_start_frame / fps_float
    rough_start_s = max(0.0, start_s - 2.0)
    exact_offset_s = start_s - rough_start_s
    safe_start_s = max(0.0, exact_offset_s - (0.5 / fps_float))

    pattern = _detect_frame_pattern(overlay_dir)
    filter_complex = (
        f"[0:v]trim=start={safe_start_s:.6f},setpts=PTS-STARTPTS,"
        f"fps={fps_str}:eof_action=pass,"
        f"trim=end_frame={render_duration_frames}[vb];"
        f"[vb][1:v]overlay=x={offset_x}:y={offset_y}:format=auto:shortest=1[outv]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{rough_start_s:.6f}", "-i", str(stretched_video),
        "-framerate", fps_str, "-start_number", "0",
        "-i", str(overlay_dir / pattern),
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-an",
        *_HEVC_NVENC_VIDEO_ARGS,
        "-video_track_timescale", "90000",
        str(output),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not output.exists() or output.stat().st_size == 0:
        raise TuberOverlayError(
            f"Composite group (seek) fail ({output}). "
            f"stderr tail:\n{(proc.stderr or '')[-1200:]}"
        )
    return output


def _probe_duration_s(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip())
    except (ValueError, AttributeError):
        return -1.0


def validate_group_output(
    video: Path,
    expected_duration_s: float,
    *,
    min_output_bytes: int = 1024,
    duration_tolerance_s: float = 0.1,
) -> None:
    """Phase O: kiểm tra tồn tại, size, duration trong tolerance."""
    if not video.exists():
        raise TuberOverlayError(f"Group output không tồn tại: {video}")
    if video.stat().st_size <= min_output_bytes:
        raise TuberOverlayError(f"Group output quá nhỏ ({video.stat().st_size}B): {video}")
    dur = _probe_duration_s(video)
    if dur < 0:
        raise TuberOverlayError(f"Không probe được duration: {video}")
    if abs(dur - expected_duration_s) > duration_tolerance_s:
        raise TuberOverlayError(
            f"Duration lệch: {dur:.3f}s vs expected {expected_duration_s:.3f}s "
            f"(tol {duration_tolerance_s}s): {video}"
        )


# ════════════════════════════════════════════════════════════════════
# DEBUG FRAME DUMP (V3)
# ════════════════════════════════════════════════════════════════════

def _dump_debug_frames(
    group: GroupJob,
    composited_video: Path,
    overlay_dir: Path,
    output_dir: Path,
    *,
    margin: int = 3,
) -> None:
    """Dump overlay + composited frames quanh group boundary.

    Ghi vào output_dir/{group_id}/:
      - overlay_{idx:06d}.png      (frame từ overlay_frames)
      - composited_{idx:06d}.png   (frame từ video_with_tuber)
      - boundary.json               (metadata để đối chiếu)
    """
    if not composited_video.exists():
        logger.warning("Debug dump skip: composited video %s không tồn tại", composited_video)
        return

    out_d = output_dir / group.group_id
    out_d.mkdir(parents=True, exist_ok=True)
    dur_frames = group.manifest.get("renderDurationFrames", 0) or 0
    end_start = max(0, dur_frames - margin)

    # Dump overlay frames (start + end)
    _sample = list(range(margin)) + list(range(end_start, dur_frames))
    for fi in _sample:
        src = overlay_dir / f"frame_{fi:06d}.png"
        if src.exists():
            dst = out_d / f"overlay_{fi:06d}.png"
            try:
                dst.write_bytes(src.read_bytes())
            except OSError:
                pass

    # Dump composited frames từ video
    for fi in _sample:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(composited_video),
            "-vf", f"select='eq(n\\,{fi})'",
            "-vframes", "1",
            str(out_d / f"composited_{fi:06d}.png"),
        ]
        subprocess.run(cmd, capture_output=True, text=True)

    # boundary.json
    (out_d / "boundary.json").write_text(
        json.dumps({
            "groupId": group.group_id,
            "groupStartFrame": group.manifest.get("groupStartFrame"),
            "groupEndFrame": group.manifest.get("groupEndFrame"),
            "renderStartFrame": group.manifest.get("renderStartFrame"),
            "renderDurationFrames": dur_frames,
            "margin": margin,
            "fps": group.manifest.get("fps"),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Debug frames dumped → %s (%d frames)", out_d, len(_sample))


# ════════════════════════════════════════════════════════════════════
# CONCAT (Phase P)
# ════════════════════════════════════════════════════════════════════

def concat_group_videos(group_videos: List[Path], output: Path, tmp_dir: Path) -> Path:
    """Phase P: concat copy các video_with_tuber.mp4 → video_stretched_with_tuber.mp4."""
    if not group_videos:
        raise TuberOverlayError("Không có group video để concat.")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    list_file = tmp_dir / "concat_video_list.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for p in group_videos:
            f.write(f"file '{Path(p).resolve().as_posix()}'\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c:v", "copy", "-an", str(output)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not output.exists():
        raise TuberOverlayError(
            f"Concat group videos fail. stderr tail:\n{(proc.stderr or '')[-1200:]}"
        )
    return output


# ════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ════════════════════════════════════════════════════════════════════

@dataclass
class GroupJob:
    """Một group đã có manifest + dir, sẵn sàng render/composite."""

    group_id: str
    group_dir: Path
    manifest_path: Path
    manifest: Dict[str, Any]


def _expected_group_duration_s(manifest: Dict[str, Any]) -> float:
    return manifest["renderDurationFrames"] / float(manifest["fps"])


def _make_mouth_lookup(group_manifest: Dict[str, Any]):
    """Trả về hàm lookup(gf: int) -> state str cho group manifest đã cho.

    Binary-search qua mouthEvents từng segment. Dùng chung cho cả
    _build_prerender_frame_list (png_sequence) và _pipe_prerender_frames (direct).
    """
    segments = group_manifest.get("segments", [])

    def lookup(gf: int) -> str:
        best = "closed"
        for seg in segments:
            events = seg.get("mouthEvents")
            if not events:
                continue
            lo, hi = 0, len(events) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                ev = events[mid]
                if ev["frame"] <= gf:
                    best = ev["state"]
                    lo = mid + 1
                else:
                    hi = mid - 1
        return best

    return lookup


def _pipe_prerender_frames(
    stretched_video: Path,
    output: Path,
    group_manifest: Dict[str, Any],
    prerender_dir: Path,
    prerender_manifest: Dict[str, Any],
    fps_str: str,
    fps_float: float,
    *,
    offset_x: int = 0,
    offset_y: int = 0,
    log_path: Optional[Path] = None,
) -> Path:
    """Direct RGB pipe: đọc prerender PNG → raw RGBA stdin → overlay seek → output.

    Thay _build_prerender_frame_list (copy PNG) + composite_group_from_stretched
    (đọc PNG sequence) bằng 1 FFmpeg process: [0:v]=video seek, [1:v]=rawvideo
    từ stdin. Không tạo overlay_frames/ — frame chảy RAM→FFmpeg.

    stderr FFmpeg ghi ra log_path (không PIPE) để tránh deadlock buffer.
    """
    try:
        from PIL import Image as _Image
    except ImportError as exc:
        raise TuberOverlayError("Direct pipe yêu cầu Pillow (PIL). Hãy cài: pip install Pillow") from exc

    from sync_engine.tuber_prerender import compute_track_frame_index, get_prerender_frame

    render_start_frame = group_manifest.get("renderStartFrame",
                                            group_manifest["groupStartFrame"])
    render_duration_frames = group_manifest["renderDurationFrames"]
    track_fps = float(prerender_manifest.get("trackFps", 30))
    track_frames = int(prerender_manifest.get("trackFrames", 170))

    # (1) Đọc kích thước THẬT từ frame prerender đầu tiên (Q4 plan)
    probe_gf = render_start_frame
    probe_idx = compute_track_frame_index(probe_gf, fps_float, track_fps, track_frames)
    probe_src = get_prerender_frame(probe_idx, "closed", prerender_dir, prerender_manifest)
    if not probe_src.exists():
        raise TuberOverlayError(f"Direct pipe: frame prerender không có: {probe_src}")
    with _Image.open(probe_src) as _im:
        W, H = _im.size

    # (2) Hybrid seek — y hệt composite_group_from_stretched
    start_s = render_start_frame / fps_float
    rough_start_s = max(0.0, start_s - 2.0)
    exact_offset_s = start_s - rough_start_s
    safe_start_s = max(0.0, exact_offset_s - (0.5 / fps_float))

    filter_complex = (
        f"[0:v]trim=start={safe_start_s:.6f},setpts=PTS-STARTPTS,"
        f"fps={fps_str}:eof_action=pass,"
        f"trim=end_frame={render_duration_frames}[vb];"
        f"[vb][1:v]overlay=x={offset_x}:y={offset_y}:format=auto:shortest=1[outv]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{rough_start_s:.6f}", "-i", str(stretched_video),
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
        "-framerate", fps_str, "-i", "pipe:0",
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-an",
        *_HEVC_NVENC_VIDEO_ARGS,
        "-video_track_timescale", "90000",
        str(output),
    ]

    # (3) Mở stderr ra file log (không PIPE) để tránh deadlock khi buffer stderr đầy
    log_fh = None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "wb")  # noqa: SIM115

    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log_fh or subprocess.DEVNULL)

        lookup = _make_mouth_lookup(group_manifest)
        end_frame = render_start_frame + render_duration_frames
        try:
            for gf in range(render_start_frame, end_frame):
                track_idx = compute_track_frame_index(gf, fps_float, track_fps, track_frames)
                state = lookup(gf)
                src = get_prerender_frame(track_idx, state, prerender_dir, prerender_manifest)
                if not src.exists():
                    src = get_prerender_frame(track_idx, "closed", prerender_dir,
                                              prerender_manifest)
                with _Image.open(src) as im:
                    if im.mode != "RGBA":
                        im = im.convert("RGBA")
                    if im.size != (W, H):
                        raise TuberOverlayError(
                            f"Direct pipe: frame {src.name} size {im.size} != ({W},{H})"
                        )
                    proc.stdin.write(im.tobytes())  # type: ignore[union-attr]
            proc.stdin.close()  # type: ignore[union-attr]
        except BrokenPipeError:
            pass  # FFmpeg đã chết — lấy returncode bên dưới
        proc.wait()
    finally:
        if log_fh:
            log_fh.close()

    if proc.returncode != 0 or not output.exists() or output.stat().st_size == 0:
        err_tail = ""
        if log_path and log_path.exists():
            try:
                err_tail = log_path.read_bytes()[-1200:].decode("utf-8", "replace")
            except OSError:
                pass
        raise TuberOverlayError(
            f"Direct pipe composite fail ({output}). stderr tail:\n{err_tail}"
        )
    return output


def _build_prerender_frame_list(
    group_manifest: Dict[str, Any],
    prerender_dir: Path,
    prerender_manifest: Dict[str, Any],
) -> Path:
    """Tạo overlay_frames từ pre-rendered character frames cho 1 group.

    Tạo symlink (hoặc copy) các pre-rendered frame vào overlay_frames/ theo
    thứ tự timeline → overlay_frames/frame_000000.png, frame_000001.png, ...

    Đọc mouthEvents trực tiếp từ group_manifest["segments"] để tránh
    cross-group collision + đảm bảo frame index đúng phạm vi group.

    Returns:
        Path tới overlay_frames dir.
    """
    from sync_engine.tuber_prerender import (
        compute_track_frame_index,
        get_prerender_frame,
    )

    overlay_dir = Path(group_manifest["overlayDir"])
    # Xóa overlay cũ nếu có
    if overlay_dir.exists():
        import shutil
        shutil.rmtree(str(overlay_dir))
    overlay_dir.mkdir(parents=True, exist_ok=True)

    fps = float(group_manifest["fps"])
    track_fps = float(prerender_manifest.get("trackFps", 30))
    track_frames = int(prerender_manifest.get("trackFrames", 170))
    start_frame = group_manifest.get("renderStartFrame", group_manifest["groupStartFrame"])
    end_frame = start_frame + group_manifest["renderDurationFrames"]

    lookup = _make_mouth_lookup(group_manifest)

    frame_idx = 0
    for gf in range(start_frame, end_frame):
        track_idx = compute_track_frame_index(gf, fps, track_fps, track_frames)
        mouth_state = lookup(gf)
        src = get_prerender_frame(track_idx, mouth_state, prerender_dir, prerender_manifest)
        if not src.exists():
            # Fallback to closed
            src = get_prerender_frame(track_idx, "closed", prerender_dir, prerender_manifest)
        # Copy vào overlay_frames với pattern frame_000000.png
        dst = overlay_dir / f"frame_{frame_idx:06d}.png"
        if src.exists():
            dst.write_bytes(src.read_bytes())
        frame_idx += 1

    return overlay_dir


def render_and_composite_groups(
    *,
    groups: List[GroupJob],
    retry_attempts: int,
    artifact_policy: Dict[str, str],
    logs_dir: Path,
    min_output_bytes: int = 1024,
    duration_tolerance_s: float = 0.1,
    prerender_dir: Optional[Path] = None,
    prerender_manifest: Optional[Dict[str, Any]] = None,
    stretched_video: Optional[Path] = None,  # required (seek source)
    source_video: Optional[Path] = None,  # stable anchor cho inputHash (resume)
    max_workers: int = 1,
    skip_done: bool = True,
    debug_frame_enabled: bool = False,
    debug_frame_margin_frames: int = 3,
    overlay_format: str = "direct",
) -> List[Path]:
    """Render → mỗi group composite/validate/cleanup, retry group fail.

    overlay_format="direct":       pipe raw RGBA → FFmpeg stdin (production).
    overlay_format="png_sequence": copy PNG → overlay_frames (debug/fallback).
    Khi max_workers > 1: xử lý groups song song (ThreadPoolExecutor).

    Trả về list video_with_tuber.mp4 theo thứ tự group. Raise TuberOverlayError nếu
    một group hết retry vẫn fail (caller fallback render_without_tuber — Phase S).
    """
    overlay_policy = artifact_policy.get("overlayFrames", "safe")

    def _process_one_group(g: GroupJob) -> Path:
        """Worker: 1 group → render overlay → composite → validate → cleanup."""
        status = st.read_status(g.group_dir) or st.new_status(g.group_id)
        overlay_dir = Path(g.manifest["overlayDir"])
        out = Path(g.manifest["videoWithTuber"])
        expected_s = _expected_group_duration_s(g.manifest)
        fps_str = g.manifest.get("fpsStr") or str(g.manifest["fps"])
        offset_x = g.manifest.get("compOffsetX", 0)
        offset_y = g.manifest.get("compOffsetY", 0)
        fps_float = float(g.manifest["fps"])

        # Skip if done + hash khớp (resume.skipDone)
        if skip_done:
            old_st = st.read_status(g.group_dir)
            if old_st and old_st.get("status") == st.STATUS_DONE:
                old_hash = old_st.get("inputHash")
                if old_hash and str(old_hash) == compute_group_input_hash(
                    g.manifest, prerender_manifest, source_video,
                ):
                    try:
                        validate_group_output(
                            out, expected_s,
                            min_output_bytes=min_output_bytes,
                            duration_tolerance_s=duration_tolerance_s,
                        )
                        logger.info("Group %s skip (done, hash khớp).", g.group_id)
                        status["status"] = st.STATUS_SKIPPED
                        st.write_status(g.group_dir, status)
                        return out
                    except TuberOverlayError:
                        logger.info(
                            "Group %s hash khớp nhưng output fail → re-render.",
                            g.group_id,
                        )

        attempt = 0
        last_err: Optional[str] = None
        ok = False
        while attempt <= retry_attempts:
            status["status"] = st.STATUS_RUNNING
            status["attempts"] = attempt
            try:
                # direct pipe hoặc png_sequence theo overlay_format.
                # Ở attempt cuối, nếu direct đã fail → fallback png_sequence (Q5)
                is_last_attempt = (attempt == retry_attempts)
                fmt = overlay_format
                if fmt == "direct" and is_last_attempt and last_err is not None:
                    logger.warning(
                        "Group %s: direct pipe fail → fallback png_sequence (last attempt).",
                        g.group_id,
                    )
                    fmt = "png_sequence"

                status["currentStep"] = st.STEP_COMPOSITING
                st.write_status(g.group_dir, status)
                if fmt == "direct":
                    _pipe_prerender_frames(
                        stretched_video,  # type: ignore[arg-type]
                        out, g.manifest, prerender_dir, prerender_manifest,  # type: ignore[arg-type]
                        fps_str, fps_float,
                        offset_x=offset_x, offset_y=offset_y,
                        log_path=logs_dir / f"direct_pipe_{g.group_id}.log",
                    )
                else:
                    # png_sequence path (debug / fallback)
                    status["currentStep"] = st.STEP_RENDERING_OVERLAY
                    st.write_status(g.group_dir, status)
                    _build_prerender_frame_list(
                        g.manifest, prerender_dir, prerender_manifest,  # type: ignore[arg-type]
                    )
                    status["currentStep"] = st.STEP_COMPOSITING
                    st.write_status(g.group_dir, status)
                    composite_group_from_stretched(
                        stretched_video,  # type: ignore[arg-type]
                        overlay_dir, out, fps_str, fps_float,
                        render_start_frame=g.manifest["renderStartFrame"],
                        render_duration_frames=g.manifest["renderDurationFrames"],
                        offset_x=offset_x, offset_y=offset_y,
                    )

                status["currentStep"] = st.STEP_VALIDATING
                st.write_status(g.group_dir, status)
                validate_group_output(
                    out, expected_s,
                    min_output_bytes=min_output_bytes,
                    duration_tolerance_s=duration_tolerance_s,
                )
                ok = True
                break
            except Exception as exc:  # noqa: BLE001 — gom mọi lỗi để retry
                last_err = str(exc)
                status["failedStep"] = status["currentStep"]
                status["lastError"] = last_err[-1000:]
                logger.warning("Group %s fail (attempt %s): %s", g.group_id, attempt, last_err)
                attempt += 1

        if not ok:
            status["status"] = st.STATUS_FAILED
            status["fallbackTriggered"] = True
            st.write_status(g.group_dir, status)
            raise TuberOverlayError(
                f"Group {g.group_id} hết retry ({retry_attempts}) vẫn fail: {last_err}"
            )

        # done + cleanup overlay frames theo policy
        status["inputHash"] = compute_group_input_hash(
            g.manifest, prerender_manifest, source_video,
        )
        status["status"] = st.STATUS_DONE
        status["currentStep"] = st.STEP_CLEANUP
        status["failedStep"] = None
        status["lastError"] = None
        st.write_status(g.group_dir, status)

        # Debug dump: overlay + composited frames quanh boundary
        if debug_frame_enabled:
            _dump_debug_frames(
                g, out, overlay_dir,
                logs_dir / "debug_frames",
                margin=debug_frame_margin_frames,
            )

        cleanup_overlay_frames(g.group_dir, overlay_policy)
        return out

    from concurrent.futures import ThreadPoolExecutor, as_completed

    if max_workers > 1:
        results: List[Tuple[int, Path]] = []
        errors: List[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            fut_map = {pool.submit(_process_one_group, g): idx for idx, g in enumerate(groups)}
            for f in as_completed(fut_map):
                idx = fut_map[f]
                g = groups[idx]
                try:
                    out = f.result()
                    results.append((idx, out))
                except TuberOverlayError as exc:
                    errors.append(str(exc))
                except Exception as exc:
                    errors.append(f"{g.group_id}: {exc}")
        if errors:
            raise TuberOverlayError(
                f"Parallel render groups: {len(errors)}/{len(groups)} group(s) fail:\n" +
                "\n".join(errors)
            )
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    else:
        group_videos: List[Path] = []
        for g in groups:
            out = _process_one_group(g)
            group_videos.append(out)
        return group_videos


def render_groups_to_video(
    *,
    groups: List[GroupJob],
    output_video: Path,
    tmp_dir: Path,
    logs_dir: Path,
    retry_attempts: int,
    artifact_policy: Dict[str, str],
    min_output_bytes: int = 1024,
    duration_tolerance_s: float = 0.1,
    prerender_dir: Optional[Path] = None,
    prerender_manifest: Optional[Dict[str, Any]] = None,
    stretched_video: Optional[Path] = None,
    source_video: Optional[Path] = None,  # stable anchor cho inputHash (resume)
    max_workers: int = 1,
    skip_done: bool = True,
    debug_frame_enabled: bool = False,
    debug_frame_margin_frames: int = 3,
    overlay_format: str = "direct",
) -> Path:
    """High-level: render/composite prerender frames → concat → output_video.

      overlay_format="direct"       → raw RGBA pipe (production, không file trung gian).
      overlay_format="png_sequence" → ghi PNG overlay_frames (debug).
    Raise TuberOverlayError nếu fail (caller fallback).
    """
    group_videos = render_and_composite_groups(
        groups=groups,
        retry_attempts=retry_attempts,
        artifact_policy=artifact_policy,
        logs_dir=logs_dir,
        min_output_bytes=min_output_bytes,
        duration_tolerance_s=duration_tolerance_s,
        prerender_dir=prerender_dir,
        prerender_manifest=prerender_manifest,
        stretched_video=stretched_video,
        source_video=source_video,
        max_workers=max_workers,
        skip_done=skip_done,
        debug_frame_enabled=debug_frame_enabled,
        debug_frame_margin_frames=debug_frame_margin_frames,
        overlay_format=overlay_format,
    )
    concat_group_videos(group_videos, output_video, tmp_dir)
    logger.info("Tuber overlay xong → %s", output_video)
    return output_video


def prepare_groups_and_base(
    *,
    config,
    timeline: List[Any],
    fps_float: float,
    fps_str: str,
    width: int,
    height: int,
    track_aspect: Optional[float] = None,
    real_total_frames: Optional[int] = None,  # clamp group theo EOF thật
) -> List[GroupJob]:
    """Phase F + H: build groups, ghi group_manifest.json mỗi group (bỏ base.mp4)."""

    from sync_engine.tuber_manifest import (
        build_render_groups, build_group_manifest, write_group_manifest,
    )

    render_groups = build_render_groups(
        timeline, fps_float, config.max_group_sec, real_total_frames=real_total_frames,
    )
    asset_id = config.asset_id()
    character = config.character
    mouth_mode = config.mouth_mode

    # mouth opts for amplitude analysis
    mouth_opts: Optional[Dict[str, Any]] = None
    if mouth_mode != "cue":
        mouth_opts = {
            "silence_db": config.mouth_silence_db,
            "min_silence_ms": config.mouth_min_silence_ms,
            "cadence_ms": config.mouth_cadence_ms,
            "num_mouth_states": len(config.mouth_states),
            "mode": mouth_mode,
            # Tầng 2 — vowel selection (spectral centroid); chỉ kích hoạt khi
            # mouth_states có "e"/"u". 3-state cũ → analyze bỏ qua, hành vi như cũ.
            "mouth_states": config.mouth_states,
            "peak_margin": config.mouth_peak_margin,
            "min_vowel_interval_ms": config.mouth_min_vowel_interval_ms,
            "vowel_low_percentile": config.mouth_vowel_low_pct,
            "vowel_high_percentile": config.mouth_vowel_high_pct,
            # Adaptive auto-leveling (V6): chống đơ miệng khi audio nhỏ
            "adaptive": config.mouth_adaptive,
            "adaptive_floor_pct": config.mouth_adaptive_floor_pct,
            "adaptive_peak_pct": config.mouth_adaptive_peak_pct,
            "adaptive_min_range_db": config.mouth_adaptive_min_range_db,
            "adaptive_gamma": config.mouth_adaptive_gamma,
        }

    jobs: List[GroupJob] = []
    for rg in render_groups:
        rg.group_id = f"group_{rg.index + 1:04d}"
        rg.group_dir = config.groups_dir / rg.group_id
        rg.group_dir.mkdir(parents=True, exist_ok=True)

        manifest = build_group_manifest(
            rg,
            fps_float=fps_float, fps_str=fps_str, width=width, height=height,
            asset_id=asset_id, character=character, mouth_mode=mouth_mode,
            group_dir=rg.group_dir,
            mouth_opts=mouth_opts,
            track_aspect=track_aspect,
        )
        mpath = write_group_manifest(manifest, rg.group_dir)
        jobs.append(GroupJob(rg.group_id, rg.group_dir, mpath, manifest))

    logger.info("Đã build %s group (base + manifest).", len(jobs))
    return jobs


def jobs_from_run_manifest(run_manifest: Dict[str, Any]) -> List[GroupJob]:
    """Repair: dựng lại GroupJob từ run_manifest.groups (đọc group_manifest.json)."""
    jobs: List[GroupJob] = []
    for mp in run_manifest.get("groups", []):
        mpath = Path(mp)
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        jobs.append(GroupJob(
            group_id=manifest["groupId"],
            group_dir=mpath.parent,
            manifest_path=mpath,
            manifest=manifest,
        ))
    return jobs


def probe_resolution(video_path: str) -> tuple:
    """Probe (width, height) của video bằng ffprobe (B3)."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0:s=x", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        w, h = proc.stdout.strip().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        raise TuberOverlayError(f"Không probe được resolution: {video_path}")


def probe_frame_count(video_path: str, fps_float: float) -> Optional[int]:
    """Đếm SỐ FRAME THỰC của video stretched → clamp group theo EOF thật.

    Ưu tiên `nb_frames` trong container (nhanh, chính xác với mp4 từ ffmpeg).
    Nếu thiếu → fallback round(duration*fps) (sai số ≤ 1 frame, vẫn nằm trong
    tolerance). KHÔNG dùng -count_frames (decode toàn bộ → chậm với video dài).
    Trả None nếu không probe được (caller bỏ qua clamp, giữ hành vi cũ).
    """
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames",
         "-of", "default=nokey=1:noprint_wrappers=1", str(video_path)],
        capture_output=True, text=True,
    )
    val = (proc.stdout or "").strip()
    if val.isdigit() and int(val) > 0:
        return int(val)
    # Fallback: duration * fps (container thiếu nb_frames)
    dur = _probe_duration_s(Path(video_path))
    if dur > 0:
        return round(dur * fps_float)
    return None


def _load_prerender_manifest(config) -> Optional[Dict[str, Any]]:
    """Load prerender_manifest.json từ prerender character dir."""
    try:
        pdir = config.prerender_character_dir
        if pdir and (pdir / "prerender_manifest.json").exists():
            return json.loads((pdir / "prerender_manifest.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _prerender_is_stale(prerender_manifest: Dict[str, Any], config) -> bool:
    """True nếu prerender hiện có thiếu mouthStates mà config yêu cầu.

    Khi user thêm 'e'/'u' vào mouthStates nhưng prerendered/ cũ chỉ có
    closed/half/open → frame ``frame-NNN_e.png``/``_u.png`` thiếu → composite
    fallback 'closed' → nguyên âm không hiện. Trả True để caller bake lại (bổ sung
    frame nguyên âm) thay vì buộc user xoá tay bằng resume.skipDone=false.
    """
    have = set(prerender_manifest.get("mouthStates") or [])
    want = set(config.mouth_states)
    return not want.issubset(have)


def _auto_run_prerender(config, width: int, height: int) -> Dict[str, Any]:
    """Tự động chạy chromakey + prerender_character() khi chưa có manifest.

    Gọi khi overlay.mode='prerender' và prerender_manifest.json chưa có.
    Nếu body-transparent/ chưa có → tự chạy FFmpeg chromakey từ bodySource trước.
    Raise TuberOverlayError nếu thiếu asset không thể tự tạo (mouth_dir, mouth_track).
    """
    from sync_engine.tuber_prerender import (
        prerender_character, compute_character_box, extract_body_transparent,
    )

    body_dir = config.body_transparent_dir()
    mouth_dir = config.mouth_dir()
    mouth_track_path = config.mouth_track_path()
    mouth_states = config.mouth_states
    out_dir = config.prerender_character_dir

    # Auto-chromakey: nếu body-transparent/ chưa có → tạo từ bodySource
    if not body_dir.is_dir() or not any(body_dir.glob("frame-*.png")):
        from sync_engine.tuber_config import _get_nested
        body_source_rel = _get_nested(config.raw, "asset.bodySource")
        body_source = config.asset_dir() / body_source_rel if body_source_rel else None
        if not body_source or not body_source.exists():
            raise TuberOverlayError(
                f"Auto-prerender thất bại: body-transparent/ chưa có và bodySource "
                f"không tìm thấy: {body_source}."
            )
        ck = config.chromakey
        logger.info(
            "Auto body-transparent: chưa có → extract từ %s (chromakey_enabled=%s)",
            body_source.name, config.chromakey_enabled,
        )
        try:
            extract_body_transparent(
                body_source=body_source,
                out_dir=body_dir,
                chroma_color=ck.get("color") or None,
                similarity=float(ck.get("similarity", 0.10)),
                blend=float(ck.get("blend", 0.10)),
                chromakey_enabled=config.chromakey_enabled,
                despill=config.chromakey_despill,
            )
        except Exception as exc:
            raise TuberOverlayError(f"Auto-chromakey thất bại: {exc}") from exc

    if not mouth_dir.is_dir():
        raise TuberOverlayError(
            f"Auto-prerender thất bại: mouth_dir không tồn tại: {mouth_dir}."
        )
    if not mouth_track_path.exists():
        raise TuberOverlayError(
            f"Auto-prerender thất bại: mouth_track không tồn tại: {mouth_track_path}."
        )

    logger.info(
        "Auto-prerender: chưa có prerender_manifest.json → tạo mới tại %s "
        "(%d mouth_states, body=%s)",
        out_dir, len(mouth_states), body_dir,
    )

    # Tính character_box từ config để crop output
    character_box = None
    try:
        import json as _json
        track = _json.loads(mouth_track_path.read_text(encoding="utf-8"))
        tw = int(track.get("width", 1920))
        th = int(track.get("height", 1080))
        track_aspect = tw / th if th > 0 else 16.0 / 9.0
        character_box = compute_character_box(config.character, width, height, track_aspect)
    except Exception as exc:
        logger.warning("Auto-prerender: không tính được character_box (%s), dùng full size.", exc)

    manifest = prerender_character(
        body_dir=body_dir,
        mouth_dir=mouth_dir,
        mouth_track_path=mouth_track_path,
        mouth_states=mouth_states,
        out_dir=out_dir,
        character_box=character_box,
        max_workers=config.max_workers,
    )
    logger.info("Auto-prerender hoàn tất: %d frames xuất ra %s", manifest.get("outputCount", 0), out_dir)
    return manifest


def run_tuber_flow_all_in(
    *,
    config,
    video_path: str,
    timeline: List[Any],
    fps_float: float,
    fps_str: str,
    base_video_stretched: str,
    mixed_audio: str,
    render_config: Dict[str, Any],
    final_render_args: Dict[str, Any],
    subtitle_synced_srt: Optional[str],
    note_overlay_final_ass: Optional[str],
    image_overlay_events: Optional[Any],
    tmp_dir: str,
) -> Path:
    """Phase D-P all-in: trả về path video_stretched_with_tuber.mp4.

    Pre-render character frames (bake nếu thiếu/stale) rồi composite lên video
    stretched. Raise TuberOverlayError nếu fail (caller fallback render_without_tuber).
    `config` đã resolve_layout() + make_dirs().
    """
    from sync_engine.tuber_manifest import (
        build_run_manifest, write_run_manifest,
    )
    from sync_engine.tuber_artifacts import (
        promote_media, promote_final_render_inputs,
        BASE_VIDEO_NAME, FINAL_AUDIO_NAME, VIDEO_WITH_TUBER_NAME,
    )

    # B3: resolution lấy từ base video stretched thật (không hardcode 1080p)
    width, height = probe_resolution(base_video_stretched)

    # Đo SỐ FRAME THỰC của video stretched → clamp group theo EOF thật (tránh
    # group cuối hụt vài frame so với timeline lý thuyết → fail tolerance).
    real_total_frames = probe_frame_count(base_video_stretched, fps_float)

    # Prerender là path duy nhất: load manifest, bake nếu thiếu/stale (vd thiếu
    # mouthStates mới e/u).
    prerender_manifest = _load_prerender_manifest(config)
    if prerender_manifest is None or _prerender_is_stale(prerender_manifest, config):
        prerender_manifest = _auto_run_prerender(config, width, height)

    # Wipe caches nếu resume.skipDone=false (debug/re-render) → bake lại
    if not config.resume_skip_done:
        for d in (config.groups_dir, config.prerender_character_dir):
            if d and Path(d).is_dir():
                import shutil
                shutil.rmtree(d, ignore_errors=True)
                logger.info("skipDone=false → xóa %s", d)
        prerender_manifest = _load_prerender_manifest(config)
        if prerender_manifest is None:
            prerender_manifest = _auto_run_prerender(config, width, height)

    if prerender_manifest is None:
        raise TuberOverlayError(
            "Không tạo được prerender_manifest cho tuber overlay (bake thất bại)."
        )
    prerender_dir = config.prerender_character_dir

    # Track aspect (từ prerender manifest hoặc mặc định 16:9)
    track_aspect = None
    if prerender_manifest:
        tw = float(prerender_manifest.get("trackWidth", 1920))
        th = float(prerender_manifest.get("trackHeight", 1080))
        track_aspect = tw / th if th > 0 else 16.0 / 9.0

    # Phase E: promote media + final_render_inputs (repairable)
    promote_media(
        base_video_src=base_video_stretched,
        final_audio_src=mixed_audio,
        media_dir=config.media_dir,
    )
    ap = config.artifact_policy()
    if ap.get("finalRenderInputs", "keep") != "delete":
        promote_final_render_inputs(
            final_render_inputs_dir=config.final_render_inputs_dir,
            subtitle_synced_srt=subtitle_synced_srt,
            note_overlay_final_ass=note_overlay_final_ass,
            image_overlay_events=image_overlay_events,
            render_config=render_config,
            final_render_args=final_render_args,
        )

    # Phase F + H: groups + manifest
    jobs = prepare_groups_and_base(
        config=config, timeline=timeline,
        fps_float=fps_float, fps_str=fps_str, width=width, height=height,
        track_aspect=track_aspect,
        real_total_frames=real_total_frames,
    )

    # run_manifest.json
    asset = {
        "assetDir": str(config.asset_dir().resolve()),
        "assetId": config.asset_id(),
    }
    video_with_tuber = config.media_dir / VIDEO_WITH_TUBER_NAME
    run_manifest = build_run_manifest(
        job_name=config.job_name, fps_float=fps_float, fps_str=fps_str,
        width=width, height=height,
        tuber_root=config.tuber_root, media_dir=config.media_dir,
        groups_dir=config.groups_dir,
        base_video=config.media_dir / BASE_VIDEO_NAME,
        final_audio=config.media_dir / FINAL_AUDIO_NAME,
        video_with_tuber=video_with_tuber,
        overlay_format=config.overlay_format,
        asset=asset,
        group_manifest_paths=[j.manifest_path for j in jobs],
        artifact_policy=ap, tuber_config_raw=config.raw,
        prerender_manifest_path=prerender_dir / "prerender_manifest.json" if prerender_dir else None,
        source_video=Path(video_path),
    )
    write_run_manifest(run_manifest, config.tuber_root)

    # Phase I-P: render + composite + concat
    return render_groups_to_video(
        groups=jobs,
        output_video=video_with_tuber,
        tmp_dir=Path(tmp_dir),
        logs_dir=config.logs_dir,
        retry_attempts=config.retry_attempts,
        artifact_policy=ap,
        min_output_bytes=int((render_config.get("tuber_validation", {}) or {}).get("minOutputBytes", 1024)),
        duration_tolerance_s=float((render_config.get("tuber_validation", {}) or {}).get("durationToleranceSec", 0.1)),
        prerender_dir=prerender_dir,
        prerender_manifest=prerender_manifest,
        stretched_video=config.media_dir / BASE_VIDEO_NAME,
        source_video=Path(video_path),
        max_workers=config.max_workers,
        skip_done=config.resume_skip_done,
        debug_frame_enabled=config.debug_frame_output_enabled,
        debug_frame_margin_frames=config.debug_frame_margin,
        overlay_format=config.overlay_format,
    )
