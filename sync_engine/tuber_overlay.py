"""
sync_engine/tuber_overlay.py
============================
Orchestration tuber overlay (Phase F base build, I render, N composite, O
validate, P concat, R retry, T cleanup). Python là orchestrator; Node/Remotion
chỉ render overlay PNG alpha.

Dùng chung sync_video (all-in) và tuber_repair (late repair).

Flow một job (bundle Remotion MỘT lần):
  prepare assets (chromakey → public) once
  build group base.mp4 (build_ffmpeg_batch_cmd) cho mỗi group
  render overlay tất cả group (1 lần bundle) → mỗi group overlay_frames/
  với mỗi group: composite overlay lên base → validate → cleanup overlay_frames
  retry group fail tới retryAttempts; hết → fallback (render_without_tuber)
  concat group video_with_tuber.mp4 → media/video_stretched_with_tuber.mp4
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from sync_engine.tuber_artifacts import cleanup_overlay_frames
from sync_engine import tuber_status as st
from sync_engine.tuber_manifest import compute_character_box
from utils.ffmpeg_probe import HEVC_NVENC_VIDEO_ARGS as _HEVC_NVENC_VIDEO_ARGS

logger = logging.getLogger("sync_video")

RENDER_RESULT_PREFIX = "__TUBER_RENDER_RESULT__="


class TuberOverlayError(RuntimeError):
    """Lỗi không thể phục hồi của tuber flow → caller fallback render_without_tuber."""


# ════════════════════════════════════════════════════════════════════
# NODE RENDER DRIVER
# ════════════════════════════════════════════════════════════════════

def _run_render_driver(
    project_dir: Path,
    manifest_paths: List[Path],
    *,
    log_path: Optional[Path] = None,
    timeout: int = 7200,
) -> Dict[str, Dict[str, Any]]:
    """Gọi scripts/render-groups.ts (bundle once, render các manifest).

    Trả về dict groupId -> result. Parse dòng __TUBER_RENDER_RESULT__.
    """
    npm = _which_npm()
    if not npm:
        raise TuberOverlayError("npm không có trong PATH — không thể gọi Remotion render driver.")

    parts = [npm, "run", "render-groups", "--", *[str(p) for p in manifest_paths]]
    logger.info("Gọi render driver: %s manifest(s)", len(manifest_paths))

    if os.name == "nt":
        cmd: Any = subprocess.list2cmdline(parts)
        shell = True
    else:
        cmd = parts
        shell = False

    proc = subprocess.run(
        cmd, cwd=str(project_dir), shell=shell,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout,
    )

    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                (proc.stdout or "") + "\n=== STDERR ===\n" + (proc.stderr or ""),
                encoding="utf-8",
            )
        except OSError:
            pass

    results: Dict[str, Dict[str, Any]] = {}
    for line in (proc.stdout or "").splitlines():
        if line.startswith(RENDER_RESULT_PREFIX):
            try:
                r = json.loads(line[len(RENDER_RESULT_PREFIX):])
                results[str(r.get("groupId"))] = r
            except json.JSONDecodeError:
                continue

    if proc.returncode != 0 and not results:
        raise TuberOverlayError(
            f"Render driver fail (exit {proc.returncode}). stderr tail:\n{(proc.stderr or '')[-1500:]}"
        )
    return results


def _which_npm() -> Optional[str]:
    import shutil
    return shutil.which("npm")


def prepare_assets(
    project_dir: Path,
    *,
    asset_id: str,
    asset_dir: Path,
    chromakey: Dict[str, Any],
    log_path: Optional[Path] = None,
    timeout: int = 1200,
) -> None:
    """Phase B1/B2: chạy npm run prepare-assets (chromakey + copy vào public)."""
    npm = _which_npm()
    if not npm:
        raise TuberOverlayError("npm không có trong PATH — không thể prepare assets.")

    args = ["--asset-id", asset_id, "--asset-dir", str(asset_dir)]
    if chromakey.get("color"):
        args += ["--color", str(chromakey["color"])]
    if chromakey.get("similarity") is not None:
        args += ["--similarity", str(chromakey["similarity"])]
    if chromakey.get("blend") is not None:
        args += ["--blend", str(chromakey["blend"])]

    parts = [npm, "run", "prepare-assets", "--", *args]
    if os.name == "nt":
        cmd: Any = subprocess.list2cmdline(parts)
        shell = True
    else:
        cmd = parts
        shell = False

    proc = subprocess.run(
        cmd, cwd=str(project_dir), shell=shell,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout,
    )
    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                (proc.stdout or "") + "\n=== STDERR ===\n" + (proc.stderr or ""),
                encoding="utf-8",
            )
        except OSError:
            pass
    if proc.returncode != 0:
        raise TuberOverlayError(
            f"prepare-assets fail (exit {proc.returncode}). stderr tail:\n{(proc.stderr or '')[-1500:]}"
        )
    logger.info("prepare-assets xong cho assetId=%s", asset_id)


# ════════════════════════════════════════════════════════════════════
# GROUP BASE BUILD (Phase F — tái dùng build_ffmpeg_batch_cmd)
# ════════════════════════════════════════════════════════════════════

def build_group_base(
    video_path: str,
    group_segments: List[Any],
    base_out: Path,
    fps_str: str,
    fps_float: float,
) -> Path:
    """Dựng group_xxxx/base.mp4 trực tiếp từ video gốc bằng cơ chế batch (B5)."""
    from sync_engine.video_processor import build_ffmpeg_batch_cmd

    base_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_ffmpeg_batch_cmd(
        video_path, str(base_out), group_segments, fps_str, fps_float,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not base_out.exists() or base_out.stat().st_size == 0:
        raise TuberOverlayError(
            f"Build group base fail ({base_out}). stderr tail:\n{(proc.stderr or '')[-1200:]}"
        )
    return base_out


# ════════════════════════════════════════════════════════════════════
# COMPOSITE (Phase N) + VALIDATE (Phase O)
# ════════════════════════════════════════════════════════════════════

def _detect_frame_pattern(overlay_dir: Path) -> str:
    """Dò pattern frame_%0Nd.png từ file thật (Remotion zero-pad theo số frame)."""
    frames = sorted(overlay_dir.glob("frame_*.png"))
    if not frames:
        raise TuberOverlayError(f"Không có overlay frame trong {overlay_dir}")
    # width = số chữ số của phần numeric (frame_00.png → 2)
    stem = frames[0].stem  # frame_00
    num_part = stem.split("_", 1)[1]
    width = len(num_part)
    return f"frame_%0{width}d.png"


def composite_group(
    base_video: Path,
    overlay_dir: Path,
    output: Path,
    fps_str: str,
    *,
    offset_x: int = 0,
    offset_y: int = 0,
) -> Path:
    """Phase N: composite overlay PNG alpha lên group base → video_with_tuber.mp4.

    Args:
        offset_x, offset_y: vị trí overlay trên base (V2: character box offset).
    """
    pattern = _detect_frame_pattern(overlay_dir)
    overlay_filter = (
        f"[0:v][1:v]overlay=x={offset_x}:y={offset_y}:format=auto:shortest=1[outv]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(base_video),
        "-framerate", fps_str, "-start_number", "0",
        "-i", str(overlay_dir / pattern),
        "-filter_complex", overlay_filter,
        "-map", "[outv]",
        "-an",
        *_HEVC_NVENC_VIDEO_ARGS,
        "-video_track_timescale", "90000",
        str(output),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not output.exists() or output.stat().st_size == 0:
        raise TuberOverlayError(
            f"Composite group fail ({output}). stderr tail:\n{(proc.stderr or '')[-1200:]}"
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


def _build_prerender_frame_list(
    group_manifest: Dict[str, Any],
    prerender_dir: Path,
    prerender_manifest: Dict[str, Any],
    mouth_events_map: Dict[str, List[Dict[str, Any]]],
) -> Path:
    """Tạo overlay_frames từ pre-rendered character frames cho 1 group.

    Tạo symlink (hoặc copy) các pre-rendered frame vào overlay_frames/ theo
    thứ tự timeline → overlay_frames/frame_000000.png, frame_000001.png, ...

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

    # Pre-compute per-segment mouth events lookup
    def _lookup_state(gf: int) -> str:
        best = "closed"
        for seg_idx, events in mouth_events_map.items():
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

    frame_idx = 0
    for gf in range(start_frame, end_frame):
        track_idx = compute_track_frame_index(gf, fps, track_fps, track_frames)
        mouth_state = _lookup_state(gf)
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
    project_dir: Path,
    groups: List[GroupJob],
    retry_attempts: int,
    artifact_policy: Dict[str, str],
    logs_dir: Path,
    min_output_bytes: int = 1024,
    duration_tolerance_s: float = 0.1,
    use_prerender: bool = False,
    prerender_dir: Optional[Path] = None,
    prerender_manifest: Optional[Dict[str, Any]] = None,
    mouth_events_map: Optional[Dict[str, Any]] = None,
) -> List[Path]:
    """Render → mỗi group composite/validate/cleanup, retry group fail.

    V1 (use_prerender=False): bundle Remotion once, render overlay, composite.
    V2 (use_prerender=True):  copy pre-rendered frames vào overlay_frames/, composite.

    Trả về list video_with_tuber.mp4 theo thứ tự group. Raise TuberOverlayError nếu
    một group hết retry vẫn fail (caller fallback render_without_tuber — Phase S).
    """
    # Init status
    for g in groups:
        st.write_status(g.group_dir, st.new_status(g.group_id))

    overlay_policy = artifact_policy.get("overlayFrames", "safe")
    group_videos: List[Path] = []

    for g in groups:
        status = st.read_status(g.group_dir) or st.new_status(g.group_id)
        base = Path(g.manifest["base"])
        overlay_dir = Path(g.manifest["overlayDir"])
        out = Path(g.manifest["videoWithTuber"])
        expected_s = _expected_group_duration_s(g.manifest)
        fps_str = g.manifest.get("fpsStr") or str(g.manifest["fps"])
        offset_x = g.manifest.get("compOffsetX", 0)
        offset_y = g.manifest.get("compOffsetY", 0)

        attempt = 0
        last_err: Optional[str] = None
        ok = False
        while attempt <= retry_attempts:
            status["status"] = st.STATUS_RUNNING
            status["attempts"] = attempt
            try:
                if use_prerender:
                    # V2: tạo overlay_frames từ pre-rendered frames
                    status["currentStep"] = st.STEP_RENDERING_OVERLAY
                    st.write_status(g.group_dir, status)
                    _build_prerender_frame_list(
                        g.manifest, prerender_dir, prerender_manifest, mouth_events_map,
                    )
                else:
                    # V1: Remotion render driver
                    # Lần đầu dùng batch results; retry thì gọi lại driver riêng
                    if attempt == 0:
                        # Mọi group composite độc lập → không cần batch results
                        pass
                    status["currentStep"] = st.STEP_RENDERING_OVERLAY
                    st.write_status(g.group_dir, status)
                    r = _run_render_driver(
                        project_dir, [g.manifest_path],
                        log_path=logs_dir / f"render_driver_{g.group_id}.log",
                    )
                    if not r.get(str(g.group_id), {}).get("ok"):
                        raise TuberOverlayError(
                            f"Render group {g.group_id} không ok: {r.get(str(g.group_id))}"
                        )

                status["currentStep"] = st.STEP_COMPOSITING
                st.write_status(g.group_dir, status)
                composite_group(
                    base, overlay_dir, out, fps_str,
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
        status["status"] = st.STATUS_DONE
        status["currentStep"] = st.STEP_CLEANUP
        status["failedStep"] = None
        status["lastError"] = None
        st.write_status(g.group_dir, status)
        cleanup_overlay_frames(g.group_dir, overlay_policy)
        group_videos.append(out)

    return group_videos


def render_groups_to_video(
    *,
    project_dir: Path,
    asset_id: str,
    asset_dir: Path,
    chromakey: Dict[str, Any],
    groups: List[GroupJob],
    output_video: Path,
    tmp_dir: Path,
    logs_dir: Path,
    retry_attempts: int,
    artifact_policy: Dict[str, str],
    min_output_bytes: int = 1024,
    duration_tolerance_s: float = 0.1,
    do_prepare_assets: bool = True,
    use_prerender: bool = False,
    prerender_dir: Optional[Path] = None,
    prerender_manifest: Optional[Dict[str, Any]] = None,
    mouth_events_map: Optional[Dict[str, Any]] = None,
) -> Path:
    """High-level: prepare assets → render/composite groups → concat → output_video.

    V1: Remotion render + composite (use_prerender=False).
    V2: Pre-rendered frames + composite (use_prerender=True, bỏ Node/Chromium).
    Raise TuberOverlayError nếu fail (caller fallback).
    """
    if not use_prerender and do_prepare_assets:
        prepare_assets(
            project_dir,
            asset_id=asset_id, asset_dir=asset_dir, chromakey=chromakey,
            log_path=logs_dir / "prepare_assets.log",
        )

    group_videos = render_and_composite_groups(
        project_dir=project_dir,
        groups=groups,
        retry_attempts=retry_attempts,
        artifact_policy=artifact_policy,
        logs_dir=logs_dir,
        min_output_bytes=min_output_bytes,
        duration_tolerance_s=duration_tolerance_s,
        use_prerender=use_prerender,
        prerender_dir=prerender_dir,
        prerender_manifest=prerender_manifest,
        mouth_events_map=mouth_events_map,
    )
    concat_group_videos(group_videos, output_video, tmp_dir)
    logger.info("Tuber overlay xong → %s", output_video)
    return output_video


def prepare_groups_and_base(
    *,
    config,
    video_path: str,
    timeline: List[Any],
    fps_float: float,
    fps_str: str,
    width: int,
    height: int,
    track_aspect: Optional[float] = None,
) -> List[GroupJob]:
    """Phase F + H: build groups, dựng base.mp4, ghi group_manifest.json mỗi group.

    `config` là TuberConfig đã resolve_layout(). Trả về list GroupJob.

    V2: nếu có mouth_mode amplitude → tự build mouthEvents trong manifest.
    V2: nếu có track_aspect → thêm compWidth/compHeight/compOffset vào manifest.
    """
    from sync_engine.tuber_manifest import (
        build_render_groups, build_group_manifest, write_group_manifest,
    )

    render_groups = build_render_groups(timeline, fps_float, config.max_group_sec)
    asset_id = config.asset_id()
    character = config.character
    mouth_mode = config.mouth_mode

    # V2: mouth opts for amplitude analysis
    mouth_opts: Optional[Dict[str, Any]] = None
    if mouth_mode != "cue":
        mouth_opts = {
            "silence_db": config.mouth_silence_db,
            "min_silence_ms": config.mouth_min_silence_ms,
            "cadence_ms": config.mouth_cadence_ms,
            "num_mouth_states": len(config.mouth_states),
        }

    jobs: List[GroupJob] = []
    for rg in render_groups:
        rg.group_id = f"group_{rg.index + 1:04d}"
        rg.group_dir = config.groups_dir / rg.group_id
        rg.group_dir.mkdir(parents=True, exist_ok=True)

        # base.mp4 (B5)
        build_group_base(
            video_path, rg.segments, rg.group_dir / "base.mp4", fps_str, fps_float,
        )

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


def _load_prerender_manifest(config) -> Optional[Dict[str, Any]]:
    """Load prerender_manifest.json từ prerender character dir."""
    try:
        pdir = config.prerender_character_dir
        if pdir and (pdir / "prerender_manifest.json").exists():
            return json.loads((pdir / "prerender_manifest.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


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
            "Auto-chromakey: body-transparent/ chưa có → extract từ %s", body_source.name
        )
        try:
            extract_body_transparent(
                body_source=body_source,
                out_dir=body_dir,
                chroma_color=ck.get("color") or None,
                similarity=float(ck.get("similarity", 0.10)),
                blend=float(ck.get("blend", 0.10)),
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

    V1: Remotion render (gọi npm prepare-assets + render-groups).
    V2: Pre-render (nếu config.asset.prerender có valid prerender_manifest.json).
    Raise TuberOverlayError nếu fail (caller fallback render_without_tuber).
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

    # Determine overlay mode (explicit config vs auto-detect)
    overlay_mode = config.overlay_mode  # "remotion" | "prerender" | "auto"
    prerender_manifest = None
    use_prerender = False

    if overlay_mode == "prerender":
        prerender_manifest = _load_prerender_manifest(config)
        if prerender_manifest is None:
            prerender_manifest = _auto_run_prerender(config, width, height)
        use_prerender = True
    elif overlay_mode == "remotion":
        pass  # use_prerender = False, không load prerender
    else:  # "auto"
        prerender_manifest = _load_prerender_manifest(config)
        use_prerender = prerender_manifest is not None

    prerender_dir = config.prerender_character_dir if use_prerender else None

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

    # Phase F + H: groups + base + manifest
    jobs = prepare_groups_and_base(
        config=config, video_path=video_path, timeline=timeline,
        fps_float=fps_float, fps_str=fps_str, width=width, height=height,
        track_aspect=track_aspect,
    )

    # run_manifest.json
    asset = {
        "assetDir": str(config.asset_dir().resolve()),
        "assetId": config.asset_id(),
    }
    remotion = {
        "projectDir": str(config.remotion_project_dir().resolve()),
        "compositionId": config.raw.get("remotion", {}).get("compositionId", "TuberOverlay"),
        "entryPoint": config.raw.get("remotion", {}).get("entryPoint", "src/index.ts"),
        "renderDriver": config.raw.get("remotion", {}).get("renderDriver", "scripts/render-groups.ts"),
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
        remotion=remotion, asset=asset,
        group_manifest_paths=[j.manifest_path for j in jobs],
        artifact_policy=ap, tuber_config_raw=config.raw,
        prerender_manifest_path=prerender_dir / "prerender_manifest.json" if prerender_dir else None,
    )
    write_run_manifest(run_manifest, config.tuber_root)

    # V2: build mouth events map for prerender
    mouth_events_map: Dict[str, Any] = {}
    if use_prerender and config.mouth_mode != "cue":
        for j in jobs:
            for seg in j.manifest.get("segments", []):
                ev = seg.get("mouthEvents")
                if ev:
                    mouth_events_map[str(seg.get("segmentIndex"))] = ev

    # Phase I-P: render + composite + concat
    return render_groups_to_video(
        project_dir=config.remotion_project_dir(),
        asset_id=config.asset_id(),
        asset_dir=config.asset_dir(),
        chromakey=config.chromakey,
        groups=jobs,
        output_video=video_with_tuber,
        tmp_dir=Path(tmp_dir),
        logs_dir=config.logs_dir,
        retry_attempts=config.retry_attempts,
        artifact_policy=ap,
        min_output_bytes=int((render_config.get("tuber_validation", {}) or {}).get("minOutputBytes", 1024)),
        duration_tolerance_s=float((render_config.get("tuber_validation", {}) or {}).get("durationToleranceSec", 0.1)),
        use_prerender=use_prerender,
        prerender_dir=prerender_dir,
        prerender_manifest=prerender_manifest,
        mouth_events_map=mouth_events_map if mouth_events_map else None,
    )
