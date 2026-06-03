"""
sync_engine/tuber_artifacts.py
==============================
Promote artifact cần thiết (Phase E) + cleanup theo artifactPolicy (Phase T).

Dùng chung sync_video (promote khi chạy all-in) và tuber_repair (đọc lại).
Không di chuyển toàn bộ tmp — chỉ copy/export artifact tuber thật sự cần.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("sync_video")

# Tên file chuẩn trong tuberRoot
BASE_VIDEO_NAME = "base_video_stretched.mp4"
FINAL_AUDIO_NAME = "final_audio_mixed.wav"
VIDEO_WITH_TUBER_NAME = "video_stretched_with_tuber.mp4"

# final_render_inputs filenames
FRI_SUBTITLE = "subtitle_synced.srt"
FRI_NOTE_ASS = "note_overlay_final.ass"
FRI_IMAGE_EVENTS = "image_overlay_events.json"
FRI_RENDER_CONFIG = "render_config.json"
FRI_MANIFEST = "final_render_manifest.json"


def promote_media(
    *, base_video_src: str, final_audio_src: str, media_dir: Path,
) -> Dict[str, Path]:
    """Phase E: copy base video + mixed audio vào tuberRoot/media."""
    media_dir.mkdir(parents=True, exist_ok=True)
    base_dst = media_dir / BASE_VIDEO_NAME
    audio_dst = media_dir / FINAL_AUDIO_NAME
    shutil.copy2(base_video_src, base_dst)
    shutil.copy2(final_audio_src, audio_dst)
    logger.info("Promote media: %s, %s", base_dst, audio_dst)
    return {"base_video": base_dst, "final_audio": audio_dst}


def serialize_image_overlay_events(events: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
    """Serialize Sequence[ImageOverlayEvent] (frozen dataclass) → list dict."""
    out: List[Dict[str, Any]] = []
    for ev in (events or []):
        if dataclasses.is_dataclass(ev):
            out.append(dataclasses.asdict(ev))
        elif isinstance(ev, dict):
            out.append(dict(ev))
    return out


def deserialize_image_overlay_events(data: List[Dict[str, Any]]) -> List[Any]:
    """Dựng lại list ImageOverlayEvent từ JSON (cho repair)."""
    from sync_engine.image_overlay import ImageOverlayEvent

    events = []
    for d in (data or []):
        events.append(ImageOverlayEvent(
            key=d["key"],
            image_path=d["image_path"],
            start_time=d["start_time"],
            end_time=d["end_time"],
            source_line=d.get("source_line", 0),
        ))
    return events


def promote_final_render_inputs(
    *,
    final_render_inputs_dir: Path,
    subtitle_synced_srt: Optional[str],
    note_overlay_final_ass: Optional[str],
    image_overlay_events: Optional[Sequence[Any]],
    render_config: Dict[str, Any],
    final_render_args: Dict[str, Any],
) -> Path:
    """Phase E (repairable): export đủ input để render_final_video muộn.

    Trả về path final_render_manifest.json. Input không tồn tại → key null.
    """
    fri = final_render_inputs_dir
    fri.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = dict(final_render_args)

    if subtitle_synced_srt and Path(subtitle_synced_srt).exists():
        dst = fri / FRI_SUBTITLE
        shutil.copy2(subtitle_synced_srt, dst)
        manifest["subtitle_synced_srt"] = str(dst.resolve())
    else:
        manifest["subtitle_synced_srt"] = None

    if note_overlay_final_ass and Path(note_overlay_final_ass).exists():
        dst = fri / FRI_NOTE_ASS
        shutil.copy2(note_overlay_final_ass, dst)
        manifest["note_overlay_synced_ass"] = str(dst.resolve())
    else:
        manifest["note_overlay_synced_ass"] = None

    # image overlay events → JSON (serialize in-memory dataclass)
    events_data = serialize_image_overlay_events(image_overlay_events)
    events_path = fri / FRI_IMAGE_EVENTS
    events_path.write_text(json.dumps(events_data, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["image_overlay_events"] = str(events_path.resolve()) if events_data else None

    # render_config snapshot
    rc_path = fri / FRI_RENDER_CONFIG
    rc_path.write_text(json.dumps(render_config, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["render_config"] = str(rc_path.resolve())

    manifest_path = fri / FRI_MANIFEST
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Promote final_render_inputs → %s", manifest_path)
    return manifest_path


def load_final_render_manifest(final_render_inputs_dir: Path) -> Dict[str, Any]:
    path = Path(final_render_inputs_dir) / FRI_MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy final_render_manifest.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def cleanup_overlay_frames(group_dir: Path, policy: str) -> None:
    """Phase T: xoá overlay_frames theo policy (safe/delete = xoá; keep = giữ)."""
    if policy not in ("safe", "delete"):
        return
    overlay_dir = Path(group_dir) / "overlay_frames"
    if overlay_dir.is_dir():
        shutil.rmtree(overlay_dir, ignore_errors=True)
        logger.info("Cleanup overlay_frames (%s): %s", policy, overlay_dir)


def cleanup_failed_group(group_dir: Path, policy: str) -> None:
    """Phase T: xoá temp group fail nếu failedGroups=delete."""
    if policy != "delete":
        return
    gd = Path(group_dir)
    if gd.is_dir():
        shutil.rmtree(gd, ignore_errors=True)
        logger.info("Cleanup failed group: %s", gd)
