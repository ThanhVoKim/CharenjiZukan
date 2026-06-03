"""
sync_engine/tuber_manifest.py
=============================
Build render groups từ timeline + export/load run_manifest.json &
group_manifest.json (Phase F + H).

Cốt lõi frame math (đơn vị nguồn = ms, frame theo fps_float):
  - segment.startFrame/endFrame là GLOBAL frame trên video stretched (mouthState.ts
    + MotionPngTuberCharacter dùng globalFrame). groupStartFrame = frame tích luỹ.
  - Số frame mỗi segment = expected_output_frames — CÙNG công thức với
    video_processor.build_ffmpeg_batch_cmd (ceil(round(stretched_dur_s*fps,4)))
    để group base (dựng bằng build_ffmpeg_batch_cmd) khớp frame với manifest.

V1 (chốt): prePaddingFrames=postPaddingFrames=0, renderStartFrame=groupStartFrame,
renderDurationFrames=groupEndFrame-groupStartFrame (M2); speechControlAudio bỏ (M1);
hasTts = block_type=="tts" and tts_clip_path (M4); blockType ∈ tts/mute/gap/tail.

Mọi runtime path trong manifest là ABSOLUTE (Phase H rule).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from sync_engine.models import TimelineSegment

logger = logging.getLogger("sync_video")

SCHEMA_VERSION = 1


def segment_output_frames(seg: TimelineSegment, fps_float: float) -> int:
    """Số frame của 1 segment sau stretch — khớp build_ffmpeg_batch_cmd."""
    duration_frames = round(((seg.orig_end - seg.orig_start) / 1000.0) * fps_float)
    stretched_duration_s = (duration_frames / fps_float) / seg.video_speed
    return math.ceil(round(stretched_duration_s * fps_float, 4))


@dataclass
class RenderGroup:
    """Một group = list segment liên tiếp + frame bounds global."""

    index: int
    segments: List[TimelineSegment]
    group_start_frame: int
    group_end_frame: int
    # đặt khi export
    group_id: str = ""
    group_dir: Optional[Path] = None

    @property
    def duration_frames(self) -> int:
        return self.group_end_frame - self.group_start_frame


def build_render_groups(
    timeline: List[TimelineSegment],
    fps_float: float,
    max_group_sec: float,
) -> List[RenderGroup]:
    """Phase F: gom timeline thành group theo duration (không cắt giữa segment).

    Quy tắc: thêm segment vào group hiện tại; nếu vượt max_group_sec thì đóng
    group cũ và mở group mới. Một segment đơn lẻ dài hơn max vẫn nằm trọn 1 group.
    """
    if not timeline:
        raise ValueError("Timeline rỗng: không thể build render groups.")

    max_frames = max_group_sec * fps_float
    groups: List[RenderGroup] = []

    cur_segs: List[TimelineSegment] = []
    cur_frames = 0
    global_frame = 0
    group_start = 0

    def _flush():
        nonlocal cur_segs, cur_frames, group_start
        if cur_segs:
            groups.append(
                RenderGroup(
                    index=len(groups),
                    segments=cur_segs,
                    group_start_frame=group_start,
                    group_end_frame=group_start + cur_frames,
                )
            )
        cur_segs = []
        cur_frames = 0

    for seg in timeline:
        seg_frames = segment_output_frames(seg, fps_float)
        if cur_segs and (cur_frames + seg_frames) > max_frames:
            _flush()
            group_start = global_frame
        cur_segs.append(seg)
        cur_frames += seg_frames
        global_frame += seg_frames

    _flush()
    return groups


def _seg_has_tts(seg: TimelineSegment) -> bool:
    return seg.block_type == "tts" and bool(seg.tts_clip_path)


def _abs(p) -> str:
    return str(Path(p).resolve())


def build_group_manifest(
    group: RenderGroup,
    *,
    fps_float: float,
    fps_str: str,
    width: int,
    height: int,
    asset_id: str,
    character: Dict[str, Any],
    mouth_mode: str,
    group_dir: Path,
) -> Dict[str, Any]:
    """Phase H: dựng dict group_manifest.json cho 1 group (absolute paths)."""
    seg_dicts: List[Dict[str, Any]] = []
    frame_cursor = group.group_start_frame
    for i, seg in enumerate(group.segments):
        n = segment_output_frames(seg, fps_float)
        start_f = frame_cursor
        end_f = frame_cursor + n
        seg_dicts.append({
            "segmentIndex": i,
            "newStartMs": round(seg.new_start, 3),
            "newEndMs": round(seg.new_end, 3),
            "startFrame": start_f,
            "endFrame": end_f,
            "blockType": seg.block_type,
            "hasTts": _seg_has_tts(seg),
        })
        frame_cursor = end_f

    overlay_dir = group_dir / "overlay_frames"
    return {
        "schemaVersion": SCHEMA_VERSION,
        "groupId": group.group_id,
        "groupIndex": group.index,
        "fps": fps_float,
        "fpsStr": fps_str,
        "width": width,
        "height": height,
        "groupStartFrame": group.group_start_frame,
        "groupEndFrame": group.group_end_frame,
        "renderStartFrame": group.group_start_frame,            # V1: padding 0 (M2)
        "renderDurationFrames": group.duration_frames,
        "prePaddingFrames": 0,
        "postPaddingFrames": 0,
        "assetId": asset_id,
        "base": _abs(group_dir / "base.mp4"),
        "overlayDir": _abs(overlay_dir),
        "videoWithTuber": _abs(group_dir / "video_with_tuber.mp4"),
        "segments": seg_dicts,
        "character": dict(character),
        "mouth": {"mode": mouth_mode},
    }


def write_group_manifest(manifest: Dict[str, Any], group_dir: Path) -> Path:
    group_dir.mkdir(parents=True, exist_ok=True)
    path = group_dir / "group_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_run_manifest(
    *,
    job_name: str,
    fps_float: float,
    fps_str: str,
    width: int,
    height: int,
    tuber_root: Path,
    media_dir: Path,
    groups_dir: Path,
    base_video: Path,
    final_audio: Path,
    video_with_tuber: Path,
    overlay_format: str,
    remotion: Dict[str, Any],
    asset: Dict[str, Any],
    group_manifest_paths: List[Path],
    artifact_policy: Dict[str, str],
    tuber_config_raw: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase H: dựng run_manifest.json (absolute paths)."""
    return {
        "schemaVersion": SCHEMA_VERSION,
        "jobName": job_name,
        "fps": fps_float,
        "fpsStr": fps_str,
        "width": width,
        "height": height,
        "tuberRoot": _abs(tuber_root),
        "mediaDir": _abs(media_dir),
        "groupsDir": _abs(groups_dir),
        "baseVideo": _abs(base_video),
        "finalAudio": _abs(final_audio),
        "videoWithTuber": _abs(video_with_tuber),
        "overlayFormat": overlay_format,
        "remotion": {
            "projectDir": _abs(remotion["projectDir"]),
            "compositionId": remotion["compositionId"],
            "entryPoint": remotion["entryPoint"],
            "renderDriver": remotion.get("renderDriver", "scripts/render-groups.ts"),
        },
        "asset": asset,
        "artifactPolicy": artifact_policy,
        "tuberConfig": tuber_config_raw,
        "groups": [_abs(p) for p in group_manifest_paths],
    }


def write_run_manifest(manifest: Dict[str, Any], tuber_root: Path) -> Path:
    tuber_root.mkdir(parents=True, exist_ok=True)
    path = tuber_root / "run_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_run_manifest(tuber_root: Path) -> Dict[str, Any]:
    path = Path(tuber_root) / "run_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy run_manifest.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
