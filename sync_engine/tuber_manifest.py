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


def compute_character_box(
    character: Dict[str, Any],
    width: int,
    height: int,
    track_aspect: float,
) -> Dict[str, int]:
    """Tính character box — mirror TS resolveCharacterBox width-priority.

    Returns {compWidth, compHeight, compOffsetX, compOffsetY}.
    """
    w_src: Optional[int] = character.get("width")
    if w_src is None and "widthRatio" in character:
        w_src = round(character["widthRatio"] * width)

    h_src: Optional[int] = character.get("height")
    if h_src is None and "heightRatio" in character:
        h_src = round(character["heightRatio"] * height)

    if w_src is not None:
        box_w = w_src
        box_h = round(w_src / track_aspect)
    elif h_src is not None:
        box_h = h_src
        box_w = round(h_src * track_aspect)
    else:
        box_h = round(0.6 * height)
        box_w = round(box_h * track_aspect)

    box_l = character.get("left")
    if box_l is None:
        box_l = round(character.get("leftRatio", 0.6) * width)

    box_t = character.get("top")
    if box_t is None:
        box_t = round(character.get("topRatio", 0.3) * height)

    return {
        "compWidth": box_w,
        "compHeight": box_h,
        "compOffsetX": int(box_l),
        "compOffsetY": int(box_t),
    }


def _build_segment_mouth_events(
    seg: TimelineSegment,
    fps_float: float,
    tts_clip_dir: Optional[Path] = None,
    mouth_opts: Optional[Dict[str, Any]] = None,
    global_start_frame: int = 0,
) -> Optional[List[Dict[str, Any]]]:
    """Build mouthEvents cho 1 segment (V2 mouth mode amplitude).

    Args:
        global_start_frame: Frame offset toàn cục trong group (từ
            seg_dict["startFrame"]). Đảm bảo event frame khớp với
            global frame mà _lookup_state() so sánh.
    """
    if not _seg_has_tts(seg):
        return None
    tts_path = None
    if seg.tts_clip_path:
        p = Path(seg.tts_clip_path)
        if not p.is_absolute() and tts_clip_dir:
            p = tts_clip_dir / p
        if p.exists():
            tts_path = p
    if not tts_path:
        return None
    try:
        from sync_engine.tuber_mouth_events import build_mouth_events_for_segment
        n = segment_output_frames(seg, fps_float)
        opts = dict(mouth_opts or {})
        return build_mouth_events_for_segment(
            global_start_frame, n, True, tts_path, fps_float, **opts,
        )
    except Exception as exc:
        logger.warning("Không build được mouthEvents cho segment: %s", exc)
        return None


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
    tts_clip_dir: Optional[Path] = None,
    mouth_opts: Optional[Dict[str, Any]] = None,
    track_aspect: Optional[float] = None,
) -> Dict[str, Any]:
    """Phase H: dựng dict group_manifest.json cho 1 group (absolute paths).

    Nếu mouth_mode != 'cue' và có mouth_opts, tự build mouthEvents khi export.
    """
    seg_dicts: List[Dict[str, Any]] = []
    frame_cursor = group.group_start_frame
    for i, seg in enumerate(group.segments):
        n = segment_output_frames(seg, fps_float)
        start_f = frame_cursor
        end_f = frame_cursor + n
        seg_dict: Dict[str, Any] = {
            "segmentIndex": i,
            "newStartMs": round(seg.new_start, 3),
            "newEndMs": round(seg.new_end, 3),
            "startFrame": start_f,
            "endFrame": end_f,
            "blockType": seg.block_type,
            "hasTts": _seg_has_tts(seg),
        }
        # TTS clip path cho Python-side mouthEvents build
        if seg.tts_clip_path:
            seg_dict["tts_clip_path"] = str(Path(seg.tts_clip_path).resolve())
        seg_dicts.append(seg_dict)
        frame_cursor = end_f

    # V2: build mouthEvents nếu mouth_mode != 'cue'
    if mouth_mode != "cue" and mouth_opts:
        mouth_events_map: Dict[str, Any] = {}
        for sd in seg_dicts:
            ev = _build_segment_mouth_events(
                group.segments[sd["segmentIndex"]],
                fps_float,
                tts_clip_dir=tts_clip_dir,
                mouth_opts=mouth_opts,
                global_start_frame=sd["startFrame"],
            )
            mouth_events_map[str(sd["segmentIndex"])] = ev
        # Nhúng vào segment dict
        for sd in seg_dicts:
            sd["mouthEvents"] = mouth_events_map.get(str(sd["segmentIndex"]))

    overlay_dir = group_dir / "overlay_frames"

    manifest: Dict[str, Any] = {
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

    # V2: nếu có track_aspect → thêm character box (compWidth/compHeight/compOffset)
    if track_aspect and track_aspect > 0:
        box = compute_character_box(character, width, height, track_aspect)
        manifest["compWidth"] = box["compWidth"]
        manifest["compHeight"] = box["compHeight"]
        manifest["compOffsetX"] = box["compOffsetX"]
        manifest["compOffsetY"] = box["compOffsetY"]

    return manifest
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
    prerender_manifest_path: Optional[Path] = None,
    source_video: Optional[Path] = None,
) -> Dict[str, Any]:
    """Phase H: dựng run_manifest.json (absolute paths).

    Args:
        prerender_manifest_path: Path đến prerender_manifest.json (V2).
    """
    manifest = {
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
    if prerender_manifest_path:
        manifest["prerenderManifest"] = _abs(prerender_manifest_path)
    if source_video is not None:
        manifest["sourceVideo"] = _abs(source_video)
    return manifest


def write_run_manifest(manifest: Dict[str, Any], tuber_root: Path) -> Path:
    tuber_root.mkdir(parents=True, exist_ok=True)
    path = tuber_root / "run_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_group_manifest(manifest: Dict[str, Any], group_dir: Path) -> Path:
    group_dir.mkdir(parents=True, exist_ok=True)
    path = group_dir / "group_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_run_manifest(tuber_root: Path) -> Dict[str, Any]:
    path = Path(tuber_root) / "run_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy run_manifest.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
