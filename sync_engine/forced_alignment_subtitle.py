#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sync_engine/forced_alignment_subtitle.py
========================================
Glue forced alignment subtitle cho `cli/sync_video.py`.

Lõi align (load model + execute) đã được tách ra `utils/forced_aligner.py`
(trung lập, dùng chung với flow OCR `cli/align_srt.py`). Module này chỉ giữ
phần phụ thuộc `render_config.json`:
- `_resolve_aligner_config()` — map render_config sang tham số hàm,
- `run_forced_alignment_subtitle()` — entry point, tự kiểm tra `enabled` và `fail_policy`.

`load_forced_aligner` và `execute_forced_alignment` được re-export ở đây để
giữ tương thích import cũ.

Schema `render_config.json` — xem `docs/sync-video-guide.md`.
"""

import sys
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger

# Lõi align trung lập — re-export để giữ tương thích import cũ.
from utils.forced_aligner import (  # noqa: F401
    load_forced_aligner,
    execute_forced_alignment,
    execute_forced_alignment_clips,
)
# segment_words_to_subtitles được re-export để mock patch trỏ tới module này vẫn chạy.
from utils.asr_subtitle_utils import segment_words_to_subtitles  # noqa: F401

logger = get_logger("sync_video")


# ═════════════════════════════════════════════════════════════════════
# Config resolution
# ═════════════════════════════════════════════════════════════════════

def _resolve_aligner_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map config từ render_config.json sang tham số hàm.

    Các key null trong JSON sẽ không override default của hàm.
    """
    return {
        "model_path": cfg.get("model_path"),          # None → dùng default hàm
        "device": cfg.get("device"),                   # None → dùng default hàm
        "dtype": cfg.get("dtype"),                     # None → dùng default hàm
        "attn_implementation": cfg.get("attn_implementation"),  # None → không truyền
        "language": cfg.get("language", "English"),
        "max_chars": cfg.get("max_chars", 42),
        "min_chars": cfg.get("min_chars", 0),
        "split_on_comma": cfg.get("split_on_comma", True),
        "offset_seconds": cfg.get("offset_seconds", 0.24),
        "keep_tts_synced_debug": cfg.get("keep_tts_synced_debug", False),
        "fail_policy": cfg.get("fail_policy", "warn"),
        "batch_size": cfg.get("batch_size", 16),
    }


# ═════════════════════════════════════════════════════════════════════
# Per-clip alignment (flow sync-video)
# ═════════════════════════════════════════════════════════════════════

def _build_clips_from_timeline(
    timeline: list,
    subtitle_segments: list[dict],
    mute_segments: list[dict],
) -> tuple[list[dict[str, Any]], list[dict]]:
    """Dựng danh sách clip TTS để align + danh sách dòng cần remap.

    Trả về (clips, remap_lines):
      - clips: mỗi clip {audio_path, text, offset_ms, audio_speed, line} cho 1 dòng
        TTS có sẵn `dubb-N.wav`. `line` = index trong tts_only (để map ngược khi fail).
      - remap_lines: các dòng KHÔNG align được (nằm trong vùng mute, hoặc thiếu clip)
        → caller remap timeline rồi gộp, đảm bảo không sót phụ đề nào.
    """
    from sync_engine.analyzer import filter_tts_subtitles

    tts_only = filter_tts_subtitles(subtitle_segments, mute_segments)

    # Dòng nằm trong vùng mute (bị filter drop) → nhận diện qua start_time, vì
    # filter_tts giữ nguyên start_time và thứ tự, chỉ đánh số lại `line`.
    tts_starts = {int(round(s["start_time"])) for s in tts_only}
    remap_lines = [
        s for s in subtitle_segments
        if int(round(s["start_time"])) not in tts_starts
    ]

    # Các TimelineSegment loại "tts" theo đúng thứ tự ↔ tts_only.
    tts_segs = [seg for seg in timeline if seg.block_type == "tts"]

    clips: list[dict[str, Any]] = []
    if len(tts_segs) != len(tts_only):
        logger.warning(
            "Số TTS timeline segment (%d) khác số dòng TTS (%d); "
            "match theo start_time để an toàn.",
            len(tts_segs), len(tts_only),
        )
        seg_by_start = {int(round(seg.orig_start)): seg for seg in tts_segs}
        pairs = [
            (line, seg_by_start.get(int(round(line["start_time"]))))
            for line in tts_only
        ]
    else:
        pairs = list(zip(tts_only, tts_segs))

    for idx, (line, seg) in enumerate(pairs):
        if seg is None or not seg.tts_clip_path:
            remap_lines.append(line)
            continue
        clips.append({
            "audio_path": seg.tts_clip_path,
            "text": line["text"],
            "offset_ms": seg.new_start,
            "audio_speed": seg.audio_speed,
            "line": idx,
        })

    return clips, remap_lines


def _run_forced_alignment_clips(
    *,
    output_srt_path: str,
    align_cfg: dict[str, Any],
    timeline: list,
    subtitle_segments: list[dict],
    mute_segments: list[dict],
    fps_float: float,
    remap_max_chars: int,
) -> Optional[dict[str, Any]]:
    """Align từng clip TTS, remap các dòng còn lại, gộp & ghi SRT hoàn chỉnh."""
    from sync_engine.analyzer import filter_tts_subtitles
    from sync_engine.timestamp_remapper import recalculate_segments
    from utils.srt_parser import segments_to_srt

    clips, remap_lines = _build_clips_from_timeline(
        timeline, subtitle_segments, mute_segments,
    )

    aligned_segments: list[dict] = []
    failed_lines: list[int] = []
    if clips:
        aligned_segments, failed_lines = execute_forced_alignment_clips(
            clips=clips,
            align_cfg=align_cfg,
        )

    # Clip align fail → đưa dòng gốc về nhóm remap (không drop).
    if failed_lines:
        tts_only = filter_tts_subtitles(subtitle_segments, mute_segments)
        remap_lines = remap_lines + [
            tts_only[k] for k in failed_lines if 0 <= k < len(tts_only)
        ]

    if not aligned_segments and not remap_lines:
        raise ValueError("Forced alignment per-clip không tạo được dòng phụ đề nào.")

    remap_segments = recalculate_segments(
        remap_lines, timeline,
        is_tts_track=False,
        max_chars=remap_max_chars,
        fps_float=fps_float,
    ) if remap_lines else []

    # Gộp + sort theo thời gian + đánh số lại line.
    final = aligned_segments + remap_segments
    final.sort(key=lambda s: (s["start_time"], s["end_time"]))
    for i, seg in enumerate(final, 1):
        seg["line"] = i

    from pathlib import Path as _Path
    _Path(output_srt_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(segments_to_srt(final))
    logger.info(
        "Đã ghi forced alignment SRT (per-clip): %s — %d dòng aligned, %d dòng remap.",
        output_srt_path, len(aligned_segments), len(remap_segments),
    )

    return {
        "subtitle_blocks": len(final),
        "aligned_blocks": len(aligned_segments),
        "remapped_blocks": len(remap_segments),
    }


# ═════════════════════════════════════════════════════════════════════
# Entry point (được gọi từ cli/sync_video.py)
# ═════════════════════════════════════════════════════════════════════

def run_forced_alignment_subtitle(
    *,
    audio_path: str,
    transcript_path: str,
    output_srt_path: str,
    render_config: dict[str, Any],
    timeline: Optional[list] = None,
    subtitle_segments: Optional[list[dict]] = None,
    mute_segments: Optional[list[dict]] = None,
    fps_float: float = 30.0,
    remap_max_chars: int = 0,
) -> Optional[dict[str, Any]]:
    """Entry point chính cho bước forced alignment subtitle.

    Được gọi từ `cli/sync_video.py` sau khi assemble audio hoàn tất.
    Tự kiểm tra `enabled` và xử lý `fail_policy`.

    Hai chế độ:
      - **Per-clip** (ưu tiên, khi có `timeline` + `subtitle_segments`): align từng
        clip TTS `dubb-N.wav` → không OOM trên video dài; dòng vùng mute / clip thiếu
        được remap timeline rồi gộp, không sót dòng nào.
      - **Mixed audio** (fallback, khi không có timeline — vd test cũ): align cả
        `audio_path` theo `transcript_path` như trước.

    Args:
        audio_path: Đường dẫn file mixed audio (chỉ dùng ở chế độ mixed).
        transcript_path: Đường dẫn file flat transcript (chỉ dùng ở chế độ mixed).
        output_srt_path: Đường dẫn file SRT output.
        render_config: Toàn bộ render config dict.
        timeline: List TimelineSegment (bật chế độ per-clip nếu có).
        subtitle_segments: Toàn bộ dòng phụ đề gốc (per-clip).
        mute_segments: Dòng mute mask (per-clip).
        fps_float: FPS để snap timestamp khi remap.
        remap_max_chars: max_chars khi wrap text cho dòng remap.

    Returns:
        Dict stats nếu thành công, None nếu bị skip hoặc lỗi với fail_policy=warn.
    """
    fa_cfg_raw = render_config.get("forced_alignment_subtitle", {}) or {}
    if not fa_cfg_raw.get("enabled", False):
        logger.info("Forced alignment subtitle không được bật trong render_config. Bỏ qua.")
        return None

    logger.info("\n--- PHASE 3.5: FORCED ALIGNMENT SUBTITLE ---")
    align_cfg = _resolve_aligner_config(fa_cfg_raw)
    fail_policy = str(align_cfg.get("fail_policy", "warn")).lower().strip()

    use_per_clip = timeline is not None and subtitle_segments is not None

    try:
        if use_per_clip:
            return _run_forced_alignment_clips(
                output_srt_path=output_srt_path,
                align_cfg=align_cfg,
                timeline=timeline,
                subtitle_segments=subtitle_segments,
                mute_segments=mute_segments or [],
                fps_float=fps_float,
                remap_max_chars=remap_max_chars,
            )
        return execute_forced_alignment(
            audio_path=audio_path,
            transcript_path=transcript_path,
            output_srt_path=output_srt_path,
            align_cfg=align_cfg,
        )
    except Exception as exc:
        if fail_policy in {"raise", "error", "fail"}:
            raise
        logger.warning(
            f"Forced alignment thất bại ({exc}). Fallback sang remap SRT do fail_policy={fail_policy}.",
            exc_info=True,
        )
        return None
