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
) -> Optional[dict[str, Any]]:
    """Entry point chính cho bước forced alignment subtitle.

    Được gọi từ `cli/sync_video.py` sau khi assemble audio hoàn tất.
    Tự kiểm tra `enabled` và xử lý `fail_policy`.

    Args:
        audio_path: Đường dẫn file mixed audio.
        transcript_path: Đường dẫn file flat transcript.
        output_srt_path: Đường dẫn file SRT output.
        render_config: Toàn bộ render config dict.

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

    try:
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
