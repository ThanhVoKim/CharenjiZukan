#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sync_engine/forced_alignment_subtitle.py
========================================
Orchestration forced alignment subtitle cho `cli/sync_video.py`.

Module này nằm trong `sync_engine/` vì nó phục vụ flow sync-video,
phụ thuộc vào `qwen_asr` (model) và `utils/asr_subtitle_utils.py` (helper chung),
nhưng không phụ thuộc ngược vào `cli/`.

Schema `render_config.json` — xem `docs/sync-video-guide.md`.
"""

import sys
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger
from utils.asr_subtitle_utils import (
    merge_punctuation,
    segment_words_to_subtitles,
    write_subtitle_srt,
)

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
# Model loading
# ═════════════════════════════════════════════════════════════════════

def load_forced_aligner(
    model_path: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    dtype_name: Optional[str] = None,
    device_map: str = "cuda:0",
    attn_implementation: Optional[str] = None,
):
    """Load Qwen3ForcedAligner model.

    Default tương đương:
        Qwen3ForcedAligner.from_pretrained(
            "Qwen/Qwen3-ForcedAligner-0.6B",
            dtype=torch.bfloat16,
            device_map="cuda:0",
        )

    Args:
        model_path: Đường dẫn model HuggingFace hoặc local.
        dtype_name: Tên dtype ("bfloat16", "float16", "float32").
                    None → torch.bfloat16.
        device_map: Device map cho model (mặc định "cuda:0").
        attn_implementation: Nếu None, không truyền tham số này.

    Returns:
        Qwen3ForcedAligner model instance.
    """
    try:
        import torch
        from qwen_asr import Qwen3ForcedAligner
    except ImportError:
        raise ImportError(
            "Thư viện 'qwen-asr' chưa được cài đặt. "
            "Vui lòng cài đặt Optional Dependency: pip install .[qwen-asr]"
        )

    # Resolve dtype
    if dtype_name is None:
        dtype = torch.bfloat16
    else:
        _DTYPE_MAP = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        dtype = _DTYPE_MAP.get(dtype_name, torch.bfloat16)

    # Build kwargs
    kwargs = {
        "dtype": dtype,
        "device_map": device_map,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    logger.info(f"Đang load Qwen3ForcedAligner: model={model_path}, dtype={dtype}, device={device_map}")
    model = Qwen3ForcedAligner.from_pretrained(model_path, **kwargs)
    return model


# ═════════════════════════════════════════════════════════════════════
# Core execution
# ═════════════════════════════════════════════════════════════════════

def execute_forced_alignment(
    *,
    audio_path: str,
    transcript_path: str,
    output_srt_path: str,
    align_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Thực thi forced alignment và ghi SRT.

    Args:
        audio_path: Đường dẫn file audio (mixed_audio.wav).
        transcript_path: Đường dẫn file text (flat_transcript.txt).
        output_srt_path: Đường dẫn file SRT output.
        align_cfg: Dict config đã resolve từ render_config.

    Returns:
        Dict stats gồm 'subtitle_blocks', 'total_words'.
    """
    # Đọc transcript
    transcript_file = Path(transcript_path)
    if not transcript_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file transcript: {transcript_path}")
    full_text = transcript_file.read_text(encoding="utf-8").strip()
    if not full_text:
        raise ValueError("File transcript rỗng, không thể align.")

    # Load model
    model_path = align_cfg.get("model_path") or "Qwen/Qwen3-ForcedAligner-0.6B"
    device = align_cfg.get("device") or "cuda:0"
    dtype_name = align_cfg.get("dtype")  # None → bfloat16
    attn_impl = align_cfg.get("attn_implementation")  # None → không truyền

    aligner = load_forced_aligner(
        model_path=model_path,
        dtype_name=dtype_name,
        device_map=device,
        attn_implementation=attn_impl,
    )

    language = align_cfg.get("language", "English")

    # Chạy alignment
    logger.info(f"Đang chạy forced alignment: audio={audio_path}, language={language}")
    results = aligner.align(
        audio=audio_path,
        text=full_text,
        language=language,
    )

    # Giải phóng model
    del aligner
    try:
        from utils.media_utils import clear_vram
        clear_vram()
        logger.info("Đã giải phóng VRAM sau forced alignment.")
    except Exception:
        pass

    # Xử lý kết quả
    if not results or not results[0]:
        raise ValueError("Forced alignment không trả về kết quả.")

    align_items = results[0]  # ForcedAlignResult items

    # Merge punctuation
    merged_words = merge_punctuation(align_items, full_text)
    logger.info(f"Forced alignment trả về {len(merged_words)} word items sau merge punctuation.")

    # Segment thành subtitle blocks
    max_chars = align_cfg.get("max_chars", 42)
    min_chars = align_cfg.get("min_chars", 0)
    split_on_comma = align_cfg.get("split_on_comma", True)

    subtitle_blocks = segment_words_to_subtitles(
        merged_words,
        max_chars=max_chars,
        min_chars=min_chars,
        split_on_comma=split_on_comma,
    )
    logger.info(f"Đã segment thành {len(subtitle_blocks)} subtitle blocks.")

    # Ghi SRT
    offset_seconds = align_cfg.get("offset_seconds", 0.24)
    write_subtitle_srt(subtitle_blocks, output_srt_path, offset_seconds=offset_seconds)
    logger.info(f"Đã ghi forced alignment SRT: {output_srt_path}")

    return {
        "subtitle_blocks": len(subtitle_blocks),
        "total_words": len(merged_words),
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
