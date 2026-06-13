#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/forced_aligner.py
=======================
Lõi forced alignment TRUNG LẬP — dùng chung cho cả flow `sync-video`
và flow OCR (`align-srt`).

Module này KHÔNG biết gì về `render_config.json`, timeline, hay `cli/`.
Nó chỉ nhận:
  - audio_path: file audio (wav) chứa giọng nói cần align,
  - transcript_path: file text PHẲNG 1 dòng (đã có dấu câu),
  - output_srt_path: nơi ghi SRT kết quả,
  - align_cfg: dict tham số (model/device/segmentation),
rồi trả về SRT đã align (text giữ nguyên, timestamp lấy từ aligner).

Glue đọc `render_config.json` nằm ở `sync_engine/forced_alignment_subtitle.py`;
glue đọc YAML của flow OCR nằm ở `cli/align_srt.py`.

Phụ thuộc: `qwen_asr` (model) + `utils/asr_subtitle_utils.py` (helper chung).
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

logger = get_logger("forced_aligner")


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
        audio_path: Đường dẫn file audio (vocals/mixed_audio.wav).
        transcript_path: Đường dẫn file text PHẲNG 1 dòng (đã có dấu).
        output_srt_path: Đường dẫn file SRT output.
        align_cfg: Dict config (model/device/segmentation) đã resolve.

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
