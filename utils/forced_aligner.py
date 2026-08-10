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

import os
import sys
from pathlib import Path
from typing import Any, Optional

# expandable_segments giảm phân mảnh VRAM khi align audio dài; phải set TRƯỚC khi
# torch khởi tạo CUDA allocator (load_forced_aligner import torch lazily bên dưới).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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


def _merged_text(merged_words: list[dict[str, Any]]) -> str:
    """Ghép text reconstructed để kiểm tra invariant với transcript nguồn."""
    return "".join(str(word.get("text", "")) for word in merged_words).strip()


def _text_preview(text: str, limit: int = 120) -> str:
    """Preview ngắn, một dòng, dùng trong log lỗi integrity."""
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


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

    # Chạy alignment — bọc try/finally để LUÔN giải phóng VRAM kể cả khi align()
    # ném OOM. Nếu không, model + activations kẹt lại trong GPU và làm hỏng các
    # bước GPU phía sau (vd FFmpeg hevc_nvenc ở final render).
    logger.info(f"Đang chạy forced alignment: audio={audio_path}, language={language}")
    try:
        results = aligner.align(
            audio=audio_path,
            text=full_text,
            language=language,
        )
    finally:
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

    # Text của forced alignment chỉ được dùng để gắn timestamp; transcript gốc
    # là SSOT cho nội dung. Nếu reconstruction không còn khớp nguyên văn, fail
    # toàn nhánh mixed để caller fallback sang remap thay vì ghi SRT bị sai chữ.
    reconstructed_text = _merged_text(merged_words)
    if reconstructed_text != full_text:
        raise ValueError(
            "Forced alignment làm thay đổi transcript sau reconstruction: "
            f"expected={_text_preview(full_text)!r}, "
            f"actual={_text_preview(reconstructed_text)!r}"
        )

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
# Per-clip execution (cho flow sync-video: align từng clip TTS dubb-N.wav)
# ═════════════════════════════════════════════════════════════════════

def _chunked(seq: list, size: int):
    """Chia list thành các batch kích thước `size` (size <= 0 → 1 batch)."""
    if size <= 0:
        yield seq
        return
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def execute_forced_alignment_clips(
    *,
    clips: list[dict[str, Any]],
    align_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[int]]:
    """Forced alignment theo TỪNG clip TTS thay vì cả mixed audio.

    Mỗi clip ngắn (vài giây) nên không bao giờ chạm giới hạn ~5 phút của
    Qwen3-ForcedAligner → không OOM, chạy được video dài bất kỳ. Word timing
    của từng clip được offset về timeline cuối qua `offset_ms` + `audio_speed`.

    Args:
        clips: list dict, mỗi clip gồm:
            - audio_path (str): đường dẫn dubb-N.wav,
            - text (str): text của dòng phụ đề tương ứng,
            - offset_ms (float): vị trí đầu clip trên timeline cuối (= seg.new_start),
            - audio_speed (float): hệ số atempo đã áp lên clip (voicevox = 1.0),
            - line (int): id dòng gốc (để báo cáo clip nào fail → caller remap).
        align_cfg: dict config (model/device/segmentation/language) đã resolve.

    Returns:
        (aligned_segments, failed_lines):
            - aligned_segments: list segment dict {line, start_time(ms), end_time(ms), text},
              line tạm thời, caller sẽ đánh số lại sau khi gộp + sort.
            - failed_lines: list `line` của các clip align rỗng → caller chuyển sang remap.
    """
    if not clips:
        return [], []

    # Load model
    model_path = align_cfg.get("model_path") or "Qwen/Qwen3-ForcedAligner-0.6B"
    device = align_cfg.get("device") or "cuda:0"
    dtype_name = align_cfg.get("dtype")
    attn_impl = align_cfg.get("attn_implementation")
    language = align_cfg.get("language", "English")
    batch_size = int(align_cfg.get("batch_size", 16) or 16)

    max_chars = align_cfg.get("max_chars", 42)
    min_chars = align_cfg.get("min_chars", 0)
    split_on_comma = align_cfg.get("split_on_comma", True)
    offset_seconds = align_cfg.get("offset_seconds", 0.24)

    aligner = load_forced_aligner(
        model_path=model_path,
        dtype_name=dtype_name,
        device_map=device,
        attn_implementation=attn_impl,
    )

    aligned_segments: list[dict[str, Any]] = []
    failed_lines: list[int] = []
    seq = 0

    def mark_failed(line: int) -> None:
        if line not in failed_lines:
            failed_lines.append(line)

    logger.info(
        f"Đang chạy forced alignment per-clip: {len(clips)} clip, "
        f"batch_size={batch_size}, language={language}"
    )
    try:
        for batch in _chunked(clips, batch_size):
            audios = [c["audio_path"] for c in batch]
            texts = [c["text"] for c in batch]
            results = aligner.align(audio=audios, text=texts, language=language)

            batch_results = list(results or [])
            if len(batch_results) != len(batch):
                logger.warning(
                    "Forced aligner trả %d result cho batch %d clip; "
                    "clip thiếu result sẽ fallback sang remap.",
                    len(batch_results), len(batch),
                )

            # Không dùng zip(): nếu model trả thiếu result, zip sẽ làm clip cuối
            # biến mất im lặng và caller không biết để remap dòng gốc.
            for result_idx, clip in enumerate(batch):
                res = batch_results[result_idx] if result_idx < len(batch_results) else None
                if not res:
                    mark_failed(clip["line"])
                    continue

                spd = clip.get("audio_speed") or 1.0
                if spd <= 0:
                    spd = 1.0
                off_ms = clip.get("offset_ms", 0.0)

                # Đưa word timing (giây, trong clip) về ms tuyệt đối trên timeline cuối.
                words = []
                for w in res:
                    if isinstance(w, dict):
                        w_text = w.get("text", "")
                        w_s = w.get("start_time", 0.0)
                        w_e = w.get("end_time", 0.0)
                    else:
                        w_text = w.text
                        w_s = w.start_time
                        w_e = w.end_time
                    words.append({
                        "text": w_text,
                        "start_time": off_ms + (w_s * 1000.0) / spd,
                        "end_time": off_ms + (w_e * 1000.0) / spd,
                    })

                merged = merge_punctuation(words, clip["text"])
                source_text = str(clip["text"]).strip()
                reconstructed_text = _merged_text(merged)
                if reconstructed_text != source_text:
                    logger.warning(
                        "Forced alignment line=%s làm thay đổi nội dung; "
                        "fallback remap. expected=%r, actual=%r",
                        clip["line"],
                        _text_preview(source_text),
                        _text_preview(reconstructed_text),
                    )
                    mark_failed(clip["line"])
                    continue

                blocks = segment_words_to_subtitles(
                    merged,
                    max_chars=max_chars,
                    min_chars=min_chars,
                    split_on_comma=split_on_comma,
                )

                emitted_for_clip = 0
                for block in blocks:
                    if not block:
                        continue
                    start_ms = block[0].get("start_time", 0.0) + offset_seconds * 1000.0
                    end_ms = block[-1].get("end_time", 0.0) + offset_seconds * 1000.0
                    start_ms = max(0.0, start_ms)
                    if end_ms <= start_ms:
                        end_ms = start_ms + 100
                    text = "".join(w.get("text", "") for w in block).strip()
                    if not text:
                        continue
                    seq += 1
                    emitted_for_clip += 1
                    aligned_segments.append({
                        "line": seq,
                        "start_time": int(round(start_ms)),
                        "end_time": int(round(end_ms)),
                        "text": text,
                    })

                if emitted_for_clip == 0:
                    logger.warning(
                        "Forced alignment line=%s không tạo block text; fallback remap.",
                        clip["line"],
                    )
                    mark_failed(clip["line"])
    finally:
        del aligner
        try:
            from utils.media_utils import clear_vram
            clear_vram()
            logger.info("Đã giải phóng VRAM sau forced alignment per-clip.")
        except Exception:
            pass

    logger.info(
        f"Forced alignment per-clip xong: {len(aligned_segments)} block, "
        f"{len(failed_lines)} clip fail (sẽ remap)."
    )
    return aligned_segments, failed_lines
