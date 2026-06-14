# -*- coding: utf-8 -*-
"""
translation/srt_translator.py
=============================
Dịch SRT theo lô bằng LLM. Wrapper mỏng quanh hạ tầng dùng chung
`llm_ai/srt_batch/` — chỉ đặc tả phần riêng của dịch thuật:
  - response_tag = "TRANSLATE_TEXT"
  - không có validator (dịch được phép đổi chữ)
  - sidecar `.txt` (segments_to_txt) cạnh SRT output.
"""

import time
import logging
from pathlib import Path

from utils.srt_parser import segments_to_txt
from llm_ai.base import BaseLLMProvider
from llm_ai.srt_batch import (
    BatchIntegrityError,
    CacheTelemetryAccumulator,
    run_srt_batch_task,
)

logger = logging.getLogger("srt_translator")

# Giữ tên cũ như alias để không break code cũ mà không import provider Gemini tại import-time.
def GeminiCaller(*args, **kwargs):
    from llm_ai.providers.gemini import GeminiProvider

    return GeminiProvider(*args, **kwargs)


def parse_gemini_response(result: str) -> str:
    from translation.response_parser import parse_translation_response

    return parse_translation_response(result)


def translate_srt_file(
    input_file: str,
    output_file: str,
    prompt_file: str,
    provider: BaseLLMProvider,
    target_language: str = "Vietnamese",
    batch_size: int = 30,
    use_full_context: bool = True,
    wait_sec: float = 0.0,
    max_workers: int = 1,
) -> dict:
    print(f"\n🤖 Provider: {provider.name}")
    print(f"🌐 Ngôn ngữ: {target_language}")
    print(f"📦 Batch  : {batch_size} block/lần | Workers: {max_workers}\n")

    t_start = time.time()
    translated_srt, stats = run_srt_batch_task(
        input_srt=input_file,
        output_srt=output_file,
        prompt_file=prompt_file,
        provider=provider,
        language=target_language,
        response_tag="TRANSLATE_TEXT",
        batch_size=batch_size,
        use_full_context=use_full_context,
        wait_sec=wait_sec,
        validator=None,
        write_srt=True,
        max_workers=max_workers,
    )
    elapsed_total = time.time() - t_start

    # Sidecar text: nối text các block thành .txt cạnh SRT output.
    txt_output = str(Path(output_file).with_suffix(".txt"))
    txt_content = segments_to_txt(translated_srt)
    Path(txt_output).write_text(txt_content, encoding="utf-8")
    logger.info(f"📄 Đã lưu text: {txt_output}")

    stats["elapsed"] = round(elapsed_total, 1)

    print(f"\n{'─'*50}")
    print(f"✅ Hoàn thành: {output_file}")
    print(f"   Batch: {stats['success']}/{stats['total_batches']} thành công | {stats['failed']} lỗi")
    print(f"   Thời gian: {elapsed_total:.1f}s")

    return stats


__all__ = [
    "translate_srt_file",
    "GeminiCaller",
    "parse_gemini_response",
    "BatchIntegrityError",
    "CacheTelemetryAccumulator",
]
