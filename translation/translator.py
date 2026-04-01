# -*- coding: utf-8 -*-
import copy
import time
import logging
from pathlib import Path
from typing import List

from utils.srt_parser import parse_srt, segments_to_srt, wrap_subtitle_text
from translation.base import BaseTranslationProvider
from translation.response_parser import parse_translation_response
from translation.gemini_provider import GeminiProvider

logger = logging.getLogger("srt_translator")

# Giữ tên cũ như alias để không break code cũ
GeminiCaller = GeminiProvider
parse_gemini_response = parse_translation_response

_GLOBAL_CONTEXT_TEMPLATE = """
# Global Context (Read-Only Reference)
The user has provided the FULL subtitle file below for context (plot, terms, gender).
**INSTRUCTIONS for Context:**
1. Read this to understand the story.
2. **DO NOT translate this section.**
3. Use this ONLY to improve the accuracy of the batch translation below.

<FULL_SOURCE_CONTEXT>
{COMPLETE_SRT_TEXT}
</FULL_SOURCE_CONTEXT>
"""

def load_prompt(prompt_file: str, target_language: str) -> str:
    content = Path(prompt_file).read_text(encoding='utf-8', errors='ignore')
    return content.replace('{lang}', target_language)

def merge_translated_batch(translated_str: str, original_batch: List[dict]) -> List[dict]:
    result = copy.deepcopy(original_batch)
    try:
        translated_blocks = parse_srt(translated_str)
        if len(translated_blocks) == len(original_batch):
            for i, block in enumerate(translated_blocks):
                result[i]['text'] = block['text']
            return result
        logger.warning(
            f"Block mismatch: gốc={len(original_batch)}, dịch={len(translated_blocks)} "
            "→ giữ nguyên batch này"
        )
    except Exception as e:
        logger.warning(f"Lỗi parse translated batch: {e} → giữ nguyên batch này")
    return result

def translate_srt_file(
    input_file: str,
    output_file: str,
    prompt_file: str,
    provider: BaseTranslationProvider,
    target_language: str = "Vietnamese",
    batch_size: int = 30,
    use_full_context: bool = True,
    wait_sec: float = 0.0,
    max_chars: int = 0,
) -> dict:
    raw_content = Path(input_file).read_text(encoding="utf-8", errors="ignore")
    srt_list = parse_srt(raw_content)
    total = len(srt_list)

    if total == 0:
        raise ValueError(f"Không đọc được block SRT nào từ: {input_file}")

    logger.info(f"📄 Đọc: {total} block SRT | Provider: {provider.name} | Batch: {batch_size}")
    print(f"\n📄 Tổng: {total} block SRT")
    print(f"🤖 Provider: {provider.name}")
    print(f"🌐 Ngôn ngữ: {target_language}")
    print(f"📦 Batch  : {batch_size} block/lần\n")

    base_prompt = load_prompt(prompt_file, target_language)

    context_block = ""
    if use_full_context:
        full_text = "\n".join([it["text"] for it in srt_list])
        context_block = _GLOBAL_CONTEXT_TEMPLATE.replace("{COMPLETE_SRT_TEXT}", full_text)
        logger.info(f"📝 Full context: {len(full_text)} ký tự")

    caller = provider
    translated_srt = copy.deepcopy(srt_list)
    batches = [srt_list[i : i + batch_size] for i in range(0, total, batch_size)]
    total_batches = len(batches)
    success_count = 0
    failed_count = 0
    t_start = time.time()

    for idx, batch in enumerate(batches):
        batch_srt_str = "\n\n".join(
            f"{item['line']}\n{item['time']}\n{item['text'].strip()}"
            for item in batch
        )

        prompt_message = (
            base_prompt
            .replace("{batch_input}", batch_srt_str)
            .replace("{context_block}", context_block)
        )

        b_start = batch[0]["line"]
        b_end = batch[-1]["line"]
        print(
            f"🔄 [{idx + 1:>3}/{total_batches}] block {b_start:>5} → {b_end:<5}",
            end="  ",
            flush=True,
        )

        try:
            t0 = time.time()
            raw_result = caller.call(prompt_message)
            translated_text = parse_translation_response(raw_result)
            elapsed_batch = time.time() - t0

            offset = idx * batch_size
            translated_batch = merge_translated_batch(translated_text, batch)
            for j, item in enumerate(translated_batch):
                translated_srt[offset + j]["text"] = item["text"]

            success_count += 1
            print(f"✅  {elapsed_batch:.1f}s")

        except Exception as e:
            failed_count += 1
            print(f"❌  {e}")
            logger.error(f"Batch {idx + 1} thất bại: {e}")

        if wait_sec > 0:
            time.sleep(wait_sec)

    if max_chars > 0:
        for item in translated_srt:
            item["text"] = wrap_subtitle_text(item["text"], max_chars)

    output_content = segments_to_srt(translated_srt)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(output_content, encoding="utf-8")

    elapsed_total = time.time() - t_start
    stats = {
        "total_blocks": total,
        "total_batches": total_batches,
        "success": success_count,
        "failed": failed_count,
        "elapsed": round(elapsed_total, 1),
        "output": output_file,
    }

    print(f"\n{'─'*50}")
    print(f"✅ Hoàn thành: {output_file}")
    print(f"   Batch: {success_count}/{total_batches} thành công | {failed_count} lỗi")
    print(f"   Thời gian: {elapsed_total:.1f}s")

    return stats
