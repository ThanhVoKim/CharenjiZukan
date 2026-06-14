# -*- coding: utf-8 -*-
"""
llm_ai/srt_batch/runner.py
==========================
Vòng lặp xử lý SRT theo LÔ (batch) bằng LLM — hạ tầng dùng chung cho cả
dịch thuật (`translation/srt_translator.py`) lẫn phục hồi dấu câu
(`cli/punctuate_srt.py`).

Khung chung (trích từ phần trùng của 2 module):
  parse_srt → build_global_context + provider.set_global_context (cache nội bộ)
  → chia batch → render_batch_prompt → provider.call → parse_tag_response
  → merge_translated_batch → validator(optional) → ghi đè text.

Khác biệt giữa các task được tham số hoá:
  - `response_tag`  : tag bao output ("TRANSLATE_TEXT" | "PUNCT_TEXT").
  - `validator`     : callable(merged_batch, original_batch) → raise BatchIntegrityError
                      nếu vi phạm (vd punctuation: chỉ thêm dấu, không đổi chữ). None = bỏ qua.

Batch thất bại sau hết retry → **giữ nguyên text gốc** (deepcopy ban đầu), không
nhận text bị đổi. Caller tự ghi sidecar (.txt/flatten) từ `result_srt_list` trả về.
"""

import copy
import time
from pathlib import Path
from typing import Any, Callable, Optional

from llm_ai.base import BaseLLMProvider
from llm_ai.retry import calculate_linear_retry_wait_seconds
from llm_ai.tasks.response_parser import parse_tag_response
from llm_ai.srt_batch.batching import (
    BatchIntegrityError,
    CacheTelemetryAccumulator,
    get_retry_attempts,
    get_retry_wait_seconds,
    merge_translated_batch,
)
from llm_ai.srt_batch.prompting import build_global_context, load_prompt, render_batch_prompt
from utils.logger import get_logger
from utils.srt_parser import parse_srt, segments_to_srt

logger = get_logger("srt_batch_runner")

Validator = Callable[[list[dict], list[dict]], None]


def run_srt_batch_task(
    input_srt: str,
    output_srt: str,
    *,
    prompt_file: str,
    provider: BaseLLMProvider,
    language: str,
    response_tag: str,
    batch_size: int = 30,
    use_full_context: bool = True,
    wait_sec: float = 0.0,
    validator: Optional[Validator] = None,
    write_srt: bool = True,
) -> tuple[list[dict], dict[str, Any]]:
    """Chạy 1 task LLM theo lô trên file SRT.

    Args:
        input_srt: SRT nguồn.
        output_srt: SRT output (chỉ ghi khi write_srt=True).
        prompt_file: Prompt template (chứa {language}/{lang}, {context_block}, {batch_input}).
        provider: LLM provider.
        language: Ngôn ngữ truyền vào prompt.
        response_tag: Tag bao output LLM (vd "TRANSLATE_TEXT", "PUNCT_TEXT").
        batch_size: Số block subtitle mỗi lần gọi LLM.
        use_full_context: Gửi kèm toàn bộ context (mạch văn xuyên block).
        wait_sec: Nghỉ giữa các batch (rate-limit).
        validator: Kiểm tra toàn vẹn sau merge; raise BatchIntegrityError → retry.
        write_srt: Ghi output_srt bằng segments_to_srt nếu True.

    Returns:
        (result_srt_list, stats) — stats: total_blocks, total_batches, success,
        failed, reverted_blocks, output.
    """
    raw_content = Path(input_srt).read_text(encoding="utf-8", errors="ignore")
    srt_list = parse_srt(raw_content)
    total = len(srt_list)
    if total == 0:
        raise ValueError(f"Không đọc được block SRT nào từ: {input_srt}")

    logger.info(
        f"📄 SRT batch task [{response_tag}]: {total} block | "
        f"Provider: {provider.name} | Lang: {language} | Batch: {batch_size}"
    )

    base_prompt = load_prompt(prompt_file, language)

    context_block = ""
    if use_full_context:
        context_block, full_text = build_global_context(srt_list)
        logger.info(f"📝 Full context: {len(full_text)} ký tự")
        # Provider hỗ trợ cache context nội bộ → tránh gửi lặp mỗi batch.
        if provider.set_global_context(context_block):
            logger.info("🧠 Provider đã nhận global context nội bộ (cache mode)")
            context_block = ""

    retry_attempts = get_retry_attempts(provider)
    retry_wait_seconds = get_retry_wait_seconds(provider)

    result_srt = copy.deepcopy(srt_list)
    batches = [srt_list[i : i + batch_size] for i in range(0, total, batch_size)]
    total_batches = len(batches)
    success_count = 0
    failed_count = 0
    reverted_blocks = 0
    cache_telemetry = CacheTelemetryAccumulator()

    for idx, batch in enumerate(batches):
        batch_srt_str = "\n\n".join(
            f"{item['line']}\n{item['time']}\n{item['text'].strip()}"
            for item in batch
        )
        prompt_message = render_batch_prompt(
            base_prompt=base_prompt,
            batch_srt_str=batch_srt_str,
            context_block=context_block,
        )

        offset = idx * batch_size
        batch_ok = False

        for attempt in range(1, retry_attempts + 1):
            try:
                raw_result = provider.call(prompt_message)
                cache_telemetry.record(provider)
                try:
                    parsed_text = parse_tag_response(raw_result, response_tag)
                except Exception as exc:
                    raise BatchIntegrityError(f"Không parse được <{response_tag}>: {exc}") from exc

                merged = merge_translated_batch(parsed_text, batch)
                if validator is not None:
                    validator(merged, batch)

                for j, item in enumerate(merged):
                    result_srt[offset + j]["text"] = item["text"]

                success_count += 1
                batch_ok = True
                logger.info(f"✅ [{idx + 1}/{total_batches}] block {batch[0]['line']}→{batch[-1]['line']}")
                break

            except BatchIntegrityError as exc:
                if attempt < retry_attempts:
                    logger.warning(f"⚠️ Batch {idx + 1} retry {attempt}/{retry_attempts}: {exc}")
                    wait_seconds = calculate_linear_retry_wait_seconds(retry_wait_seconds, attempt)
                    if wait_seconds > 0:
                        time.sleep(wait_seconds)
                    continue
                logger.error(f"❌ Batch {idx + 1} thất bại sau {retry_attempts} lần: {exc}")
            except Exception as exc:
                logger.error(f"❌ Batch {idx + 1} lỗi: {exc}")
                break

        if not batch_ok:
            # Giữ nguyên text gốc cho cả batch — KHÔNG nhận text bị đổi.
            failed_count += 1
            reverted_blocks += len(batch)

        if wait_sec > 0:
            time.sleep(wait_sec)

    if write_srt:
        output_content = segments_to_srt(result_srt)
        Path(output_srt).parent.mkdir(parents=True, exist_ok=True)
        Path(output_srt).write_text(output_content, encoding="utf-8")
        logger.info(f"📄 Đã ghi SRT: {output_srt}")

    cache_summary = cache_telemetry.summary_line()
    if cache_summary:
        logger.info(cache_summary)

    stats = {
        "total_blocks": total,
        "total_batches": total_batches,
        "success": success_count,
        "failed": failed_count,
        "reverted_blocks": reverted_blocks,
        "output": output_srt,
    }
    return result_srt, stats
