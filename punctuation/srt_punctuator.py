# -*- coding: utf-8 -*-
"""
punctuation/srt_punctuator.py
=============================
Phục hồi dấu câu cho SRT (text OCR) bằng LLM — chạy theo LÔ (batch), mô phỏng
`translation/srt_translator.py`.

Khác translate ở hai điểm cốt lõi:
1. LLM **chỉ được thêm dấu câu**, KHÔNG được đổi/thêm/bớt ký tự chữ. Sau mỗi batch,
   `_content_signature()` strip toàn bộ dấu câu (Unicode category `P*`) + whitespace
   rồi so khớp output vs input theo từng dòng. Lệch → `BatchIntegrityError` → retry;
   hết retry thì GIỮ NGUYÊN dòng gốc (không dấu) thay vì nhận text bị đổi.
2. Trung lập ngôn ngữ: `language` truyền vào prompt; validator không hardcode bộ dấu CJK.

Output:
- `restore_punctuation_srt(...)` ghi `_punct.srt` (cùng cấu trúc, chỉ thêm dấu).
- `flatten_srt_to_text(...)` nối text các block thành **1 dòng** cho forced aligner.
"""

import copy
import time
import unicodedata
from pathlib import Path
from typing import Any

from utils.logger import get_logger
from utils.srt_parser import parse_srt, segments_to_srt, is_cjk
from llm_ai.base import BaseLLMProvider
from llm_ai.tasks.response_parser import parse_tag_response
from llm_ai.retry import calculate_linear_retry_wait_seconds
from translation.batching import (
    BatchIntegrityError,
    CacheTelemetryAccumulator,
    get_retry_attempts,
    get_retry_wait_seconds,
    merge_translated_batch,
)
from translation.prompting import build_global_context, render_batch_prompt

logger = get_logger("srt_punctuator")

PUNCT_TAG = "PUNCT_TEXT"


# ═════════════════════════════════════════════════════════════════════
# Validator — chống ảo giác (chỉ thêm dấu, không đổi chữ)
# ═════════════════════════════════════════════════════════════════════

def _content_signature(text: str) -> str:
    """Trả về phần 'nội dung chữ' của text: bỏ dấu câu (Unicode P*) + whitespace.

    Dùng để xác nhận LLM chỉ chèn dấu mà không thêm/bớt/đổi ký tự chữ.
    Trung lập ngôn ngữ — không phụ thuộc bộ dấu CJK hardcode.
    """
    return "".join(
        ch
        for ch in text
        if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    )


def _validate_content_preserved(punctuated_batch: list[dict], original_batch: list[dict]) -> None:
    """Raise BatchIntegrityError nếu bất kỳ dòng nào bị đổi nội dung chữ."""
    for idx, (new_item, old_item) in enumerate(zip(punctuated_batch, original_batch)):
        if _content_signature(new_item["text"]) != _content_signature(old_item["text"]):
            raise BatchIntegrityError(
                f"Dòng {old_item.get('line', idx)} bị đổi nội dung chữ "
                f"(không chỉ thêm dấu): gốc={old_item['text']!r} → out={new_item['text']!r}"
            )


# ═════════════════════════════════════════════════════════════════════
# Prompt loading
# ═════════════════════════════════════════════════════════════════════

def _load_punct_prompt(prompt_file: str, language: str) -> str:
    """Đọc prompt template, thay {language}/{lang} bằng ngôn ngữ nguồn."""
    content = Path(prompt_file).read_text(encoding="utf-8", errors="ignore")
    return content.replace("{language}", language).replace("{lang}", language)


# ═════════════════════════════════════════════════════════════════════
# Core: restore punctuation
# ═════════════════════════════════════════════════════════════════════

def restore_punctuation_srt(
    input_srt: str,
    output_srt: str,
    prompt_file: str,
    provider: BaseLLMProvider,
    language: str = "Chinese",
    batch_size: int = 30,
    use_full_context: bool = True,
    wait_sec: float = 0.0,
) -> dict[str, Any]:
    """Phục hồi dấu câu cho SRT bằng LLM, ghi `output_srt`.

    Args:
        input_srt: SRT nguồn (text OCR chưa có dấu).
        output_srt: SRT output (đã thêm dấu).
        prompt_file: Prompt template (chứa {language}, {context_block}, {batch_input}).
        provider: LLM provider (vd vertexai).
        language: Ngôn ngữ nguồn (Chinese/Japanese/English/...).
        batch_size: Số block subtitle mỗi lần gọi LLM.
        use_full_context: Gửi kèm toàn bộ context (mạch văn xuyên block).
        wait_sec: Nghỉ giữa các batch (rate-limit).

    Returns:
        Dict stats: total_blocks, total_batches, success, failed, reverted, output.
    """
    raw_content = Path(input_srt).read_text(encoding="utf-8", errors="ignore")
    srt_list = parse_srt(raw_content)
    total = len(srt_list)
    if total == 0:
        raise ValueError(f"Không đọc được block SRT nào từ: {input_srt}")

    logger.info(f"📄 Punctuation: {total} block | Provider: {provider.name} | Lang: {language} | Batch: {batch_size}")

    base_prompt = _load_punct_prompt(prompt_file, language)

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

    punctuated_srt = copy.deepcopy(srt_list)
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
                    punctuated_text = parse_tag_response(raw_result, PUNCT_TAG)
                except Exception as exc:
                    raise BatchIntegrityError(f"Không parse được <{PUNCT_TAG}>: {exc}") from exc

                merged = merge_translated_batch(punctuated_text, batch)
                _validate_content_preserved(merged, batch)

                for j, item in enumerate(merged):
                    punctuated_srt[offset + j]["text"] = item["text"]

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
            # Giữ nguyên text gốc (chưa dấu) cho cả batch — KHÔNG nhận text bị đổi.
            failed_count += 1
            reverted_blocks += len(batch)

        if wait_sec > 0:
            time.sleep(wait_sec)

    output_content = segments_to_srt(punctuated_srt)
    Path(output_srt).parent.mkdir(parents=True, exist_ok=True)
    Path(output_srt).write_text(output_content, encoding="utf-8")
    logger.info(f"📄 Đã ghi punctuation SRT: {output_srt}")

    cache_summary = cache_telemetry.summary_line()
    if cache_summary:
        logger.info(cache_summary)

    return {
        "total_blocks": total,
        "total_batches": total_batches,
        "success": success_count,
        "failed": failed_count,
        "reverted_blocks": reverted_blocks,
        "output": output_srt,
    }


# ═════════════════════════════════════════════════════════════════════
# Flatten: SRT → flat text (1 dòng) cho forced aligner
# ═════════════════════════════════════════════════════════════════════

def flatten_srt_to_text(input_srt: str, output_txt: str) -> str:
    """Nối text các block SRT thành MỘT dòng liên tục cho forced aligner.

    Aligner (`merge_punctuation`) duyệt full_text theo ký tự và sẽ nuốt `\\n`
    vào phụ đề, nên transcript phải phẳng 1 dòng.

    Separator phụ thuộc ngôn ngữ:
      - CJK (Trung/Nhật/Hàn): nối trực tiếp, bỏ mọi khoảng trắng (dấu câu là ranh giới).
      - Latin: nối bằng một khoảng trắng, collapse whitespace.

    Args:
        input_srt: File SRT (lý tưởng là `_punct.srt` đã có dấu).
        output_txt: File text phẳng output.

    Returns:
        Chuỗi flat text đã ghi.
    """
    raw_content = Path(input_srt).read_text(encoding="utf-8", errors="ignore")
    segments = parse_srt(raw_content)

    joined_all = "".join(seg.get("text", "") for seg in segments)
    cjk = is_cjk(joined_all)

    chunks = []
    for seg in segments:
        text = seg.get("text", "")
        if not text:
            continue
        if cjk:
            # Bỏ toàn bộ whitespace/newline trong câu CJK.
            text = "".join(text.split())
        else:
            text = " ".join(text.split())
        if text:
            chunks.append(text)

    sep = "" if cjk else " "
    flat_text = sep.join(chunks).strip()

    path = Path(output_txt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(flat_text, encoding="utf-8")
    logger.info(f"📄 Đã ghi flat transcript: {output_txt} ({len(flat_text)} ký tự)")
    return flat_text
