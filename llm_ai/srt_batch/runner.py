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

Concurrency (max_workers > 1):
  Batch ĐẦU chạy tuần tự ("warm-up") để provider kịp tạo/ấm cache (Vertex
  CachedContent) hoặc anchor R0 (OpenAI Responses) TRƯỚC khi fan-out — tránh cả
  wave đầu cùng cold-start cache. Sau đó các batch còn lại chạy song song qua
  ThreadPoolExecutor. Telemetry đọc thread-local trong worker rồi trả về main
  thread cộng dồn đơn luồng (không cần khoá). Một `_RateLimiter` tuỳ chọn (theo
  `wait_sec`) đặt trần nhịp phát request để không vượt RPM khi song song.
"""

import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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


@dataclass
class _BatchOutcome:
    """Kết quả xử lý 1 batch — trả từ worker về main thread.

    texts: danh sách text đã merge (cùng độ dài batch) nếu thành công, None nếu
    thất bại (caller giữ nguyên text gốc). telemetry: record token của call cuối.
    """

    idx: int
    ok: bool
    texts: Optional[list[str]]
    telemetry: Optional[dict]


class _RateLimiter:
    """Đặt trần nhịp phát request: tối đa 1 request mỗi `min_interval` giây.

    Dùng khi chạy song song để không vượt RPM. Mỗi thread "đặt chỗ" một mốc thời
    gian tăng dần dưới khoá (nhanh) rồi ngủ NGOÀI khoá tới mốc đó → các request
    được giãn đều mà không pin thread nào ngủ trong khi giữ khoá.
    """

    def __init__(self, min_interval: float) -> None:
        self._min_interval = max(0.0, float(min_interval or 0.0))
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            target = max(now, self._next_allowed)
            self._next_allowed = target + self._min_interval
        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def _process_one_batch(
    idx: int,
    batch: list[dict],
    *,
    base_prompt: str,
    context_block: str,
    provider: BaseLLMProvider,
    response_tag: str,
    validator: Optional[Validator],
    retry_attempts: int,
    retry_wait_seconds: float,
    total_batches: int,
    rate_limiter: Optional[_RateLimiter] = None,
) -> _BatchOutcome:
    """Xử lý 1 batch (render → call → parse → merge → validate) với retry tính toàn vẹn.

    Thuần hàm theo nghĩa KHÔNG ghi vào state chia sẻ (result_srt/counter); chỉ trả
    `_BatchOutcome` để main thread áp dụng. An toàn chạy trong ThreadPoolExecutor.
    """
    batch_srt_str = "\n".join(
        f"{item['line']}. {item['text'].strip()}"
        for item in batch
    )
    prompt_message = render_batch_prompt(
        base_prompt=base_prompt,
        batch_srt_str=batch_srt_str,
        context_block=context_block,
    )

    telemetry: Optional[dict] = None

    for attempt in range(1, retry_attempts + 1):
        try:
            if rate_limiter is not None:
                rate_limiter.acquire()
            raw_result = provider.call(prompt_message)

            # Đọc telemetry NGAY trong worker (thread-local) rồi mang về main thread.
            record = getattr(provider, "last_telemetry_record", None)
            if record:
                telemetry = dict(record)

            try:
                parsed_text = parse_tag_response(raw_result, response_tag)
            except Exception as exc:
                raise BatchIntegrityError(f"Không parse được <{response_tag}>: {exc}") from exc

            merged = merge_translated_batch(parsed_text, batch)
            if validator is not None:
                validator(merged, batch)

            logger.info(
                f"✅ [{idx + 1}/{total_batches}] block {batch[0]['line']}→{batch[-1]['line']}"
            )
            return _BatchOutcome(
                idx=idx,
                ok=True,
                texts=[item["text"] for item in merged],
                telemetry=telemetry,
            )

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

    return _BatchOutcome(idx=idx, ok=False, texts=None, telemetry=telemetry)


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
    max_workers: int = 1,
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
        wait_sec: Tuần tự = nghỉ giữa các batch; song song = trần nhịp phát request
            (tối đa 1 request mỗi wait_sec giây qua _RateLimiter).
        validator: Kiểm tra toàn vẹn sau merge; raise BatchIntegrityError → retry.
        write_srt: Ghi output_srt bằng segments_to_srt nếu True.
        max_workers: Số batch chạy song song. 1 = tuần tự (mặc định, giữ nguyên hành
            vi cũ). >1 = warm-up batch đầu tuần tự rồi fan-out phần còn lại.

    Returns:
        (result_srt_list, stats) — stats: total_blocks, total_batches, success,
        failed, reverted_blocks, output.
    """
    raw_content = Path(input_srt).read_text(encoding="utf-8", errors="ignore")
    srt_list = parse_srt(raw_content)
    total = len(srt_list)
    if total == 0:
        raise ValueError(f"Không đọc được block SRT nào từ: {input_srt}")

    workers = max(1, int(max_workers or 1))
    logger.info(
        f"📄 SRT batch task [{response_tag}]: {total} block | "
        f"Provider: {provider.name} | Lang: {language} | Batch: {batch_size} | Workers: {workers}"
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
    cache_telemetry = CacheTelemetryAccumulator()

    counters = {"success": 0, "failed": 0, "reverted_blocks": 0}

    def _apply_outcome(outcome: _BatchOutcome) -> None:
        """Áp kết quả 1 batch vào result_srt + cộng telemetry. Chạy ở MAIN thread."""
        if outcome.telemetry:
            cache_telemetry.record_dict(outcome.telemetry)
        batch = batches[outcome.idx]
        offset = outcome.idx * batch_size
        if outcome.ok and outcome.texts is not None:
            for j, text in enumerate(outcome.texts):
                result_srt[offset + j]["text"] = text
            counters["success"] += 1
        else:
            # Giữ nguyên text gốc cho cả batch — KHÔNG nhận text bị đổi.
            counters["failed"] += 1
            counters["reverted_blocks"] += len(batch)

    def _make_kwargs(rate_limiter: Optional[_RateLimiter]) -> dict[str, Any]:
        return dict(
            base_prompt=base_prompt,
            context_block=context_block,
            provider=provider,
            response_tag=response_tag,
            validator=validator,
            retry_attempts=retry_attempts,
            retry_wait_seconds=retry_wait_seconds,
            total_batches=total_batches,
            rate_limiter=rate_limiter,
        )

    if workers == 1 or total_batches <= 1:
        # Tuần tự: giữ nguyên hành vi cũ (wait_sec = nghỉ giữa batch).
        kwargs = _make_kwargs(rate_limiter=None)
        for idx, batch in enumerate(batches):
            _apply_outcome(_process_one_batch(idx, batch, **kwargs))
            if wait_sec > 0:
                time.sleep(wait_sec)
    else:
        # Song song: warm-up batch 0 tuần tự để tạo/ấm cache + anchor TRƯỚC khi fan-out.
        rate_limiter = _RateLimiter(wait_sec)
        kwargs = _make_kwargs(rate_limiter=rate_limiter)
        _apply_outcome(_process_one_batch(0, batches[0], **kwargs))

        logger.info(f"🚀 Fan-out {total_batches - 1} batch còn lại với {workers} workers")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_one_batch, idx, batches[idx], **kwargs): idx
                for idx in range(1, total_batches)
            }
            for future in as_completed(futures):
                _apply_outcome(future.result())

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
        "success": counters["success"],
        "failed": counters["failed"],
        "reverted_blocks": counters["reverted_blocks"],
        "output": output_srt,
    }
    return result_srt, stats
