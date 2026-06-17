"""
llm_ai/srt_batch/
=================
Hạ tầng xử lý SRT theo lô (batch) bằng LLM — dùng chung cho dịch thuật và
phục hồi dấu câu. Bao gồm: vòng lặp batch + context cache + integrity retry,
prompting, telemetry. Trung lập với loại task (tham số hoá qua tag + validator).
"""

from llm_ai.srt_batch.batching import (
    BatchIntegrityError,
    CacheTelemetryAccumulator,
    get_retry_attempts,
    get_retry_wait_seconds,
    merge_translated_batch,
    parse_numbered_lines,
)
from llm_ai.srt_batch.prompting import (
    build_global_context,
    load_prompt,
    render_batch_prompt,
)
from llm_ai.srt_batch.runner import run_srt_batch_task

__all__ = [
    "run_srt_batch_task",
    "BatchIntegrityError",
    "CacheTelemetryAccumulator",
    "merge_translated_batch",
    "parse_numbered_lines",
    "get_retry_attempts",
    "get_retry_wait_seconds",
    "build_global_context",
    "load_prompt",
    "render_batch_prompt",
]
