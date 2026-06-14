#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/llm_ai/test_srt_batch_concurrency.py
==========================================
Test cho fan-out song song của srt_batch runner + các cơ chế an-toàn-thread đi kèm:
exponential backoff (retry.py), telemetry thread-local + tổng hợp main thread, anchor
lock double-check (openai provider), và FallbackLLMProvider thread-safe.

Cấu trúc layers:
  Layer 1 — Unit: exponential backoff math, _RateLimiter, CacheTelemetryAccumulator.record_dict
  Layer 2 — Component: run_srt_batch_task tuần tự vs song song cho cùng kết quả; warm-up;
                       telemetry không bị clobber; fallback chain song song.

Không cần GPU/FFmpeg/network — toàn bộ provider được fake bằng Python thuần.

Cách chạy:
    pytest tests/llm_ai/test_srt_batch_concurrency.py -v -k "Layer1"
    pytest tests/llm_ai/test_srt_batch_concurrency.py -v -k "Layer2"
"""

import sys
import threading
import time
from pathlib import Path

import pytest

# ── Project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm_ai.base import BaseLLMProvider  # noqa: E402
from llm_ai.retry import calculate_exponential_retry_wait_seconds  # noqa: E402
from llm_ai.srt_batch.batching import CacheTelemetryAccumulator  # noqa: E402
from llm_ai.srt_batch.runner import _RateLimiter, run_srt_batch_task  # noqa: E402


# ═════════════════════════════════════════════════════════════════════
# SHARED FIXTURES & FAKES
# ═════════════════════════════════════════════════════════════════════


def _make_srt(num_blocks: int) -> str:
    """Sinh SRT num_blocks block, mỗi block 1 giây."""
    lines = []
    for i in range(1, num_blocks + 1):
        start = f"00:00:{i:02d},000"
        end = f"00:00:{i + 1:02d},000"
        lines.append(f"{i}\n{start} --> {end}\nLine {i}")
    return "\n\n".join(lines) + "\n"


@pytest.fixture()
def sample_srt_path(tmp_path: Path):
    def _factory(num_blocks: int) -> Path:
        path = tmp_path / f"in_{num_blocks}.srt"
        path.write_text(_make_srt(num_blocks), encoding="utf-8")
        return path

    return _factory


@pytest.fixture()
def prompt_path(tmp_path: Path) -> Path:
    content = "Translate to {language}\n{context_block}\nINPUT:\n{batch_input}\n"
    path = tmp_path / "prompt.txt"
    path.write_text(content, encoding="utf-8")
    return path


class FakeProvider(BaseLLMProvider):
    """Provider giả: trả lại đúng các block đầu vào (echo) bao trong tag.

    Đếm số call đồng thời (max_concurrent) và độ trễ giả lập để ép overlap khi
    chạy song song. telemetry per-call lưu thread-local đúng như provider thật.
    """

    def __init__(self, *, tag: str = "TRANSLATE_TEXT", delay: float = 0.0,
                 prompt_tokens: int = 100, cached_tokens: int = 90):
        self._tag = tag
        self._delay = delay
        self._prompt_tokens = prompt_tokens
        self._cached_tokens = cached_tokens
        self._tls = threading.local()
        self._lock = threading.Lock()
        self.call_count = 0
        self.concurrent = 0
        self.max_concurrent = 0
        self.global_context_calls = 0

    @property
    def name(self) -> str:
        return "FakeProvider"

    def set_global_context(self, context: str) -> bool:
        self.global_context_calls += 1
        return True  # giả lập cache mode

    @property
    def last_telemetry_record(self):
        return getattr(self._tls, "record", None)

    def call(self, message: str) -> str:
        with self._lock:
            self.call_count += 1
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self._delay:
                time.sleep(self._delay)
            # Parse các block "N\nTIME\nTEXT" từ message INPUT để echo lại đúng.
            input_part = message.split("INPUT:", 1)[-1]
            blocks = [b for b in input_part.strip().split("\n\n") if b.strip()]
            body = "\n\n".join(blocks)
            self._tls.record = {
                "prompt_tokens": self._prompt_tokens,
                "cached_tokens": self._cached_tokens,
                "output_tokens": 10,
            }
            return f"<{self._tag}>\n{body}\n</{self._tag}>"
        finally:
            with self._lock:
                self.concurrent -= 1


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# ═════════════════════════════════════════════════════════════════════


class TestLayer1_ExponentialBackoff:
    def test_doubles_each_attempt(self):
        # base=1, jitter=0 -> 1, 2, 4, 8
        waits = [
            calculate_exponential_retry_wait_seconds(1, n, max_wait_seconds=100, jitter_seconds=0)
            for n in (1, 2, 3, 4)
        ]
        assert waits == [1.0, 2.0, 4.0, 8.0]

    def test_truncated_at_max(self):
        w = calculate_exponential_retry_wait_seconds(
            1, 10, max_wait_seconds=5, jitter_seconds=0
        )
        assert w == 5.0

    def test_jitter_within_bounds(self):
        for _ in range(50):
            w = calculate_exponential_retry_wait_seconds(
                1, 1, max_wait_seconds=100, jitter_seconds=1.0
            )
            assert 1.0 <= w <= 2.0

    def test_zero_base_floored(self):
        # base 0 -> floored 0.5 để exponential vẫn tăng
        w = calculate_exponential_retry_wait_seconds(0, 1, jitter_seconds=0)
        assert w == 0.5


class TestLayer1_RateLimiter:
    def test_zero_interval_no_delay(self):
        rl = _RateLimiter(0)
        start = time.monotonic()
        for _ in range(5):
            rl.acquire()
        assert time.monotonic() - start < 0.05

    def test_spaces_requests(self):
        rl = _RateLimiter(0.05)
        start = time.monotonic()
        for _ in range(3):
            rl.acquire()
        # 3 acquire: mốc 0, 0.05, 0.10 -> tối thiểu ~0.10s
        assert time.monotonic() - start >= 0.09


class TestLayer1_CacheTelemetryRecordDict:
    def test_record_dict_accumulates(self):
        acc = CacheTelemetryAccumulator()
        acc.record_dict({"prompt_tokens": 100, "cached_tokens": 90, "output_tokens": 10})
        acc.record_dict({"prompt_tokens": 50, "cached_tokens": 40, "output_tokens": 5})
        assert acc.samples == 2
        assert acc.prompt_tokens == 150
        assert acc.cached_tokens == 130
        assert acc.output_tokens == 15

    def test_record_dict_ignores_empty(self):
        acc = CacheTelemetryAccumulator()
        acc.record_dict(None)
        acc.record_dict({})
        assert acc.samples == 0

    def test_summary_line_hit_pct(self):
        acc = CacheTelemetryAccumulator()
        acc.record_dict({"prompt_tokens": 100, "cached_tokens": 90, "output_tokens": 10})
        line = acc.summary_line()
        assert "90%" in line


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — COMPONENT TESTS
# ═════════════════════════════════════════════════════════════════════


class TestLayer2_RunnerConcurrency:
    def _run(self, provider, sample_srt_path, prompt_path, tmp_path, *, workers, blocks=20):
        in_srt = sample_srt_path(blocks)
        out_srt = tmp_path / f"out_w{workers}.srt"
        result, stats = run_srt_batch_task(
            input_srt=str(in_srt),
            output_srt=str(out_srt),
            prompt_file=str(prompt_path),
            provider=provider,
            language="Vietnamese",
            response_tag="TRANSLATE_TEXT",
            batch_size=5,
            use_full_context=True,
            wait_sec=0.0,
            max_workers=workers,
            write_srt=True,
        )
        return result, stats

    def test_sequential_and_parallel_same_result(
        self, sample_srt_path, prompt_path, tmp_path
    ):
        seq_provider = FakeProvider()
        par_provider = FakeProvider(delay=0.01)
        seq_result, seq_stats = self._run(
            seq_provider, sample_srt_path, prompt_path, tmp_path, workers=1
        )
        par_result, par_stats = self._run(
            par_provider, sample_srt_path, prompt_path, tmp_path, workers=4
        )
        # Kết quả text giống hệt nhau bất kể thứ tự/song song.
        assert [b["text"] for b in seq_result] == [b["text"] for b in par_result]
        assert seq_stats["success"] == par_stats["success"]
        assert seq_stats["total_batches"] == par_stats["total_batches"]

    def test_parallel_actually_overlaps(self, sample_srt_path, prompt_path, tmp_path):
        provider = FakeProvider(delay=0.02)
        self._run(provider, sample_srt_path, prompt_path, tmp_path, workers=4, blocks=20)
        # 4 batch (block 2-4) fan-out song song -> phải có overlap thực sự.
        assert provider.max_concurrent >= 2

    def test_warmup_first_batch_sequential(self, sample_srt_path, prompt_path, tmp_path):
        provider = FakeProvider(delay=0.02)
        # Ghi số call ĐANG bay (chưa kể chính mình) tại thời điểm vào mỗi call.
        observed = []
        orig_call = provider.call

        def spy(message):
            observed.append(provider.concurrent)
            return orig_call(message)

        provider.call = spy  # type: ignore
        self._run(provider, sample_srt_path, prompt_path, tmp_path, workers=4, blocks=20)
        # Warm-up (call đầu) phải đơn độc: không có call nào khác đang bay.
        assert observed[0] == 0
        # Fan-out sau đó phải có overlap thật: ít nhất 1 call thấy call khác đang bay.
        assert max(observed) >= 1

    def test_telemetry_aggregated_correctly(self, sample_srt_path, prompt_path, tmp_path):
        # 20 block / batch 5 = 4 batch. prompt=100 cached=90 mỗi batch.
        provider = FakeProvider(delay=0.01, prompt_tokens=100, cached_tokens=90)
        _, stats = self._run(
            provider, sample_srt_path, prompt_path, tmp_path, workers=4, blocks=20
        )
        assert stats["total_batches"] == 4
        assert stats["success"] == 4
        # Telemetry main-thread không bị clobber: đếm đủ đúng 4 sample dù song song
        # (gián tiếp xác nhận qua việc mọi batch thành công + provider gọi đúng số lần).
        assert provider.call_count == 4

    def test_global_context_set_once(self, sample_srt_path, prompt_path, tmp_path):
        provider = FakeProvider(delay=0.005)
        self._run(provider, sample_srt_path, prompt_path, tmp_path, workers=4, blocks=20)
        # set_global_context chỉ gọi 1 lần trước fan-out.
        assert provider.global_context_calls == 1


class TestLayer2_FallbackChainThreadSafe:
    def test_concurrent_calls_thread_safe(self):
        """FallbackLLMProvider.call() chạy đồng thời nhiều thread không vỡ state."""
        from concurrent.futures import ThreadPoolExecutor
        from llm_ai.provider_chain import FallbackLLMProvider

        primary = FakeProvider()
        chain = FallbackLLMProvider([primary], names=["primary"])
        chain.set_global_context("ctx")

        def worker(i):
            return chain.call(f"INPUT:\n{i}\n00:00:01,000 --> 00:00:02,000\nLine {i}")

        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(worker, range(40)))

        assert len(results) == 40
        assert all("TRANSLATE_TEXT" in r for r in results)
        # Context chỉ set lên provider active 1 lần dù gọi song song.
        assert primary.global_context_calls == 1
