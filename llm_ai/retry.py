from __future__ import annotations

import random
from typing import Any, Callable


def calculate_linear_retry_wait_seconds(
    retry_wait_seconds: float | int,
    attempt_number: int,
) -> float:
    """Tính thời gian chờ retry tuyến tính theo số attempt đã thất bại.

    Ví dụ: retry_wait_seconds=10, attempt_number=2 -> 20 giây.
    """
    try:
        base_wait = float(retry_wait_seconds)
    except (TypeError, ValueError):
        base_wait = 0.0

    try:
        attempt = int(attempt_number)
    except (TypeError, ValueError):
        attempt = 1

    return max(0.0, base_wait) * max(1, attempt)


def build_linear_retry_wait(retry_wait_seconds: float | int) -> Callable[[Any], float]:
    """Tạo wait strategy cho tenacity: wait = retry_wait_seconds * attempt_number."""

    def _wait(retry_state: Any) -> float:
        return calculate_linear_retry_wait_seconds(
            retry_wait_seconds,
            getattr(retry_state, "attempt_number", 1),
        )

    return _wait


def calculate_exponential_retry_wait_seconds(
    retry_wait_seconds: float | int,
    attempt_number: int,
    *,
    max_wait_seconds: float = 60.0,
    jitter_seconds: float = 1.0,
) -> float:
    """Tính thời gian chờ retry theo lũy thừa (exponential backoff) + jitter.

    Phù hợp với rate-limit 429 khi chạy nhiều request song song: thời gian chờ
    tăng gấp đôi mỗi lần thất bại (truncated tại max_wait_seconds) và cộng thêm
    một lượng ngẫu nhiên (jitter) để các thread cùng dính 429 KHÔNG retry đồng
    thời (tránh "retry storm").

    Ví dụ base=1: attempt 1→~1s, 2→~2s, 3→~4s, 4→~8s (+ jitter 0..1s).
    """
    try:
        base_wait = float(retry_wait_seconds)
    except (TypeError, ValueError):
        base_wait = 0.0
    # Floor base để exponential thực sự tăng kể cả khi config để 0.
    if base_wait <= 0:
        base_wait = 0.5

    try:
        attempt = int(attempt_number)
    except (TypeError, ValueError):
        attempt = 1
    attempt = max(1, attempt)

    raw = base_wait * (2 ** (attempt - 1))
    capped = min(raw, max(0.0, max_wait_seconds))
    jitter = random.uniform(0.0, max(0.0, jitter_seconds))
    return capped + jitter


def build_exponential_retry_wait(
    retry_wait_seconds: float | int,
    *,
    max_wait_seconds: float = 60.0,
    jitter_seconds: float = 1.0,
) -> Callable[[Any], float]:
    """Tạo wait strategy exponential-backoff + jitter cho tenacity."""

    def _wait(retry_state: Any) -> float:
        return calculate_exponential_retry_wait_seconds(
            retry_wait_seconds,
            getattr(retry_state, "attempt_number", 1),
            max_wait_seconds=max_wait_seconds,
            jitter_seconds=jitter_seconds,
        )

    return _wait
