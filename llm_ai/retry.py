from __future__ import annotations

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
