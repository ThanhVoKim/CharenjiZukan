from typing import List

from utils.srt_parser import parse_srt


class BatchIntegrityError(RuntimeError):
    """Lỗi dữ liệu phản hồi không toàn vẹn (thiếu block/sai format)."""


def merge_translated_batch(translated_str: str, original_batch: List[dict]) -> List[dict]:
    import copy

    try:
        translated_blocks = parse_srt(translated_str)
    except Exception as exc:
        raise BatchIntegrityError(f"Lỗi parse translated batch: {exc}") from exc

    if len(translated_blocks) != len(original_batch):
        raise BatchIntegrityError(
            f"Block mismatch: gốc={len(original_batch)}, dịch={len(translated_blocks)}"
        )

    result = copy.deepcopy(original_batch)
    for idx, block in enumerate(translated_blocks):
        result[idx]["text"] = block["text"]
    return result


def get_retry_attempts(provider: object) -> int:
    raw_retry_attempts = getattr(provider, "_retry_attempts", 3)
    try:
        retry_attempts = int(raw_retry_attempts)
    except (TypeError, ValueError):
        retry_attempts = 3
    return max(1, retry_attempts)


def get_retry_wait_seconds(provider: object) -> float:
    raw_retry_wait_seconds = getattr(provider, "_retry_wait_seconds", 0)
    try:
        retry_wait_seconds = float(raw_retry_wait_seconds)
    except (TypeError, ValueError):
        retry_wait_seconds = 0.0
    return max(0.0, retry_wait_seconds)
