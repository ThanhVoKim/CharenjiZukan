from typing import List, Mapping

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


class CacheTelemetryAccumulator:
    """Cộng dồn telemetry token qua các batch để đo hiệu quả context cache.

    Đọc `provider.last_telemetry_record` sau mỗi batch. Hỗ trợ cả Vertex
    (`cached_tokens` = cached_content_token_count) lẫn OpenAI Responses
    (`cached_tokens` từ prompt_tokens_details). Provider không bật telemetry
    sẽ trả None và được bỏ qua an toàn.
    """

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0
        self.samples = 0

    def record(self, provider: object) -> None:
        self.record_dict(getattr(provider, "last_telemetry_record", None))

    def record_dict(self, record: object) -> None:
        """Cộng dồn trực tiếp từ một telemetry record (dict).

        Dùng ở chế độ chạy song song: worker đọc telemetry (thread-local) rồi trả
        record về main thread, main thread gọi record_dict đơn luồng -> không cần
        khoá cho phép cộng (`+=`).
        """
        if not record or not isinstance(record, Mapping):
            return
        self.samples += 1
        self.prompt_tokens += int(record.get("prompt_tokens") or 0)
        self.cached_tokens += int(record.get("cached_tokens") or 0)
        self.output_tokens += int(record.get("output_tokens") or 0)

    def summary_line(self) -> str | None:
        if self.samples == 0 or self.prompt_tokens == 0:
            return None
        hit_pct = 100.0 * self.cached_tokens / self.prompt_tokens
        return (
            f"🧠 Cache: {self.cached_tokens:,}/{self.prompt_tokens:,} prompt tokens "
            f"hit ({hit_pct:.0f}%) | output {self.output_tokens:,}"
        )


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
