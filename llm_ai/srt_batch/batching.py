import copy
import re
from typing import Dict, List, Mapping


class BatchIntegrityError(RuntimeError):
    """Lỗi dữ liệu phản hồi không toàn vẹn (thiếu block/sai format)."""


# Dòng mở một block mới: "<số>. <text>" (cho phép khoảng trắng đầu dòng, dấu cách
# sau dấu chấm tùy chọn). Số này là `line` (chỉ số SRT toàn cục) — neo chống lệch hàng.
_NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)\.\s?(.*)$")


def parse_numbered_lines(text: str) -> Dict[int, str]:
    """Parse output LLM dạng numbered-line `"N. text"` thành {N: text}.

    Dòng khớp `_NUMBERED_LINE_RE` mở một entry mới theo số N. Dòng KHÔNG khớp được
    nối tiếp vào entry hiện tại (an toàn khi text bị LLM xuống dòng giữa chừng).
    Số trùng lặp → BatchIntegrityError (không thể map an toàn).
    """
    result: Dict[int, str] = {}
    current: int | None = None
    for raw in text.splitlines():
        match = _NUMBERED_LINE_RE.match(raw)
        if match:
            num = int(match.group(1))
            if num in result:
                raise BatchIntegrityError(f"Số dòng lặp trong output: {num}")
            result[num] = match.group(2).strip()
            current = num
        elif current is not None and raw.strip():
            # Orphan line: nối vào entry đang mở (giữ một space ngăn cách).
            result[current] = f"{result[current]} {raw.strip()}".strip()
    return result


def merge_translated_batch(translated_str: str, original_batch: List[dict]) -> List[dict]:
    """Map output numbered-line của LLM vào batch gốc THEO SỐ dòng (line), không theo vị trí.

    Tập số parse được phải khớp ĐÚNG tập `item['line']` của batch (đủ, không thừa/thiếu)
    → chống lệch hàng. Timestamp giữ nguyên từ original_batch (chỉ ghi đè `text`).
    """
    parsed = parse_numbered_lines(translated_str)

    expected = {item["line"] for item in original_batch}
    got = set(parsed.keys())
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise BatchIntegrityError(
            f"Line mismatch: thiếu={missing}, thừa={extra} "
            f"(gốc={len(expected)} số, parse được={len(got)} số)"
        )

    result = copy.deepcopy(original_batch)
    for item in result:
        item["text"] = parsed[item["line"]]
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
