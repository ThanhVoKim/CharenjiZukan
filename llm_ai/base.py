from abc import ABC, abstractmethod
from typing import Any, Mapping


class BaseLLMProvider(ABC):
    """Interface chung cho các provider sinh nội dung bằng LLM."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tên provider/model dùng cho log và CLI."""
        ...

    @abstractmethod
    def call(self, message: str) -> str:
        """Gửi prompt/message tới provider và trả text response."""
        ...

    def generate(self, message: str) -> str:
        """Alias trung tính hơn cho call(), dùng trong các flow LLM generic."""
        return self.call(message)

    def set_global_context(self, context: str) -> bool:
        """
        Tuỳ chọn preload global context vào phía provider.

        Returns:
            True nếu provider đã tiếp nhận context thành công theo cơ chế nội bộ,
            khi đó caller có thể bỏ chèn context vào từng prompt.
            False nếu provider không hỗ trợ hoặc không thể set context.
        """
        return False

    def compact_state(self, response_id: str | None = None) -> Any:
        """Tuỳ chọn compact provider-side state nếu provider hỗ trợ."""
        return None

    @property
    def last_response_id(self) -> str | None:
        """Response id gần nhất nếu provider/stateful API có trả về."""
        return None

    @property
    def last_telemetry_record(self) -> Mapping[str, Any] | None:
        """Telemetry metadata gần nhất nếu provider có bật observability."""
        return None


# Backward-compatible alias cho các import cũ trong translation/*.
BaseTranslationProvider = BaseLLMProvider
