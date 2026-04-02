from abc import ABC, abstractmethod


class BaseTranslationProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def call(self, message: str) -> str:
        ...

    def set_global_context(self, context: str) -> bool:
        """
        Tuỳ chọn preload global context vào phía provider (ví dụ context cache).

        Returns:
            True nếu provider đã tiếp nhận context thành công theo cơ chế nội bộ,
            khi đó caller có thể bỏ chèn context vào từng prompt batch.
            False nếu provider không hỗ trợ hoặc không thể set context.
        """
        return False
