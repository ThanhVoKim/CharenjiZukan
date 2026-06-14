from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from llm_ai.base import BaseLLMProvider
from llm_ai.openai_compat import OpenAICompatCapabilityError

logger = logging.getLogger("llm_ai")
ProviderFactory = Callable[[], BaseLLMProvider]
ProviderSource = BaseLLMProvider | ProviderFactory

PROVIDER_CHAIN_OVERRIDE_KEYS = (
    "base_url",
    "model",
    "system_prompt",
    "temperature",
    "max_tokens",
    "max_output_tokens",
    "thinking_budget",
    "budget",
    "request_timeout",
    "retry_attempts",
    "retry_wait_seconds",
    "generation_config",
    "safety_settings",
    "cache_ttl_seconds",
    "profile_name",
    "api_mode",
    "capability_flags",
    "request_options",
    "stateful_options",
    "telemetry",
    "task_name",
)


@dataclass(frozen=True)
class ProviderFailure:
    """Thông tin lỗi của một provider trong provider_chain."""

    chain_name: str
    provider_name: str
    error_type: str
    message: str


class ProviderChainError(RuntimeError):
    """Raise khi toàn bộ provider trong fallback chain đều thất bại."""

    def __init__(self, failures: Sequence[ProviderFailure]):
        self.failures = list(failures)
        details = "; ".join(
            f"{failure.chain_name} ({failure.provider_name}) -> "
            f"{failure.error_type}: {failure.message}"
            for failure in self.failures
        )
        super().__init__(f"Tất cả provider trong provider_chain đều thất bại: {details}")


def normalize_provider_chain(raw_chain: Any) -> list[dict[str, Any]]:
    """Validate và chuẩn hoá provider_chain từ task config YAML."""
    if not raw_chain:
        return []
    if not isinstance(raw_chain, list):
        raise ValueError("provider_chain phải là list các object provider config")

    normalized: list[dict[str, Any]] = []
    for idx, raw_entry in enumerate(raw_chain, start=1):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"provider_chain[{idx}] phải là object/dict")

        entry = dict(raw_entry)
        provider_type = str(entry.get("provider") or "").lower().strip()
        if not provider_type:
            raise ValueError(f"provider_chain[{idx}] thiếu khóa provider")
        if provider_type not in {"gemini", "openai", "vertexai"}:
            raise ValueError(
                f"provider_chain[{idx}] provider không hợp lệ: {provider_type}. "
                "Các provider hỗ trợ: gemini, openai, vertexai"
            )

        entry["provider"] = provider_type
        entry.setdefault("name", f"{idx}_{provider_type}")
        normalized.append(entry)

    return normalized


def apply_provider_chain_entry_overrides(
    provider_config: dict[str, Any],
    entry: Mapping[str, Any],
    *,
    task_system_prompt: str | None = None,
) -> dict[str, Any]:
    """Merge task system_prompt và override từ một provider_chain entry."""
    cfg = dict(provider_config or {})

    if task_system_prompt is not None:
        cfg["system_prompt"] = task_system_prompt

    for key in PROVIDER_CHAIN_OVERRIDE_KEYS:
        if key in entry and entry.get(key) is not None:
            # generation_config được deep-merge để override lẻ (vd thinking_level)
            # không xoá temperature/max_output_tokens của base provider config.
            if key == "generation_config" and isinstance(entry.get(key), dict):
                merged_gen_cfg = dict(cfg.get("generation_config") or {})
                merged_gen_cfg.update(entry.get(key))
                cfg["generation_config"] = merged_gen_cfg
            else:
                cfg[key] = entry.get(key)

    return cfg


class FallbackLLMProvider(BaseLLMProvider):
    """Provider wrapper thử lần lượt nhiều provider theo provider_chain.

    Mỗi concrete provider vẫn tự retry theo cấu hình riêng. Wrapper này chỉ chuyển
    sang provider kế tiếp sau khi provider hiện tại đã raise lỗi cuối cùng.

    Provider có thể truyền vào dạng instance hoặc factory lazy. Lazy factory giúp
    fallback config không kéo optional dependency hoặc credential cho tới khi thật
    sự cần dùng provider đó.
    """

    def __init__(
        self,
        providers: Sequence[ProviderSource],
        names: Sequence[str] | None = None,
        *,
        retry_attempts: int | None = None,
        retry_wait_seconds: float | int | None = None,
    ):
        if not providers:
            raise ValueError("provider_chain cần ít nhất 1 provider khả dụng")

        self._sources = list(providers)
        self._providers: list[BaseLLMProvider | None] = [
            source if isinstance(source, BaseLLMProvider) else None
            for source in self._sources
        ]
        self._chain_names = list(names or [self._source_display_name(i) for i in range(len(providers))])
        if len(self._chain_names) != len(self._sources):
            raise ValueError("Số lượng names phải khớp số lượng providers")

        self._active_index = 0
        # Global context được chain tự quản (xem set_global_context). Mỗi provider
        # khi trở thành active sẽ được hỏi set_global_context: nếu provider tự cache
        # được (True) thì gửi message nguyên; nếu không (False) chain prepend inline.
        self._global_context: str | None = None
        self._context_applied: set[int] = set()
        self._needs_inline: dict[int, bool] = {}
        # Khoá bảo vệ trạng thái chuyển fallback khi chạy batch song song: chốt
        # active_index + áp context được thực hiện trong khoá, còn network call
        # (phần chậm) chạy NGOÀI khoá để các batch vẫn song song thật sự.
        self._lock = threading.RLock()
        # Lưu provider mà mỗi thread vừa gọi -> last_telemetry_record đọc đúng số
        # liệu của chính thread đó (không bị nhầm sang provider thread khác đang dùng).
        self._call_local = threading.local()
        first_provider = self._providers[0]
        self._retry_attempts = retry_attempts if retry_attempts is not None else getattr(first_provider, "_retry_attempts", 3)
        self._retry_wait_seconds = (
            retry_wait_seconds
            if retry_wait_seconds is not None
            else getattr(first_provider, "_retry_wait_seconds", 0)
        )

    def _source_display_name(self, index: int) -> str:
        source = self._sources[index]
        if isinstance(source, BaseLLMProvider):
            return source.name
        return f"provider_{index + 1}"

    def _get_provider(self, index: int) -> BaseLLMProvider:
        provider = self._providers[index]
        if provider is not None:
            return provider

        source = self._sources[index]
        if not callable(source):
            raise TypeError(f"provider_chain[{index + 1}] không phải provider hoặc factory")

        provider = source()
        self._providers[index] = provider
        return provider

    @property
    def name(self) -> str:
        chain = " -> ".join(self._chain_names)
        active = self._chain_names[self._active_index]
        return f"FallbackChain(active={active}; chain={chain})"

    @property
    def active_provider_name(self) -> str:
        provider = self._providers[self._active_index]
        return provider.name if provider is not None else self._chain_names[self._active_index]

    @property
    def last_telemetry_record(self) -> Mapping[str, Any] | None:
        """Telemetry của provider mà thread hiện tại vừa gọi (delegate xuống dưới).

        Đọc per-thread (`_call_local`) để khi chạy song song không nhầm sang
        provider của thread khác; fallback về provider active nếu thread chưa gọi.
        """
        provider = getattr(self._call_local, "provider", None)
        if provider is None:
            provider = self._providers[self._active_index]
        return provider.last_telemetry_record if provider is not None else None

    @property
    def provider_count(self) -> int:
        return len(self._sources)

    def advance_to_next_provider(self, reason: str = "") -> bool:
        """Chuyển active provider sang provider kế tiếp nếu còn fallback."""
        if self._active_index + 1 >= len(self._sources):
            return False

        old_name = self._chain_names[self._active_index]
        self._active_index += 1
        new_name = self._chain_names[self._active_index]
        if reason:
            logger.warning(
                "[ProviderChain] Chuyển fallback %s -> %s. Lý do: %s",
                old_name,
                new_name,
                reason,
            )
        else:
            logger.warning("[ProviderChain] Chuyển fallback %s -> %s", old_name, new_name)
        return True

    def _ensure_context_on_active(self) -> None:
        """Đảm bảo provider active đã nhận global context (cache hoặc đánh dấu inline)."""
        if self._global_context is None:
            return
        index = self._active_index
        if index in self._context_applied:
            return
        provider = self._get_provider(index)
        cached = provider.set_global_context(self._global_context)
        self._needs_inline[index] = not cached
        self._context_applied.add(index)
        if cached:
            logger.info(
                "[ProviderChain] %s đã nhận global context nội bộ (cache mode)",
                self._chain_names[index],
            )
        else:
            logger.info(
                "[ProviderChain] %s không cache được, dùng inline context",
                self._chain_names[index],
            )

    def _prepare_message(self, message: str) -> str:
        """Prepend global context nếu provider active không tự cache được."""
        if self._global_context and self._needs_inline.get(self._active_index):
            return f"{self._global_context}\n\n{message}"
        return message

    def call(self, message: str) -> str:
        failures: list[ProviderFailure] = []
        last_exc: Exception | None = None

        while True:
            # Chốt snapshot trạng thái dưới khoá (nhanh), rồi gọi network NGOÀI khoá.
            with self._lock:
                index = self._active_index
                if index >= len(self._sources):
                    break
                chain_name = self._chain_names[index]
                provider = self._get_provider(index)
                self._ensure_context_on_active()
                prepared = self._prepare_message(message)
                # Ghi provider của thread này để telemetry đọc đúng nguồn.
                self._call_local.provider = provider

            try:
                return provider.call(prepared)
            except OpenAICompatCapabilityError as exc:
                logger.error(
                    "[ProviderChain] Lỗi capability/config ở %s (%s); không fallback để tránh chạy sai profile: %s",
                    chain_name,
                    provider.name,
                    exc,
                )
                raise
            except Exception as exc:
                last_exc = exc
                failures.append(
                    ProviderFailure(
                        chain_name=chain_name,
                        provider_name=provider.name,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                logger.error(
                    "[ProviderChain] Provider thất bại sau retry: %s (%s) - %s",
                    chain_name,
                    provider.name,
                    exc,
                )

                with self._lock:
                    # Chỉ thread đầu tiên thất bại tại index này mới advance; thread
                    # khác thấy index đã đổi -> vòng lặp lại trên provider mới.
                    if self._active_index == index:
                        if not self.advance_to_next_provider(str(exc)):
                            break  # đang ở provider cuối, hết fallback
                    # else: thread khác đã advance, lặp lại để dùng active mới.

        raise ProviderChainError(failures) from last_exc

    def set_global_context(self, context: str) -> bool:
        """Chain tự quản global context cho mọi provider trong fallback chain.

        Trả True để caller (workflow SRT) bỏ inline context khỏi từng prompt; chain
        sẽ tự lo việc cấp context cho provider active: provider nào tự cache được
        (vd Vertex explicit cache, OpenAI Responses anchor) thì gửi message nguyên,
        provider không cache được thì chain prepend context inline ngay trước khi gọi.
        Việc set_global_context lên provider được hoãn tới khi provider thật sự
        active để tránh eager instantiate toàn bộ fallback (kéo credential/dependency).
        """
        if not context or not context.strip():
            return False

        self._global_context = context
        self._context_applied.clear()
        self._needs_inline.clear()
        return True
