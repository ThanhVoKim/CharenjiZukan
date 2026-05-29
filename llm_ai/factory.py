from pathlib import Path
from typing import Any, Dict, Optional

from llm_ai.base import BaseLLMProvider


def load_provider_config(config_path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("pyyaml chưa cài. Chạy: pip install PyYAML>=6.0") from exc

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Provider config không tồn tại: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def create_provider(
    provider_type: str,
    config: Optional[Dict[str, Any]],
    secrets: Dict[str, Any],
) -> BaseLLMProvider:
    cfg = config or {}
    resolved_provider_type = (provider_type or cfg.get("provider") or "").lower().strip()

    if resolved_provider_type == "gemini":
        from llm_ai.providers.gemini import GeminiProvider

        return GeminiProvider(
            api_keys=secrets.get("api_keys", []),
            model=cfg.get("model", "gemini-3-flash-preview"),
            thinking_budget=cfg.get("thinking_budget", 8192),
            system_prompt=cfg.get("system_prompt", ""),
            temperature=cfg.get("temperature", 1),
            max_output_tokens=cfg.get("max_output_tokens", cfg.get("max_tokens", 65530)),
            retry_attempts=cfg.get("retry_attempts", 3),
            retry_wait_seconds=cfg.get("retry_wait_seconds", 5),
            safety_settings=cfg.get("safety_settings", {}),
        )

    if resolved_provider_type == "openai":
        from llm_ai.providers.openai import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            api_key=secrets.get("api_key", ""),
            base_url=cfg.get("base_url", "https://api.openai.com/v1"),
            model=cfg.get("model", "gpt-4o-mini"),
            temperature=cfg.get("temperature", 1),
            max_tokens=cfg.get("max_tokens", cfg.get("max_output_tokens", 8192)),
            system_prompt=cfg.get("system_prompt", ""),
            request_timeout=cfg.get("request_timeout", 120),
            retry_attempts=cfg.get("retry_attempts", 3),
            retry_wait_seconds=cfg.get("retry_wait_seconds", 5),
            profile_config=cfg,
        )

    if resolved_provider_type == "vertexai":
        from llm_ai.providers.vertexai import VertexAIProvider

        return VertexAIProvider(
            project_id=cfg.get("project_id", ""),
            location=cfg.get("location", "us-central1"),
            model=cfg.get("model", "gemini-3-flash-preview"),
            generation_config=cfg.get("generation_config", {}),
            safety_settings=cfg.get("safety_settings", {}),
            system_prompt=cfg.get("system_prompt", ""),
            request_timeout=cfg.get("request_timeout", 180),
            retry_attempts=cfg.get("retry_attempts", 3),
            retry_wait_seconds=cfg.get("retry_wait_seconds", 10),
            cache_ttl_seconds=cfg.get("cache_ttl_seconds", 3600),
        )

    raise ValueError(
        f"Provider không hợp lệ: '{provider_type}'. "
        f"Các provider hỗ trợ: gemini, openai, vertexai"
    )


def create_llm_provider(*args, **kwargs) -> BaseLLMProvider:
    return create_provider(*args, **kwargs)
