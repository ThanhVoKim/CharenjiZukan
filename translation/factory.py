import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from translation.base import BaseTranslationProvider

def load_provider_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Provider config không tồn tại: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def create_provider(
    provider_type: str,
    config: Optional[Dict[str, Any]],
    secrets: Dict[str, Any],
) -> BaseTranslationProvider:
    cfg = config or {}

    if provider_type == "gemini":
        from translation.gemini_provider import GeminiProvider
        return GeminiProvider(
            api_keys=secrets.get("api_keys", []),
            model=cfg.get("model", "gemini-3-flash-preview"),
            thinking_budget=cfg.get("thinking_budget", 8192),
        )

    elif provider_type == "openai":
        from translation.openai_provider import OpenAICompatibleProvider
        return OpenAICompatibleProvider(
            api_key=secrets.get("api_key", ""),
            base_url=cfg.get("base_url", "https://api.openai.com/v1"),
            model=cfg.get("model", "gpt-4o-mini"),
            temperature=cfg.get("temperature", 1),
            max_tokens=cfg.get("max_tokens", 8192),
            system_prompt=cfg.get("system_prompt", ""),
            request_timeout=cfg.get("request_timeout", 120),
            retry_attempts=cfg.get("retry_attempts", 3),
            retry_wait_seconds=cfg.get("retry_wait_seconds", 5),
        )

    elif provider_type == "vertexai":
        from translation.vertexai_provider import VertexAIProvider
        return VertexAIProvider(
            project_id=cfg.get("project_id", ""),
            location=cfg.get("location", "us-central1"),
            model=cfg.get("model", "gemini-1.5-pro"),
            generation_config=cfg.get("generation_config", {}),
            safety_settings=cfg.get("safety_settings", {}),
            system_prompt=cfg.get("system_prompt", ""),
            request_timeout=cfg.get("request_timeout", 180),
            retry_attempts=cfg.get("retry_attempts", 3),
            retry_wait_seconds=cfg.get("retry_wait_seconds", 10),
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider_type}'. "
            f"Các provider hỗ trợ: gemini, openai, vertexai"
        )
