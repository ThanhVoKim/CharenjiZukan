#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/llm_ai/test_openai_compat_capability_probe.py
==================================================
Opt-in real endpoint probe for OpenAI-compatible profiles.

Cấu trúc layers:
  Layer 4 — Real API Tests (cần OPENAI_API_KEY và explicit allow-cost env)

Cách chạy:
    OPENAI_COMPAT_PROFILE=config/llm/openai_compat.yaml \
    OPENAI_COMPAT_PROBE_ALLOW_COST=1 \
    OPENAI_API_KEY=... \
    pytest tests/llm_ai/test_openai_compat_capability_probe.py -v -s -k "Layer4"
"""

import os
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_ai.factory import create_provider, load_provider_config  # noqa: E402
from llm_ai.openai_compat import (  # noqa: E402
    PROBE_STATUS_ERROR,
    PROBE_STATUS_VERIFIED,
    OpenAICompatProfile,
    write_capability_report,
)


def _require_probe_env() -> tuple[Path, str]:
    if os.getenv("OPENAI_COMPAT_PROBE_ALLOW_COST") != "1":
        pytest.skip("Set OPENAI_COMPAT_PROBE_ALLOW_COST=1 để cho phép real endpoint probe có thể tốn phí")

    raw_profile = os.getenv("OPENAI_COMPAT_PROFILE")
    if not raw_profile:
        pytest.skip("OPENAI_COMPAT_PROFILE chưa được set")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY chưa được set")

    profile_path = Path(raw_profile)
    if not profile_path.is_absolute():
        profile_path = PROJECT_ROOT / profile_path
    if not profile_path.exists():
        pytest.skip(f"OPENAI_COMPAT_PROFILE không tồn tại: {profile_path}")

    return profile_path, api_key


class TestLayer4_OpenAICompatRealEndpointProbe:
    """Probe thật, luôn opt-in bằng env để tránh tốn chi phí ngoài ý muốn."""

    def test_openai_compatible_profile_basic_generation_probe(self):
        profile_path, api_key = _require_probe_env()
        pytest.importorskip("openai", reason="openai package required for real endpoint probe")

        config: dict[str, Any] = load_provider_config(str(profile_path))
        model_override = os.getenv("OPENAI_COMPAT_PROBE_MODEL")
        if model_override:
            config["model"] = model_override

        profile = OpenAICompatProfile.from_config(config)
        provider = create_provider("openai", config, {"api_key": api_key})

        capabilities: dict[str, Any] = {}
        errors_sanitized: list[str] = []
        telemetry_summary: dict[str, Any] = {}
        status = PROBE_STATUS_ERROR

        try:
            result = provider.call("Return exactly this token and nothing else: OK")
            status = PROBE_STATUS_VERIFIED if result.strip() else PROBE_STATUS_ERROR
            capabilities[f"{profile.api_mode}_basic"] = {
                "status": status,
                "response_non_empty": bool(result.strip()),
            }
            last_telemetry = provider.last_telemetry_record or {}
            telemetry_summary = {
                "usage_seen": any(
                    key in last_telemetry
                    for key in ("input_tokens", "output_tokens", "total_tokens")
                ),
                "cache_headers_seen": "cache_headers" in last_telemetry or "cache_status" in last_telemetry,
                "cached_tokens_seen": last_telemetry.get("cached_tokens") is not None,
            }
        except Exception as exc:  # pragma: no cover - opt-in real endpoint path
            errors_sanitized.append(f"{type(exc).__name__}: {exc}")
            capabilities[f"{profile.api_mode}_basic"] = {
                "status": PROBE_STATUS_ERROR,
                "error_type": type(exc).__name__,
            }

        report_path = write_capability_report(
            profile_name=profile.profile_name,
            base_url=profile.base_url,
            model=profile.model,
            capabilities=capabilities,
            telemetry_summary=telemetry_summary,
            errors_sanitized=errors_sanitized,
        )

        assert report_path.exists()
        assert capabilities[f"{profile.api_mode}_basic"]["status"] == PROBE_STATUS_VERIFIED
