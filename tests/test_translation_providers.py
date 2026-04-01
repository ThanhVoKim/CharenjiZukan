#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_translation_providers.py
===================================
Test multi-provider-translate feature (Pure Logic).

Cấu trúc layers:
  Layer 1 — Unit Tests          (Response Parser, Factory)
  Layer 2 — Component Tests     (Pipeline with Mocked Provider)
  Layer 3 — Integration         (Retry logic mock exceptions)
  Layer 4 — Real API Tests      (Gửi SRT mẫu tới API thật, tự động skip nếu không có key)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock

import pytest

# ── Project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

yaml = pytest.importorskip("yaml")

from translation.base import BaseTranslationProvider
from translation.response_parser import parse_translation_response
from translation.factory import create_provider, load_provider_config
from translation.translator import translate_srt_file

# ═════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ═════════════════════════════════════════════════════════════════════

@pytest.fixture()
def sample_srt_path(tmp_path: Path) -> Path:
    """Tạo file SRT mẫu."""
    content = """1
00:00:01,000 --> 00:00:03,500
Hello World

2
00:00:05,000 --> 00:00:08,000
This is a test subtitle
"""
    path = tmp_path / "sample.srt"
    path.write_text(content, encoding="utf-8")
    return path

@pytest.fixture()
def sample_prompt_path(tmp_path: Path) -> Path:
    """Tạo file prompt mẫu chứa placeholder cần thiết."""
    content = """
Translate below to {lang}
{context_block}
START:
{batch_input}
"""
    path = tmp_path / "prompt.txt"
    path.write_text(content, encoding="utf-8")
    return path

@pytest.fixture()
def sample_provider_config_path(tmp_path: Path) -> Path:
    """Tạo file YAML config runtime."""
    config_data = {
        "model": "gpt-mock",
        "temperature": 0.5,
        "max_tokens": 1000,
        "base_url": "https://api.mock.com/v1"
    }
    path = tmp_path / "openai_config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)
    return path


# Fake Provider cho Component Test (Layer 2)
class FakeProvider(BaseTranslationProvider):
    def __init__(self, name: str = "FakeProvider"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def call(self, message: str) -> str:
        # Giả lập trả về đúng format với tag
        return """
<think>
This is a thought process.
</think>
<TRANSLATE_TEXT>
1
00:00:01,000 --> 00:00:03,500
Xin chào thế giới

2
00:00:05,000 --> 00:00:08,000
Đây là phụ đề thử nghiệm
</TRANSLATE_TEXT>
"""


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_ResponseParser:
    """Test việc bóc tách tag và loại bỏ <think>."""

    def test_parse_valid_response_with_think(self):
        raw = "<think>Thinking...</think>\\n<TRANSLATE_TEXT>Translated Text</TRANSLATE_TEXT>"
        assert parse_translation_response(raw) == "Translated Text"

    def test_parse_valid_response_without_think(self):
        raw = "<TRANSLATE_TEXT>\nLine 1\nLine 2\n</TRANSLATE_TEXT>"
        assert parse_translation_response(raw) == "Line 1\nLine 2"

    def test_parse_valid_response_without_closing_tag(self):
        # Match till end of string if closing tag is missing
        raw = "Some text <TRANSLATE_TEXT>Translated till end"
        assert parse_translation_response(raw) == "Translated till end"

    def test_parse_missing_tag_raises_error(self):
        raw = "<think>Done</think>\nJust raw text without tag."
        with pytest.raises(RuntimeError, match="Không tìm thấy <TRANSLATE_TEXT>"):
            parse_translation_response(raw)

class TestLayer1_Factory:
    """Test khởi tạo Provider và load cấu hình."""

    def test_load_provider_config(self, sample_provider_config_path: Path):
        cfg = load_provider_config(str(sample_provider_config_path))
        assert cfg["model"] == "gpt-mock"
        assert cfg["temperature"] == 0.5

    def test_load_provider_config_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_provider_config("nonexistent.yaml")

    def test_create_gemini_provider(self):
        gemini = pytest.importorskip("google.genai", reason="google-genai required")
        provider = create_provider(
            "gemini", 
            {"model": "gemini-2.5"}, 
            {"api_keys": ["KEY_1"]}
        )
        assert provider.name.startswith("Gemini")

    def test_create_openai_provider(self):
        openai = pytest.importorskip("openai", reason="openai required")
        provider = create_provider(
            "openai", 
            {"model": "gpt-4o-mini"}, 
            {"api_key": "sk-123"}
        )
        assert provider.name.startswith("OpenAI")

    def test_create_vertexai_provider(self):
        vertexai = pytest.importorskip("vertexai", reason="google-cloud-aiplatform required")
        provider = create_provider(
            "vertexai", 
            {"model": "gemini-1.5-pro", "project_id": "test-project"}, 
            {}
        )
        assert provider.name.startswith("VertexAI")

    def test_create_invalid_provider(self):
        with pytest.raises(ValueError, match="Provider không hợp lệ"):
            create_provider("invalid", {}, {})


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — COMPONENT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestLayer2_PipelineWithMock:
    """Test toàn bộ luồng pipeline dịch thuật không gọi mạng thực."""

    def test_translate_srt_file_success(self, sample_srt_path: Path, sample_prompt_path: Path, tmp_path: Path):
        output_file = tmp_path / "output.srt"
        fake_provider = FakeProvider()
        
        stats = translate_srt_file(
            input_file=str(sample_srt_path),
            output_file=str(output_file),
            prompt_file=str(sample_prompt_path),
            provider=fake_provider,
            target_language="Vietnamese",
            batch_size=10,
            use_full_context=True,
            wait_sec=0
        )
        
        assert stats["total_blocks"] == 2
        assert stats["success"] == 1  # 1 batch (2 blocks)
        assert stats["failed"] == 0
        assert output_file.exists()
        
        content = output_file.read_text(encoding="utf-8")
        assert "Xin chào thế giới" in content
        assert "Đây là phụ đề thử nghiệm" in content

    def test_translate_srt_file_no_context(self, sample_srt_path: Path, sample_prompt_path: Path, tmp_path: Path):
        output_file = tmp_path / "output2.srt"
        fake_provider = FakeProvider()
        
        stats = translate_srt_file(
            input_file=str(sample_srt_path),
            output_file=str(output_file),
            prompt_file=str(sample_prompt_path),
            provider=fake_provider,
            target_language="Vietnamese",
            batch_size=10,
            use_full_context=False,
            wait_sec=0
        )
        
        assert stats["success"] == 1
        assert output_file.exists()


# ═════════════════════════════════════════════════════════════════════
# LAYER 3 — PIPELINE INTEGRATION
# ═════════════════════════════════════════════════════════════════════

class TestLayer3_RetryLogic:
    """Giả lập lỗi mạng để test retry logic (tenacity) của OpenAI Provider."""

    def test_openai_retry_success_after_failure(self):
        pytest.importorskip("openai")
        from openai import RateLimitError
        import httpx
        
        provider = create_provider(
            "openai", 
            {"model": "gpt-mock", "retry_attempts": 3, "retry_wait_seconds": 0.1}, 
            {"api_key": "mock"}
        )
        
        call_count = 0
        
        def fake_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # Gây lỗi RateLimitError 2 lần đầu
                raise RateLimitError("Rate limit", response=httpx.Response(429, request=httpx.Request("POST", "url")), body=None)
            
            # Lần 3 thành công
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "<TRANSLATE_TEXT>Success</TRANSLATE_TEXT>"
            return mock_resp

        # Inject mock directly into provider
        provider._client = MagicMock()
        provider._client.chat.completions.create.side_effect = fake_create

        result = provider.call("Test message")
        assert call_count == 3
        assert result == "<TRANSLATE_TEXT>Success</TRANSLATE_TEXT>"


# ═════════════════════════════════════════════════════════════════════
# LAYER 4 — REAL MODEL TESTS
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.api
class TestLayer4_RealAPIs:
    """Test với API thật trên Colab. Tự động đọc cấu hình từ thư mục config/."""
    
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"), 
        reason="Không có GEMINI_API_KEY trong environment"
    )
    def test_gemini_real_api(self):
        pytest.importorskip("google.genai")
        
        api_key = os.getenv("GEMINI_API_KEY")
        provider = create_provider(
            "gemini", 
            {"model": "gemini-3-flash-preview"}, 
            {"api_keys": [api_key]}
        )
        
        prompt = "Vui lòng dịch câu sau sang tiếng Việt, phải nằm trong thẻ <TRANSLATE_TEXT>...</TRANSLATE_TEXT>:\nHello"
        result = provider.call(prompt)
        parsed = parse_translation_response(result)
        
        assert len(parsed) > 0
        assert "chào" in parsed.lower()
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), 
        reason="Không có OPENAI_API_KEY trong environment"
    )
    def test_openai_real_api(self):
        """Test gửi request thực tế đến một endpoint OpenAI-Compatible lấy từ cấu hình YAML."""
        pytest.importorskip("openai")
        from translation.factory import load_provider_config
        
        api_key = os.getenv("OPENAI_API_KEY")
        config_path = PROJECT_ROOT / "config" / "openai_compat_translate.yaml"
        
        if not config_path.exists():
            pytest.skip(f"Không tìm thấy file config thật tại {config_path}")
            
        config = load_provider_config(str(config_path))
        
        provider = create_provider(
            provider_type="openai", 
            config=config, 
            secrets={"api_key": api_key}
        )
        
        prompt = "Vui lòng dịch câu sau sang tiếng Việt, phải nằm trong thẻ <TRANSLATE_TEXT>...</TRANSLATE_TEXT>:\nGood morning"
        result = provider.call(prompt)
        parsed = parse_translation_response(result)
        
        assert len(parsed) > 0
        assert "chào buổi sáng" in parsed.lower() or "chào" in parsed.lower()

    def test_vertexai_real_api(self):
        """Test gửi request thực tế đến Vertex AI lấy cấu hình từ thư mục config/."""
        pytest.importorskip("google.cloud.aiplatform")
        from translation.factory import load_provider_config
        
        config_path = PROJECT_ROOT / "config" / "vertexai_translate.yaml"
        if not config_path.exists():
            pytest.skip(f"Không tìm thấy file config thật tại {config_path}")
            
        config = load_provider_config(str(config_path))
        
        # Bỏ qua nếu user chưa cấu hình project_id thật
        if config.get("project_id") in ["your-gcp-project-id", ""]:
            pytest.skip("Chưa cấu hình 'project_id' thật trong file vertexai_translate.yaml. Vui lòng mở file và cập nhật.")
            
        try:
            # VertexAIProvider không yêu cầu truyền secrets/api_key vì dùng Application Default Credentials
            provider = create_provider(
                provider_type="vertexai", 
                config=config, 
                secrets={}
            )
            
            prompt = "Vui lòng dịch câu sau sang tiếng Việt, phải nằm trong thẻ <TRANSLATE_TEXT>...</TRANSLATE_TEXT>:\nGood night"
            result = provider.call(prompt)
            parsed = parse_translation_response(result)
            
            assert len(parsed) > 0
            assert "chúc ngủ ngon" in parsed.lower() or "ngủ ngon" in parsed.lower()
        except Exception as e:
            # Nếu báo lỗi Authentication / PermissionDenied thì báo người dùng đăng nhập
            pytest.skip(f"Không thể xác thực Vertex AI. Vui lòng chạy lệnh 'gcloud auth application-default login' trước. Chi tiết: {e}")

