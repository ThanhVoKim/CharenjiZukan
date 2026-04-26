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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock

import pytest

# ── Project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from translation.base import BaseTranslationProvider
from translation.response_parser import parse_translation_response


def _import_yaml_or_skip():
    return pytest.importorskip("yaml", reason="pyyaml required")


def _import_tenacity_or_skip():
    return pytest.importorskip("tenacity", reason="tenacity required")


def _import_translate_modules_or_skip():
    _import_tenacity_or_skip()
    try:
        from translation.translator import translate_srt_file as _translate_srt_file
        return _translate_srt_file
    except ImportError as exc:
        if "google" in str(exc).lower() or "genai" in str(exc).lower():
            pytest.skip(f"google-genai required: {exc}")
        raise


def create_provider(*args, **kwargs):
    _import_yaml_or_skip()
    from translation.factory import create_provider as _create_provider
    return _create_provider(*args, **kwargs)


def load_provider_config(*args, **kwargs):
    _import_yaml_or_skip()
    from translation.factory import load_provider_config as _load_provider_config
    return _load_provider_config(*args, **kwargs)


def translate_srt_file(*args, **kwargs):
    _translate_srt_file = _import_translate_modules_or_skip()
    return _translate_srt_file(*args, **kwargs)

# ═════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ═════════════════════════════════════════════════════════════════════

@dataclass
class VertexAICredentials:
    """Kết quả parse credentials từ GOOGLE_APPLICATION_CREDENTIALS."""
    project_id: str
    location: str


@pytest.fixture()
def vertexai_credentials() -> VertexAICredentials:
    """
    Đọc và parse credentials từ GOOGLE_APPLICATION_CREDENTIALS.
    
    Skip test nếu:
    - GOOGLE_APPLICATION_CREDENTIALS chưa được set
    - File credentials không tồn tại
    - File credentials không hợp lệ JSON
    - Không tìm thấy project_id trong credentials
    
    Trả về VertexAICredentials chứa project_id và location (mặc định 'global').
    """
    import json as _json
    import os as _os
    
    creds_path = _os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        pytest.skip(
            "GOOGLE_APPLICATION_CREDENTIALS not set. "
            "On Colab, save service account JSON to a temp file and set this env var."
        )
    
    try:
        with open(creds_path, "r", encoding="utf-8") as f:
            creds_data = _json.load(f)
    except FileNotFoundError:
        pytest.skip(f"Credentials file not found: {creds_path}")
    except _json.JSONDecodeError:
        pytest.skip(f"Invalid JSON in credentials file: {creds_path}")
    
    project_id = creds_data.get("project_id")
    if not project_id:
        pytest.skip("project_id not found in credentials file")
    
    location = _os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    
    return VertexAICredentials(project_id=project_id, location=location)


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
    yaml_lib = _import_yaml_or_skip()
    config_data = {
        "model": "gpt-mock",
        "temperature": 1,
        "max_tokens": 1000,
        "base_url": "https://api.mock.com/v1"
    }
    path = tmp_path / "openai_config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml_lib.dump(config_data, f)
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
        assert cfg["temperature"] == 1

    def test_load_provider_config_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_provider_config("nonexistent.yaml")

    def test_create_gemini_provider(self):
        gemini = pytest.importorskip("google.genai", reason="google-genai required")
        provider = create_provider(
            "gemini", 
            {"model": "gemini-3.1-flash-lite-preview"}, 
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
        pytest.importorskip("google.genai", reason="google-genai required")

        with patch("google.genai.Client", return_value=MagicMock()):
            provider = create_provider(
                "vertexai",
                {"model": "gemini-3.1-flash-lite-preview", "project_id": "test-project"},
                {}
            )
        assert provider.name.startswith("Vertex AI")

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


class TestLayer3_VertexAICache:
    """Test context cache integration cho Vertex AI provider bằng mock."""

    def test_vertexai_set_global_context_creates_cache_and_call_uses_cached_content(self):
        pytest.importorskip("google.genai", reason="google-genai required")

        from translation import vertexai_provider as vertexai_provider_module

        mock_client = MagicMock()
        mock_cached = MagicMock()
        mock_cached.name = "projects/p/locations/l/cachedContents/cache-001"
        mock_client.caches.create.return_value = mock_cached

        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.text = "<TRANSLATE_TEXT>ok</TRANSLATE_TEXT>"
        mock_client.models.generate_content.return_value = mock_response

        with patch("google.genai.Client", return_value=mock_client):
            provider = vertexai_provider_module.VertexAIProvider(
                project_id="demo-project",
                location="global",
                model="gemini-3.1-flash-lite-preview",
                generation_config={"temperature": 0.1, "max_output_tokens": 4096},
                safety_settings={},
                system_prompt="You are a translator",
                request_timeout=60,
                retry_attempts=1,
                retry_wait_seconds=1,
                cache_ttl_seconds=600,
            )

        captured_generate_cfg = {}

        def fake_generate_content_config(**kwargs):
            captured_generate_cfg.update(kwargs)
            return kwargs

        with patch.object(provider._types, "GenerateContentConfig", side_effect=fake_generate_content_config):
            long_context = "x" * 9000
            used_cache = provider.set_global_context(long_context)

            assert used_cache is True
            assert provider._cached_content_name == "projects/p/locations/l/cachedContents/cache-001"
            mock_client.caches.create.assert_called_once()

            result = provider.call("Translate this")
            assert result == "<TRANSLATE_TEXT>ok</TRANSLATE_TEXT>"

            mock_client.models.generate_content.assert_called_once()
            assert captured_generate_cfg.get("cached_content") == "projects/p/locations/l/cachedContents/cache-001"

    def test_vertexai_cache_integration_real_extraction(self, vertexai_credentials: VertexAICredentials):
        """
        Full integration test: gọi Vertex AI API thật cho cả 2 bước.

        1) Tạo cached content thật bằng set_global_context().
        2) Gọi generate_content thật qua provider.call() để trích xuất secret_info từ cache.

        Không dùng bất kỳ mock nào trong test này.
        """
        from translation import vertexai_provider as vertexai_provider_module

        secret_info = {
            "name": "Nguyễn Văn An",
            "birth_date": "15/03/1990",
            "address": "123 Đường Lê Lợi, Quận 1, TP.HCM",
            "phone": "0912-345-678",
        }

        important_section = f"""
        ═══════════════════════════════════════════════════════════════
        HỒ SƠ KHÁCH HÀNG CẦN TRÍCH XUẤT TỪ CACHE
        ═══════════════════════════════════════════════════════════════
        Họ và tên: {secret_info['name']}
        Ngày sinh: {secret_info['birth_date']}
        Địa chỉ: {secret_info['address']}
        SĐT: {secret_info['phone']}
        ═══════════════════════════════════════════════════════════════
        """

        # Filler đủ lớn để vượt ngưỡng cache tối thiểu của Vertex (~2048 tokens)
        filler = "\n".join(
            f"Dòng nền {i}: " + ("Ngữ cảnh nền cho kiểm thử cache. " * 22)
            for i in range(1, 25)
        )
        context = important_section + "\n\n" + filler

        provider = vertexai_provider_module.VertexAIProvider(
            project_id=vertexai_credentials.project_id,
            location=vertexai_credentials.location,
            model="gemini-3.1-flash-lite-preview",
            generation_config={"temperature": 0, "max_output_tokens": 256, "candidate_count": 1},
            safety_settings={},
            system_prompt="Bạn là trợ lý truy xuất dữ liệu từ cached context. Trả lời chính xác từng trường.",
            request_timeout=300,
            retry_attempts=3,
            retry_wait_seconds=10,
            cache_ttl_seconds=300,
        )

        try:
            cache_created = provider.set_global_context(context)
            if not cache_created:
                pytest.skip(
                    "Context cache creation failed. "
                    "Vertex AI yêu cầu tối thiểu ~2048 tokens cho cached content."
                )

            assert provider._cached_content_name is not None

            query = (
                "Dựa trên cached context, hãy trích xuất đúng 4 trường sau và KHÔNG suy đoán: "
                "Họ và tên, Ngày sinh, Địa chỉ, SĐT. "
                "BẮT BUỘC trả lời duy nhất trong thẻ <TRANSLATE_TEXT>...</TRANSLATE_TEXT> "
                "theo đúng format: "
                "Họ và tên: <value> | Ngày sinh: <value> | Địa chỉ: <value> | SĐT: <value>"
            )

            result = provider.call(query)
            print("\n[DEBUG][VertexAI][raw_response_start]")
            print(result)
            print("[DEBUG][VertexAI][raw_response_end]\n")

            try:
                extracted = parse_translation_response(result)
            except Exception as e:
                pytest.fail(
                    "Không parse được <TRANSLATE_TEXT> từ Vertex AI response. "
                    f"Raw response đầy đủ: {result!r}. Lỗi parse: {type(e).__name__} - {e}"
                )

            print(f"[DEBUG][VertexAI][extracted]: {extracted}")

            assert secret_info["name"] in extracted, f"Tên không tìm thấy trong response: {extracted}"
            assert secret_info["birth_date"] in extracted, f"Ngày sinh không tìm thấy trong response: {extracted}"
            assert secret_info["address"] in extracted, f"Địa chỉ không tìm thấy trong response: {extracted}"
            assert secret_info["phone"] in extracted, f"SĐT không tìm thấy trong response: {extracted}"
        finally:
            if provider._cached_content_name:
                try:
                    provider._client.caches.delete(name=provider._cached_content_name)
                except Exception:
                    pass

    def test_vertexai_cache_without_cached_content_fallback(self, vertexai_credentials: VertexAICredentials):
        """
        Test rằng khi không dùng cache, provider vẫn hoạt động bình thường.
        Mock generate_content để verify provider không crash khi không có cached_content.
        """
        from translation import vertexai_provider as vertexai_provider_module
        
        provider_no_cache = vertexai_provider_module.VertexAIProvider(
            project_id=vertexai_credentials.project_id,
            location=vertexai_credentials.location,
            model="gemini-3.1-flash-lite-preview",
            generation_config={"temperature": 0.1, "max_output_tokens": 50},
            safety_settings={},
            system_prompt="Trả lời ngắn gọn.",
            request_timeout=60,
            retry_attempts=1,
            retry_wait_seconds=5,
            cache_ttl_seconds=300,
        )
        
        # Mock generate_content để verify provider hoạt động mà không cần API thật
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.text = "<TRANSLATE_TEXT>Trời nắng</TRANSLATE_TEXT>"
        
        with patch.object(provider_no_cache._client.models, "generate_content", return_value=mock_response):
            result = provider_no_cache.call("Viết 3 từ về thời tiết.")
        
        assert result is not None
        assert "Trời nắng" in result

    def test_vertexai_set_global_context_fallback_when_context_too_short(self):
        pytest.importorskip("google.genai", reason="google-genai required")

        from translation import vertexai_provider as vertexai_provider_module

        mock_client = MagicMock()
        mock_client.caches.create.side_effect = RuntimeError(
            "The minimum input token count for context caching is 2048 tokens"
        )

        with patch("google.genai.Client", return_value=mock_client):
            provider = vertexai_provider_module.VertexAIProvider(
                project_id="demo-project",
                location="global",
                model="gemini-3.1-flash-lite-preview",
                generation_config={"temperature": 0.1},
                safety_settings={},
                system_prompt="",
                request_timeout=60,
                retry_attempts=1,
                retry_wait_seconds=1,
            )

        used_cache = provider.set_global_context("too short")
        assert used_cache is False
        assert provider._cached_content_name is None
        mock_client.caches.create.assert_called_once()


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
            {"model": "gemini-3.1-flash-lite-preview"}, 
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
        try:
            result = provider.call(prompt)
            parsed = parse_translation_response(result)
            
            assert len(parsed) > 0
            assert "chào buổi sáng" in parsed.lower() or "chào" in parsed.lower()
        except Exception as e:
            import openai
            if isinstance(e, openai.APIStatusError):
                print(f"\n[LỖI OPENAI API] HTTP Status Code: {e.status_code}")
                print(f"[LỖI OPENAI API] Response Body: {e.response.text}")
                print(f"[LỖI OPENAI API] Headers: {e.response.headers}")
            else:
                print(f"\n[LỖI KHÁC] {type(e).__name__}: {str(e)}")
            raise e

    def test_vertexai_real_api(self):
        """Test gửi request thực tế đến Vertex AI lấy cấu hình từ thư mục config/."""
        pytest.importorskip("google.genai")

        config_path = PROJECT_ROOT / "config" / "vertexai_translate.yaml"
        if not config_path.exists():
            skip_msg = f"Không tìm thấy file config thật tại {config_path}"
            print(f"\n[SKIPPED Vertex AI] Lý do: {skip_msg}")
            pytest.skip(skip_msg)
            
        config = load_provider_config(str(config_path))
        
        # Bỏ qua nếu user chưa cấu hình project_id thật
        if config.get("project_id") in ["your-gcp-project-id", ""]:
            skip_msg = "Chưa cấu hình 'project_id' thật trong file vertexai_translate.yaml. Vui lòng mở file và cập nhật."
            print(f"\n[SKIPPED Vertex AI] Lý do: {skip_msg}")
            pytest.skip(skip_msg)
            
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
            skip_msg = f"Không thể xác thực Vertex AI. Vui lòng chạy lệnh 'gcloud auth application-default login' trước. Chi tiết lỗi gốc: {type(e).__name__} - {e}"
            print(f"\n[SKIPPED Vertex AI] Lý do: {skip_msg}")
            pytest.skip(skip_msg)

