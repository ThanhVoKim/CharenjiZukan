#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/sync_engine/test_forced_alignment_subtitle.py
====================================================
Mock integration test cho sync_engine/forced_alignment_subtitle.py.

Layer 1: Unit tests — config resolution, fail policy, entry point logic.
Layer 2: Component tests — mock aligner, verify SRT output end-to-end.

Không cần GPU, không cần FFmpeg.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import tempfile
import os

# ═════════════════════════════════════════════════════════════════════
# Layer 1: Unit Tests — Config Resolution & Entry Point Logic
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.Layer1
class TestResolveAlignerConfig:
    """Test _resolve_aligner_config() map JSON config sang function params."""

    def test_defaults_from_empty_config(self):
        """Config rỗng → tất cả dùng default."""
        from sync_engine.forced_alignment_subtitle import _resolve_aligner_config

        result = _resolve_aligner_config({})
        assert result["model_path"] is None
        assert result["device"] is None
        assert result["dtype"] is None
        assert result["attn_implementation"] is None
        assert result["language"] == "English"
        assert result["max_chars"] == 42
        assert result["min_chars"] == 0
        assert result["split_on_comma"] is True
        assert result["offset_seconds"] == 0.24
        assert result["keep_tts_synced_debug"] is False
        assert result["fail_policy"] == "warn"

    def test_null_model_keys_pass_through(self):
        """Các key model null trong JSON → None (không override default hàm)."""
        from sync_engine.forced_alignment_subtitle import _resolve_aligner_config

        cfg = {
            "model_path": None,
            "device": None,
            "dtype": None,
            "attn_implementation": None,
        }
        result = _resolve_aligner_config(cfg)
        assert result["model_path"] is None
        assert result["device"] is None
        assert result["dtype"] is None
        assert result["attn_implementation"] is None

    def test_custom_values_override_defaults(self):
        """Giá trị custom trong JSON override default."""
        from sync_engine.forced_alignment_subtitle import _resolve_aligner_config

        cfg = {
            "model_path": "custom/model-1B",
            "device": "cuda:1",
            "dtype": "float16",
            "attn_implementation": "sdpa",
            "language": "Japanese",
            "max_chars": 30,
            "min_chars": 5,
            "split_on_comma": False,
            "offset_seconds": 0.1,
            "keep_tts_synced_debug": True,
            "fail_policy": "raise",
        }
        result = _resolve_aligner_config(cfg)
        assert result["model_path"] == "custom/model-1B"
        assert result["device"] == "cuda:1"
        assert result["dtype"] == "float16"
        assert result["attn_implementation"] == "sdpa"
        assert result["language"] == "Japanese"
        assert result["max_chars"] == 30
        assert result["min_chars"] == 5
        assert result["split_on_comma"] is False
        assert result["offset_seconds"] == 0.1
        assert result["keep_tts_synced_debug"] is True
        assert result["fail_policy"] == "raise"

    def test_partial_config_keeps_defaults(self):
        """Config chỉ override 1 số key, các key còn lại giữ default."""
        from sync_engine.forced_alignment_subtitle import _resolve_aligner_config

        cfg = {"language": "Chinese", "max_chars": 50}
        result = _resolve_aligner_config(cfg)
        assert result["language"] == "Chinese"
        assert result["max_chars"] == 50
        assert result["model_path"] is None
        assert result["fail_policy"] == "warn"


@pytest.mark.Layer1
class TestRunForcedAlignmentSubtitleEntryPoint:
    """Test run_forced_alignment_subtitle() entry point logic."""

    def test_disabled_returns_none(self):
        """enabled=false → return None, không gọi aligner."""
        from sync_engine.forced_alignment_subtitle import run_forced_alignment_subtitle

        cfg = {"forced_alignment_subtitle": {"enabled": False}}
        result = run_forced_alignment_subtitle(
            audio_path="/tmp/audio.wav",
            transcript_path="/tmp/text.txt",
            output_srt_path="/tmp/out.srt",
            render_config=cfg,
        )
        assert result is None

    def test_missing_config_returns_none(self):
        """Không có block forced_alignment_subtitle → return None."""
        from sync_engine.forced_alignment_subtitle import run_forced_alignment_subtitle

        result = run_forced_alignment_subtitle(
            audio_path="/tmp/audio.wav",
            transcript_path="/tmp/text.txt",
            output_srt_path="/tmp/out.srt",
            render_config={},
        )
        assert result is None

    def test_null_config_returns_none(self):
        """forced_alignment_subtitle: null → return None."""
        from sync_engine.forced_alignment_subtitle import run_forced_alignment_subtitle

        result = run_forced_alignment_subtitle(
            audio_path="/tmp/audio.wav",
            transcript_path="/tmp/text.txt",
            output_srt_path="/tmp/out.srt",
            render_config={"forced_alignment_subtitle": None},
        )
        assert result is None

    def test_fail_policy_warn_returns_none_on_error(self):
        """fail_policy=warn → exception bị catch, return None."""
        from sync_engine.forced_alignment_subtitle import run_forced_alignment_subtitle

        cfg = {
            "forced_alignment_subtitle": {
                "enabled": True,
                "fail_policy": "warn",
            }
        }
        with patch(
            "sync_engine.forced_alignment_subtitle.execute_forced_alignment",
            side_effect=RuntimeError("align failed"),
        ):
            result = run_forced_alignment_subtitle(
                audio_path="/tmp/audio.wav",
                transcript_path="/tmp/text.txt",
                output_srt_path="/tmp/out.srt",
                render_config=cfg,
            )
            assert result is None

    def test_fail_policy_raise_propagates_exception(self):
        """fail_policy=raise → exception không bị catch."""
        from sync_engine.forced_alignment_subtitle import run_forced_alignment_subtitle

        cfg = {
            "forced_alignment_subtitle": {
                "enabled": True,
                "fail_policy": "raise",
            }
        }
        with patch(
            "sync_engine.forced_alignment_subtitle.execute_forced_alignment",
            side_effect=RuntimeError("align failed"),
        ):
            with pytest.raises(RuntimeError, match="align failed"):
                run_forced_alignment_subtitle(
                    audio_path="/tmp/audio.wav",
                    transcript_path="/tmp/text.txt",
                    output_srt_path="/tmp/out.srt",
                    render_config=cfg,
                )

    def test_fail_policy_error_propagates_exception(self):
        """fail_policy=error → exception không bị catch."""
        from sync_engine.forced_alignment_subtitle import run_forced_alignment_subtitle

        cfg = {
            "forced_alignment_subtitle": {
                "enabled": True,
                "fail_policy": "error",
            }
        }
        with patch(
            "sync_engine.forced_alignment_subtitle.execute_forced_alignment",
            side_effect=FileNotFoundError("model not found"),
        ):
            with pytest.raises(FileNotFoundError, match="model not found"):
                run_forced_alignment_subtitle(
                    audio_path="/tmp/audio.wav",
                    transcript_path="/tmp/text.txt",
                    output_srt_path="/tmp/out.srt",
                    render_config=cfg,
                )

    def test_fail_policy_fail_propagates_exception(self):
        """fail_policy=fail → exception không bị catch."""
        from sync_engine.forced_alignment_subtitle import run_forced_alignment_subtitle

        cfg = {
            "forced_alignment_subtitle": {
                "enabled": True,
                "fail_policy": "fail",
            }
        }
        with patch(
            "sync_engine.forced_alignment_subtitle.execute_forced_alignment",
            side_effect=ValueError("empty transcript"),
        ):
            with pytest.raises(ValueError, match="empty transcript"):
                run_forced_alignment_subtitle(
                    audio_path="/tmp/audio.wav",
                    transcript_path="/tmp/text.txt",
                    output_srt_path="/tmp/out.srt",
                    render_config=cfg,
                )

    def test_success_returns_stats(self):
        """Alignment thành công → return dict stats."""
        from sync_engine.forced_alignment_subtitle import run_forced_alignment_subtitle

        cfg = {
            "forced_alignment_subtitle": {
                "enabled": True,
                "fail_policy": "warn",
            }
        }
        expected_stats = {"subtitle_blocks": 5, "total_words": 30}
        with patch(
            "sync_engine.forced_alignment_subtitle.execute_forced_alignment",
            return_value=expected_stats,
        ):
            result = run_forced_alignment_subtitle(
                audio_path="/tmp/audio.wav",
                transcript_path="/tmp/text.txt",
                output_srt_path="/tmp/out.srt",
                render_config=cfg,
            )
            assert result == expected_stats


# ═════════════════════════════════════════════════════════════════════
# Layer 2: Component Tests — Mock Aligner End-to-End
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.Layer2
class TestExecuteForcedAlignmentMocked:
    """Test execute_forced_alignment() với mock Qwen3ForcedAligner."""

    @staticmethod
    def _make_fake_align_items():
        """Tạo fake ForcedAlignItem objects (dùng dict để mock)."""
        # Mô phỏng kết quả align: mỗi item có .text, .start_time, .end_time
        items = []
        words = [
            ("Hello", 0.0, 0.5),
            (",", 0.5, 0.5),
            ("world", 0.5, 1.0),
            ("!", 1.0, 1.0),
            ("This", 1.0, 1.3),
            ("is", 1.3, 1.5),
            ("a", 1.5, 1.6),
            ("test", 1.6, 2.0),
            (".", 2.0, 2.0),
        ]
        for text, start, end in words:
            item = MagicMock()
            item.text = text
            item.start_time = start
            item.end_time = end
            items.append(item)
        return items

    @staticmethod
    def _make_render_config_enabled(**overrides):
        """Tạo render_config với forced_alignment_subtitle enabled."""
        fa_cfg = {
            "enabled": True,
            "model_path": None,
            "device": None,
            "dtype": None,
            "attn_implementation": None,
            "language": "English",
            "max_chars": 42,
            "min_chars": 0,
            "split_on_comma": True,
            "offset_seconds": 0.24,
            "keep_tts_synced_debug": False,
            "fail_policy": "warn",
        }
        fa_cfg.update(overrides)
        return {"forced_alignment_subtitle": fa_cfg}

    def test_end_to_end_produces_srt(self, tmp_path):
        """Mock aligner → execute_forced_alignment → SRT file tồn tại và hợp lệ."""
        from sync_engine.forced_alignment_subtitle import run_forced_alignment_subtitle

        # Chuẩn bị input files
        transcript = tmp_path / "flat_transcript.txt"
        transcript.write_text("Hello, world! This is a test.", encoding="utf-8")

        audio = tmp_path / "mixed_audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 100)  # fake wav header

        output_srt = tmp_path / "output_synced.srt"

        cfg = self._make_render_config_enabled()

        fake_items = self._make_fake_align_items()

        # Mock: load_forced_aligner → trả về mock aligner
        mock_aligner = MagicMock()
        mock_aligner.align.return_value = [fake_items]

        with patch(
            "sync_engine.forced_alignment_subtitle.load_forced_aligner",
            return_value=mock_aligner,
        ), patch("utils.media_utils.clear_vram"):
            result = run_forced_alignment_subtitle(
                audio_path=str(audio),
                transcript_path=str(transcript),
                output_srt_path=str(output_srt),
                render_config=cfg,
            )

        # Verify SRT file tồn tại
        assert output_srt.exists(), "SRT file phải được tạo"

        # Verify stats
        assert result is not None
        assert "subtitle_blocks" in result
        assert "total_words" in result
        assert result["total_words"] > 0

        # Verify SRT content có format hợp lệ (ít nhất 1 block)
        srt_content = output_srt.read_text(encoding="utf-8")
        assert "-->" in srt_content, "SRT phải chứa timestamp separator"
        # Block đầu tiên phải bắt đầu bằng "1"
        assert srt_content.startswith("1\n"), "SRT block đầu tiên phải đánh số 1"

    def test_missing_transcript_raises_file_not_found(self, tmp_path):
        """Transcript file không tồn tại → FileNotFoundError."""
        from sync_engine.forced_alignment_subtitle import execute_forced_alignment

        align_cfg = {
            "language": "English",
            "max_chars": 42,
            "min_chars": 0,
            "split_on_comma": True,
            "offset_seconds": 0.24,
        }

        with pytest.raises(FileNotFoundError, match="transcript"):
            execute_forced_alignment(
                audio_path=str(tmp_path / "audio.wav"),
                transcript_path=str(tmp_path / "nonexistent.txt"),
                output_srt_path=str(tmp_path / "out.srt"),
                align_cfg=align_cfg,
            )

    def test_empty_transcript_raises_value_error(self, tmp_path):
        """Transcript rỗng → ValueError."""
        from sync_engine.forced_alignment_subtitle import execute_forced_alignment

        # Tạo file rỗng
        empty_transcript = tmp_path / "empty.txt"
        empty_transcript.write_text("", encoding="utf-8")

        align_cfg = {
            "language": "English",
            "max_chars": 42,
            "min_chars": 0,
            "split_on_comma": True,
            "offset_seconds": 0.24,
        }

        with pytest.raises(ValueError, match="rỗng"):
            execute_forced_alignment(
                audio_path=str(tmp_path / "audio.wav"),
                transcript_path=str(empty_transcript),
                output_srt_path=str(tmp_path / "out.srt"),
                align_cfg=align_cfg,
            )

    def test_aligner_returns_empty_raises_value_error(self, tmp_path):
        """Aligner trả về kết quả rỗng → ValueError."""
        from sync_engine.forced_alignment_subtitle import execute_forced_alignment

        transcript = tmp_path / "text.txt"
        transcript.write_text("Some text here.", encoding="utf-8")

        align_cfg = {
            "language": "English",
            "max_chars": 42,
            "min_chars": 0,
            "split_on_comma": True,
            "offset_seconds": 0.24,
        }

        mock_aligner = MagicMock()
        mock_aligner.align.return_value = [[]]  # empty results

        with patch(
            "sync_engine.forced_alignment_subtitle.load_forced_aligner",
            return_value=mock_aligner,
        ), patch("utils.media_utils.clear_vram"):
            with pytest.raises(ValueError, match="không trả về kết quả"):
                execute_forced_alignment(
                    audio_path=str(tmp_path / "audio.wav"),
                    transcript_path=str(transcript),
                    output_srt_path=str(tmp_path / "out.srt"),
                    align_cfg=align_cfg,
                )

    def test_aligner_returns_none_raises_value_error(self, tmp_path):
        """Aligner trả về None → ValueError."""
        from sync_engine.forced_alignment_subtitle import execute_forced_alignment

        transcript = tmp_path / "text.txt"
        transcript.write_text("Some text here.", encoding="utf-8")

        align_cfg = {
            "language": "English",
            "max_chars": 42,
            "min_chars": 0,
            "split_on_comma": True,
            "offset_seconds": 0.24,
        }

        mock_aligner = MagicMock()
        mock_aligner.align.return_value = None

        with patch(
            "sync_engine.forced_alignment_subtitle.load_forced_aligner",
            return_value=mock_aligner,
        ), patch("utils.media_utils.clear_vram"):
            with pytest.raises(ValueError, match="không trả về kết quả"):
                execute_forced_alignment(
                    audio_path=str(tmp_path / "audio.wav"),
                    transcript_path=str(transcript),
                    output_srt_path=str(tmp_path / "out.srt"),
                    align_cfg=align_cfg,
                )

    def test_vram_cleanup_after_alignment(self, tmp_path):
        """Sau alignment, clear_vram phải được gọi."""
        from sync_engine.forced_alignment_subtitle import execute_forced_alignment

        transcript = tmp_path / "text.txt"
        transcript.write_text("Hello world.", encoding="utf-8")

        align_cfg = {
            "language": "English",
            "max_chars": 42,
            "min_chars": 0,
            "split_on_comma": True,
            "offset_seconds": 0.24,
        }

        fake_items = self._make_fake_align_items()
        mock_aligner = MagicMock()
        mock_aligner.align.return_value = [fake_items]

        with patch(
            "sync_engine.forced_alignment_subtitle.load_forced_aligner",
            return_value=mock_aligner,
        ) as mock_load, patch(
            "utils.media_utils.clear_vram"
        ) as mock_clear:
            execute_forced_alignment(
                audio_path=str(tmp_path / "audio.wav"),
                transcript_path=str(transcript),
                output_srt_path=str(tmp_path / "out.srt"),
                align_cfg=align_cfg,
            )

        # Verify aligner was deleted (del aligner) and clear_vram called
        mock_clear.assert_called_once()

    def test_custom_segmentation_params(self, tmp_path):
        """Custom max_chars, min_chars, split_on_comma được truyền đúng."""
        from sync_engine.forced_alignment_subtitle import run_forced_alignment_subtitle

        transcript = tmp_path / "flat_transcript.txt"
        transcript.write_text("Hello, world! This is a test.", encoding="utf-8")

        audio = tmp_path / "mixed_audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 100)

        output_srt = tmp_path / "output_synced.srt"

        cfg = self._make_render_config_enabled(
            max_chars=20,
            min_chars=5,
            split_on_comma=False,
            offset_seconds=0.1,
        )

        fake_items = self._make_fake_align_items()
        mock_aligner = MagicMock()
        mock_aligner.align.return_value = [fake_items]

        with patch(
            "sync_engine.forced_alignment_subtitle.load_forced_aligner",
            return_value=mock_aligner,
        ), patch("utils.media_utils.clear_vram"), patch(
            "sync_engine.forced_alignment_subtitle.segment_words_to_subtitles"
        ) as mock_segment:
            mock_segment.return_value = [
                {
                    "index": 1,
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "text": "Hello, world!",
                }
            ]
            run_forced_alignment_subtitle(
                audio_path=str(audio),
                transcript_path=str(transcript),
                output_srt_path=str(output_srt),
                render_config=cfg,
            )

        # Verify segment_words_to_subtitles được gọi với đúng params
        mock_segment.assert_called_once()
        call_kwargs = mock_segment.call_args
        assert call_kwargs.kwargs.get("max_chars") == 20
        assert call_kwargs.kwargs.get("min_chars") == 5
        assert call_kwargs.kwargs.get("split_on_comma") is False


@pytest.mark.Layer1
class TestLoadForcedAligner:
    """Test load_forced_aligner() — mock torch và Qwen3ForcedAligner."""

    def test_default_params(self):
        """Default params → gọi from_pretrained với bfloat16, cuda:0."""
        mock_qwen_cls = MagicMock()
        mock_model = MagicMock()
        mock_qwen_cls.from_pretrained.return_value = mock_model

        mock_torch = MagicMock()
        mock_torch.bfloat16 = "bfloat16"

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "qwen_asr": MagicMock(Qwen3ForcedAligner=mock_qwen_cls),
        }):
            from sync_engine.forced_alignment_subtitle import load_forced_aligner

            result = load_forced_aligner()

        mock_qwen_cls.from_pretrained.assert_called_once_with(
            "Qwen/Qwen3-ForcedAligner-0.6B",
            dtype="bfloat16",
            device_map="cuda:0",
        )
        assert result == mock_model

    def test_custom_dtype_name(self):
        """dtype_name='float16' → gọi với torch.float16."""
        mock_qwen_cls = MagicMock()
        mock_model = MagicMock()
        mock_qwen_cls.from_pretrained.return_value = mock_model

        mock_torch = MagicMock()
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "qwen_asr": MagicMock(Qwen3ForcedAligner=mock_qwen_cls),
        }):
            from sync_engine.forced_alignment_subtitle import load_forced_aligner

            result = load_forced_aligner(dtype_name="float16")

        call_kwargs = mock_qwen_cls.from_pretrained.call_args
        assert call_kwargs.kwargs.get("dtype") == "float16" or call_kwargs[1].get("dtype") == "float16"

    def test_custom_device(self):
        """device_map='cuda:1' → truyền đúng."""
        mock_qwen_cls = MagicMock()
        mock_model = MagicMock()
        mock_qwen_cls.from_pretrained.return_value = mock_model

        mock_torch = MagicMock()
        mock_torch.bfloat16 = "bfloat16"

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "qwen_asr": MagicMock(Qwen3ForcedAligner=mock_qwen_cls),
        }):
            from sync_engine.forced_alignment_subtitle import load_forced_aligner

            load_forced_aligner(device_map="cuda:1")

        call_kwargs = mock_qwen_cls.from_pretrained.call_args
        assert call_kwargs.kwargs.get("device_map") == "cuda:1" or call_kwargs[1].get("device_map") == "cuda:1"

    def test_attn_implementation_included_when_not_none(self):
        """attn_implementation không None → truyền vào kwargs."""
        mock_qwen_cls = MagicMock()
        mock_model = MagicMock()
        mock_qwen_cls.from_pretrained.return_value = mock_model

        mock_torch = MagicMock()
        mock_torch.bfloat16 = "bfloat16"

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "qwen_asr": MagicMock(Qwen3ForcedAligner=mock_qwen_cls),
        }):
            from sync_engine.forced_alignment_subtitle import load_forced_aligner

            load_forced_aligner(attn_implementation="sdpa")

        call_kwargs = mock_qwen_cls.from_pretrained.call_args
        assert call_kwargs.kwargs.get("attn_implementation") == "sdpa" or call_kwargs[1].get("attn_implementation") == "sdpa"

    def test_attn_implementation_omitted_when_none(self):
        """attn_implementation=None → không truyền key này."""
        mock_qwen_cls = MagicMock()
        mock_model = MagicMock()
        mock_qwen_cls.from_pretrained.return_value = mock_model

        mock_torch = MagicMock()
        mock_torch.bfloat16 = "bfloat16"

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "qwen_asr": MagicMock(Qwen3ForcedAligner=mock_qwen_cls),
        }):
            from sync_engine.forced_alignment_subtitle import load_forced_aligner

            load_forced_aligner(attn_implementation=None)

        call_kwargs = mock_qwen_cls.from_pretrained.call_args
        # attn_implementation không nên xuất hiện trong kwargs
        assert "attn_implementation" not in call_kwargs.kwargs or call_kwargs.kwargs.get("attn_implementation") is None

    def test_import_error_when_qwen_asr_missing(self):
        """qwen_asr chưa cài → ImportError."""
        with patch.dict("sys.modules", {"qwen_asr": None, "torch": MagicMock()}):
            from sync_engine.forced_alignment_subtitle import load_forced_aligner

            with pytest.raises(ImportError, match="qwen-asr"):
                load_forced_aligner()
