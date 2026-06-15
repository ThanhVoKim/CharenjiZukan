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
            "utils.forced_aligner.load_forced_aligner",
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
            "utils.forced_aligner.load_forced_aligner",
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
            "utils.forced_aligner.load_forced_aligner",
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
            "utils.forced_aligner.load_forced_aligner",
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
            "utils.forced_aligner.load_forced_aligner",
            return_value=mock_aligner,
        ), patch("utils.media_utils.clear_vram"), patch(
            "utils.forced_aligner.segment_words_to_subtitles"
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


# ═════════════════════════════════════════════════════════════════════
# Layer 1: Per-Clip Logic & Completeness (Pure Logic — no GPU/FFmpeg)
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.Layer1_PerClip
class TestLayer1_PerClipBuildClips:
    """Test _build_clips_from_timeline: dựng clips từ timeline, xác định dòng remap."""

    @staticmethod
    def _make_timeline_segment(orig_start, new_start, clip_path=None, audio_speed=1.0, block_type="tts",
                               orig_end=None, new_end=None):
        seg = MagicMock()
        seg.block_type = block_type
        seg.orig_start = float(orig_start)
        seg.orig_end = float(orig_end if orig_end is not None else orig_start + 2000)
        seg.new_start = float(new_start)
        seg.new_end = float(new_end if new_end is not None else new_start + 2000)
        seg.audio_speed = float(audio_speed)
        seg.tts_clip_path = clip_path
        return seg

    def test_mute_zone_lines_go_to_remap(self, tmp_path):
        """Dòng subtitle nằm trong vùng mute → remap_lines, không phải clips."""
        from sync_engine.forced_alignment_subtitle import _build_clips_from_timeline

        # 2 dòng phụ đề: 1 TTS (start=1000ms), 1 mute (start=5000ms)
        subtitle_segments = [
            {"line": 1, "start_time": 1000, "end_time": 3000, "text": "Hello world"},
            {"line": 2, "start_time": 5000, "end_time": 7000, "text": "Mute line"},
        ]
        mute_segments = [
            {"line": 1, "start_time": 5000, "end_time": 7000, "text": "Mute line"},
        ]
        wav = tmp_path / "dubb-0.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 44)

        timeline = [
            self._make_timeline_segment(1000, 10000, clip_path=str(wav)),
            self._make_timeline_segment(5000, 60000, clip_path=None, block_type="mute"),
        ]

        clips, remap_lines = _build_clips_from_timeline(timeline, subtitle_segments, mute_segments)

        assert len(clips) == 1
        assert clips[0]["text"] == "Hello world"
        assert len(remap_lines) == 1
        assert remap_lines[0]["text"] == "Mute line"

    def test_missing_clip_file_goes_to_remap(self, tmp_path):
        """Clip file không tồn tại → dòng đó vào remap, không phải clips."""
        from sync_engine.forced_alignment_subtitle import _build_clips_from_timeline

        subtitle_segments = [
            {"line": 1, "start_time": 1000, "end_time": 3000, "text": "Hello"},
        ]
        # tts_clip_path = None → không có clip
        timeline = [
            self._make_timeline_segment(1000, 10000, clip_path=None),
        ]

        clips, remap_lines = _build_clips_from_timeline(timeline, subtitle_segments, [])

        assert len(clips) == 0
        assert len(remap_lines) == 1

    def test_offset_ms_taken_from_new_start(self, tmp_path):
        """offset_ms của clip = seg.new_start (ms trên timeline cuối)."""
        from sync_engine.forced_alignment_subtitle import _build_clips_from_timeline

        subtitle_segments = [
            {"line": 1, "start_time": 1000, "end_time": 3000, "text": "Test"},
        ]
        wav = tmp_path / "dubb-0.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 44)
        timeline = [
            self._make_timeline_segment(1000, 99000, clip_path=str(wav), audio_speed=1.5),
        ]

        clips, _ = _build_clips_from_timeline(timeline, subtitle_segments, [])

        assert len(clips) == 1
        assert clips[0]["offset_ms"] == 99000.0
        assert clips[0]["audio_speed"] == 1.5

    def test_resolve_aligner_config_batch_size_default(self):
        """batch_size không có trong config → mặc định 16."""
        from sync_engine.forced_alignment_subtitle import _resolve_aligner_config

        result = _resolve_aligner_config({})
        assert result["batch_size"] == 16

    def test_resolve_aligner_config_batch_size_custom(self):
        """batch_size custom được map đúng."""
        from sync_engine.forced_alignment_subtitle import _resolve_aligner_config

        result = _resolve_aligner_config({"batch_size": 8})
        assert result["batch_size"] == 8


# ═════════════════════════════════════════════════════════════════════
# Layer 2: Per-Clip Align + Remap Merge (Mock aligner, no GPU/FFmpeg)
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.Layer2_PerClip
class TestLayer2_PerClipAlignMerge:
    """Test per-clip alignment flow: offset đúng, gộp aligned+remap, không sót dòng."""

    @staticmethod
    def _make_word(text, start_s, end_s):
        w = MagicMock()
        w.text = text
        w.start_time = start_s
        w.end_time = end_s
        return w

    @staticmethod
    def _make_timeline_segment(orig_start, new_start, clip_path=None, audio_speed=1.0, block_type="tts",
                               orig_end=None, new_end=None):
        seg = MagicMock()
        seg.block_type = block_type
        seg.orig_start = float(orig_start)
        _orig_end = float(orig_end if orig_end is not None else orig_start + 3000)
        seg.orig_end = _orig_end
        _new_end = float(new_end if new_end is not None else new_start + 5000)
        seg.new_start = float(new_start)
        seg.new_end = _new_end
        seg.new_chunk_dur = _new_end - float(new_start)
        seg.audio_speed = float(audio_speed)
        seg.tts_clip_path = clip_path
        return seg

    def test_per_clip_offsets_applied_voicevox(self, tmp_path):
        """Clip TTS (audio_speed=1.0, voicevox) → word times offset đúng về new_start."""
        from sync_engine.forced_alignment_subtitle import run_forced_alignment_subtitle

        wav = tmp_path / "dubb-0.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 44)

        subtitle_segments = [
            {"line": 1, "start_time": 1000, "end_time": 3000, "text": "Hello world"},
        ]
        timeline_seg = self._make_timeline_segment(1000, 50000, clip_path=str(wav), audio_speed=1.0)

        # Aligner trả word timing tính từ đầu clip (giây)
        fake_words = [self._make_word("Hello", 0.1, 0.4), self._make_word("world", 0.5, 0.9)]
        mock_aligner = MagicMock()
        mock_aligner.align.return_value = [fake_words]

        render_config = {"forced_alignment_subtitle": {"enabled": True, "fail_policy": "warn"}}
        output_srt = tmp_path / "out.srt"

        with patch("utils.forced_aligner.load_forced_aligner", return_value=mock_aligner), \
             patch("utils.media_utils.clear_vram"):
            result = run_forced_alignment_subtitle(
                audio_path=str(tmp_path / "audio.wav"),
                transcript_path=str(tmp_path / "transcript.txt"),
                output_srt_path=str(output_srt),
                render_config=render_config,
                timeline=[timeline_seg],
                subtitle_segments=subtitle_segments,
                mute_segments=[],
                fps_float=30.0,
            )

        assert result is not None
        assert output_srt.exists()
        srt_content = output_srt.read_text(encoding="utf-8")
        # SRT phải có timestamp, offset_ms = 50000ms = 50s, word 0.1s → ~50.1s
        assert "-->" in srt_content

    def test_mute_lines_present_in_output(self, tmp_path):
        """Dòng phụ đề vùng mute → vẫn xuất hiện trong SRT cuối (remap)."""
        from sync_engine.forced_alignment_subtitle import run_forced_alignment_subtitle

        wav = tmp_path / "dubb-0.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 44)

        subtitle_segments = [
            {"line": 1, "start_time": 1000, "end_time": 3000, "text": "TTS line"},
            {"line": 2, "start_time": 5000, "end_time": 7000, "text": "Mute zone line"},
        ]
        mute_segments = [
            {"line": 1, "start_time": 5000, "end_time": 7000, "text": "Mute zone line"},
        ]
        tts_seg = self._make_timeline_segment(1000, 10000, clip_path=str(wav), audio_speed=1.0,
                                              orig_end=3000, new_end=15000)
        mute_seg = self._make_timeline_segment(5000, 60000, block_type="mute", audio_speed=1.0,
                                               orig_end=7000, new_end=65000)

        fake_words = [self._make_word("TTS", 0.0, 0.3), self._make_word("line", 0.3, 0.6)]
        mock_aligner = MagicMock()
        mock_aligner.align.return_value = [fake_words]

        render_config = {"forced_alignment_subtitle": {"enabled": True, "fail_policy": "warn"}}
        output_srt = tmp_path / "out.srt"

        with patch("utils.forced_aligner.load_forced_aligner", return_value=mock_aligner), \
             patch("utils.media_utils.clear_vram"):
            result = run_forced_alignment_subtitle(
                audio_path=str(tmp_path / "audio.wav"),
                transcript_path=str(tmp_path / "transcript.txt"),
                output_srt_path=str(output_srt),
                render_config=render_config,
                timeline=[tts_seg, mute_seg],
                subtitle_segments=subtitle_segments,
                mute_segments=mute_segments,
                fps_float=30.0,
            )

        assert result is not None
        srt_content = output_srt.read_text(encoding="utf-8")
        assert "Mute zone line" in srt_content, "Dòng vùng mute phải có trong SRT output"

    def test_vram_freed_on_clip_oom(self, tmp_path):
        """Khi align() ném OOM → clear_vram vẫn được gọi (try/finally)."""
        from utils.forced_aligner import execute_forced_alignment_clips

        clips = [{"audio_path": str(tmp_path / "clip.wav"), "text": "Hello", "offset_ms": 0.0,
                  "audio_speed": 1.0, "line": 0}]
        mock_aligner = MagicMock()
        mock_aligner.align.side_effect = RuntimeError("CUDA out of memory")

        with patch("utils.forced_aligner.load_forced_aligner", return_value=mock_aligner), \
             patch("utils.media_utils.clear_vram") as mock_clear:
            with pytest.raises(RuntimeError, match="CUDA out of memory"):
                execute_forced_alignment_clips(
                    clips=clips,
                    align_cfg={"language": "Japanese", "max_chars": 42, "min_chars": 0,
                               "split_on_comma": True, "offset_seconds": 0.0, "batch_size": 16},
                )
            mock_clear.assert_called_once()

    def test_empty_clip_result_skipped_no_abort(self, tmp_path):
        """Clip align trả rỗng → skip dòng đó, không abort toàn bộ."""
        from utils.forced_aligner import execute_forced_alignment_clips

        wav1 = tmp_path / "c0.wav"
        wav1.write_bytes(b"RIFF" + b"\x00" * 44)
        wav2 = tmp_path / "c1.wav"
        wav2.write_bytes(b"RIFF" + b"\x00" * 44)

        clips = [
            {"audio_path": str(wav1), "text": "Hello world", "offset_ms": 0.0, "audio_speed": 1.0, "line": 0},
            {"audio_path": str(wav2), "text": "Empty", "offset_ms": 5000.0, "audio_speed": 1.0, "line": 1},
        ]

        word = MagicMock()
        word.text = "Hello"
        word.start_time = 0.0
        word.end_time = 0.5

        # Clip 0 trả kết quả, clip 1 trả rỗng
        mock_aligner = MagicMock()
        mock_aligner.align.return_value = [[word, MagicMock(text="world", start_time=0.5, end_time=1.0)], []]

        with patch("utils.forced_aligner.load_forced_aligner", return_value=mock_aligner), \
             patch("utils.media_utils.clear_vram"):
            aligned, failed = execute_forced_alignment_clips(
                clips=clips,
                align_cfg={"language": "Japanese", "max_chars": 42, "min_chars": 0,
                           "split_on_comma": True, "offset_seconds": 0.0, "batch_size": 16},
            )

        # Clip 0 có output, clip 1 (rỗng) vào failed_lines
        assert len(aligned) >= 1
        assert 1 in failed

    def test_fallback_to_mixed_audio_when_no_timeline(self, tmp_path):
        """Không có timeline → dùng nhánh cũ (execute_forced_alignment)."""
        from sync_engine.forced_alignment_subtitle import run_forced_alignment_subtitle

        transcript = tmp_path / "transcript.txt"
        transcript.write_text("Hello world", encoding="utf-8")
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 44)

        render_config = {"forced_alignment_subtitle": {"enabled": True, "fail_policy": "warn"}}

        word = MagicMock()
        word.text = "Hello"
        word.start_time = 0.1
        word.end_time = 0.4
        mock_aligner = MagicMock()
        mock_aligner.align.return_value = [[word]]

        with patch("utils.forced_aligner.load_forced_aligner", return_value=mock_aligner), \
             patch("utils.media_utils.clear_vram"):
            result = run_forced_alignment_subtitle(
                audio_path=str(audio),
                transcript_path=str(transcript),
                output_srt_path=str(tmp_path / "out.srt"),
                render_config=render_config,
                # timeline=None → nhánh cũ
            )

        # Nhánh cũ chạy, aligner.align gọi với audio path (không phải list clip)
        assert mock_aligner.align.called
        call_args = mock_aligner.align.call_args
        audio_arg = call_args.kwargs.get("audio") or call_args[1].get("audio") or call_args[0][0]
        # Nhánh cũ: audio là 1 path string, không phải list
        assert isinstance(audio_arg, str)
