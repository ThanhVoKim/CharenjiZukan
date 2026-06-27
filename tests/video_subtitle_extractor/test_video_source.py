#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/video_subtitle_extractor/test_video_source.py
====================================================
Unit tests cho video_subtitle_extractor/video_source.py.

Layer 1 — Unit Tests (không cần GPU, FFmpeg thật, hay video thật).
Tất cả subprocess và shutil.which được mock hoàn toàn.

Cách chạy:
    pytest tests/video_subtitle_extractor/test_video_source.py -v -k "Layer1"
"""

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ── Project root ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Helpers để import video_source mà không cần cv2 ──────────────────

def _import_video_source():
    """Import video_source module với utils.ffmpeg_probe được mock."""
    import importlib.util

    # Fake utils.ffmpeg_probe nếu chưa có
    if "utils.ffmpeg_probe" not in sys.modules:
        fake_ffmpeg_probe = types.ModuleType("utils.ffmpeg_probe")
        fake_ffmpeg_probe.detect_hevc_nvenc = lambda: False
        fake_ffmpeg_probe.HEVC_NVENC_VIDEO_ARGS = ["-c:v", "hevc_nvenc", "-preset", "p4", "-tune", "hq", "-cq", "28"]
        sys.modules["utils"] = sys.modules.get("utils", types.ModuleType("utils"))
        sys.modules["utils.ffmpeg_probe"] = fake_ffmpeg_probe

    spec = importlib.util.spec_from_file_location(
        "video_source",
        PROJECT_ROOT / "video_subtitle_extractor" / "video_source.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# Không cần GPU, FFmpeg, hay video thật.
# ═════════════════════════════════════════════════════════════════════


class TestLayer1_ProbeVideoCodec:
    """Unit tests cho probe_video_codec()."""

    def setup_method(self):
        self.vs = _import_video_source()

    def test_returns_empty_when_ffprobe_missing(self):
        with patch("shutil.which", return_value=None):
            result = self.vs.probe_video_codec("any.mp4")
        assert result == ""

    def test_returns_codec_name_lowercase(self):
        mock_result = MagicMock(stdout="AV1\n", returncode=0)
        with patch("shutil.which", return_value="/usr/bin/ffprobe"), \
             patch("subprocess.run", return_value=mock_result):
            result = self.vs.probe_video_codec("video.av1")
        assert result == "av1"

    def test_returns_empty_on_timeout(self):
        import subprocess
        with patch("shutil.which", return_value="/usr/bin/ffprobe"), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffprobe", 15)):
            result = self.vs.probe_video_codec("video.mp4")
        assert result == ""

    def test_returns_empty_on_oserror(self):
        with patch("shutil.which", return_value="/usr/bin/ffprobe"), \
             patch("subprocess.run", side_effect=OSError("not found")):
            result = self.vs.probe_video_codec("video.mp4")
        assert result == ""

    def test_strips_whitespace(self):
        mock_result = MagicMock(stdout="  h264  \n", returncode=0)
        with patch("shutil.which", return_value="/usr/bin/ffprobe"), \
             patch("subprocess.run", return_value=mock_result):
            result = self.vs.probe_video_codec("video.mp4")
        assert result == "h264"


class TestLayer1_PrepareOpencvSource_NoTranscode:
    """Các trường hợp KHÔNG cần transcode."""

    def setup_method(self):
        self.vs = _import_video_source()

    def test_h264_returns_original_path(self):
        with patch.object(self.vs, "probe_video_codec", return_value="h264"), \
             patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            path, tmp = self.vs.prepare_opencv_source("video.mp4")
        assert path == "video.mp4"
        assert tmp is None

    def test_hevc_returns_original_path(self):
        with patch.object(self.vs, "probe_video_codec", return_value="hevc"), \
             patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            path, tmp = self.vs.prepare_opencv_source("video.mp4")
        assert path == "video.mp4"
        assert tmp is None

    def test_empty_codec_returns_original_path(self):
        with patch.object(self.vs, "probe_video_codec", return_value=""), \
             patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            path, tmp = self.vs.prepare_opencv_source("video.mp4")
        assert path == "video.mp4"
        assert tmp is None

    def test_no_ffmpeg_returns_original_path(self):
        with patch("shutil.which", return_value=None):
            path, tmp = self.vs.prepare_opencv_source("video.mp4")
        assert path == "video.mp4"
        assert tmp is None

    def test_custom_transcode_codecs_not_matched(self):
        """Codec không nằm trong custom list → không transcode."""
        with patch.object(self.vs, "probe_video_codec", return_value="av1"), \
             patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            # Truyền danh sách rỗng → không có codec nào cần transcode
            path, tmp = self.vs.prepare_opencv_source("video.mp4", transcode_codecs=())
        assert path == "video.mp4"
        assert tmp is None


class TestLayer1_PrepareOpencvSource_Transcode:
    """Các trường hợp CÓ transcode (AV1)."""

    def setup_method(self):
        self.vs = _import_video_source()

    def test_av1_triggers_transcode_and_returns_temp(self):
        mock_run = MagicMock(returncode=0, stderr="", stdout="")
        with patch.object(self.vs, "probe_video_codec", return_value="av1"), \
             patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("subprocess.run", return_value=mock_run), \
             patch("os.close"), \
             patch("tempfile.mkstemp", return_value=(0, "/tmp/ocr_test.mp4")):
            path, tmp = self.vs.prepare_opencv_source("video.av1")

        assert path == "/tmp/ocr_test.mp4"
        assert tmp == "/tmp/ocr_test.mp4"

    def test_transcode_failure_falls_back(self):
        mock_run = MagicMock(returncode=1, stderr="codec error", stdout="")
        safe_remove_calls = []
        with patch.object(self.vs, "probe_video_codec", return_value="av1"), \
             patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("subprocess.run", return_value=mock_run), \
             patch("os.close"), \
             patch("tempfile.mkstemp", return_value=(0, "/tmp/fail.mp4")), \
             patch.object(self.vs, "_safe_remove", side_effect=safe_remove_calls.append):
            path, tmp = self.vs.prepare_opencv_source("video.av1")

        assert path == "video.av1"
        assert tmp is None
        assert "/tmp/fail.mp4" in safe_remove_calls

    def test_transcode_timeout_falls_back(self):
        import subprocess
        with patch.object(self.vs, "probe_video_codec", return_value="av1"), \
             patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 3600)), \
             patch("os.close"), \
             patch("tempfile.mkstemp", return_value=(0, "/tmp/timeout.mp4")), \
             patch.object(self.vs, "_safe_remove"):
            path, tmp = self.vs.prepare_opencv_source("video.av1")

        assert path == "video.av1"
        assert tmp is None

    def test_av1_uses_libx264_when_no_nvenc(self):
        """Khi NVENC không available → command dùng libx264."""
        import importlib
        import types

        # Fake ffmpeg_probe: nvenc không available
        fake_probe = types.ModuleType("utils.ffmpeg_probe")
        fake_probe.detect_hevc_nvenc = lambda: False
        fake_probe.HEVC_NVENC_VIDEO_ARGS = ["-c:v", "hevc_nvenc"]
        sys.modules["utils.ffmpeg_probe"] = fake_probe

        vs = _import_video_source()

        captured_cmd = []
        mock_run = MagicMock(returncode=0, stderr="", stdout="")

        def capture_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return mock_run

        with patch.object(vs, "probe_video_codec", return_value="av1"), \
             patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("subprocess.run", side_effect=capture_run), \
             patch("os.close"), \
             patch("tempfile.mkstemp", return_value=(0, "/tmp/x264.mp4")):
            vs.prepare_opencv_source("video.av1")

        assert "libx264" in captured_cmd
        assert "hevc_nvenc" not in captured_cmd

    def test_av1_uses_nvenc_when_available(self):
        """Khi NVENC available → command dùng hevc_nvenc."""
        import types

        fake_probe = types.ModuleType("utils.ffmpeg_probe")
        fake_probe.detect_hevc_nvenc = lambda: True
        fake_probe.HEVC_NVENC_VIDEO_ARGS = ["-c:v", "hevc_nvenc", "-preset", "p4", "-tune", "hq", "-cq", "28"]
        sys.modules["utils.ffmpeg_probe"] = fake_probe

        vs = _import_video_source()

        captured_cmd = []
        mock_run = MagicMock(returncode=0, stderr="", stdout="")

        def capture_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return mock_run

        with patch.object(vs, "probe_video_codec", return_value="av1"), \
             patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("subprocess.run", side_effect=capture_run), \
             patch("os.close"), \
             patch("tempfile.mkstemp", return_value=(0, "/tmp/nvenc.mp4")):
            vs.prepare_opencv_source("video.av1")

        assert "hevc_nvenc" in captured_cmd
        assert "libx264" not in captured_cmd

    def test_transcode_drops_audio(self):
        """Transcode command phải có -an (bỏ audio, chỉ cần video để OCR)."""
        mock_run = MagicMock(returncode=0, stderr="", stdout="")
        captured_cmd = []

        def capture_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return mock_run

        vs = _import_video_source()
        with patch.object(vs, "probe_video_codec", return_value="av1"), \
             patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("subprocess.run", side_effect=capture_run), \
             patch("os.close"), \
             patch("tempfile.mkstemp", return_value=(0, "/tmp/noaudio.mp4")):
            vs.prepare_opencv_source("video.av1")

        assert "-an" in captured_cmd


class TestLayer1_GenerateWarnings:
    """Unit tests cho SubtitleWriter.generate_warnings()."""

    def setup_method(self):
        import importlib.util, types, logging
        if "utils.logger" not in sys.modules:
            sys.modules["utils"] = sys.modules.get("utils", types.ModuleType("utils"))
            fake_logger = types.ModuleType("utils.logger")
            fake_logger.get_logger = logging.getLogger
            sys.modules["utils.logger"] = fake_logger

        spec = importlib.util.spec_from_file_location(
            "subtitle_writer",
            PROJECT_ROOT / "video_subtitle_extractor" / "subtitle_writer.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.SubtitleEntry = mod.SubtitleEntry
        self.SubtitleWriter = mod.SubtitleWriter

    def _make_entry(self, idx, start, end, text, frame_count=10):
        return self.SubtitleEntry(index=idx, start_time=start, end_time=end, text=text, frame_count=frame_count)

    def test_short_duration_entry_is_flagged(self, tmp_path):
        raw = [self._make_entry(1, 1.0, 1.2, "short", frame_count=2)]
        w = self.SubtitleWriter()
        out = str(tmp_path / "warn.txt")
        has_warn = w.generate_warnings(raw, [], out, frame_interval=3, min_frames=15)
        content = open(out, encoding="utf-8").read()
        assert has_warn
        assert "short" in content
        assert "SHORT-DURATION" in content

    def test_long_duration_entry_not_flagged_in_short_section(self, tmp_path):
        raw = [self._make_entry(1, 1.0, 5.0, "long_text", frame_count=20)]
        w = self.SubtitleWriter()
        out = str(tmp_path / "warn.txt")
        has_warn = w.generate_warnings(raw, raw, out, frame_interval=3, min_frames=15)
        content = open(out, encoding="utf-8").read()
        short_section = content.split("ENGLISH")[0]
        assert "long_text" not in short_section

    def test_english_entry_flagged(self, tmp_path):
        raw = [self._make_entry(1, 1.0, 5.0, "Hello World", frame_count=20)]
        w = self.SubtitleWriter()
        out = str(tmp_path / "warn.txt")
        has_warn = w.generate_warnings(raw, raw, out, frame_interval=3, min_frames=15)
        content = open(out, encoding="utf-8").read()
        assert has_warn
        assert "Hello World" in content
        assert "ENGLISH/NUMBER" in content

    def test_no_warnings_returns_false(self, tmp_path):
        raw = [self._make_entry(1, 1.0, 5.0, "正常字幕", frame_count=20)]
        w = self.SubtitleWriter()
        out = str(tmp_path / "warn.txt")
        has_warn = w.generate_warnings(raw, raw, out, frame_interval=3, min_frames=15)
        assert not has_warn
        assert Path(out).exists()  # file vẫn được tạo

    def test_neighbor_text_shown_in_short_warning(self, tmp_path):
        raw = [
            self._make_entry(1, 1.0, 4.0, "PREV_TEXT", frame_count=15),
            self._make_entry(2, 5.0, 5.2, "SHORT_TEXT", frame_count=2),
            self._make_entry(3, 6.0, 9.0, "NEXT_TEXT", frame_count=15),
        ]
        w = self.SubtitleWriter()
        out = str(tmp_path / "warn.txt")
        w.generate_warnings(raw, raw, out, frame_interval=3, min_frames=15)
        content = open(out, encoding="utf-8").read()
        assert "PREV_TEXT" in content
        assert "NEXT_TEXT" in content

    def test_frame_count_default_is_1(self):
        e = self.SubtitleEntry(index=1, start_time=0.0, end_time=1.0, text="x")
        assert e.frame_count == 1

    def test_short_check_uses_frame_count_times_interval(self, tmp_path):
        # frame_count=4, frame_interval=4 → 4*4=16 >= 15 → không flag
        raw = [self._make_entry(1, 1.0, 5.0, "borderline", frame_count=4)]
        w = self.SubtitleWriter()
        out = str(tmp_path / "warn.txt")
        has_warn = w.generate_warnings(raw, raw, out, frame_interval=4, min_frames=15)
        content = open(out, encoding="utf-8").read()
        short_section = content.split("ENGLISH")[0]
        assert "borderline" not in short_section

    def test_file_always_created_even_with_no_entries(self, tmp_path):
        w = self.SubtitleWriter()
        out = str(tmp_path / "warn.txt")
        w.generate_warnings([], [], out)
        assert Path(out).exists()
