#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/utils/test_video_cutter.py
=================================
Tests for utils/video_cutter.py — Pre-cut video remove SRT logic.

Cấu trúc layers:
  Layer 1 — Unit Tests          (pure logic, no I/O, no subprocess)
  Layer 2 — Component Tests     (mocked subprocess, file I/O via tmp_path)

Cách chạy:
    pytest tests/utils/test_video_cutter.py -v -k "Layer1"
    pytest tests/utils/test_video_cutter.py -v -k "Layer2"
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.video_cutter import (
    RemoveRange,
    KeepRange,
    VideoInfo,
    apply_safe_margin,
    normalize_and_merge,
    expand_to_keyframes,
    snap_to_frame_grid,
    invert_to_keep_ranges,
    build_hybrid_copy_part_cmd,
    build_reencode_part_cmd,
    parse_remove_srt,
    _build_manifest,
    MIN_KEEP_MS,
)


# ═════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ═════════════════════════════════════════════════════════════════════

@pytest.fixture()
def remove_srt_with_text(tmp_path: Path) -> Path:
    content = """1
00:00:12,500 --> 00:00:18,000
CUT intro mistake

2
00:03:10,000 --> 00:03:25,200
CUT sponsor
"""
    p = tmp_path / "remove.srt"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture()
def remove_srt_no_text(tmp_path: Path) -> Path:
    content = """1
00:00:12,500 --> 00:00:18,000

2
00:03:10,000 --> 00:03:25,200
"""
    p = tmp_path / "remove_no_text.srt"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture()
def basic_remove_ranges() -> list:
    return [
        RemoveRange(start_ms=12500, end_ms=18000, line=1, text="CUT intro"),
        RemoveRange(start_ms=190000, end_ms=205200, line=2, text="CUT sponsor"),
    ]


@pytest.fixture()
def video_info_with_audio() -> VideoInfo:
    return VideoInfo(
        duration_ms=600000, fps=30.0, has_audio=True,
        video_bitrate=5000000, video_codec="h264",
    )


@pytest.fixture()
def video_info_no_audio() -> VideoInfo:
    return VideoInfo(
        duration_ms=600000, fps=30.0, has_audio=False,
        video_bitrate=5000000, video_codec="h264",
    )


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_ParseRemoveSrt:
    """Test 1 & 2: Parse remove SRT with and without text."""

    def test_parse_srt_with_text(self, remove_srt_with_text):
        ranges = parse_remove_srt(str(remove_srt_with_text))
        assert len(ranges) == 2
        assert ranges[0].start_ms == 12500
        assert ranges[0].end_ms == 18000
        assert ranges[0].line == 1
        assert ranges[0].text == "CUT intro mistake"
        assert ranges[1].start_ms == 190000
        assert ranges[1].end_ms == 205200
        assert ranges[1].line == 2
        assert ranges[1].text == "CUT sponsor"

    def test_parse_srt_without_text(self, remove_srt_no_text):
        ranges = parse_remove_srt(str(remove_srt_no_text))
        assert len(ranges) == 2
        assert ranges[0].start_ms == 12500
        assert ranges[0].end_ms == 18000
        assert ranges[0].line == 1
        assert ranges[1].start_ms == 190000
        assert ranges[1].end_ms == 205200


class TestLayer1_SafeMargin:
    """Test 3: Apply safe margin and clamp to source duration."""

    def test_basic_margin(self):
        ranges = [RemoveRange(start_ms=10000, end_ms=20000, line=1)]
        result = apply_safe_margin(ranges, safe_margin_ms=100, source_duration_ms=600000)
        assert result[0].start_ms == 9900
        assert result[0].end_ms == 20100

    def test_clamp_to_zero(self):
        ranges = [RemoveRange(start_ms=50, end_ms=5000, line=1)]
        result = apply_safe_margin(ranges, safe_margin_ms=100, source_duration_ms=600000)
        assert result[0].start_ms == 0

    def test_clamp_to_duration(self):
        ranges = [RemoveRange(start_ms=599900, end_ms=600000, line=1)]
        result = apply_safe_margin(ranges, safe_margin_ms=200, source_duration_ms=600000)
        assert result[0].end_ms == 600000

    def test_large_margin(self):
        ranges = [RemoveRange(start_ms=1000, end_ms=2000, line=1)]
        result = apply_safe_margin(ranges, safe_margin_ms=5000, source_duration_ms=10000)
        assert result[0].start_ms == 0
        assert result[0].end_ms == 7000


class TestLayer1_NormalizeAndMerge:
    """Test 4: Normalize and merge overlapping remove ranges."""

    def test_sorted_output(self):
        ranges = [
            RemoveRange(start_ms=20000, end_ms=25000, line=2),
            RemoveRange(start_ms=5000, end_ms=10000, line=1),
        ]
        result = normalize_and_merge(ranges, source_duration_ms=100000)
        assert result[0].start_ms == 5000
        assert result[1].start_ms == 20000

    def test_merge_overlapping(self):
        ranges = [
            RemoveRange(start_ms=5000, end_ms=15000, line=1),
            RemoveRange(start_ms=12000, end_ms=20000, line=2),
        ]
        result = normalize_and_merge(ranges, source_duration_ms=100000)
        assert len(result) == 1
        assert result[0].start_ms == 5000
        assert result[0].end_ms == 20000

    def test_merge_adjacent(self):
        ranges = [
            RemoveRange(start_ms=5000, end_ms=10000, line=1),
            RemoveRange(start_ms=10000, end_ms=15000, line=2),
        ]
        result = normalize_and_merge(ranges, source_duration_ms=100000)
        assert len(result) == 1
        assert result[0].end_ms == 15000

    def test_clamp_negative_start(self):
        ranges = [RemoveRange(start_ms=-500, end_ms=5000, line=1)]
        result = normalize_and_merge(ranges, source_duration_ms=100000)
        assert result[0].start_ms == 0

    def test_clamp_beyond_duration(self):
        ranges = [RemoveRange(start_ms=95000, end_ms=110000, line=1)]
        result = normalize_and_merge(ranges, source_duration_ms=100000)
        assert result[0].end_ms == 100000

    def test_skip_invalid_range(self):
        ranges = [
            RemoveRange(start_ms=10000, end_ms=5000, line=1),
            RemoveRange(start_ms=20000, end_ms=25000, line=2),
        ]
        result = normalize_and_merge(ranges, source_duration_ms=100000)
        assert len(result) == 1
        assert result[0].line == 2

    def test_empty_input(self):
        assert normalize_and_merge([], source_duration_ms=100000) == []

    def test_no_merge_when_separate(self):
        ranges = [
            RemoveRange(start_ms=5000, end_ms=10000, line=1),
            RemoveRange(start_ms=20000, end_ms=25000, line=2),
        ]
        result = normalize_and_merge(ranges, source_duration_ms=100000)
        assert len(result) == 2


class TestLayer1_KeyframeExpansion:
    """Test 5: Keyframe expansion conservative for hybrid-copy."""

    def test_expand_to_nearest_keyframes(self):
        ranges = [RemoveRange(start_ms=5500, end_ms=8500, line=1)]
        keyframes = [0, 2000, 4000, 6000, 8000, 10000, 12000]
        result = expand_to_keyframes(ranges, keyframes, source_duration_ms=60000)
        assert len(result) == 1
        assert result[0].start_ms <= 5500
        assert result[0].end_ms >= 8500

    def test_fail_on_no_keyframes(self):
        ranges = [RemoveRange(start_ms=5000, end_ms=10000, line=1)]
        with pytest.raises(RuntimeError, match="No keyframes found"):
            expand_to_keyframes(ranges, [], source_duration_ms=60000)

    def test_expand_start_to_previous_keyframe(self):
        ranges = [RemoveRange(start_ms=7000, end_ms=9000, line=1)]
        keyframes = [0, 5000, 10000, 15000]
        result = expand_to_keyframes(ranges, keyframes, source_duration_ms=60000)
        assert result[0].start_ms == 5000

    def test_expand_end_to_next_keyframe(self):
        ranges = [RemoveRange(start_ms=7000, end_ms=9000, line=1)]
        keyframes = [0, 5000, 10000, 15000]
        result = expand_to_keyframes(ranges, keyframes, source_duration_ms=60000)
        assert result[0].end_ms == 10000

    def test_clamp_expanded_to_duration(self):
        ranges = [RemoveRange(start_ms=55000, end_ms=59000, line=1)]
        keyframes = [0, 10000, 20000, 30000, 40000, 50000]
        result = expand_to_keyframes(ranges, keyframes, source_duration_ms=60000)
        assert result[0].end_ms <= 60000

    def test_merge_after_expansion(self):
        ranges = [
            RemoveRange(start_ms=6000, end_ms=8000, line=1),
            RemoveRange(start_ms=9000, end_ms=11000, line=2),
        ]
        keyframes = [0, 5000, 10000, 15000]
        result = expand_to_keyframes(ranges, keyframes, source_duration_ms=60000)
        assert len(result) == 1


class TestLayer1_FrameSnap:
    """Test 6: Frame snap for reencode-smooth."""

    def test_snap_to_grid(self):
        ranges = [RemoveRange(start_ms=1015, end_ms=2017, line=1)]
        result = snap_to_frame_grid(ranges, fps=30.0, source_duration_ms=60000)
        assert len(result) == 1
        start_frame = round(1015 / 1000.0 * 30)
        expected_start = (start_frame / 30) * 1000
        assert abs(result[0].start_ms - expected_start) < 0.01

    def test_clamp_within_duration(self):
        ranges = [RemoveRange(start_ms=59900, end_ms=60100, line=1)]
        result = snap_to_frame_grid(ranges, fps=30.0, source_duration_ms=60000)
        assert result[0].end_ms <= 60000


class TestLayer1_InvertKeepRanges:
    """Test 7 & 8: Invert remove ranges to keep ranges, drop short keeps."""

    def test_basic_invert(self):
        removes = [RemoveRange(start_ms=10000, end_ms=20000, line=1)]
        keeps = invert_to_keep_ranges(removes, source_duration_ms=60000)
        assert len(keeps) == 2
        assert keeps[0].start_ms == 0
        assert keeps[0].end_ms == 10000
        assert keeps[1].start_ms == 20000
        assert keeps[1].end_ms == 60000

    def test_clean_timeline_offsets(self):
        removes = [RemoveRange(start_ms=10000, end_ms=20000, line=1)]
        keeps = invert_to_keep_ranges(removes, source_duration_ms=60000)
        assert keeps[0].clean_start_ms == 0
        assert keeps[0].clean_end_ms == 10000
        assert keeps[1].clean_start_ms == 10000
        assert keeps[1].clean_end_ms == 50000

    def test_remove_at_start(self):
        removes = [RemoveRange(start_ms=0, end_ms=5000, line=1)]
        keeps = invert_to_keep_ranges(removes, source_duration_ms=60000)
        assert len(keeps) == 1
        assert keeps[0].start_ms == 5000

    def test_remove_at_end(self):
        removes = [RemoveRange(start_ms=55000, end_ms=60000, line=1)]
        keeps = invert_to_keep_ranges(removes, source_duration_ms=60000)
        assert len(keeps) == 1
        assert keeps[0].end_ms == 55000

    def test_drop_short_keep_ranges(self):
        removes = [
            RemoveRange(start_ms=0, end_ms=9970, line=1),
            RemoveRange(start_ms=10000, end_ms=60000, line=2),
        ]
        keeps = invert_to_keep_ranges(removes, source_duration_ms=60000, min_keep_ms=MIN_KEEP_MS)
        assert len(keeps) == 0

    def test_keep_range_above_min(self):
        removes = [
            RemoveRange(start_ms=0, end_ms=9900, line=1),
            RemoveRange(start_ms=10000, end_ms=60000, line=2),
        ]
        keeps = invert_to_keep_ranges(removes, source_duration_ms=60000, min_keep_ms=50)
        assert len(keeps) == 1
        assert keeps[0].start_ms == 9900
        assert keeps[0].end_ms == 10000

    def test_remove_entire_video(self):
        removes = [RemoveRange(start_ms=0, end_ms=60000, line=1)]
        keeps = invert_to_keep_ranges(removes, source_duration_ms=60000)
        assert len(keeps) == 0

    def test_multiple_removes(self):
        removes = [
            RemoveRange(start_ms=10000, end_ms=15000, line=1),
            RemoveRange(start_ms=30000, end_ms=35000, line=2),
        ]
        keeps = invert_to_keep_ranges(removes, source_duration_ms=60000)
        assert len(keeps) == 3
        assert keeps[0].start_ms == 0
        assert keeps[0].end_ms == 10000
        assert keeps[1].start_ms == 15000
        assert keeps[1].end_ms == 30000
        assert keeps[2].start_ms == 35000
        assert keeps[2].end_ms == 60000


class TestLayer1_FFmpegCommands:
    """Test 13: FFmpeg commands use proper option/value separation."""

    def test_hybrid_copy_cmd_separation(self):
        keep = KeepRange(start_ms=5000, end_ms=15000)
        cmd = build_hybrid_copy_part_cmd(
            "input.mp4", "output.mp4", keep,
            audio_bitrate="256k",
        )
        assert "-c:v" in cmd
        assert "copy" in cmd
        assert "-c:a" in cmd
        assert "aac" in cmd
        assert "-b:a" in cmd
        assert "256k" in cmd
        assert "-ar" in cmd
        assert "48000" in cmd
        assert "-ac" in cmd
        assert "2" in cmd
        for i, part in enumerate(cmd):
            assert not (part.startswith("-") and len(part) > 4 and part[1:].replace(":", "").isalnum()
                        and not part.startswith("-c:") and not part.startswith("-b:")
                        and not part.startswith("-af")), \
                f"Possible stuck option/value: {part}"

    def test_reencode_cmd_separation(self):
        keep = KeepRange(start_ms=5000, end_ms=15000)
        cmd = build_reencode_part_cmd(
            "input.mp4", "output.mp4", keep,
            cq=28, preset="p4",
        )
        assert "-cq" in cmd
        idx = cmd.index("-cq")
        assert cmd[idx + 1] == "28"
        assert "-preset" in cmd
        assert "p4" in cmd
        assert "-rc" in cmd
        assert "vbr" in cmd
        assert "-tune" in cmd
        assert "hq" in cmd

    def test_reencode_no_maxrate_when_none(self):
        keep = KeepRange(start_ms=0, end_ms=10000)
        cmd = build_reencode_part_cmd(
            "input.mp4", "output.mp4", keep,
            maxrate=None, bufsize=None,
        )
        assert "-maxrate" not in cmd
        assert "-bufsize" not in cmd

    def test_reencode_with_maxrate(self):
        keep = KeepRange(start_ms=0, end_ms=10000)
        cmd = build_reencode_part_cmd(
            "input.mp4", "output.mp4", keep,
            maxrate=5750000, bufsize=11500000,
        )
        assert "-maxrate" in cmd
        assert "5750000" in cmd
        assert "-bufsize" in cmd
        assert "11500000" in cmd

    def test_hybrid_copy_has_video_copy(self):
        keep = KeepRange(start_ms=0, end_ms=10000)
        cmd = build_hybrid_copy_part_cmd("in.mp4", "out.mp4", keep)
        idx_cv = cmd.index("-c:v")
        assert cmd[idx_cv + 1] == "copy"

    def test_reencode_has_hevc_nvenc(self):
        keep = KeepRange(start_ms=0, end_ms=10000)
        cmd = build_reencode_part_cmd("in.mp4", "out.mp4", keep)
        idx_cv = cmd.index("-c:v")
        assert cmd[idx_cv + 1] == "hevc_nvenc"

    def test_audio_fade_filter_present(self):
        keep = KeepRange(start_ms=0, end_ms=10000)
        cmd = build_hybrid_copy_part_cmd(
            "in.mp4", "out.mp4", keep,
            audio_fade_ms=10, audio_fade_enabled=True,
        )
        assert "-af" in cmd
        af_idx = cmd.index("-af")
        assert "afade" in cmd[af_idx + 1]

    def test_audio_fade_disabled(self):
        keep = KeepRange(start_ms=0, end_ms=10000)
        cmd = build_hybrid_copy_part_cmd(
            "in.mp4", "out.mp4", keep,
            audio_fade_enabled=False,
        )
        assert "-af" not in cmd

    def test_no_bv_zero_in_reencode(self):
        keep = KeepRange(start_ms=0, end_ms=10000)
        cmd = build_reencode_part_cmd("in.mp4", "out.mp4", keep, cq=28)
        assert "-b:v" not in cmd

    def test_cq_always_28(self):
        keep = KeepRange(start_ms=0, end_ms=10000)
        cmd = build_reencode_part_cmd("in.mp4", "out.mp4", keep, cq=28)
        idx = cmd.index("-cq")
        assert cmd[idx + 1] == "28"


class TestLayer1_ManifestPath:
    """Test 9: Default manifest path next to output with suffix."""

    def test_default_manifest_suffix(self):
        out = Path("clean.mp4")
        result = out.with_name(f"{out.stem}_cut_manifest.json")
        assert result.name == "clean_cut_manifest.json"


class TestLayer1_ManifestContent:
    """Test 10: Manifest has all required fields."""

    def test_manifest_fields_hybrid_copy(self, video_info_with_audio, basic_remove_ranges):
        keep_ranges = [
            KeepRange(start_ms=0, end_ms=12500, clean_start_ms=0, clean_end_ms=12500),
            KeepRange(start_ms=18000, end_ms=190000, clean_start_ms=12500, clean_end_ms=184500),
        ]

        manifest = _build_manifest(
            input_path="source.mp4",
            output_path="clean.mp4",
            remove_srt_path="remove.srt",
            method="hybrid-copy",
            safe_margin_ms=100,
            audio_fade_ms=10,
            audio_fade_enabled=True,
            info=video_info_with_audio,
            source_ranges=basic_remove_ranges,
            normalized_ranges=basic_remove_ranges,
            expanded_ranges=basic_remove_ranges,
            keep_ranges=keep_ranges,
            expected_clean_ms=197000,
            actual_ms=196980,
            drift_ms=-20,
            hevc_cq=28,
            hevc_preset="p4",
            maxrate=None,
            bufsize=None,
            audio_bitrate="256k",
            warnings=[],
        )

        assert manifest["version"] == 1
        assert manifest["input_video"] == "source.mp4"
        assert manifest["output_video"] == "clean.mp4"
        assert manifest["method"] == "hybrid-copy"
        assert manifest["safe_margin_ms"] == 100
        assert manifest["audio_fade_ms"] == 10
        assert manifest["audio_fade_enabled"] is True
        assert manifest["source_duration_ms"] == 600000
        assert manifest["expected_clean_duration_ms"] == 197000
        assert manifest["actual_output_duration_ms"] == 196980
        assert manifest["duration_drift_ms"] == -20
        assert manifest["fps"] == 30.0
        assert len(manifest["source_remove_ranges"]) == 2
        assert len(manifest["normalized_remove_ranges"]) == 2
        assert len(manifest["expanded_remove_ranges"]) == 2
        assert len(manifest["keep_ranges"]) == 2
        assert "encoder" in manifest
        assert manifest["encoder"]["video"] == "copy"
        assert manifest["encoder"]["audio"] == "aac"
        assert manifest["encoder"]["audio_bitrate"] == "256k"
        assert "warnings" in manifest

    def test_manifest_fields_reencode_smooth(self, video_info_with_audio, basic_remove_ranges):
        keep_ranges = [
            KeepRange(start_ms=0, end_ms=12500, clean_start_ms=0, clean_end_ms=12500),
        ]

        manifest = _build_manifest(
            input_path="source.mp4",
            output_path="clean.mp4",
            remove_srt_path="remove.srt",
            method="reencode-smooth",
            safe_margin_ms=100,
            audio_fade_ms=10,
            audio_fade_enabled=True,
            info=video_info_with_audio,
            source_ranges=basic_remove_ranges,
            normalized_ranges=basic_remove_ranges,
            expanded_ranges=basic_remove_ranges,
            keep_ranges=keep_ranges,
            expected_clean_ms=12500,
            actual_ms=12480,
            drift_ms=-20,
            hevc_cq=28,
            hevc_preset="p4",
            maxrate=5750000,
            bufsize=11500000,
            audio_bitrate="256k",
            warnings=[],
        )

        encoder = manifest["encoder"]
        assert encoder["video"] == "hevc_nvenc"
        assert encoder["hevc_nvenc_available"] is True
        assert encoder["cq"] == 28
        assert encoder["preset"] == "p4"
        assert encoder["tune"] == "hq"
        assert encoder["maxrate"] == 5750000
        assert encoder["bufsize"] == 11500000
        assert encoder["maxrate_used"] is True

    def test_manifest_drift_detection_fields(self, video_info_with_audio, basic_remove_ranges):
        keep_ranges = [KeepRange(start_ms=0, end_ms=10000)]
        manifest = _build_manifest(
            input_path="s.mp4", output_path="c.mp4", remove_srt_path="r.srt",
            method="hybrid-copy", safe_margin_ms=100, audio_fade_ms=10,
            audio_fade_enabled=True, info=video_info_with_audio,
            source_ranges=basic_remove_ranges, normalized_ranges=basic_remove_ranges,
            expanded_ranges=basic_remove_ranges, keep_ranges=keep_ranges,
            expected_clean_ms=10000, actual_ms=9950, drift_ms=-50,
            hevc_cq=28, hevc_preset="p4", maxrate=None, bufsize=None,
            audio_bitrate="256k", warnings=["test warning"],
        )
        assert "expected_clean_duration_ms" in manifest
        assert "actual_output_duration_ms" in manifest
        assert "duration_drift_ms" in manifest
        assert manifest["duration_drift_ms"] == -50
        assert manifest["warnings"] == ["test warning"]


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — COMPONENT TESTS (mocked subprocess, run_pre_cut logic)
# ═════════════════════════════════════════════════════════════════════

class TestLayer2_HybridCopyNoHevcDependency:
    """Test 11: hybrid-copy does not require hevc_nvenc."""

    def test_hybrid_copy_does_not_probe_hevc(self):
        keep = KeepRange(start_ms=0, end_ms=10000)
        cmd = build_hybrid_copy_part_cmd("in.mp4", "out.mp4", keep)
        assert "hevc_nvenc" not in cmd
        assert cmd[cmd.index("-c:v") + 1] == "copy"


class TestLayer2_ReencodeSmoothFailFast:
    """Test 12: reencode-smooth fail-fast if hevc_nvenc not available."""

    @patch("utils.video_cutter.detect_hevc_nvenc", return_value=False)
    @patch("utils.video_cutter.probe_video_info")
    def test_fail_fast_no_hevc(self, mock_probe, mock_detect, video_info_with_audio, tmp_path):
        mock_probe.return_value = video_info_with_audio

        srt_file = tmp_path / "remove.srt"
        srt_file.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nCUT\n",
            encoding="utf-8",
        )

        from utils.video_cutter import run_pre_cut
        with pytest.raises(RuntimeError, match="hevc_nvenc not available"):
            run_pre_cut(
                input_path="input.mp4",
                output_path=str(tmp_path / "output.mp4"),
                remove_srt_path=str(srt_file),
                method="reencode-smooth",
            )


class TestLayer2_FailFastNoAudio:
    """Test 14: Fail-fast if input has no audio stream."""

    @patch("utils.video_cutter.probe_video_info")
    def test_fail_fast_no_audio(self, mock_probe, video_info_no_audio, tmp_path):
        mock_probe.return_value = video_info_no_audio

        srt_file = tmp_path / "remove.srt"
        srt_file.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nCUT\n",
            encoding="utf-8",
        )

        from utils.video_cutter import run_pre_cut
        with pytest.raises(RuntimeError, match="no audio stream"):
            run_pre_cut(
                input_path="input.mp4",
                output_path=str(tmp_path / "output.mp4"),
                remove_srt_path=str(srt_file),
            )


class TestLayer2_FailFastRemoveAll:
    """Test 15: Fail-fast if remove SRT removes everything."""

    @patch("utils.video_cutter.query_keyframes", return_value=[0, 5000, 10000])
    @patch("utils.video_cutter.probe_video_info")
    def test_fail_fast_no_content_remaining(self, mock_probe, mock_kf, video_info_with_audio, tmp_path):
        video_info_with_audio.duration_ms = 10000
        mock_probe.return_value = video_info_with_audio

        srt_file = tmp_path / "remove.srt"
        srt_file.write_text(
            "1\n00:00:00,000 --> 00:00:10,000\nCUT ALL\n",
            encoding="utf-8",
        )

        from utils.video_cutter import run_pre_cut
        with pytest.raises(RuntimeError, match="No content remaining"):
            run_pre_cut(
                input_path="input.mp4",
                output_path=str(tmp_path / "output.mp4"),
                remove_srt_path=str(srt_file),
            )


class TestLayer2_TempCleanup:
    """Test 16: Temp files cleaned after concat (unless --keep-tmp)."""

    @patch("utils.video_cutter.probe_output_duration_ms", return_value=8000.0)
    @patch("utils.video_cutter.concat_parts")
    @patch("subprocess.run")
    @patch("utils.video_cutter.query_keyframes", return_value=[0, 5000, 10000, 15000, 20000])
    @patch("utils.video_cutter.probe_video_info")
    def test_cleanup_default(self, mock_probe, mock_kf, mock_subproc, mock_concat, mock_dur, video_info_with_audio, tmp_path):
        video_info_with_audio.duration_ms = 20000
        mock_probe.return_value = video_info_with_audio

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_subproc.return_value = mock_result

        srt_file = tmp_path / "remove.srt"
        srt_file.write_text(
            "1\n00:00:05,000 --> 00:00:10,000\nCUT\n",
            encoding="utf-8",
        )

        out_path = tmp_path / "output.mp4"
        tmp_dir = tmp_path / "output_precut_tmp"

        fake_part = tmp_dir / "keep_0000.mp4"
        tmp_dir.mkdir(exist_ok=True)
        fake_part.write_bytes(b"fake_video_data" * 1000)

        from utils.video_cutter import run_pre_cut
        run_pre_cut(
            input_path="input.mp4",
            output_path=str(out_path),
            remove_srt_path=str(srt_file),
            keep_tmp=False,
        )

        assert not tmp_dir.exists(), "Temp directory should be cleaned up by default"

    @patch("utils.video_cutter.probe_output_duration_ms", return_value=8000.0)
    @patch("utils.video_cutter.concat_parts")
    @patch("subprocess.run")
    @patch("utils.video_cutter.query_keyframes", return_value=[0, 5000, 10000, 15000, 20000])
    @patch("utils.video_cutter.probe_video_info")
    def test_keep_tmp(self, mock_probe, mock_kf, mock_subproc, mock_concat, mock_dur, video_info_with_audio, tmp_path):
        video_info_with_audio.duration_ms = 20000
        mock_probe.return_value = video_info_with_audio

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subproc.return_value = mock_result

        srt_file = tmp_path / "remove.srt"
        srt_file.write_text(
            "1\n00:00:05,000 --> 00:00:10,000\nCUT\n",
            encoding="utf-8",
        )

        out_path = tmp_path / "output.mp4"
        tmp_dir = tmp_path / "output_precut_tmp"

        fake_part = tmp_dir / "keep_0000.mp4"
        tmp_dir.mkdir(exist_ok=True)
        fake_part.write_bytes(b"fake_video_data" * 1000)

        from utils.video_cutter import run_pre_cut
        run_pre_cut(
            input_path="input.mp4",
            output_path=str(out_path),
            remove_srt_path=str(srt_file),
            keep_tmp=True,
        )

        assert tmp_dir.exists(), "Temp directory should be kept when --keep-tmp"
