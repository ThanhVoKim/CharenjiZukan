#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/sync_engine/test_audio_assembler.py
=========================================
Test Component Phase 3: Assemble Audio Track.
`build_mute_ranges`, `build_ambient_mask`, `resolve_audio_policies`,
`assemble_audio_track`, `compress_tts_clip`.

Cấu trúc layers:
  Layer 1 — Unit Tests          (policy + mask logic)
  Layer 2 — Component Tests     (test với synthetic audio)

Cách chạy từng layer:
    pytest tests/sync_engine/test_audio_assembler.py -v -k "Layer1"
    pytest tests/sync_engine/test_audio_assembler.py -v -k "Layer2"
"""

import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ── Project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sync_engine import audio_assembler
from sync_engine.audio_assembler import (
    _build_mute_volume_filter,
    assemble_audio_track,
    build_ambient_mask,
    build_mute_ranges,
    compress_tts_clip,
    resolve_audio_policies,
)
from sync_engine.models import TimelineSegment


# ═════════════════════════════════════════════════════════════════════
# SHARED FIXTURES / HELPERS
# ═════════════════════════════════════════════════════════════════════


def _segment(
    orig_start: float,
    orig_end: float,
    new_start: float,
    new_end: float,
    block_type: str,
    *,
    video_speed: float = 1.0,
    audio_speed: float = 1.0,
    tts_clip_path: str | None = None,
    tts_duration: float = 0.0,
) -> TimelineSegment:
    return TimelineSegment(
        orig_start=orig_start,
        orig_end=orig_end,
        new_start=new_start,
        new_end=new_end,
        video_speed=video_speed,
        audio_speed=audio_speed,
        new_chunk_dur=new_end - new_start,
        block_type=block_type,
        tts_clip_path=tts_clip_path,
        tts_duration=tts_duration,
    )


@pytest.fixture(scope="module")
def synthetic_ambient_wav(tmp_path_factory) -> Path:
    """WAV 1 giây silence tạo bằng FFmpeg."""
    tmp_dir = tmp_path_factory.mktemp("audio")
    path = tmp_dir / "ambient.wav"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            "1.0",
            str(path),
        ],
        check=True,
        capture_output=True,
    )

    return path


@pytest.fixture(scope="module")
def short_tts_wav(tmp_path_factory) -> Path:
    """WAV 0.5 giây tạo bằng FFmpeg (dùng làm mock TTS clip)."""
    tmp_dir = tmp_path_factory.mktemp("audio_tts")
    path = tmp_dir / "short_tts.wav"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=48000:duration=0.5",
            "-ac",
            "2",
            str(path),
        ],
        check=True,
        capture_output=True,
    )

    return path


def _get_duration_ffprobe(wav_path: str) -> float:
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        wav_path,
    ]
    res = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
    return float(res.stdout.strip())


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# ═════════════════════════════════════════════════════════════════════


class TestLayer1_AudioAssemblerPolicyLogic:
    def test_resolve_audio_policies_defaults(self):
        assert resolve_audio_policies({}) == {
            "global_bgm": "off",
            "mute_audio": "original",
            "ambient": "exclude_mute",
        }

    def test_resolve_audio_policies_accepts_aliases(self):
        cfg = {
            "audio_policies": {
                "global_bgm": "whole video",
                "mute_audio": "instrumental",
                "ambient": "exclude-mute",
            }
        }

        assert resolve_audio_policies(cfg) == {
            "global_bgm": "whole_video",
            "mute_audio": "instrumental",
            "ambient": "exclude_mute",
        }

    def test_resolve_audio_policies_maps_legacy_flags(self):
        cfg = {"audio_separator": {"extract_bgm": True, "extract_vocals": True}}

        assert resolve_audio_policies(cfg) == {
            "global_bgm": "whole_video",
            "mute_audio": "vocals",
            "ambient": "exclude_mute",
        }

    def test_resolve_audio_policies_explicit_block_overrides_legacy_flags(self):
        cfg = {
            "audio_policies": {
                "global_bgm": "off",
                "mute_audio": "silence",
                "ambient": "whole_video",
            },
            "audio_separator": {"extract_bgm": True, "extract_vocals": True},
        }

        assert resolve_audio_policies(cfg) == {
            "global_bgm": "off",
            "mute_audio": "silence",
            "ambient": "whole_video",
        }

    def test_resolve_audio_policies_rejects_invalid_value(self):
        with pytest.raises(ValueError, match=r"audio_policies\.global_bgm không hợp lệ"):
            resolve_audio_policies({"audio_policies": {"global_bgm": "mute_only"}})


class TestLayer1_AudioAssemblerMaskLogic:
    def test_build_mute_ranges(self):
        timeline = [
            _segment(0.0, 10_000.0, 0.0, 10_000.0, "gap"),
            _segment(10_000.0, 20_000.0, 10_000.0, 30_000.0, "mute", video_speed=0.5),
            _segment(20_000.0, 30_000.0, 30_000.0, 40_000.0, "tts"),
        ]

        assert build_mute_ranges(timeline) == [(10.0, 30.0)]

    def test_build_ambient_mask(self):
        timeline = [
            _segment(0.0, 10_000.0, 0.0, 10_000.0, "gap"),
            _segment(10_000.0, 20_000.0, 10_000.0, 30_000.0, "mute", video_speed=0.5),
            _segment(20_000.0, 30_000.0, 30_000.0, 40_000.0, "tts"),
        ]

        mask = build_ambient_mask(timeline, total_ms=50_000.0)
        assert mask == [(0.0, 10_000.0), (30_000.0, 50_000.0)]

    def test_build_mute_volume_filter_without_mute_ranges(self):
        assert _build_mute_volume_filter([], 0.25) == "volume=0.250000"

    def test_build_mute_volume_filter_with_mute_ranges(self):
        volume_filter = _build_mute_volume_filter([(10.0, 30.0), (45.5, 50.25)], 0.35)

        assert volume_filter.startswith("volume='if(")
        assert "between(t,10.000,30.000)" in volume_filter
        assert "between(t,45.500,50.250)" in volume_filter
        assert ", 0, 0.350000)" in volume_filter
        assert volume_filter.endswith("':eval=frame")


class TestLayer1_AudioAssemblerVolumeConfigLogic:
    def test_compress_tts_clip_applies_non_voicevox_volume_filter(self):
        with patch("sync_engine.audio_assembler.subprocess.run") as run_mock:
            compress_tts_clip(
                "input.wav",
                1.0,
                "output.wav",
                tts_provider="edge",
                target_dur_s=1.0,
                non_voicevox_tts_volume=1.75,
            )

        cmd = run_mock.call_args.args[0]
        filter_str = cmd[cmd.index("-filter:a") + 1]
        assert "volume=1.750000" in filter_str
        assert "apad=whole_dur=1.000000" in filter_str

    def test_compress_tts_clip_ignores_render_volume_for_voicevox_family(self):
        with patch("sync_engine.audio_assembler.subprocess.run") as run_mock:
            compress_tts_clip(
                "input.wav",
                1.0,
                "output.wav",
                tts_provider="voicevox_nemo",
                target_dur_s=1.0,
                non_voicevox_tts_volume=3.0,
            )

        cmd = run_mock.call_args.args[0]
        filter_str = cmd[cmd.index("-filter:a") + 1]
        assert "volume=" not in filter_str
        assert "apad=whole_dur=1.000000" in filter_str

    def test_assemble_audio_track_passes_mute_audio_volume_to_original_mute_chunk(self, tmp_path):
        timeline = [_segment(0.0, 1000.0, 0.0, 1000.0, "mute")]
        output_wav = str(tmp_path / "mixed.wav")
        finalized_volumes = []

        def fake_finalize(*args, **kwargs):
            finalized_volumes.append(kwargs["volume"])
            Path(args[1]).write_bytes(b"fake wav")

        with (
            patch.object(audio_assembler, "extract_quoted_audio", return_value=0.0),
            patch.object(audio_assembler, "_finalize_audio_chunk", side_effect=fake_finalize),
            patch.object(audio_assembler, "_generate_silence_chunk"),
            patch.object(audio_assembler.concurrent.futures, "ThreadPoolExecutor") as executor_cls,
            patch.object(audio_assembler.concurrent.futures, "as_completed", side_effect=lambda futures: futures),
            patch.object(audio_assembler.subprocess, "run"),
            patch.object(audio_assembler.shutil, "copy"),
        ):
            executor = executor_cls.return_value.__enter__.return_value
            executor.submit.side_effect = lambda fn, *args, **kwargs: _ImmediateFuture(fn(*args, **kwargs))

            assemble_audio_track(
                timeline=timeline,
                video_path="fake_video.mp4",
                ambient_path=None,
                output_path=output_wav,
                tmp_dir=str(tmp_path),
                audio_mix_config={"mute_audio_volume": 0.42},
                audio_policies={"mute_audio": "original", "ambient": "off", "global_bgm": "off"},
            )

        assert finalized_volumes == [0.42]


class _ImmediateFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — COMPONENT TESTS
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg không có trong PATH")
class TestLayer2_AudioAssemblerIntegration:
    def test_compress_tts_clip_with_target_dur(self, short_tts_wav, tmp_path):
        """Kiểm tra compress_tts_clip với target_dur_s áp dụng apad/atrim."""
        out_path = tmp_path / "compressed.wav"

        compress_tts_clip(
            str(short_tts_wav),
            1.0,
            str(out_path),
            tts_provider="edge",
            target_dur_s=1.5,
        )

        assert out_path.exists()
        dur = _get_duration_ffprobe(str(out_path))
        assert abs(dur - 1.5) < 0.01

        out_path_short = tmp_path / "compressed_short.wav"
        compress_tts_clip(
            str(short_tts_wav),
            1.0,
            str(out_path_short),
            tts_provider="edge",
            target_dur_s=0.2,
        )
        dur_short = _get_duration_ffprobe(str(out_path_short))
        assert abs(dur_short - 0.2) < 0.01

    def test_assemble_audio_track_single(self, synthetic_ambient_wav, tmp_path):
        """Test track TTS + ambient với explicit audio_policies mới."""
        timeline = [
            _segment(
                0.0,
                1000.0,
                0.0,
                1000.0,
                "tts",
                tts_clip_path=str(synthetic_ambient_wav),
                tts_duration=1000.0,
            ),
        ]

        output_wav = str(tmp_path / "mixed.wav")
        assemble_audio_track(
            timeline=timeline,
            video_path="fake_video.mp4",
            ambient_path=str(synthetic_ambient_wav),
            output_path=output_wav,
            tmp_dir=str(tmp_path),
            audio_policies={"ambient": "whole_video"},
        )

        out_path = Path(output_wav)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

        duration_s = _get_duration_ffprobe(output_wav)
        assert abs(duration_s - 1.0) < 0.1

    def test_assemble_audio_track_multi_segment_concat(self, short_tts_wav, synthetic_ambient_wav, tmp_path):
        """
        Kiểm tra cơ chế concat với multi-segment timeline:
        - Gap 1s
        - TTS 1.5s (dùng clip gốc 0.5s)
        - Tail 0.5s
        Tổng = 3.0s
        """
        timeline = [
            _segment(0.0, 1000.0, 0.0, 1000.0, "gap"),
            _segment(
                1000.0,
                2000.0,
                1000.0,
                2500.0,
                "tts",
                video_speed=0.666,
                tts_clip_path=str(short_tts_wav),
                tts_duration=500.0,
            ),
            _segment(2000.0, 2500.0, 2500.0, 3000.0, "tail"),
        ]

        output_wav = str(tmp_path / "mixed_concat.wav")
        assemble_audio_track(
            timeline=timeline,
            video_path="fake_video.mp4",
            ambient_path=str(synthetic_ambient_wav),
            output_path=output_wav,
            tmp_dir=str(tmp_path),
            audio_policies={"ambient": "exclude_mute"},
        )

        out_path = Path(output_wav)
        assert out_path.exists()

        duration_s = _get_duration_ffprobe(output_wav)
        assert abs(duration_s - 3.0) < 0.05
