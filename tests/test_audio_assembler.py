#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_audio_assembler.py
===========================
Test Component Phase 3: Assemble Audio Track.
build_ambient_mask, assemble_audio_track, compress_tts_clip.

Cấu trúc layers:
  Layer 1 — Unit Tests          (Test mask logic)
  Layer 2 — Component Tests     (Test với synthetic audio)

Cách chạy từng layer:
    pytest tests/test_audio_assembler.py -v -k "Layer1"
    pytest tests/test_audio_assembler.py -v -k "Layer2"
"""

import sys
import shutil
import subprocess
from pathlib import Path

import pytest

# ── Project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sync_engine.models import TimelineSegment
from sync_engine.audio_assembler import build_ambient_mask, assemble_audio_track, compress_tts_clip

# ═════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ═════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def synthetic_ambient_wav(tmp_path_factory) -> Path:
    """WAV 1 giây silence tạo bằng FFmpeg."""
    tmp_dir = tmp_path_factory.mktemp("audio")
    path = tmp_dir / "ambient.wav"

    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "anullsrc=r=48000:cl=stereo",
        "-t", "1.0",
        str(path),
    ], check=True, capture_output=True)

    return path

@pytest.fixture(scope="module")
def short_tts_wav(tmp_path_factory) -> Path:
    """WAV 0.5 giây tạo bằng FFmpeg (dùng làm mock TTS clip)."""
    tmp_dir = tmp_path_factory.mktemp("audio_tts")
    path = tmp_dir / "short_tts.wav"

    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        # Dùng sine wave beep thay vì silence để dễ phân biệt
        "-i", "sine=frequency=1000:sample_rate=48000:duration=0.5",
        "-ac", "2",
        str(path),
    ], check=True, capture_output=True)

    return path

def _get_duration_ffprobe(wav_path: str) -> float:
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        wav_path,
    ]
    res = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
    return float(res.stdout.strip())

# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_AudioAssemblerUnit:
    def test_build_ambient_mask(self):
        timeline = [
            TimelineSegment(orig_start=0, orig_end=10, new_start=0, new_end=10, video_speed=1, audio_speed=1, new_chunk_dur=10, block_type="gap", tts_clip_path=None, tts_duration=0),
            TimelineSegment(orig_start=10, orig_end=20, new_start=10, new_end=30, video_speed=0.5, audio_speed=1, new_chunk_dur=20, block_type="mute", tts_clip_path=None, tts_duration=0),
            TimelineSegment(orig_start=20, orig_end=30, new_start=30, new_end=40, video_speed=1, audio_speed=1, new_chunk_dur=10, block_type="tts", tts_clip_path=None, tts_duration=0),
        ]
        total_ms = 50.0
        mask = build_ambient_mask(timeline, total_ms)
        
        # Mute từ 10->30
        # Nên ambient sẽ phát ở 0->10, và 30->50
        assert mask == [(0.0, 10.0), (30.0, 50.0)]


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — COMPONENT TESTS
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg không có trong PATH")
class TestLayer2_AudioAssemblerIntegration:
    def test_compress_tts_clip_with_target_dur(self, short_tts_wav, tmp_path):
        """Kiểm tra compress_tts_clip với target_dur_s áp dụng apad/atrim."""
        out_path = tmp_path / "compressed.wav"
        
        # Clip gốc 0.5s, target 1.5s -> FFmpeg phải apad thêm 1.0s silence
        compress_tts_clip(
            str(short_tts_wav), 1.0, str(out_path), 
            tts_provider="edge", target_dur_s=1.5
        )
        
        assert out_path.exists()
        dur = _get_duration_ffprobe(str(out_path))
        # Phải chính xác 1.5s
        assert abs(dur - 1.5) < 0.01

        # Test case 2: target ngắn hơn clip gốc -> phải atrim
        out_path_short = tmp_path / "compressed_short.wav"
        compress_tts_clip(
            str(short_tts_wav), 1.0, str(out_path_short), 
            tts_provider="edge", target_dur_s=0.2
        )
        dur_short = _get_duration_ffprobe(str(out_path_short))
        assert abs(dur_short - 0.2) < 0.01

    def test_assemble_audio_track_single(self, synthetic_ambient_wav, tmp_path):
        """Test layer ambient + tts mà không cần video (fake path video)."""
        timeline = [
            TimelineSegment(
                orig_start=0.0, orig_end=1000.0,
                new_start=0.0, new_end=1000.0,
                video_speed=1.0, audio_speed=1.0, new_chunk_dur=1000.0,
                block_type="tts", tts_clip_path=str(synthetic_ambient_wav), tts_duration=1000.0
            ),
        ]
        
        output_wav = str(tmp_path / "mixed.wav")
        assemble_audio_track(
            timeline=timeline,
            video_path="fake_video.mp4", # Sẽ bị bỏ qua vì không có block "mute"
            ambient_path=str(synthetic_ambient_wav),
            output_path=output_wav,
            tmp_dir=str(tmp_path)
        )
        
        out_path = Path(output_wav)
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        
        duration_s = _get_duration_ffprobe(output_wav)
        assert abs(duration_s - 1.0) < 0.1

    def test_assemble_audio_track_multi_segment_concat(self, short_tts_wav, synthetic_ambient_wav, tmp_path):
        """
        Kiểm tra cơ chế Concat với multi-segment timeline:
        - Gap 1s
        - TTS 1.5s (dùng clip gốc 0.5s)
        - Tail 0.5s
        Tổng = 3.0s
        """
        timeline = [
            TimelineSegment(
                orig_start=0.0, orig_end=1000.0, new_start=0.0, new_end=1000.0,
                video_speed=1.0, audio_speed=1.0, new_chunk_dur=1000.0, # 1.0s
                block_type="gap", tts_clip_path=None, tts_duration=0.0
            ),
            TimelineSegment(
                orig_start=1000.0, orig_end=2000.0, new_start=1000.0, new_end=2500.0,
                video_speed=0.666, audio_speed=1.0, new_chunk_dur=1500.0, # 1.5s
                block_type="tts", tts_clip_path=str(short_tts_wav), tts_duration=500.0
            ),
            TimelineSegment(
                orig_start=2000.0, orig_end=2500.0, new_start=2500.0, new_end=3000.0,
                video_speed=1.0, audio_speed=1.0, new_chunk_dur=500.0, # 0.5s
                block_type="tail", tts_clip_path=None, tts_duration=0.0
            ),
        ]
        
        output_wav = str(tmp_path / "mixed_concat.wav")
        assemble_audio_track(
            timeline=timeline,
            video_path="fake_video.mp4",
            ambient_path=str(synthetic_ambient_wav),
            output_path=output_wav,
            tmp_dir=str(tmp_path)
        )
        
        out_path = Path(output_wav)
        assert out_path.exists()
        
        # Verify tổng duration phải chính xác 3.0s
        duration_s = _get_duration_ffprobe(output_wav)
        assert abs(duration_s - 3.0) < 0.05

