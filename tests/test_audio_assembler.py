#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_audio_assembler.py
===========================
Test Component Phase 3: Assemble Audio Track.
build_ambient_mask, assemble_audio_track.

Cấu trúc layers:
  Layer 1 — Unit Tests          (Test mask logic)
  Layer 2 — Component Tests     (Test với synthetic audio)

Cách chạy từng layer:
    pytest tests/test_audio_assembler.py -v -k "Layer1"
    pytest tests/test_audio_assembler.py -v -k "Layer2"
"""

import sys
import shutil
from pathlib import Path

import pytest

# ── Project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sync_engine.models import TimelineSegment
from sync_engine.audio_assembler import build_ambient_mask, assemble_audio_track

pydub = pytest.importorskip("pydub", reason="pip install pydub")
from pydub import AudioSegment


# ═════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ═════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def synthetic_ambient_wav(tmp_path_factory) -> Path:
    """WAV 1 giây silence."""
    tmp_dir = tmp_path_factory.mktemp("audio")
    path = tmp_dir / "ambient.wav"
    silence = AudioSegment.silent(duration=1000, frame_rate=48000)
    silence.set_channels(2).export(str(path), format="wav")
    return path


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
    def test_assemble_audio_track(self, synthetic_ambient_wav, tmp_path):
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
        
        # Kiểm tra duration
        result = AudioSegment.from_file(output_wav)
        assert len(result) == 1000  # 1s
