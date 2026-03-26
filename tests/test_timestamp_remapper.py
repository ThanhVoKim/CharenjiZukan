#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_timestamp_remapper.py
===========================
Test Unit Phase 4: Timestamp remapping.
recalculate_srt, recalculate_ass.

Cấu trúc layers:
  Layer 1 — Unit Tests

Cách chạy từng layer:
    pytest tests/test_timestamp_remapper.py -v -k "Layer1"
"""

import sys
from pathlib import Path

import pytest

# ── Project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sync_engine.models import TimelineSegment
from sync_engine.timestamp_remapper import recalculate_srt, recalculate_ass
from utils.srt_parser import parse_srt_file as parse_srt


# ═════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ═════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="function")
def sample_srt_path(tmp_path: Path) -> Path:
    content = (
        "1\n"
        "00:00:10,000 --> 00:00:15,000\n"
        "Test subtitle 1\n\n"
        "2\n"
        "00:00:20,000 --> 00:00:30,000\n"
        "Test subtitle 2\n"
    )
    p = tmp_path / "test.srt"
    p.write_text(content, encoding="utf-8")
    return p

@pytest.fixture(scope="function")
def sample_ass_path(tmp_path: Path) -> Path:
    content = (
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:10.00,0:00:15.00,Default,,0,0,0,,Long text that should be wrapped because it exceeds fifteen characters\n"
    )
    p = tmp_path / "test.ass"
    p.write_text(content, encoding="utf-8")
    return p


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_TimestampRemapperUnit:
    def test_recalculate_srt(self, sample_srt_path, tmp_path):
        timeline = [
            TimelineSegment(
                orig_start=0.0, orig_end=10000.0,
                new_start=0.0, new_end=10000.0,
                video_speed=1.0, audio_speed=1.0, new_chunk_dur=10000.0,
                block_type="gap", tts_clip_path=None, tts_duration=0
            ),
            # Sub1 stretched
            TimelineSegment(
                orig_start=10000.0, orig_end=20000.0,
                new_start=10000.0, new_end=30000.0,
                video_speed=0.5, audio_speed=1.0, new_chunk_dur=20000.0,
                block_type="tts", tts_clip_path=None, tts_duration=12000.0
            ),
            # Sub2 normal
            TimelineSegment(
                orig_start=20000.0, orig_end=30000.0,
                new_start=30000.0, new_end=40000.0,
                video_speed=1.0, audio_speed=1.0, new_chunk_dur=10000.0,
                block_type="tts", tts_clip_path=None, tts_duration=8000.0
            )
        ]
        
        segments = parse_srt(str(sample_srt_path))
        
        # 1. is_tts_track=False (hardsub, remap linear)
        out_srt_1 = tmp_path / "out1.srt"
        recalculate_srt(segments, timeline, str(out_srt_1), is_tts_track=False)
        out_segs_1 = parse_srt(str(out_srt_1))
        
        # Orig sub1: 10k -> 15k
        # new_start = 10k + (15k-10k)/10k * 20k = 10k + 10k = 20k
        assert out_segs_1[0]["start_time"] == 10000.0
        assert out_segs_1[0]["end_time"] == 20000.0
        
        # 2. is_tts_track=True (softsub, neo end_time theo tts_duration)
        out_srt_2 = tmp_path / "out2.srt"
        recalculate_srt(segments, timeline, str(out_srt_2), is_tts_track=True)
        out_segs_2 = parse_srt(str(out_srt_2))
        
        # Sub1 start: 10000. tts_duration = 12000.
        assert out_segs_2[0]["start_time"] == 10000.0
        assert out_segs_2[0]["end_time"] == 10000.0 + 12000.0

    def test_recalculate_ass(self, sample_ass_path, tmp_path):
        timeline = [
            TimelineSegment(
                orig_start=0.0, orig_end=10000.0,
                new_start=0.0, new_end=10000.0,
                video_speed=1.0, audio_speed=1.0, new_chunk_dur=10000.0,
                block_type="gap", tts_clip_path=None, tts_duration=0
            ),
            TimelineSegment(
                orig_start=10000.0, orig_end=20000.0,
                new_start=10000.0, new_end=30000.0,
                video_speed=0.5, audio_speed=1.0, new_chunk_dur=20000.0,
                block_type="tts", tts_clip_path=None, tts_duration=12000.0
            )
        ]
        
        out_ass = tmp_path / "out.ass"
        recalculate_ass(str(sample_ass_path), timeline, str(out_ass), max_chars_per_line=15)
        
        lines = out_ass.read_text(encoding="utf-8").splitlines()
        dialogue = [line for line in lines if line.startswith("Dialogue:")][0]
        
        # Expected new timestamps for 10s->15s (10000->15000)
        # 10s -> 10s, 15s -> 20s
        # ASS Format: 0:00:10.00,0:00:20.00
        assert "0:00:10.00,0:00:20.00" in dialogue
        
        # Kiểm tra wrap_text (Text ở part index 9)
        # text gốc: "Long text that should be wrapped because it exceeds fifteen characters"
        parts = dialogue.split(",", 9)
        text = parts[9]
        # Text phải có newline \N vì bị wrap
        assert "\\N" in text
