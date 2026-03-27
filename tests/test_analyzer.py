#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_analyzer.py
===========================
Test core logic Phase 0 và 1 của TTS-Video Sync:
filter_tts_subtitles, classify_and_compute_slots, compute_speeds, build_timeline_map.

Cấu trúc layers:
  Layer 1 — Unit Tests          (không cần GPU/FFmpeg/video)

Cách chạy từng layer:
    pytest tests/test_analyzer.py -v -k "Layer1"
"""

import sys
from pathlib import Path
from typing import List

import pytest

# ── Project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sync_engine.models import SubBlock, TimelineSegment
from sync_engine.analyzer import (
    filter_tts_subtitles,
    classify_and_compute_slots,
    compute_speeds,
    build_timeline_map,
    remap_timestamp
)

# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_AnalyzerFilterAndClassify:
    """Test filter_tts_subtitles và classify_and_compute_slots."""

    def test_filter_tts_subtitles(self):
        subs = [
            {"line": 1, "start_time": 0, "end_time": 5000, "text": "A"},
            {"line": 2, "start_time": 6000, "end_time": 10000, "text": "B"},
            {"line": 3, "start_time": 11000, "end_time": 15000, "text": "C"},
        ]
        mutes = [
            {"start_time": 4000, "end_time": 7000, "text": "[MUTE]"}
        ]
        
        filtered = filter_tts_subtitles(subs, mutes)
        assert len(filtered) == 1
        assert filtered[0]["text"] == "C"
        assert filtered[0]["line"] == 1  # Được đánh số lại

    def test_classify_and_compute_slots(self):
        subs = [
            {"line": 1, "start_time": 10000, "end_time": 15000, "text": "Sub1"},
            {"line": 2, "start_time": 20000, "end_time": 25000, "text": "Sub2"},
            {"line": 3, "start_time": 45000, "end_time": 50000, "text": "Sub4"},
        ]
        mutes = [
            {"line": 1, "start_time": 30000, "end_time": 40000, "text": "Mute1"}
        ]
        video_dur = 60000.0

        blocks = classify_and_compute_slots(subs, mutes, video_dur)
        
        # HEAD GAP [0->10000]
        assert blocks[0].type == "gap"
        assert blocks[0].start_time == 0
        assert blocks[0].slot_duration == 10000
        
        # Sub1 [10000->20000]
        assert blocks[1].type == "tts"
        assert blocks[1].start_time == 10000
        assert blocks[1].slot_duration == 10000
        assert blocks[1].hard_limit_ms is None
        
        # Sub2 [20000->30000] slot_duration is limited by next mute
        assert blocks[2].type == "tts"
        assert blocks[2].start_time == 20000
        assert blocks[2].slot_duration == 10000
        assert blocks[2].hard_limit_ms == 10000
        
        # Mute1 [30000->40000]
        assert blocks[3].type == "mute"
        assert blocks[3].start_time == 30000
        assert blocks[3].slot_duration == 10000
        
        # GAP [40000->45000]
        assert blocks[4].type == "gap"
        assert blocks[4].start_time == 40000
        assert blocks[4].slot_duration == 5000
        
        # Sub4 [45000->60000] slot_duration is limited by video_dur
        assert blocks[5].type == "tts"
        assert blocks[5].start_time == 45000
        assert blocks[5].slot_duration == 15000
        assert blocks[5].hard_limit_ms is None


class TestLayer1_AnalyzerSpeedsAndTimeline:
    """Test compute_speeds, build_timeline_map và remap_timestamp."""

    def test_compute_speeds(self):
        # Case 1: tts_ms <= slot_ms
        vs, as_, nd = compute_speeds(tts_ms=8000, slot_ms=10000, cap=0.5)
        assert vs == 1.0 and as_ == 1.0 and nd == 10000

        # Case 2: slot_ms < tts_ms <= slot_ms / cap
        vs, as_, nd = compute_speeds(tts_ms=15000, slot_ms=10000, cap=0.5)
        assert vs == 10000 / 15000  # 0.666
        assert as_ == 1.0
        assert nd == 15000
        
        # Case 3: tts_ms > slot_ms / cap
        vs, as_, nd = compute_speeds(tts_ms=25000, slot_ms=10000, cap=0.5)
        assert vs == 0.5
        assert nd == 20000  # 10000 / 0.5
        assert as_ == 25000 / 20000  # 1.25

    def test_compute_speeds_with_hard_limit(self):
        """Hard limit nhỏ hơn slot → effective = hard_limit."""
        vs, as_, nd = compute_speeds(tts_ms=9000, slot_ms=10000, hard_limit_ms=8000, cap=0.5)
        # effective=8000, max_str=16000, tts=9000 > 8000 → Case 2
        assert vs == pytest.approx(8000/9000, rel=1e-3)
        assert as_ == 1.0
        assert nd  == 9000

    def test_build_timeline_map(self):
        blocks = [
            SubBlock(type="gap", start_time=0, end_time=10000, slot_duration=10000, hard_limit_ms=None, tts_clip_path=None, tts_duration=0),
            SubBlock(type="tts", start_time=10000, end_time=15000, slot_duration=10000, hard_limit_ms=None, tts_clip_path="dubb-0.wav", tts_duration=15000),
            SubBlock(type="tts", start_time=20000, end_time=25000, slot_duration=10000, hard_limit_ms=10000, tts_clip_path="dubb-1.wav", tts_duration=8000),
            SubBlock(type="mute", start_time=30000, end_time=40000, slot_duration=10000, hard_limit_ms=None, tts_clip_path=None, tts_duration=0),
        ]
        speeds = [
            (1.0, 1.0, 10000), # gap
            (10/15, 1.0, 15000), # tts1 (stretched)
            (1.0, 1.0, 10000), # tts2
            (1.0, 1.0, 10000), # mute
        ]
        video_dur = 50000.0
        
        timeline = build_timeline_map(blocks, speeds, video_dur)
        
        assert len(timeline) == 5 # 4 blocks + 1 tail
        
        assert timeline[0].block_type == "gap"
        assert timeline[0].new_chunk_dur == 10000
        assert timeline[0].new_start == 0
        
        assert timeline[1].block_type == "tts"
        assert timeline[1].new_chunk_dur == 15000
        assert timeline[1].new_start == 10000
        assert timeline[1].new_end == 25000
        
        assert timeline[2].block_type == "tts"
        assert timeline[2].new_start == 25000
        assert timeline[2].new_end == 35000
        
        assert timeline[3].block_type == "mute"
        assert timeline[3].new_start == 35000
        assert timeline[3].new_end == 45000
        
        assert timeline[4].block_type == "tail"
        assert timeline[4].orig_start == 40000
        assert timeline[4].orig_end == 50000
        assert timeline[4].new_chunk_dur == 10000
        assert timeline[4].new_start == 45000
        assert timeline[4].new_end == 55000

    def test_remap_timestamp(self):
        timeline = [
            TimelineSegment(orig_start=0, orig_end=10, new_start=0, new_end=10, video_speed=1, audio_speed=1, new_chunk_dur=10, block_type="gap", tts_clip_path=None, tts_duration=0),
            TimelineSegment(orig_start=10, orig_end=20, new_start=10, new_end=30, video_speed=0.5, audio_speed=1, new_chunk_dur=20, block_type="tts", tts_clip_path=None, tts_duration=0),
            TimelineSegment(orig_start=20, orig_end=30, new_start=30, new_end=40, video_speed=1, audio_speed=1, new_chunk_dur=10, block_type="mute", tts_clip_path=None, tts_duration=0),
        ]
        
        assert remap_timestamp(5, timeline) == 5
        assert remap_timestamp(10, timeline) == 10
        assert remap_timestamp(15, timeline) == 20 # (15-10)/(20-10) * 20 + 10 = 0.5 * 20 + 10 = 20
        assert remap_timestamp(20, timeline) == 30
        assert remap_timestamp(25, timeline) == 35
        assert remap_timestamp(35, timeline) == 45 # Extrapolate past end
        assert remap_timestamp(-5, timeline) == -5 # Extrapolate before start
