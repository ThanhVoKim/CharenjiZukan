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
        assert len(filtered) == 2
        assert filtered[0]["text"] == "A"
        assert filtered[0]["end_time"] == 4000
        assert filtered[1]["text"] == "C"

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
            TimelineSegment(orig_start=0, orig_end=1000, new_start=0, new_end=1000, video_speed=1, audio_speed=1, new_chunk_dur=1000, block_type="gap", tts_clip_path=None, tts_duration=0),
            TimelineSegment(orig_start=1000, orig_end=2000, new_start=1000, new_end=3000, video_speed=0.5, audio_speed=1, new_chunk_dur=2000, block_type="tts", tts_clip_path=None, tts_duration=0),
            TimelineSegment(orig_start=2000, orig_end=3000, new_start=3000, new_end=4000, video_speed=1, audio_speed=1, new_chunk_dur=1000, block_type="mute", tts_clip_path=None, tts_duration=0),
        ]
        
        # Test interpolate
        # orig_ms=500 -> in seg 0. ratio=0.5. raw_new=500. new_frames=round(500/1000*30)=15. return 15/30*1000 = 500.0
        assert remap_timestamp(500, timeline, fps_float=30.0) == 500.0
        # orig_ms=1500 -> in seg 1. ratio=0.5. raw_new=1000 + 0.5*2000 = 2000. new_frames=round(2000/1000*30)=60. return 60/30*1000 = 2000.0
        assert remap_timestamp(1500, timeline, fps_float=30.0) == 2000.0 
        
        # Test boundary
        assert remap_timestamp(1000, timeline, fps_float=30.0) == 1000.0
        assert remap_timestamp(2000, timeline, fps_float=30.0) == 3000.0
        
        # Test extrapolate (trước/sau timeline)
        # orig_ms=-500 -> before start. first seg new_start=0, orig_start=0. raw_new = 0 - (0 - -500) = -500. new_frames = round(-500/1000*30) = -15. return -15/30*1000 = -500.0
        assert remap_timestamp(-500, timeline, fps_float=30.0) == -500.0
        # orig_ms=3500 -> after end. last seg new_end=4000, orig_end=3000. raw_new = 4000 + (3500 - 3000) = 4500. new_frames = round(4500/1000*30) = 135. return 135/30*1000 = 4500.0
        assert remap_timestamp(3500, timeline, fps_float=30.0) == 4500.0
