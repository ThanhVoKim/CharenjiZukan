#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/sync_engine/test_tuber_mouth_events.py
=============================================
Test cho sync_engine/tuber_mouth_events.py.

Cấu trúc layers:
  Layer 1 — Unit: amplitude analysis, state mapping, silence merge
  Layer 2 — Component: TTS clip thật → mouthEvents
  Layer 3 — Integration: mouthEvents trong manifest → composite

Cách chạy:
    pytest tests/sync_engine/test_tuber_mouth_events.py -v -k "Layer1"
    pytest tests/sync_engine/test_tuber_mouth_events.py -v -k "Layer2"
    pytest tests/sync_engine/test_tuber_mouth_events.py -v -k "Layer3"
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sync_engine.tuber_mouth_events import (
    analyze_tts_amplitude,
    build_mouth_events_for_segment,
    build_mouth_events_for_segments,
    _read_wav_rms,
    _rms_to_db,
    _state_from_amplitude,
    _merge_short_silence,
)


# ══════════════════════════════════════════════════════════════════════════
# LAYER 1 — Unit tests
# ══════════════════════════════════════════════════════════════════════════

class TestLayer1_AmplitudeAnalysis:
    """Unit: _state_from_amplitude, _merge_short_silence, _rms_to_db."""

    def test_silence_is_closed(self):
        assert _state_from_amplitude(0.0) == "closed"
        assert _state_from_amplitude(0.001) in ("closed", "half")

    def test_half_state(self):
        state = _state_from_amplitude(0.1)
        assert state == "half"

    def test_open_state(self):
        state = _state_from_amplitude(0.8)
        assert state == "open"

    def test_3_state_distribution(self):
        for amp in [0.001, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0]:
            s = _state_from_amplitude(amp, num_states=3)
            assert s in ("closed", "half", "open"), f"amp={amp} -> {s}"

    def test_rms_to_db_inf_silence(self):
        assert _rms_to_db(0) == -float("inf")

    def test_rms_to_db_positive(self):
        db = _rms_to_db(32768)
        assert db == 0.0  # full scale = 0 dB

    def test_merge_short_silence_noop(self):
        events = [
            {"frame": 0, "state": "closed"},
            {"frame": 30, "state": "open"},
        ]
        merged = _merge_short_silence(events, 10)
        assert merged == events  # không có closed ở giữa

    def test_merge_short_silence_removes(self):
        events = [
            {"frame": 0, "state": "open"},
            {"frame": 10, "state": "closed"},
            {"frame": 15, "state": "open"},  # closed quá ngắn (5 frames < 10 min)
        ]
        merged = _merge_short_silence(events, 10)
        # closed bị bỏ → open được nối với open → merge thành 1 element
        assert len(merged) == 1
        assert merged[0]["state"] == "open"

    def test_merge_short_silence_keeps_long(self):
        events = [
            {"frame": 0, "state": "open"},
            {"frame": 10, "state": "closed"},
            {"frame": 25, "state": "open"},  # closed 15 frames >= 10 min
        ]
        merged = _merge_short_silence(events, 10)
        assert len(merged) == 3

    def test_merge_short_silence_keeps_early_onset_frame(self):
        """Regression: gộp 2 đoạn 'open' quanh 1 'closed' ngắn KHÔNG được dời
        onset 'open' về frame sau.

        Bug cũ: bước merge consecutive same-state ghi đè
        ``merged[-1]['frame'] = ev['frame']`` → onset 'open' bị dời từ frame 7
        sang 39 (~1s ở 30fps) → miệng mở trễ so với audio (cả EdgeTTS lẫn
        Voicevox). Dữ liệu dựng lại đúng từ dubb-2.wav thật.
        """
        events = [
            {"frame": 0, "state": "closed"},
            {"frame": 7, "state": "open"},     # onset thật của speech
            {"frame": 35, "state": "closed"},  # nghỉ ngắn 4 frame (< min 6)
            {"frame": 39, "state": "open"},    # tiếp tục nói
        ]
        merged = _merge_short_silence(events, 6)
        open_events = [e for e in merged if e["state"] == "open"]
        assert len(open_events) == 1, f"2 đoạn open phải gộp thành 1: {merged}"
        assert open_events[0]["frame"] == 7, (
            f"onset 'open' phải giữ frame 7 (sớm nhất), không bị dời về sau: {merged}"
        )

    def test_empty_rms_raises_for_nonexistent(self):
        with pytest.raises(RuntimeError):
            _read_wav_rms(Path(tempfile.gettempdir()) / "nonexistent_xyz.wav", 30)

    def test_build_segment_no_tts(self):
        ev = build_mouth_events_for_segment(0, 100, False, None, 30)
        assert ev is None

    def test_build_segment_notts_no_file(self):
        ev = build_mouth_events_for_segment(0, 100, True, Path("nonexistent.wav"), 30)
        assert ev is None


class TestLayer1_HybridMode:
    """Unit: nhánh hybrid cadence debounce (phân tích source code)."""

    def test_hybrid_accepts_mode_param(self):
        """analyze_tts_amplitude nhận mode param."""
        from sync_engine import tuber_mouth_events
        import inspect
        sig = inspect.signature(tuber_mouth_events.analyze_tts_amplitude)
        assert "mode" in sig.parameters

    def test_hybrid_debounce_in_source(self):
        """Hybrid: source code chứa logic debounce (mode=='hybrid')."""
        from sync_engine import tuber_mouth_events
        import inspect
        src = inspect.getsource(tuber_mouth_events.analyze_tts_amplitude)
        assert 'mode == "hybrid"' in src, "Source phải chứa hybrid debounce"
        assert "cadence_frames" in src, "Source phải tính cadence_frames"
        assert "silence_frames" not in src

    def test_hybrid_source_allows_silence_override(self):
        """Hybrid source: chuyển sang closed KHÔNG bị debounce."""
        from sync_engine import tuber_mouth_events
        import inspect
        src = inspect.getsource(tuber_mouth_events.analyze_tts_amplitude)
        # state != "closed" → non-closed mới bị debounce
        assert 'state != "closed"' in src


# ══════════════════════════════════════════════════════════════════════════
# LAYER 2 — Component tests (cần TTS WAV thật)
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not os.environ.get("TTS_CLIP_DIR"),
                    reason="Set TTS_CLIP_DIR để test với WAV thật")
class TestLayer2_RealAudio:
    """Component: analyze_tts_amplitude với TTS WAV thật."""

    def test_analyze_tts_amplitude(self):
        tts_dir = Path(os.environ["TTS_CLIP_DIR"])
        clips = list(tts_dir.glob("*.wav"))
        if not clips:
            pytest.skip("Không có WAV file trong TTS_CLIP_DIR")
        events = analyze_tts_amplitude(clips[0], 30)
        assert len(events) > 0
        for ev in events:
            assert "frame" in ev
            assert ev["state"] in ("closed", "half", "open")

    def test_events_have_continuous_frames(self):
        tts_dir = Path(os.environ["TTS_CLIP_DIR"])
        clips = list(tts_dir.glob("*.wav"))
        if not clips:
            pytest.skip("Không có WAV file trong TTS_CLIP_DIR")
        events = analyze_tts_amplitude(clips[0], 30)
        frames = [ev["frame"] for ev in events]
        for i in range(1, len(frames)):
            assert frames[i] > frames[i - 1], "Frames không liên tục"

    def test_loud_audio_produces_open_state(self):
        tts_dir = Path(os.environ["TTS_CLIP_DIR"])
        clips = list(tts_dir.glob("*.wav"))
        if not clips:
            pytest.skip("Không có WAV file trong TTS_CLIP_DIR")
        events = analyze_tts_amplitude(clips[0], 30, silence_db=-60.0)
        # Với ngưỡng thấp -60dB, hầu hết các phần sẽ có state không phải closed
        open_states = [e for e in events if e["state"] != "closed"]
        assert len(open_states) > 0, "Audio > -60dB phải tạo được non-closed state"


# ══════════════════════════════════════════════════════════════════════════
# LAYER 3 — Integration tests
# ══════════════════════════════════════════════════════════════════════════

class TestLayer3_MouthEventsIntegration:
    """Integration: mouthEvents build and format."""

    def test_build_mouth_events_for_segments_empty(self):
        segs = []
        result = build_mouth_events_for_segments(segs, 30)
        assert result == []

    def test_build_mouth_events_for_segments_no_tts(self):
        segs = [
            {"segmentIndex": 0, "startFrame": 0, "endFrame": 50,
             "hasTts": False, "blockType": "mute"},
        ]
        result = build_mouth_events_for_segments(segs, 30)
        assert result[0]["mouthEvents"] is None

    def test_build_mouth_events_with_missing_clip_path(self):
        segs = [
            {"segmentIndex": 0, "startFrame": 0, "endFrame": 50,
             "hasTts": True, "blockType": "tts",
             "tts_clip_path": "/nonexistent/path.wav"},
        ]
        result = build_mouth_events_for_segments(segs, 30)
        assert result[0]["mouthEvents"] is None

    def test_segment_mouth_events_structure(self):
        # Không có TTS file thật → kiểm tra cấu trúc output đúng
        segs = [
            {"segmentIndex": 0, "startFrame": 0, "endFrame": 50,
             "hasTts": True, "blockType": "tts",
             "tts_clip_path": "/nonexistent.wav"},
            {"segmentIndex": 1, "startFrame": 50, "endFrame": 100,
             "hasTts": False, "blockType": "mute"},
        ]
        result = build_mouth_events_for_segments(segs, 30)
        assert len(result) == 2
        # segment không TTS → null
        assert result[1]["mouthEvents"] is None
        # segment TTS thiếu file → null
        assert result[0]["mouthEvents"] is None

    def test_json_serializable(self):
        """Verify mouthEvents có thể serialize được JSON."""
        events = [
            {"frame": 0, "state": "closed"},
            {"frame": 10, "state": "half"},
            {"frame": 20, "state": "open"},
        ]
        json_str = json.dumps(events)
        loaded = json.loads(json_str)
        assert loaded == events
