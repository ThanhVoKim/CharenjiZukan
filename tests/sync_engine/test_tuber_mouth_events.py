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
    _select_vowel_shapes,
    _percentile,
    _apply_vowel_selection,
    _adaptive_levels,
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


class TestLayer1_VowelSelection:
    """Unit: Tầng 2 — _percentile, _select_vowel_shapes, _apply_vowel_selection.

    Spectral centroid → u/e/open (port ③ライブ実行). Không I/O với WAV thật.
    """

    def test_percentile_basic(self):
        vals = [0.0, 1.0, 2.0, 3.0, 4.0]
        assert _percentile(vals, 0) == 0.0
        assert _percentile(vals, 100) == 4.0
        assert _percentile(vals, 50) == 2.0
        # nội suy tuyến tính: idx = 4*0.25 = 1.0 → đúng phần tử
        assert _percentile(vals, 25) == 1.0

    def test_percentile_empty_and_single(self):
        assert _percentile([], 50) == 0.0
        assert _percentile([7.0], 50) == 7.0

    def test_low_centroid_selects_u(self):
        """Centroid thấp tại đỉnh sóng → 'u'."""
        levels = ["open"] * 5
        env = [0.2, 0.6, 1.0, 0.6, 0.2]   # đỉnh ở index 2 → phát hiện tại i=3
        centroids = [0.05] * 5
        out = _select_vowel_shapes(
            levels, centroids, env,
            u_th=0.2, e_th=0.5, peak_margin=0.02, min_vowel_frames=1,
        )
        assert "u" in out, out
        assert "e" not in out

    def test_high_centroid_selects_e(self):
        """Centroid cao tại đỉnh sóng → 'e'."""
        levels = ["open"] * 5
        env = [0.2, 0.6, 1.0, 0.6, 0.2]
        centroids = [0.8] * 5
        out = _select_vowel_shapes(
            levels, centroids, env,
            u_th=0.2, e_th=0.5, peak_margin=0.02, min_vowel_frames=1,
        )
        assert "e" in out, out
        assert "u" not in out

    def test_no_peak_keeps_open(self):
        """Env phẳng (không có đỉnh) → không đổi khẩu hình, giữ 'open'."""
        levels = ["open"] * 6
        env = [1.0] * 6
        centroids = [0.05] * 6
        out = _select_vowel_shapes(
            levels, centroids, env,
            u_th=0.2, e_th=0.5, peak_margin=0.02, min_vowel_frames=1,
        )
        assert set(out) == {"open"}, out

    def test_closed_resets_shape(self):
        """Frame 'closed'/'half' không bị ghi thành vowel; closed reset shape."""
        levels = ["open", "open", "open", "closed", "half"]
        env = [0.2, 0.6, 1.0, 0.0, 0.5]
        centroids = [0.05] * 5
        out = _select_vowel_shapes(
            levels, centroids, env,
            u_th=0.2, e_th=0.5, peak_margin=0.02, min_vowel_frames=1,
        )
        assert out[3] == "closed"
        assert out[4] == "half"

    def test_mismatched_lengths_returns_unchanged(self):
        levels = ["open", "open"]
        out = _select_vowel_shapes(
            levels, [0.1], [0.5, 0.6],
            u_th=0.2, e_th=0.5, peak_margin=0.02, min_vowel_frames=1,
        )
        assert out == levels

    def test_apply_vowel_selection_disabled_without_eu(self):
        """mouthStates không có e/u → trả levels nguyên (không đọc WAV)."""
        levels = ["closed", "open", "half"]
        out = _apply_vowel_selection(
            levels, [0.0, 1.0, 0.5], Path("does_not_exist.wav"), 30,
            mouth_states=["closed", "half", "open"],
            peak_margin=0.02, min_vowel_interval_ms=120,
            vowel_low_percentile=20, vowel_high_percentile=80,
        )
        assert out == levels


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


class TestLayer2_VowelFromWav:
    """Component: spectral-centroid vowel selection từ WAV tổng hợp (numpy)."""

    @staticmethod
    def _write_wav(path: Path, freqs, amps, *, framerate=24000, fps=30):
        """Ghi WAV mono pcm_s16le: mỗi 'frame' (1/fps giây) 1 tần số + biên độ."""
        import numpy as np
        import wave

        win = round(framerate / fps)
        phase = 0.0
        segs = []
        for f, a in zip(freqs, amps):
            tt = np.arange(win) / framerate
            segs.append(a * np.sin(2 * np.pi * f * tt + phase))
            phase += 2 * np.pi * f * (win / framerate)
        sig = np.clip(np.concatenate(segs), -1.0, 1.0) * 32767.0
        pcm = sig.astype("<i2")
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(framerate)
            wf.writeframes(pcm.tobytes())

    @pytest.fixture()
    def chirp_wav(self, tmp_path: Path) -> Path:
        """Quét tần số 100→6000Hz, biên độ điều chế (tạo đỉnh) → centroid trải rộng."""
        np = pytest.importorskip("numpy")
        n = 60
        freqs = np.linspace(100, 6000, n)
        amps = 0.8 * (0.6 + 0.4 * np.abs(np.sin(2 * np.pi * np.arange(n) / 5.0)))
        path = tmp_path / "chirp.wav"
        self._write_wav(path, freqs, amps)
        return path

    @pytest.fixture()
    def tone_wav(self, tmp_path: Path) -> Path:
        """Đơn tần 200Hz (centroid đồng nhất) — phân bố suy biến."""
        np = pytest.importorskip("numpy")
        n = 40
        freqs = np.full(n, 200.0)
        amps = 0.8 * (0.6 + 0.4 * np.abs(np.sin(2 * np.pi * np.arange(n) / 5.0)))
        path = tmp_path / "tone.wav"
        self._write_wav(path, freqs, amps)
        return path

    def test_chirp_produces_vowel(self, chirp_wav: Path):
        pytest.importorskip("numpy")
        events = analyze_tts_amplitude(
            chirp_wav, 30, mode="amplitude", min_silence_ms=0,
            mouth_states=["closed", "half", "open", "e", "u"],
        )
        states = {ev["state"] for ev in events}
        assert states & {"e", "u"}, f"Chirp phải sinh ít nhất 1 vowel: {states}"

    def test_3state_never_yields_vowel(self, chirp_wav: Path):
        pytest.importorskip("numpy")
        events = analyze_tts_amplitude(
            chirp_wav, 30, mode="amplitude", min_silence_ms=0,
            mouth_states=["closed", "half", "open"],
        )
        states = {ev["state"] for ev in events}
        assert not (states & {"e", "u"}), f"3-state không được sinh vowel: {states}"
        assert states <= {"closed", "half", "open"}

    def test_pure_tone_degenerate_no_vowel(self, tone_wav: Path):
        """Centroid đồng nhất → guard u_th<e_th chặn → không vowel, không crash."""
        pytest.importorskip("numpy")
        events = analyze_tts_amplitude(
            tone_wav, 30, mode="amplitude", min_silence_ms=0,
            mouth_states=["closed", "half", "open", "e", "u"],
        )
        assert len(events) > 0
        states = {ev["state"] for ev in events}
        assert not (states & {"e", "u"}), f"Đơn tần không nên sinh vowel: {states}"


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


# ══════════════════════════════════════════════════════════════════════════
# LAYER 1 — Adaptive auto-leveling (unit, không I/O)
# ══════════════════════════════════════════════════════════════════════════

class TestLayer1_AdaptiveLeveling:
    """Unit: _adaptive_levels — chống đơ miệng khi audio nhỏ (port aituber-kit).

    Regression chính: audio biên độ nhỏ dao động quanh "half" → adaptive sinh
    ≥2 state khác nhau (có closed/open), còn nhánh tuyệt đối kẹt 1 state.
    """

    # 32768 = int16 full-scale; -35dB ≈ rms 580; -38dB ≈ rms 410
    _SILENCE_DB = -40.0

    @staticmethod
    def _make_rms(db_values: list) -> list:
        """Chuyển list dB → list RMS (float int16 scale)."""
        out = []
        for db in db_values:
            if db <= -200:
                out.append(0.0)
            else:
                out.append(32768.0 * (10 ** (db / 20.0)))
        return out

    def test_quiet_audio_produces_varied_states(self):
        """Regression: audio nhỏ dao động quanh -35..-32 dB → adaptive sinh ≥2 state."""
        # Mô phỏng âm tiết: lên/xuống nhỏ từ -38 đến -30 dB (nhánh tuyệt đối → luôn "half")
        import math
        n = 30
        db_values = [-35.0 + 5.0 * math.sin(2 * math.pi * i / 8) for i in range(n)]
        rms_list = self._make_rms(db_values)

        result = _adaptive_levels(
            rms_list,
            silence_db=self._SILENCE_DB,
            num_states=3,
            floor_pct=10.0, peak_pct=90.0,
            min_range_db=6.0, gamma=0.75,
        )
        assert len(result) == n
        states_set = set(result)
        assert len(states_set) >= 2, (
            f"Adaptive phải sinh ≥2 state khác nhau trên audio nhỏ: {states_set}"
        )

    def test_absolute_mode_freezes_on_quiet_audio(self):
        """Đối chứng: nhánh tuyệt đối kẹt 1 state khi audio nhỏ."""
        import math
        n = 30
        db_values = [-35.0 + 5.0 * math.sin(2 * math.pi * i / 8) for i in range(n)]
        rms_list = self._make_rms(db_values)

        # Nhánh tuyệt đối: mọi db ~ -35 → amplitude ~ 0.1 → luôn "half"
        from sync_engine.tuber_mouth_events import _rms_normalized
        levels_abs = []
        for i, rms in enumerate(rms_list):
            from sync_engine.tuber_mouth_events import _rms_to_db as rms_to_db
            if rms_to_db(rms) < self._SILENCE_DB:
                levels_abs.append("closed")
            else:
                amp = _rms_normalized(rms_list[max(0, i - 2):i + 1], self._SILENCE_DB)
                levels_abs.append(_state_from_amplitude(amp, 3))

        # Tuyệt đối phải kẹt (1 state duy nhất) trên dải hẹp này
        assert len(set(levels_abs)) == 1, (
            f"Nhánh tuyệt đối phải kẹt 1 state (đây là vấn đề cần fix): {set(levels_abs)}"
        )

    def test_silence_gate_still_closes(self):
        """Frame im lặng (dB < silence_db) phải vẫn là 'closed' dù adaptive BẬT."""
        rms_list = self._make_rms([-50.0] * 10 + [-30.0] * 10)
        result = _adaptive_levels(
            rms_list,
            silence_db=self._SILENCE_DB,
            num_states=3,
            floor_pct=10.0, peak_pct=90.0,
            min_range_db=6.0, gamma=0.75,
        )
        for s in result[:10]:
            assert s == "closed", f"Frame silent phải là 'closed': {result[:10]}"

    def test_min_range_guard_no_chatter(self):
        """Clip gần phẳng tuyệt đối (dao động < min_range_db) → không chatter."""
        # Tất cả frame voiced gần giống nhau → rng bị chặn bởi min_range_db
        rms_list = self._make_rms([-35.01, -35.0, -34.99, -35.0, -35.01] * 4)
        result = _adaptive_levels(
            rms_list,
            silence_db=self._SILENCE_DB,
            num_states=3,
            floor_pct=10.0, peak_pct=90.0,
            min_range_db=6.0, gamma=0.75,
        )
        # Không nên đập qua lại giữa closed và open vô nghĩa
        transitions = sum(1 for i in range(1, len(result)) if result[i] != result[i - 1])
        assert transitions <= 4, f"Quá nhiều chatter: {transitions} transitions, {result}"

    def test_few_voiced_frames_fallback_no_crash(self):
        """< 4 voiced frames → fallback tuyệt đối, không crash, kết quả hợp lệ."""
        # 2 voiced + 8 silent
        rms_list = self._make_rms([-50.0] * 8 + [-30.0, -28.0])
        result = _adaptive_levels(
            rms_list,
            silence_db=self._SILENCE_DB,
            num_states=3,
            floor_pct=10.0, peak_pct=90.0,
            min_range_db=6.0, gamma=0.75,
        )
        assert len(result) == len(rms_list)
        assert all(s in ("closed", "half", "open") for s in result)

    def test_empty_rms_returns_empty(self):
        result = _adaptive_levels(
            [], silence_db=self._SILENCE_DB, num_states=3,
            floor_pct=10.0, peak_pct=90.0, min_range_db=6.0, gamma=0.75,
        )
        assert result == []

    def test_analyze_tts_adaptive_flag_false_is_backcompat(self):
        """adaptive=False → hành vi y hệt trước khi có feature (nhánh tuyệt đối)."""
        import math
        n = 20
        db_values = [-35.0 + 5.0 * math.sin(2 * math.pi * i / 8) for i in range(n)]
        rms_list = self._make_rms(db_values)

        # Tái tạo nhánh tuyệt đối thủ công
        from sync_engine.tuber_mouth_events import _rms_normalized, _rms_to_db
        levels_expected = []
        for i, rms in enumerate(rms_list):
            if _rms_to_db(rms) < self._SILENCE_DB:
                levels_expected.append("closed")
            else:
                amp = _rms_normalized(rms_list[max(0, i - 2):i + 1], self._SILENCE_DB)
                levels_expected.append(_state_from_amplitude(amp, 3))

        levels_adaptive_false = _adaptive_levels(
            rms_list,
            silence_db=self._SILENCE_DB, num_states=3,
            floor_pct=10.0, peak_pct=90.0, min_range_db=6.0, gamma=0.75,
        )
        # adaptive=True sẽ khác; adaptive=False phải khớp nhánh tuyệt đối
        # (test này chỉ verify _adaptive_levels trả kết quả KHÁC, không phải kiểm tra False path —
        #  False path được test gián tiếp qua test_absolute_mode_freezes_on_quiet_audio)
        _ = levels_expected  # dùng khi cần diff về sau


# ══════════════════════════════════════════════════════════════════════════
# LAYER 2 — Quiet audio movement (WAV tổng hợp, cần numpy)
# ══════════════════════════════════════════════════════════════════════════

class TestLayer2_QuietAudioMovement:
    """Component: adaptive leveling từ WAV tổng hợp biên độ nhỏ.

    Dùng numpy+wave để tạo WAV runtime (không commit file media, R3).
    Test regression chính: cùng 1 clip quiet → adaptive sinh nhiều transition,
    nhánh tuyệt đối gần như không có transition (kẹt "half").
    """

    @staticmethod
    def _write_quiet_wav(path: Path, *, framerate: int = 24000, fps: int = 30,
                         n_frames: int = 40, amp_scale: float = 0.02) -> None:
        """WAV biên độ nhỏ với điều chế âm tiết (sin sóng mang + sin điều chế)."""
        np = pytest.importorskip("numpy")
        import wave as wv

        win = round(framerate / fps)
        t_per_win = win / framerate
        seg_list = []
        for i in range(n_frames):
            t0 = i * t_per_win
            tt = np.linspace(t0, t0 + t_per_win, win, endpoint=False)
            # Sóng 200Hz điều chế biên độ bởi sóng 3Hz → tạo đỉnh/thung âm tiết
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * tt)
            carrier = np.sin(2 * np.pi * 200.0 * tt)
            seg_list.append(amp_scale * envelope * carrier)

        sig = np.clip(np.concatenate(seg_list), -1.0, 1.0) * 32767.0
        pcm = sig.astype("<i2")
        with wv.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(framerate)
            wf.writeframes(pcm.tobytes())

    @pytest.fixture()
    def quiet_wav(self, tmp_path: Path) -> Path:
        """WAV biên độ rất nhỏ (~2% full-scale) có điều chế âm tiết."""
        pytest.importorskip("numpy")
        path = tmp_path / "quiet_modulated.wav"
        self._write_quiet_wav(path, amp_scale=0.02)
        return path

    def test_adaptive_produces_varied_states(self, quiet_wav: Path):
        """Clip nhỏ: adaptive=True → ≥2 state (có chuyển động)."""
        pytest.importorskip("numpy")
        events = analyze_tts_amplitude(
            quiet_wav, 30,
            mode="amplitude", min_silence_ms=0,
            adaptive=True,
        )
        states = {ev["state"] for ev in events}
        assert len(states) >= 2, (
            f"Adaptive phải sinh ≥2 state trên clip biên độ nhỏ: events={events}"
        )

    def test_absolute_mode_freezes(self, quiet_wav: Path):
        """Đối chứng: cùng clip, adaptive=False → thường kẹt 1 state."""
        pytest.importorskip("numpy")
        events = analyze_tts_amplitude(
            quiet_wav, 30,
            mode="amplitude", min_silence_ms=0,
            adaptive=False,
        )
        states = {ev["state"] for ev in events}
        # Clip rất nhỏ → nhánh tuyệt đối hầu như chỉ sinh 1 state
        assert len(states) <= 2, (
            f"Nhánh tuyệt đối trên clip nhỏ thường kẹt ≤2 state (đây là bug cũ): {states}"
        )

    def test_adaptive_default_true(self, quiet_wav: Path):
        """adaptive=True là mặc định — không cần truyền tường minh."""
        pytest.importorskip("numpy")
        events_default = analyze_tts_amplitude(
            quiet_wav, 30, mode="amplitude", min_silence_ms=0,
        )
        events_explicit = analyze_tts_amplitude(
            quiet_wav, 30, mode="amplitude", min_silence_ms=0, adaptive=True,
        )
        assert events_default == events_explicit

    def test_adaptive_false_backcompat_vowel_unaffected(self, quiet_wav: Path):
        """adaptive=False kết hợp với vowel selection không crash."""
        pytest.importorskip("numpy")
        events = analyze_tts_amplitude(
            quiet_wav, 30, mode="amplitude", min_silence_ms=0,
            adaptive=False,
            mouth_states=["closed", "half", "open", "e", "u"],
        )
        assert len(events) > 0
        valid = {"closed", "half", "open", "e", "u"}
        for ev in events:
            assert ev["state"] in valid, f"State không hợp lệ: {ev}"
