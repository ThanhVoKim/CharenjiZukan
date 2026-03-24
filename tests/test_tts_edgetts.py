#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_tts_edgetts.py
===========================
Test cho chức năng Text-to-Speech bằng EdgeTTS và Strip Silence logic.

Cấu trúc layers:
  Layer 1 — Unit Tests          (Logic strip_audio_silence thuần túy)
  Layer 2 — Component Tests     (EdgeTTSEngine flow convert mp3 sang wav và strip)
  Layer 3 — Pipeline Integration (Test end-to-end với file SRT chạy qua run_tts)
  Layer 4 — Real Model Tests    (Không áp dụng)

Cách chạy từng layer:
    pytest tests/test_tts_edgetts.py -v -k "Layer1"
    pytest tests/test_tts_edgetts.py -v -k "Layer2"
    pytest tests/test_tts_edgetts.py -v -k "Layer3"
"""

import asyncio
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Lazy imports ─────────────────────────────────────────────────────
pydub = pytest.importorskip("pydub")
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from cli.tts_srt import run_tts
from tts_edgetts import EdgeTTSEngine, convert_to_wav, strip_audio_silence

# ═════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ═════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def synthetic_wav_with_silence(tmp_path_factory) -> Path:
    """
    Tạo file WAV mô phỏng EdgeTTS output:
    [Silence 500ms] + [Audio thực 1000ms] + [Silence 500ms]
    """
    tmp_dir = tmp_path_factory.mktemp("tts_audio")
    path = tmp_dir / "sample_with_silence.wav"
    
    # Tạo silence 500ms (dùng mức âm thanh -60 dBFS)
    silence = AudioSegment.silent(duration=500, frame_rate=48000)
    # Pydub silent function might result in completely empty (-inf dB), 
    # to be safe for detect_nonsilent, we can just use the truly silent one.
    
    # Tạo sine wave giả audio thực 1000ms
    from pydub.generators import Sine
    audio = Sine(440).to_audio_segment(duration=1000).set_frame_rate(48000)
    
    combined = silence + audio + silence
    combined = combined.set_channels(2)
    combined.export(str(path), format="wav")
    
    return path


@pytest.fixture(scope="module")
def synthetic_wav_all_silent(tmp_path_factory) -> Path:
    """
    Tạo file WAV rỗng hoàn toàn dài 1000ms.
    """
    tmp_dir = tmp_path_factory.mktemp("tts_audio_silent")
    path = tmp_dir / "all_silent.wav"
    
    silence = AudioSegment.silent(duration=1000, frame_rate=48000).set_channels(2)
    silence.export(str(path), format="wav")
    
    return path

@pytest.fixture
def mock_edgetts_communicate():
    """Mock edge_tts.Communicate để không gọi mạng."""
    with patch("tts_edgetts.Communicate") as mock_comm:
        instance = MagicMock()
        # Mock hàm save trả về future hoàn thành ngay lập tức
        # Nó sẽ tạo ra 1 file rỗng tại path truyền vào để mô phỏng tải xong
        async def fake_save(path):
            Path(path).touch()
        
        instance.save.side_effect = fake_save
        mock_comm.return_value = instance
        yield mock_comm

# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_StripSilence:
    """Kiểm tra hàm strip_audio_silence."""

    def test_strip_silence_only_tail(self, synthetic_wav_with_silence, tmp_path):
        """Kiểm tra xem nó có CHỈ cắt phần đuôi không (start_ms = 0)."""
        # Copy ra tmp để không ảnh hưởng file gốc
        test_wav = tmp_path / "test_only_tail.wav"
        shutil.copy(synthetic_wav_with_silence, test_wav)
        
        original_seg = AudioSegment.from_file(test_wav)
        original_len = len(original_seg)
        
        assert original_len == 2000  # 500 + 1000 + 500
        
        # Gọi hàm cắt silence
        trimmed_ms = strip_audio_silence(
            str(test_wav),
            silence_thresh_dbfs=-50,
            min_silence_len_ms=100,
            keep_padding_ms=30,
        )
        
        trimmed_seg = AudioSegment.from_file(test_wav)
        trimmed_len = len(trimmed_seg)
        
        # Original: silence(500) + audio(1000) + silence(500)
        # Non-silent block detected: [500, 1500]
        # start_ms = 0  --> Giữ nguyên phần đầu.
        # end_ms = min(2000, 1500 + 30) = 1530
        # Expected new length: 1530
        # Expected trimmed_ms: 2000 - 1530 = 470
        
        assert trimmed_len < original_len
        assert abs(trimmed_len - 1530) < 5
        assert abs(trimmed_ms - 470) < 5
        
        # Kiểm tra phần đầu vẫn là silence
        first_500ms = trimmed_seg[:500]
        nonsilent_in_first = detect_nonsilent(first_500ms, silence_thresh=-50, min_silence_len=100)
        assert len(nonsilent_in_first) == 0  # Toàn bộ phần đầu vẫn là im lặng

    def test_strip_silence_all_silent(self, synthetic_wav_all_silent, tmp_path):
        """Kiểm tra fallback khi file chỉ toàn im lặng (TTS sinh file rỗng)."""
        test_wav = tmp_path / "test_all_silent.wav"
        shutil.copy(synthetic_wav_all_silent, test_wav)
        
        original_len = len(AudioSegment.from_file(test_wav))
        
        trimmed_ms = strip_audio_silence(str(test_wav))
        
        trimmed_len = len(AudioSegment.from_file(test_wav))
        
        assert trimmed_ms == 0
        assert original_len == trimmed_len


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — COMPONENT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestLayer2_EdgeTTSEngine:
    """Kiểm tra EdgeTTSEngine và luồng convert + strip silence."""

    @patch("tts_edgetts.convert_to_wav")
    def test_engine_run_with_strip_silence(self, mock_convert, mock_edgetts_communicate, synthetic_wav_with_silence, tmp_path):
        """Test engine hoạt động bình thường, sinh mp3 giả và convert sang wav, sau đó strip."""
        
        # Chuẩn bị queue test
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        filename = str(cache_dir / "test01.wav")
        
        queue_tts = [{
            "text": "Hello world",
            "filename": filename
        }]
        
        # Giả lập convert_to_wav: nó copy synthetic_wav_with_silence vào filename.wav
        def fake_convert(mp3, wav):
            shutil.copy(synthetic_wav_with_silence, wav)
            return True
            
        mock_convert.side_effect = fake_convert
        
        # Chạy engine bật tính năng cắt silence
        engine = EdgeTTSEngine(
            queue_tts=queue_tts,
            voice="en-US-JennyNeural",
            strip_silence=True,
            keep_padding_ms=0,
        )
        
        stats = engine.run()
        
        assert stats["ok"] == 1
        assert stats["err"] == 0
        
        assert Path(filename).exists()
        
        # Vì synthetic_wav_with_silence có độ dài 2000ms
        # Với keep_padding_ms=0, start=0, đoạn non-silent kết thúc ở 1500ms
        # Đoạn wav xuất ra cuối cùng sẽ có độ dài 1500ms thay vì 2000ms.
        final_seg = AudioSegment.from_file(filename)
        assert abs(len(final_seg) - 1500) < 5

    @patch("tts_edgetts.convert_to_wav")
    def test_engine_run_without_strip_silence(self, mock_convert, mock_edgetts_communicate, synthetic_wav_with_silence, tmp_path):
        """Test engine khi cờ strip_silence=False."""
        
        cache_dir = tmp_path / "cache2"
        cache_dir.mkdir()
        filename = str(cache_dir / "test02.wav")
        
        queue_tts = [{
            "text": "Hello world 2",
            "filename": filename
        }]
        
        def fake_convert(mp3, wav):
            shutil.copy(synthetic_wav_with_silence, wav)
            return True
            
        mock_convert.side_effect = fake_convert
        
        # Chạy engine TẮT tính năng cắt silence
        engine = EdgeTTSEngine(
            queue_tts=queue_tts,
            voice="en-US-JennyNeural",
            strip_silence=False,
        )
        
        engine.run()
        
        final_seg = AudioSegment.from_file(filename)
        # Giữ nguyên độ dài 2000ms
        assert len(final_seg) == 2000


# ═════════════════════════════════════════════════════════════════════
# LAYER 3 — PIPELINE INTEGRATION
# ═════════════════════════════════════════════════════════════════════

class TestLayer3_TTS_CLI:
    """Test CLI pipeline run_tts."""

    @patch("cli.tts_srt.EdgeTTSEngine.run")
    def test_cli_full_pipeline_no_autorate(self, mock_engine_run, synthetic_wav_with_silence, tmp_path):
        """
        Mô phỏng chạy CLI tts_srt.py.
        Không cần test việc chạy thật model AI, chỉ quan tâm luồng dữ liệu 
        từ đọc srt -> ghép âm thanh -> tạo file cuối.
        """
        pytest.importorskip("ffmpeg", reason="Cần cài đặt thư viện ffmpeg cho subprocess (tuy nhiên pydub và subprocess đều dùng nó)")
        if not shutil.which("ffmpeg"):
            pytest.skip("FFmpeg không có trong hệ thống")

        srt_path = tmp_path / "input.srt"
        srt_path.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n", encoding="utf-8")
        
        out_wav = tmp_path / "out.wav"
        cache_dir = tmp_path / "cache_cli"
        
        # Mô phỏng Engine.run() đã thực hiện xong và ghi file WAV
        # Để SpeedRate có thể xử lý, mình cần ghi file wav thật cho item
        def fake_engine_run():
            wav1 = cache_dir / "dubb-0.wav"
            wav1.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(synthetic_wav_with_silence, wav1)
            return {"ok": 1, "err": 0}
            
        mock_engine_run.side_effect = fake_engine_run
        
        result = run_tts(
            input_file=str(srt_path),
            output_file=str(out_wav),
            voice="en-US-JennyNeural",
            voice_autorate=False,
            cache_folder=str(cache_dir),
            strip_silence=True,  # Test logic default có strip
        )
        
        assert result["tts_ok"] == 1
        assert out_wav.exists()
        assert out_wav.stat().st_size > 0
