#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_tts_edgetts.py
===========================
Test cho chức năng Text-to-Speech bằng EdgeTTS và clean_tail logic.

Cấu trúc layers:
  Layer 1 — Unit Tests          (Logic _apply_clean_tail: trim silence + fade)
  Layer 2 — Component Tests     (EdgeTTSEngine flow convert mp3 sang wav và clean_tail)
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

from tts.edgetts import EdgeTTSEngine, convert_to_wav

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
    with patch("tts.edgetts.Communicate") as mock_comm:
        instance = MagicMock()
        # Mock hàm save trả về future hoàn thành ngay lập tức
        # Nó sẽ tạo ra 1 file rỗng tại path truyền vào để mô phỏng tải xong
        async def fake_save(path):
            with open(path, "wb") as f:
                f.write(b"dummy mp3 data")
        
        instance.save.side_effect = fake_save
        mock_comm.return_value = instance
        yield mock_comm

# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_CleanTail:
    """Kiểm tra EdgeTTSEngine._apply_clean_tail (librosa trim + fade)."""

    def test_clean_tail_trims_silence(self, synthetic_wav_with_silence, tmp_path):
        """_apply_clean_tail cắt silence 2 đầu — kết quả ~1000ms (chỉ phần audio)."""
        pytest.importorskip("librosa")  # trim yêu cầu librosa; skip nếu không có

        test_wav = tmp_path / "test_clean.wav"
        shutil.copy(synthetic_wav_with_silence, test_wav)

        assert len(AudioSegment.from_file(test_wav)) == 2000  # 500+1000+500

        EdgeTTSEngine._apply_clean_tail(str(test_wav), top_db=30.0, fade_ms=0.0)

        result_len = len(AudioSegment.from_file(test_wav))
        # librosa trim cắt 500ms silence đầu và cuối → còn ~1000ms
        assert result_len < 2000
        assert abs(result_len - 1000) < 50  # tolerance cho frame-based trim

    def test_clean_tail_all_silent_kept(self, synthetic_wav_all_silent, tmp_path):
        """File toàn silence → giữ nguyên (librosa trim ra rỗng → fallback giữ gốc)."""
        test_wav = tmp_path / "test_clean_silent.wav"
        shutil.copy(synthetic_wav_all_silent, test_wav)

        original_len = len(AudioSegment.from_file(test_wav))
        EdgeTTSEngine._apply_clean_tail(str(test_wav), top_db=30.0, fade_ms=0.0)

        result_len = len(AudioSegment.from_file(test_wav))
        assert abs(result_len - original_len) < 50  # toàn silence → không xén


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — COMPONENT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestLayer2_EdgeTTSEngine:
    """Kiểm tra EdgeTTSEngine và luồng convert + strip silence."""

    @patch("tts.edgetts.convert_to_wav")
    def test_engine_run_with_clean_tail(self, mock_convert, mock_edgetts_communicate, synthetic_wav_with_silence, tmp_path):
        """Test engine bật clean_tail: trim silence 2 đầu → kết quả ~1000ms."""
        pytest.importorskip("librosa")  # trim yêu cầu librosa
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        filename = str(cache_dir / "test01.wav")

        queue_tts = [{"text": "Hello world", "filename": filename}]

        def fake_convert(mp3, wav):
            shutil.copy(synthetic_wav_with_silence, wav)
            return True

        mock_convert.side_effect = fake_convert

        engine = EdgeTTSEngine(
            queue_tts=queue_tts,
            voice="en-US-JennyNeural",
            clean_tail=True,
        )

        stats = engine.run()

        assert stats["ok"] == 1
        assert stats["err"] == 0
        assert Path(filename).exists()

        # silence(500)+audio(1000)+silence(500) → librosa trim → ~1000ms
        final_seg = AudioSegment.from_file(filename)
        assert abs(len(final_seg) - 1000) < 50

    @patch("tts.edgetts.convert_to_wav")
    def test_engine_run_without_clean_tail(self, mock_convert, mock_edgetts_communicate, synthetic_wav_with_silence, tmp_path):
        """Test engine khi clean_tail=False: giữ nguyên 2000ms."""
        cache_dir = tmp_path / "cache2"
        cache_dir.mkdir()
        filename = str(cache_dir / "test02.wav")

        queue_tts = [{"text": "Hello world 2", "filename": filename}]

        def fake_convert(mp3, wav):
            shutil.copy(synthetic_wav_with_silence, wav)
            return True

        mock_convert.side_effect = fake_convert

        engine = EdgeTTSEngine(
            queue_tts=queue_tts,
            voice="en-US-JennyNeural",
            clean_tail=False,
        )

        engine.run()

        final_seg = AudioSegment.from_file(filename)
        assert len(final_seg) == 2000


# ═════════════════════════════════════════════════════════════════════
# LAYER 3 — PIPELINE INTEGRATION
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.skip(
    reason="run_tts đã bị refactor thành cli.tts.run_task (signature khác). "
    "Pipeline mới được cover bởi tests/cli/test_tts_refactor.py."
)
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
