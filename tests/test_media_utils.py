#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_media_utils.py — Unit tests cho utils/media_utils.py
"""

import pytest
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# TESTS - TYPE DETECTION
# ─────────────────────────────────────────────────────────────────────

class TestDetectMediaType:
    """Tests cho detect_media_type function."""
    
    def test_video_extensions(self):
        """Test video file detection."""
        from utils.media_utils import detect_media_type
        
        video_files = [
            "video.mp4", "movie.mkv", "clip.mov",
            "video.avi", "stream.webm", "video.flv", "video.wmv"
        ]
        
        for f in video_files:
            assert detect_media_type(f) == "video", f"Failed for {f}"
    
    def test_audio_extensions(self):
        """Test audio file detection."""
        from utils.media_utils import detect_media_type
        
        audio_files = [
            "audio.wav", "music.mp3", "sound.m4a",
            "audio.flac", "audio.aac", "audio.ogg", "audio.wma"
        ]
        
        for f in audio_files:
            assert detect_media_type(f) == "audio", f"Failed for {f}"
    
    def test_srt_extensions(self):
        """Test SRT file detection."""
        from utils.media_utils import detect_media_type
        
        assert detect_media_type("subtitle.srt") == "srt"
        assert detect_media_type("SUBTITLE.SRT") == "srt"  # Case insensitive
    
    def test_ass_extensions(self):
        """Test ASS file detection."""
        from utils.media_utils import detect_media_type
        
        assert detect_media_type("overlay.ass") == "ass"
        assert detect_media_type("OVERLAY.ASS") == "ass"  # Case insensitive
    
    def test_unknown_extension(self):
        """Test unknown file type."""
        from utils.media_utils import detect_media_type
        
        assert detect_media_type("file.xyz") == "unknown"
        assert detect_media_type("file.txt") == "unknown"
        assert detect_media_type("file.json") == "unknown"


# ─────────────────────────────────────────────────────────────────────
# TESTS - TIME SCALING
# ─────────────────────────────────────────────────────────────────────

class TestScaleTimeMs:
    """Tests cho scale_time_ms function."""
    
    def test_slow_down(self):
        """Test slow down (speed < 1.0)."""
        from utils.media_utils import scale_time_ms
        
        # 10s at 0.65x → ~15.385s
        assert scale_time_ms(10000, 0.65) == 15385
        # 5s at 0.5x → 10s
        assert scale_time_ms(5000, 0.5) == 10000
        # 1s at 0.65x → ~1.538s
        assert scale_time_ms(1000, 0.65) == 1538
    
    def test_speed_up(self):
        """Test speed up (speed > 1.0)."""
        from utils.media_utils import scale_time_ms
        
        # 10s at 1.5x → ~6.667s
        assert scale_time_ms(10000, 1.5) == 6667
        # 10s at 2.0x → 5s
        assert scale_time_ms(10000, 2.0) == 5000
        # 5s at 1.2x → ~4.167s
        assert scale_time_ms(5000, 1.2) == 4167
    
    def test_no_change(self):
        """Test no change (speed = 1.0)."""
        from utils.media_utils import scale_time_ms
        
        assert scale_time_ms(10000, 1.0) == 10000
        assert scale_time_ms(5000, 1.0) == 5000
    
    def test_minimum_value(self):
        """Test minimum return value is 1."""
        from utils.media_utils import scale_time_ms
        
        # Even with very small input, should return >= 1
        assert scale_time_ms(1, 0.65) >= 1
        assert scale_time_ms(0, 0.65) == 1  # 0 → 1 (minimum)
    
    def test_invalid_speed(self):
        """Test invalid speed raises error."""
        from utils.media_utils import scale_time_ms
        
        with pytest.raises(ValueError):
            scale_time_ms(10000, 0)
        
        with pytest.raises(ValueError):
            scale_time_ms(10000, -1.0)


# ─────────────────────────────────────────────────────────────────────
# TESTS - ASS TIMESTAMP CONVERSION
# ─────────────────────────────────────────────────────────────────────

class TestAssTimestampConversion:
    """Tests cho ASS timestamp conversion functions."""
    
    def test_parse_ass_timestamp_basic(self):
        """Test basic ASS timestamp parsing."""
        from utils.media_utils import parse_ass_timestamp_to_ms
        
        assert parse_ass_timestamp_to_ms("0:00:00.00") == 0
        assert parse_ass_timestamp_to_ms("0:00:01.00") == 1000
        assert parse_ass_timestamp_to_ms("0:01:00.00") == 60000
        assert parse_ass_timestamp_to_ms("1:00:00.00") == 3600000
    
    def test_parse_ass_timestamp_with_centiseconds(self):
        """Test ASS timestamp with centiseconds."""
        from utils.media_utils import parse_ass_timestamp_to_ms
        
        assert parse_ass_timestamp_to_ms("0:00:01.50") == 1500
        assert parse_ass_timestamp_to_ms("0:01:24.23") == 84230
        assert parse_ass_timestamp_to_ms("1:30:45.50") == 5445500
    
    def test_ms_to_ass_timestamp_basic(self):
        """Test basic ms to ASS timestamp conversion."""
        from utils.media_utils import ms_to_ass_timestamp
        
        assert ms_to_ass_timestamp(0) == "0:00:00.00"
        assert ms_to_ass_timestamp(1000) == "0:00:01.00"
        assert ms_to_ass_timestamp(60000) == "0:01:00.00"
        assert ms_to_ass_timestamp(3600000) == "1:00:00.00"
    
    def test_ms_to_ass_timestamp_with_centiseconds(self):
        """Test ms to ASS timestamp with centiseconds."""
        from utils.media_utils import ms_to_ass_timestamp
        
        assert ms_to_ass_timestamp(1500) == "0:00:01.50"
        assert ms_to_ass_timestamp(84230) == "0:01:24.23"
        assert ms_to_ass_timestamp(5445500) == "1:30:45.50"
    
    def test_round_trip_conversion(self):
        """Test round-trip conversion."""
        from utils.media_utils import parse_ass_timestamp_to_ms, ms_to_ass_timestamp
        
        test_cases = [
            "0:00:00.00",
            "0:00:01.00",
            "0:01:24.23",
            "1:30:45.50",
            "0:10:30.75",
        ]
        
        for ts in test_cases:
            ms = parse_ass_timestamp_to_ms(ts)
            back = ms_to_ass_timestamp(ms)
            assert back == ts, f"Round-trip failed for {ts}: got {back}"
    
    def test_parse_invalid_timestamp(self):
        """Test parsing invalid timestamp raises error."""
        from utils.media_utils import parse_ass_timestamp_to_ms
        
        with pytest.raises(ValueError):
            parse_ass_timestamp_to_ms("invalid")
        
        # Note: "00:01:24" is actually valid - it parses as 0:01:24 with 0 centiseconds
        # Test a truly invalid format
        with pytest.raises(ValueError):
            parse_ass_timestamp_to_ms("00:01")  # Missing seconds


# ─────────────────────────────────────────────────────────────────────
# TESTS - OUTPUT PATH HELPER
# ─────────────────────────────────────────────────────────────────────

class TestGetDefaultOutputPath:
    """Tests cho get_default_output_path function."""
    
    def test_slow_down_naming(self):
        """Test output naming for slow down."""
        from utils.media_utils import get_default_output_path
        
        assert get_default_output_path("video.mp4", 0.65) == "video_slow.mp4"
        assert get_default_output_path("audio.wav", 0.5) == "audio_slow.wav"
        assert get_default_output_path("subtitle.srt", 0.8) == "subtitle_slow.srt"
    
    def test_speed_up_naming(self):
        """Test output naming for speed up."""
        from utils.media_utils import get_default_output_path
        
        assert get_default_output_path("video.mp4", 1.5) == "video_fast.mp4"
        assert get_default_output_path("audio.wav", 2.0) == "audio_fast.wav"
        assert get_default_output_path("subtitle.srt", 1.2) == "subtitle_fast.srt"
    
    def test_no_change_naming(self):
        """Test output naming for no change."""
        from utils.media_utils import get_default_output_path
        
        assert get_default_output_path("video.mp4", 1.0) == "video_copy.mp4"
        assert get_default_output_path("audio.wav", 1.0) == "audio_copy.wav"
    
    def test_with_path(self):
        """Test output naming with path."""
        from utils.media_utils import get_default_output_path
        from pathlib import Path
        
        # Use Path for cross-platform compatibility
        result = get_default_output_path("/path/to/video.mp4", 0.65)
        expected = str(Path("/path/to/video_slow.mp4"))
        assert result == expected
        
        result = get_default_output_path("./audio.wav", 1.5)
        expected = str(Path("./audio_fast.wav"))
        assert result == expected


# ─────────────────────────────────────────────────────────────────────
# TESTS - SRT TIMESTAMP SCALING
# ─────────────────────────────────────────────────────────────────────

class TestScaleSrtTimestamps:
    """Tests cho scale_srt_timestamps function."""
    
    def test_basic_scaling(self, tmp_path):
        """Test basic SRT timestamp scaling."""
        from utils.media_utils import scale_srt_timestamps
        
        # Create test SRT file
        srt_content = """1
00:00:10,000 --> 00:00:15,000
Test subtitle line 1

2
00:00:20,000 --> 00:00:25,000
Test subtitle line 2
"""
        input_file = tmp_path / "test.srt"
        input_file.write_text(srt_content, encoding='utf-8')
        
        output_file = tmp_path / "test_slow.srt"
        
        # Scale
        count = scale_srt_timestamps(str(input_file), str(output_file), 0.5)
        
        assert count == 2
        assert output_file.exists()
        
        # Check content
        output_content = output_file.read_text(encoding='utf-8')
        
        # Original: 10s → 20s at 0.5x
        assert "00:00:20,000 --> 00:00:30,000" in output_content
        # Original: 20s → 40s at 0.5x
        assert "00:00:40,000 --> 00:00:50,000" in output_content
        # Text should be preserved
        assert "Test subtitle line 1" in output_content
        assert "Test subtitle line 2" in output_content
    
    def test_text_preserved(self, tmp_path):
        """Test that text content is preserved."""
        from utils.media_utils import scale_srt_timestamps
        
        srt_content = """1
00:00:01,000 --> 00:00:03,000
Hello world!
Line 2 of subtitle

2
00:00:05,000 --> 00:00:08,000
日本語テスト
"""
        input_file = tmp_path / "test.srt"
        input_file.write_text(srt_content, encoding='utf-8')
        
        output_file = tmp_path / "test_slow.srt"
        
        scale_srt_timestamps(str(input_file), str(output_file), 0.65)
        
        output_content = output_file.read_text(encoding='utf-8')
        
        assert "Hello world!" in output_content
        assert "Line 2 of subtitle" in output_content
        assert "日本語テスト" in output_content


# ─────────────────────────────────────────────────────────────────────
# TESTS - ASS TIMESTAMP SCALING
# ─────────────────────────────────────────────────────────────────────

class TestScaleAssTimestamps:
    """Tests cho scale_ass_timestamps function."""
    
    def test_basic_scaling(self, tmp_path):
        """Test basic ASS timestamp scaling."""
        from utils.media_utils import scale_ass_timestamps
        
        # Create test ASS file
        ass_content = """[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:10.00,0:00:15.00,Default,,0,0,0,,Test dialogue 1
Dialogue: 0,0:00:20.00,0:00:25.00,Default,,0,0,0,,Test dialogue 2
"""
        input_file = tmp_path / "test.ass"
        input_file.write_text(ass_content, encoding='utf-8')
        
        output_file = tmp_path / "test_slow.ass"
        
        # Scale with 0.5x (double duration)
        count = scale_ass_timestamps(str(input_file), str(output_file), 0.5)
        
        assert count == 2
        assert output_file.exists()
        
        # Check content
        output_content = output_file.read_text(encoding='utf-8')
        
        # Original: 10s → 20s at 0.5x
        assert "0:00:20.00,0:00:30.00" in output_content
        # Original: 20s → 40s at 0.5x
        assert "0:00:40.00,0:00:50.00" in output_content
        # Text should be preserved
        assert "Test dialogue 1" in output_content
        assert "Test dialogue 2" in output_content
        # Script info should be preserved
        assert "[Script Info]" in output_content
        assert "ScriptType: v4.00+" in output_content
    
    def test_text_with_comma_preserved(self, tmp_path):
        """Test that text with commas is preserved."""
        from utils.media_utils import scale_ass_timestamps
        
        ass_content = """[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello, World, How are you?
"""
        input_file = tmp_path / "test.ass"
        input_file.write_text(ass_content, encoding='utf-8')
        
        output_file = tmp_path / "test_slow.ass"
        
        scale_ass_timestamps(str(input_file), str(output_file), 0.65)
        
        output_content = output_file.read_text(encoding='utf-8')
        
        # Text with commas should be preserved
        assert "Hello, World, How are you?" in output_content
    
    def test_non_dialogue_lines_preserved(self, tmp_path):
        """Test that non-dialogue lines are preserved."""
        from utils.media_utils import scale_ass_timestamps
        
        ass_content = """[Script Info]
Title: Test Video
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize
Style: Default,Arial,20

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Test
Comment: This is a comment
"""
        input_file = tmp_path / "test.ass"
        input_file.write_text(ass_content, encoding='utf-8')
        
        output_file = tmp_path / "test_slow.ass"
        
        scale_ass_timestamps(str(input_file), str(output_file), 0.65)
        
        output_content = output_file.read_text(encoding='utf-8')
        
        # Non-dialogue content should be preserved
        assert "Title: Test Video" in output_content
        assert "PlayResX: 1920" in output_content
        assert "Style: Default,Arial,20" in output_content
        assert "Comment: This is a comment" in output_content


# ─────────────────────────────────────────────────────────────────────
# TESTS - RUBBERBAND CHECK
# ─────────────────────────────────────────────────────────────────────

class TestCheckRubberbandAvailable:
    """Tests cho check_rubberband_available function."""
    
    def test_returns_tuple(self):
        """Test function returns tuple."""
        from utils.media_utils import check_rubberband_available
        
        result = check_rubberband_available()
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)


# ─────────────────────────────────────────────────────────────────────
# INTEGRATION TESTS (require hardware)
# ─────────────────────────────────────────────────────────────────────

class TestAudioStretching:
    """Tests cho audio stretching - cần FFmpeg/rubberband."""
    
    @pytest.mark.usefixtures("skip_if_weak_hardware", "check_ffmpeg_available")
    def test_stretch_audio_atempo(self, tmp_path):
        """Test audio stretching with FFmpeg atempo."""
        # Skip if pydub not available
        pytest.importorskip("pydub")
        
        from utils.media_utils import stretch_audio_atempo
        from pydub import AudioSegment
        
        # Create a simple test audio (1 second of silence)
        test_audio = tmp_path / "test.wav"
        silence = AudioSegment.silent(duration=1000, frame_rate=48000)
        silence.export(str(test_audio), format="wav")
        
        output = tmp_path / "test_slow.wav"
        
        # Stretch with 0.5x (double duration)
        stretch_audio_atempo(str(test_audio), str(output), 0.5)
        
        assert output.exists()
        
        # Check duration
        result_audio = AudioSegment.from_file(str(output))
        duration_ms = len(result_audio)
        
        # Should be approximately 2000ms (±100ms tolerance)
        assert 1900 <= duration_ms <= 2100, f"Expected ~2000ms, got {duration_ms}ms"