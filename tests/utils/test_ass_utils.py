#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_ass_utils.py — Unit tests for ASS utilities module

Run tests:
    uv run pytest tests/test_ass_utils.py -v
    uv run python -m pytest tests/test_ass_utils.py -v
"""

import pytest
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.ass_utils import (
    srt_timestamp_to_ass,
    ass_timestamp_to_srt,
    wrap_text,
    normalize_newlines,
    create_dialogue_line,
    parse_ass_file,
    write_ass_file,
    convert_srt_segments_to_ass_dialogues,
)


# ─────────────────────────────────────────────────────────────
# TEST TIMESTAMP CONVERSION
# ─────────────────────────────────────────────────────────────

class TestSrtTimestampToAss:
    """Tests cho hàm srt_timestamp_to_ass."""
    
    def test_basic_conversion(self):
        """Test basic timestamp conversion."""
        assert srt_timestamp_to_ass("00:01:24,233") == "0:01:24.23"
    
    def test_zero_hours(self):
        """Test timestamp with zero hours."""
        assert srt_timestamp_to_ass("00:00:01,000") == "0:00:01.00"
    
    def test_single_digit_hour(self):
        """Test timestamp with single digit hour."""
        assert srt_timestamp_to_ass("01:30:45,500") == "1:30:45.50"
    
    def test_double_digit_hour(self):
        """Test timestamp with double digit hour."""
        assert srt_timestamp_to_ass("10:30:45,500") == "10:30:45.50"
    
    def test_milliseconds_rounding(self):
        """Test milliseconds to centiseconds conversion."""
        # 233ms -> 23 centiseconds
        assert srt_timestamp_to_ass("00:00:00,233") == "0:00:00.23"
        # 999ms -> 99 centiseconds
        assert srt_timestamp_to_ass("00:00:00,999") == "0:00:00.99"
        # 001ms -> 00 centiseconds
        assert srt_timestamp_to_ass("00:00:00,001") == "0:00:00.00"
    
    def test_invalid_format(self):
        """Test invalid timestamp format raises error."""
        with pytest.raises(ValueError):
            srt_timestamp_to_ass("invalid")
        
        with pytest.raises(ValueError):
            srt_timestamp_to_ass("00:01:24")  # Missing milliseconds


class TestAssTimestampToSrt:
    """Tests cho hàm ass_timestamp_to_srt."""
    
    def test_basic_conversion(self):
        """Test basic timestamp conversion."""
        assert ass_timestamp_to_srt("0:01:24.23") == "00:01:24,230"
    
    def test_single_digit_hour(self):
        """Test timestamp with single digit hour."""
        assert ass_timestamp_to_srt("1:30:45.50") == "01:30:45,500"
    
    def test_double_digit_hour(self):
        """Test timestamp with double digit hour."""
        assert ass_timestamp_to_srt("10:30:45.50") == "10:30:45,500"
    
    def test_centiseconds_to_milliseconds(self):
        """Test centiseconds to milliseconds conversion."""
        # 23cs -> 230ms
        assert ass_timestamp_to_srt("0:00:00.23") == "00:00:00,230"
        # 99cs -> 990ms
        assert ass_timestamp_to_srt("0:00:00.99") == "00:00:00,990"
        # 00cs -> 000ms
        assert ass_timestamp_to_srt("0:00:00.00") == "00:00:00,000"
    
    def test_invalid_format(self):
        """Test invalid timestamp format raises error."""
        with pytest.raises(ValueError):
            ass_timestamp_to_srt("invalid")


class TestTimestampRoundTrip:
    """Tests cho round-trip conversion."""
    
    def test_roundtrip_srt_to_ass_to_srt(self):
        """Test SRT -> ASS -> SRT round-trip."""
        # Note: Some precision is lost (milliseconds -> centiseconds)
        original_srt = "00:01:24,230"
        ass = srt_timestamp_to_ass(original_srt)
        back_to_srt = ass_timestamp_to_srt(ass)
        assert back_to_srt == original_srt


# ─────────────────────────────────────────────────────────────
# TEST TEXT PROCESSING
# ─────────────────────────────────────────────────────────────

class TestNormalizeNewlines:
    """Tests cho hàm normalize_newlines."""
    
    def test_unix_newline(self):
        """Test Unix newline (\\n) conversion."""
        assert normalize_newlines("Line1\nLine2") == "Line1\\NLine2"
    
    def test_windows_newline(self):
        """Test Windows newline (\\r\\n) conversion."""
        assert normalize_newlines("Line1\r\nLine2") == "Line1\\NLine2"
    
    def test_mac_newline(self):
        """Test Mac newline (\\r) conversion."""
        assert normalize_newlines("Line1\rLine2") == "Line1\\NLine2"
    
    def test_already_ass_newline(self):
        """Test already ASS newline (\\N) stays unchanged."""
        assert normalize_newlines("Line1\\NLine2") == "Line1\\NLine2"
    
    def test_multiple_newlines(self):
        """Test multiple newlines."""
        assert normalize_newlines("Line1\nLine2\nLine3") == "Line1\\NLine2\\NLine3"
    
    def test_mixed_newlines(self):
        """Test mixed newline types."""
        assert normalize_newlines("Line1\nLine2\r\nLine3") == "Line1\\NLine2\\NLine3"
    
    def test_no_newline(self):
        """Test text without newline."""
        assert normalize_newlines("SingleLine") == "SingleLine"


class TestWrapText:
    """Tests cho hàm wrap_text."""
    
    def test_short_text(self):
        """Test text shorter than max_chars."""
        assert wrap_text("Short", 14) == "Short"
    
    def test_exact_length(self):
        """Test text exactly at max_chars."""
        assert wrap_text("12345678901234", 14) == "12345678901234"
    
    def test_long_text_wrap(self):
        """Test text longer than max_chars gets wrapped."""
        # 16 chars -> wrap at 14
        result = wrap_text("1234567890123456", 14)
        assert result == "12345678901234\\N56"
    
    def test_japanese_text_wrap(self):
        """Test Japanese text wrapping."""
        # 16 chars -> wrap at 14
        result = wrap_text("サムのギアリストサムの", 14)
        assert result == "サムのギアリストサ\\Nムの"
    
    def test_existing_newline_preserved(self):
        """Test existing \\N is preserved."""
        assert wrap_text("Line1\\NLine2", 14) == "Line1\\NLine2"
    
    def test_existing_newline_with_long_line(self):
        """Test existing \\N with long line gets wrapped."""
        result = wrap_text("1234567890123456\\NShort", 14)
        assert result == "12345678901234\\N56\\NShort"
    
    def test_multiple_existing_newlines(self):
        """Test multiple existing \\N preserved."""
        assert wrap_text("A\\NB\\NC", 14) == "A\\NB\\NC"
    
    def test_very_long_text(self):
        """Test very long text wrapping."""
        # 30 chars -> should wrap into 3 lines of 14, 14, 2
        result = wrap_text("123456789012345678901234567890", 14)
        assert result == "12345678901234\\N56789012345678\\N90"


# ─────────────────────────────────────────────────────────────
# TEST DIALOGUE LINE CREATION
# ─────────────────────────────────────────────────────────────

class TestCreateDialogueLine:
    """Tests cho hàm create_dialogue_line."""
    
    def test_basic_dialogue(self):
        """Test basic dialogue line creation."""
        result = create_dialogue_line("0:01:24.23", "0:01:27.57", "Hello\\NWorld")
        expected = "Dialogue: 0,0:01:24.23,0:01:27.57,NoteStyle,,0,0,0,,Hello\\NWorld"
        assert result == expected
    
    def test_custom_style(self):
        """Test dialogue with custom style."""
        result = create_dialogue_line("0:01:00.00", "0:01:05.00", "Test", style="CustomStyle")
        assert "CustomStyle" in result
    
    def test_custom_layer(self):
        """Test dialogue with custom layer."""
        result = create_dialogue_line("0:01:00.00", "0:01:05.00", "Test", layer=5)
        assert result.startswith("Dialogue: 5,")
    
    def test_custom_margins(self):
        """Test dialogue with custom margins."""
        result = create_dialogue_line("0:01:00.00", "0:01:05.00", "Test", margin_l=10, margin_r=20, margin_v=30)
        assert ",10,20,30," in result
    
    def test_empty_text(self):
        """Test dialogue with empty text."""
        result = create_dialogue_line("0:01:00.00", "0:01:05.00", "")
        assert result.endswith(",")  # Empty text at end


# ─────────────────────────────────────────────────────────────
# TEST ASS FILE PARSING AND WRITING
# ─────────────────────────────────────────────────────────────

class TestParseAssFile:
    """Tests cho hàm parse_ass_file."""
    
    def test_parse_sample_ass(self):
        """Test parsing sample.ass file."""
        sample_path = PROJECT_ROOT / "assets" / "sample.ass"
        if not sample_path.exists():
            pytest.skip("sample.ass not found")
        
        result = parse_ass_file(str(sample_path))
        
        # Check structure
        assert 'script_info' in result
        assert 'styles' in result
        assert 'events_format' in result
        assert 'dialogues' in result
        
        # Check script info
        assert result['script_info']['ScriptType'] == 'v4.00+'
        assert result['script_info']['PlayResX'] == '1920'
        assert result['script_info']['PlayResY'] == '1080'
        
        # Check styles
        assert len(result['styles']) >= 1
        assert 'NoteStyle' in result['styles'][0]
        
        # Check events format
        assert result['events_format'].startswith('Format:')
        
        # Check dialogues
        assert len(result['dialogues']) >= 1
    
    def test_parse_nonexistent_file(self):
        """Test parsing nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            parse_ass_file("nonexistent.ass")


class TestWriteAssFile:
    """Tests cho hàm write_ass_file."""
    
    def test_write_ass_file(self, tmp_path):
        """Test writing ASS file."""
        output_file = tmp_path / "test_output.ass"
        
        script_info = {
            'ScriptType': 'v4.00+',
            'PlayResX': '1920',
            'PlayResY': '1080',
            'WrapStyle': '1',
            'Collisions': 'Normal',
        }
        
        styles = [
            "Style: NoteStyle,Noto Sans CJK JP,60,&H00000000,&H000000FF,&H00FFFFFF,&H80000000,1,0,0,0,100,100,0,0,1,4,2,7,110,1120,50,1"
        ]
        
        events_format = "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
        
        dialogues = [
            "Dialogue: 0,0:01:00.00,0:01:05.00,NoteStyle,,0,0,0,,Test Text"
        ]
        
        write_ass_file(
            str(output_file),
            script_info=script_info,
            styles=styles,
            events_format=events_format,
            dialogues=dialogues,
        )
        
        # Check file exists
        assert output_file.exists()
        
        # Check content
        content = output_file.read_text(encoding='utf-8')
        assert "[Script Info]" in content
        assert "[V4+ Styles]" in content
        assert "[Events]" in content
        assert "Test Text" in content


# ─────────────────────────────────────────────────────────────
# TEST HIGH-LEVEL CONVERSION
# ─────────────────────────────────────────────────────────────

class TestConvertSrtSegmentsToAssDialogues:
    """Tests cho hàm convert_srt_segments_to_ass_dialogues."""
    
    def test_basic_conversion(self):
        """Test basic segment conversion."""
        segments = [
            {
                'line': 1,
                'start_time': 84233,  # 00:01:24,233
                'end_time': 87566,    # 00:01:27,566
                'startraw': '00:01:24,233',
                'endraw': '00:01:27,566',
                'text': 'Hello World',
            }
        ]
        
        dialogues = convert_srt_segments_to_ass_dialogues(segments)
        
        assert len(dialogues) == 1
        assert "0:01:24.23" in dialogues[0]
        assert "0:01:27.56" in dialogues[0]
        assert "Hello World" in dialogues[0]
    
    def test_multiline_text(self):
        """Test multiline text conversion."""
        segments = [
            {
                'line': 1,
                'start_time': 0,
                'end_time': 5000,
                'startraw': '00:00:00,000',
                'endraw': '00:00:05,000',
                'text': 'Line 1\nLine 2',
            }
        ]
        
        dialogues = convert_srt_segments_to_ass_dialogues(segments)
        
        assert len(dialogues) == 1
        assert "Line 1\\NLine 2" in dialogues[0]
    
    def test_text_wrapping(self):
        """Test text wrapping in conversion."""
        segments = [
            {
                'line': 1,
                'start_time': 0,
                'end_time': 5000,
                'startraw': '00:00:00,000',
                'endraw': '00:00:05,000',
                'text': '12345678901234567890',  # 20 chars
            }
        ]
        
        dialogues = convert_srt_segments_to_ass_dialogues(segments, max_chars=14)
        
        assert len(dialogues) == 1
        assert "12345678901234\\N567890" in dialogues[0]
    
    def test_custom_style(self):
        """Test custom style in conversion."""
        segments = [
            {
                'line': 1,
                'start_time': 0,
                'end_time': 5000,
                'startraw': '00:00:00,000',
                'endraw': '00:00:05,000',
                'text': 'Test',
            }
        ]
        
        dialogues = convert_srt_segments_to_ass_dialogues(segments, style="CustomStyle")
        
        assert "CustomStyle" in dialogues[0]
    
    def test_conversion_without_raw_timestamps(self):
        """Test conversion using milliseconds when raw timestamps not available."""
        segments = [
            {
                'line': 1,
                'start_time': 84233,  # 00:01:24,233
                'end_time': 87566,    # 00:01:27,566
                'text': 'Hello World',
            }
        ]
        
        dialogues = convert_srt_segments_to_ass_dialogues(segments)
        
        assert len(dialogues) == 1
        assert "0:01:24.23" in dialogues[0]


# ─────────────────────────────────────────────────────────────
# RUN TESTS
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])