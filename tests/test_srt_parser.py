#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for utils/srt_parser.py
"""

import pytest
from utils.srt_parser import parse_srt, parse_srt_file, ts_to_ms, segments_to_srt


class TestTsToMs:
    """Tests for ts_to_ms function."""
    
    def test_basic_timestamp(self):
        """Test basic timestamp conversion."""
        assert ts_to_ms("00:01:24,233") == 84233
    
    def test_with_dot(self):
        """Test timestamp with dot separator."""
        assert ts_to_ms("00:01:24.233") == 84233
    
    def test_hours(self):
        """Test timestamp with hours."""
        assert ts_to_ms("01:30:45,500") == 5445500
    
    def test_zero(self):
        """Test zero timestamp."""
        assert ts_to_ms("00:00:00,000") == 0


class TestParseSrt:
    """Tests for parse_srt function."""
    
    def test_single_segment(self):
        """Test parsing single segment."""
        content = """1
00:01:24,233 --> 00:01:27,566
Hello world"""
        segments = parse_srt(content)
        assert len(segments) == 1
        assert segments[0]['line'] == 1
        assert segments[0]['start_time'] == 84233
        assert segments[0]['end_time'] == 87566
        assert segments[0]['text'] == "Hello world"
    
    def test_multiple_segments(self):
        """Test parsing multiple segments."""
        content = """1
00:01:24,233 --> 00:01:27,566
Hello world

2
00:02:30,000 --> 00:02:35,500
This is a test"""
        segments = parse_srt(content)
        assert len(segments) == 2
        assert segments[0]['line'] == 1
        assert segments[1]['line'] == 2
    
    def test_html_tags_removed(self):
        """Test HTML tags are removed."""
        content = """1
00:01:24,233 --> 00:01:27,566
<b>Hello</b> <i>world</i>"""
        segments = parse_srt(content)
        assert segments[0]['text'] == "Hello world"
    
    def test_multiline_text(self):
        """Test multiline text."""
        content = """1
00:01:24,233 --> 00:01:27,566
Line 1
Line 2"""
        segments = parse_srt(content)
        assert segments[0]['text'] == "Line 1\nLine 2"
    
    def test_empty_text_skipped(self):
        """Test segments with empty text are skipped."""
        content = """1
00:01:24,233 --> 00:01:27,566

2
00:02:30,000 --> 00:02:35,500
Valid text"""
        segments = parse_srt(content)
        assert len(segments) == 1
        assert segments[0]['line'] == 2
    
    def test_invalid_timestamp_skipped(self):
        """Test segments with invalid timestamp are skipped."""
        content = """1
invalid --> timestamp
Hello world

2
00:02:30,000 --> 00:02:35,500
Valid text"""
        segments = parse_srt(content)
        assert len(segments) == 1


class TestSegmentsToSrt:
    """Tests for segments_to_srt function."""
    
    def test_single_segment(self):
        """Test converting single segment to SRT."""
        segments = [{'line': 1, 'start_time': 0, 'end_time': 1000, 'text': 'Hello'}]
        result = segments_to_srt(segments)
        assert '1' in result
        assert '00:00:00,000 --> 00:00:01,000' in result
        assert 'Hello' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])