#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for cli/merge_srt.py
"""

import pytest
import tempfile
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cli.merge_srt import check_overlap, merge_srt_segments, merge_srt_files


class TestCheckOverlap:
    """Tests for check_overlap function."""
    
    def test_no_overlap(self):
        """Test segments with no overlap."""
        segments = [
            {'line': 1, 'start_time': 0, 'end_time': 1000, 'text': 'A'},
            {'line': 2, 'start_time': 1000, 'end_time': 2000, 'text': 'B'},
            {'line': 3, 'start_time': 2000, 'end_time': 3000, 'text': 'C'},
        ]
        overlaps = check_overlap(segments)
        assert len(overlaps) == 0
    
    def test_with_overlap(self):
        """Test segments with overlap."""
        segments = [
            {'line': 1, 'start_time': 0, 'end_time': 1500, 'text': 'A'},
            {'line': 2, 'start_time': 1000, 'end_time': 2000, 'text': 'B'},
        ]
        overlaps = check_overlap(segments)
        assert len(overlaps) == 1
        assert overlaps[0][0] == 0  # index of first segment
        assert overlaps[0][1] == 1  # index of second segment
    
    def test_multiple_overlaps(self):
        """Test segments with multiple overlaps."""
        segments = [
            {'line': 1, 'start_time': 0, 'end_time': 1500, 'text': 'A'},
            {'line': 2, 'start_time': 1000, 'end_time': 2500, 'text': 'B'},
            {'line': 3, 'start_time': 2000, 'end_time': 3500, 'text': 'C'},
        ]
        overlaps = check_overlap(segments)
        assert len(overlaps) == 2
    
    def test_empty_segments(self):
        """Test empty segments list."""
        overlaps = check_overlap([])
        assert len(overlaps) == 0
    
    def test_single_segment(self):
        """Test single segment (no overlap possible)."""
        segments = [
            {'line': 1, 'start_time': 0, 'end_time': 1000, 'text': 'A'},
        ]
        overlaps = check_overlap(segments)
        assert len(overlaps) == 0


class TestMergeSrtSegments:
    """Tests for merge_srt_segments function."""
    
    def test_basic_merge(self):
        """Test basic merge of two segment lists."""
        commentary = [
            {'line': 1, 'start_time': 0, 'end_time': 10000, 'text': 'Bình luận đoạn 1'},
            {'line': 2, 'start_time': 20000, 'end_time': 30000, 'text': 'Bình luận đoạn 2'},
        ]
        quoted = [
            {'line': 1, 'start_time': 10000, 'end_time': 20000, 'text': 'Video trích dẫn 1'},
            {'line': 2, 'start_time': 30000, 'end_time': 40000, 'text': 'Video trích dẫn 2'},
        ]
        
        merged = merge_srt_segments(commentary, quoted, check_overlaps=False)
        
        assert len(merged) == 4
        # Check sorted by start_time
        assert merged[0]['start_time'] == 0
        assert merged[1]['start_time'] == 10000
        assert merged[2]['start_time'] == 20000
        assert merged[3]['start_time'] == 30000
        # Check line numbers are renumbered
        assert merged[0]['line'] == 1
        assert merged[1]['line'] == 2
        assert merged[2]['line'] == 3
        assert merged[3]['line'] == 4
    
    def test_empty_commentary(self):
        """Test merge with empty commentary."""
        commentary = []
        quoted = [
            {'line': 1, 'start_time': 10000, 'end_time': 20000, 'text': 'Video trích dẫn 1'},
        ]
        
        merged = merge_srt_segments(commentary, quoted, check_overlaps=False)
        
        assert len(merged) == 1
        assert merged[0]['text'] == 'Video trích dẫn 1'
    
    def test_empty_quoted(self):
        """Test merge with empty quoted."""
        commentary = [
            {'line': 1, 'start_time': 0, 'end_time': 10000, 'text': 'Bình luận đoạn 1'},
        ]
        quoted = []
        
        merged = merge_srt_segments(commentary, quoted, check_overlaps=False)
        
        assert len(merged) == 1
        assert merged[0]['text'] == 'Bình luận đoạn 1'
    
    def test_both_empty(self):
        """Test merge with both lists empty."""
        merged = merge_srt_segments([], [], check_overlaps=False)
        assert len(merged) == 0
    
    def test_same_timestamp(self):
        """Test merge with same timestamp in both files."""
        commentary = [
            {'line': 1, 'start_time': 10000, 'end_time': 15000, 'text': 'Text A'},
        ]
        quoted = [
            {'line': 1, 'start_time': 10000, 'end_time': 15000, 'text': 'Text B'},
        ]
        
        merged = merge_srt_segments(commentary, quoted, check_overlaps=False)
        
        # Both segments kept, commentary first (stable sort)
        assert len(merged) == 2
        assert merged[0]['text'] == 'Text A'
        assert merged[1]['text'] == 'Text B'
    
    def test_gap_between_segments(self):
        """Test merge with gap between segments."""
        commentary = [
            {'line': 1, 'start_time': 10000, 'end_time': 15000, 'text': 'A'},
        ]
        quoted = [
            {'line': 1, 'start_time': 20000, 'end_time': 25000, 'text': 'B'},
        ]
        
        merged = merge_srt_segments(commentary, quoted, check_overlaps=False)
        
        assert len(merged) == 2
        # Gap is preserved (no silence segment added)
        assert merged[0]['end_time'] == 15000
        assert merged[1]['start_time'] == 20000
    
    def test_overlap_detection(self, caplog):
        """Test overlap detection logs error."""
        commentary = [
            {'line': 1, 'start_time': 0, 'end_time': 15000, 'text': 'A'},
        ]
        quoted = [
            {'line': 1, 'start_time': 10000, 'end_time': 20000, 'text': 'B'},
        ]
        
        merged = merge_srt_segments(commentary, quoted, check_overlaps=True)
        
        # Check overlap was detected and logged
        assert 'OVERLAP DETECTED' in caplog.text
        assert len(merged) == 2  # Both segments still kept


class TestMergeSrtFiles:
    """Tests for merge_srt_files function."""
    
    def test_basic_merge_files(self):
        """Test basic file merge."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create commentary file
            commentary_path = Path(tmpdir) / 'commentary.srt'
            commentary_path.write_text('''1
00:00:00,000 --> 00:00:10,000
Bình luận đoạn 1

2
00:00:20,000 --> 00:00:30,000
Bình luận đoạn 2
''', encoding='utf-8')
            
            # Create quoted file
            quoted_path = Path(tmpdir) / 'quoted.srt'
            quoted_path.write_text('''1
00:00:10,000 --> 00:00:20,000
Video trích dẫn 1

2
00:00:30,000 --> 00:00:40,000
Video trích dẫn 2
''', encoding='utf-8')
            
            # Merge
            output_path = Path(tmpdir) / 'merged.srt'
            count = merge_srt_files(
                str(commentary_path),
                str(quoted_path),
                str(output_path),
                check_overlaps=False
            )
            
            assert count == 4
            assert output_path.exists()
            
            # Check output content
            content = output_path.read_text(encoding='utf-8')
            assert 'Bình luận đoạn 1' in content
            assert 'Video trích dẫn 1' in content
    
    def test_empty_commentary_file(self):
        """Test merge with empty commentary file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty commentary file
            commentary_path = Path(tmpdir) / 'commentary.srt'
            commentary_path.write_text('', encoding='utf-8')
            
            # Create quoted file
            quoted_path = Path(tmpdir) / 'quoted.srt'
            quoted_path.write_text('''1
00:00:10,000 --> 00:00:20,000
Video trích dẫn 1
''', encoding='utf-8')
            
            # Merge
            output_path = Path(tmpdir) / 'merged.srt'
            count = merge_srt_files(
                str(commentary_path),
                str(quoted_path),
                str(output_path),
                check_overlaps=False
            )
            
            assert count == 1
            content = output_path.read_text(encoding='utf-8')
            assert 'Video trích dẫn 1' in content
    
    def test_file_not_found(self):
        """Test merge with non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            commentary_path = Path(tmpdir) / 'nonexistent.srt'
            quoted_path = Path(tmpdir) / 'quoted.srt'
            quoted_path.write_text('1\n00:00:10,000 --> 00:00:20,000\nTest', encoding='utf-8')
            
            with pytest.raises(FileNotFoundError):
                merge_srt_files(
                    str(commentary_path),
                    str(quoted_path),
                    str(Path(tmpdir) / 'merged.srt')
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])