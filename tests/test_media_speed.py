#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_media_speed.py — Unit tests cho cli/media_speed.py
"""

import pytest
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# TESTS - CLI PARSER
# ─────────────────────────────────────────────────────────────────────

class TestBuildParser:
    """Tests cho build_parser function."""
    
    def test_parser_exists(self):
        """Test parser can be created."""
        from cli.media_speed import build_parser
        parser = build_parser()
        assert parser is not None
    
    def test_parser_has_required_args(self):
        """Test parser has required arguments."""
        from cli.media_speed import build_parser
        parser = build_parser()
        
        actions = {a.dest for a in parser._actions}
        assert "input" in actions
        assert "output" in actions
        assert "speed" in actions
        assert "type" in actions
        assert "no_keep_audio" in actions
        assert "verbose" in actions
    
    def test_default_values(self):
        """Test default values for arguments."""
        from cli.media_speed import build_parser
        parser = build_parser()
        
        args = parser.parse_args(["--input", "test.mp4"])
        
        assert args.speed == 0.65
        assert args.type == "auto"
        assert args.no_keep_audio is False
        assert args.verbose is False
        assert args.output is None
    
    def test_speed_argument(self):
        """Test speed argument parsing."""
        from cli.media_speed import build_parser
        parser = build_parser()
        
        # Valid speeds
        args = parser.parse_args(["--input", "test.mp4", "--speed", "0.5"])
        assert args.speed == 0.5
        
        args = parser.parse_args(["--input", "test.mp4", "-s", "1.5"])
        assert args.speed == 1.5
    
    def test_type_argument(self):
        """Test type argument parsing."""
        from cli.media_speed import build_parser
        parser = build_parser()
        
        for type_val in ["auto", "video", "audio", "srt", "ass"]:
            args = parser.parse_args(["--input", "test.xyz", "--type", type_val])
            assert args.type == type_val
    
    def test_no_keep_audio_flag(self):
        """Test --no-keep-audio flag."""
        from cli.media_speed import build_parser
        parser = build_parser()
        
        args = parser.parse_args(["--input", "test.mp4"])
        assert args.no_keep_audio is False
        
        args = parser.parse_args(["--input", "test.mp4", "--no-keep-audio"])
        assert args.no_keep_audio is True
    
    def test_verbose_flag(self):
        """Test --verbose flag."""
        from cli.media_speed import build_parser
        parser = build_parser()
        
        args = parser.parse_args(["--input", "test.mp4"])
        assert args.verbose is False
        
        args = parser.parse_args(["--input", "test.mp4", "-v"])
        assert args.verbose is True


# ─────────────────────────────────────────────────────────────────────
# TESTS - MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────

class TestMain:
    """Tests cho main function."""
    
    def test_missing_input(self, monkeypatch, capsys):
        """Test main exits with error if input missing."""
        from cli.media_speed import main
        
        monkeypatch.setattr("sys.argv", ["media_speed.py", "--input", "nonexistent.mp4"])
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 1
    
    def test_invalid_speed(self, monkeypatch, tmp_path):
        """Test main exits with error if speed invalid."""
        from cli.media_speed import main
        
        # Create a test file
        test_file = tmp_path / "test.mp4"
        test_file.write_text("dummy")
        
        monkeypatch.setattr("sys.argv", ["media_speed.py", "--input", str(test_file), "--speed", "0"])
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 1
    
    def test_unknown_type_auto_detect(self, monkeypatch, tmp_path):
        """Test main exits with error if type unknown."""
        from cli.media_speed import main
        
        # Create a test file with unknown extension
        test_file = tmp_path / "test.xyz"
        test_file.write_text("dummy")
        
        monkeypatch.setattr("sys.argv", ["media_speed.py", "--input", str(test_file)])
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 1


# ─────────────────────────────────────────────────────────────────────
# INTEGRATION TESTS (require hardware)
# ─────────────────────────────────────────────────────────────────────

class TestMediaSpeedCLI:
    """Integration tests cho media_speed CLI - cần FFmpeg."""
    
    @pytest.mark.usefixtures("skip_if_weak_hardware", "check_ffmpeg_available")
    def test_srt_scaling(self, monkeypatch, tmp_path):
        """Test SRT scaling via CLI."""
        from cli.media_speed import main
        
        # Create test SRT
        srt_content = """1
00:00:10,000 --> 00:00:15,000
Test subtitle

2
00:00:20,000 --> 00:00:25,000
Another subtitle
"""
        input_file = tmp_path / "test.srt"
        input_file.write_text(srt_content, encoding='utf-8')
        
        output_file = tmp_path / "test_slow.srt"
        
        monkeypatch.setattr("sys.argv", [
            "media_speed.py",
            "--input", str(input_file),
            "--output", str(output_file),
            "--speed", "0.5",
        ])
        
        main()
        
        assert output_file.exists()
        
        # Check content
        output_content = output_file.read_text(encoding='utf-8')
        assert "00:00:20,000 --> 00:00:30,000" in output_content
        assert "Test subtitle" in output_content
    
    @pytest.mark.usefixtures("skip_if_weak_hardware", "check_ffmpeg_available")
    def test_ass_scaling(self, monkeypatch, tmp_path):
        """Test ASS scaling via CLI."""
        from cli.media_speed import main
        
        # Create test ASS
        ass_content = """[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:10.00,0:00:15.00,Default,,0,0,0,,Test dialogue
"""
        input_file = tmp_path / "test.ass"
        input_file.write_text(ass_content, encoding='utf-8')
        
        output_file = tmp_path / "test_slow.ass"
        
        monkeypatch.setattr("sys.argv", [
            "media_speed.py",
            "--input", str(input_file),
            "--output", str(output_file),
            "--speed", "0.5",
        ])
        
        main()
        
        assert output_file.exists()
        
        # Check content
        output_content = output_file.read_text(encoding='utf-8')
        assert "0:00:20.00,0:00:30.00" in output_content
        assert "Test dialogue" in output_content
    
    @pytest.mark.usefixtures("skip_if_weak_hardware", "check_ffmpeg_available")
    def test_default_output_naming(self, monkeypatch, tmp_path):
        """Test default output naming."""
        from cli.media_speed import main
        
        # Create test SRT
        srt_content = """1
00:00:01,000 --> 00:00:02,000
Test
"""
        input_file = tmp_path / "video.srt"
        input_file.write_text(srt_content, encoding='utf-8')
        
        # Change to tmp_path directory
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            monkeypatch.setattr("sys.argv", [
                "media_speed.py",
                "--input", "video.srt",
                "--speed", "0.65",
            ])
            
            main()
            
            # Check default output was created
            expected_output = tmp_path / "video_slow.srt"
            assert expected_output.exists()
        finally:
            os.chdir(original_cwd)