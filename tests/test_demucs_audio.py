#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_demucs_audio.py — Unit tests cho cli/demucs_audio.py
"""

import pytest
import multiprocessing
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def skip_if_weak_hardware():
    """Skip tất cả tests trong module nếu hardware yếu."""
    from cli.demucs_audio import check_hardware_requirements
    can_run, reason = check_hardware_requirements()
    if not can_run:
        pytest.skip(f"Skipped: {reason}")
    return True


# ─────────────────────────────────────────────────────────────────────
# TESTS - Hardware Check (luôn chạy)
# ─────────────────────────────────────────────────────────────────────

class TestCheckHardwareRequirements:
    """Tests cho check_hardware_requirements function."""

    def test_returns_tuple(self):
        """Test function returns correct tuple format."""
        from cli.demucs_audio import check_hardware_requirements
        can_run, reason = check_hardware_requirements()
        assert isinstance(can_run, bool)
        assert isinstance(reason, str)

    def test_cpu_cores_detection(self):
        """Test CPU cores được detect đúng."""
        cores = multiprocessing.cpu_count()
        assert cores >= 1  # Ít nhất 1 core

    def test_reason_not_empty(self):
        """Test reason không rỗng."""
        from cli.demucs_audio import check_hardware_requirements
        can_run, reason = check_hardware_requirements()
        assert len(reason) > 0


class TestCheckDemucsInstalled:
    """Tests cho check_demucs_installed function."""

    def test_returns_boolean(self):
        """Test function returns boolean."""
        from cli.demucs_audio import check_demucs_installed
        result = check_demucs_installed()
        assert isinstance(result, bool)


class TestGetDevice:
    """Tests cho get_device function."""

    def test_returns_string(self):
        """Test function returns string."""
        from cli.demucs_audio import get_device
        result = get_device()
        assert isinstance(result, str)

    def test_returns_valid_device(self):
        """Test function returns valid device."""
        from cli.demucs_audio import get_device
        result = get_device()
        assert result in ["cuda", "cpu"] or result.startswith("cuda:")


# ─────────────────────────────────────────────────────────────────────
# TESTS - CLI Arguments (luôn chạy)
# ─────────────────────────────────────────────────────────────────────

class TestBuildParser:
    """Tests cho build_parser function."""

    def test_parser_exists(self):
        """Test parser can be created."""
        from cli.demucs_audio import build_parser
        parser = build_parser()
        assert parser is not None

    def test_parser_has_required_args(self):
        """Test parser has required arguments."""
        from cli.demucs_audio import build_parser
        parser = build_parser()
        
        # Check required arguments
        actions = {a.dest for a in parser._actions}
        assert "input" in actions
        assert "output" in actions
        assert "model" in actions
        assert "stems" in actions
        assert "keep" in actions
        assert "device" in actions
        assert "verbose" in actions

    def test_default_values(self):
        """Test default values for arguments."""
        from cli.demucs_audio import build_parser
        parser = build_parser()
        
        # Parse with minimal args
        args = parser.parse_args(["--input", "test.wav"])
        
        assert args.model == "htdemucs"
        assert args.stems == 2
        assert args.keep == "bgm"
        assert args.device is None
        assert args.verbose is False

    def test_stems_choices(self):
        """Test stems argument accepts valid values."""
        from cli.demucs_audio import build_parser
        parser = build_parser()
        
        # Valid values
        args = parser.parse_args(["--input", "test.wav", "--stems", "2"])
        assert args.stems == 2
        
        args = parser.parse_args(["--input", "test.wav", "--stems", "4"])
        assert args.stems == 4

    def test_keep_choices(self):
        """Test keep argument accepts valid values."""
        from cli.demucs_audio import build_parser
        parser = build_parser()
        
        args = parser.parse_args(["--input", "test.wav", "--keep", "bgm"])
        assert args.keep == "bgm"
        
        args = parser.parse_args(["--input", "test.wav", "--keep", "vocals"])
        assert args.keep == "vocals"

    def test_model_choices(self):
        """Test model argument accepts valid values."""
        from cli.demucs_audio import build_parser
        parser = build_parser()
        
        valid_models = ["htdemucs", "htdemucs_ft", "mdx", "mdx_extra"]
        for model in valid_models:
            args = parser.parse_args(["--input", "test.wav", "--model", model])
            assert args.model == model


# ─────────────────────────────────────────────────────────────────────
# TESTS - Integration (chỉ chạy nếu hardware đủ)
# ─────────────────────────────────────────────────────────────────────

class TestSeparateAudio:
    """Tests cho separate_audio function - chỉ chạy nếu hardware đủ."""

    @pytest.mark.usefixtures("skip_if_weak_hardware")
    def test_separate_audio_requires_demucs(self):
        """Test separate_audio requires demucs installed."""
        from cli.demucs_audio import check_demucs_installed
        if not check_demucs_installed():
            pytest.skip("Demucs not installed")

    @pytest.mark.usefixtures("skip_if_weak_hardware")
    def test_separate_audio_invalid_input(self, tmp_path):
        """Test separate_audio raises error for invalid input."""
        from cli.demucs_audio import check_demucs_installed
        if not check_demucs_installed():
            pytest.skip("Demucs not installed")
        
        from cli.demucs_audio import separate_audio
        
        non_existent = tmp_path / "non_existent.wav"
        output = tmp_path / "output.wav"
        
        with pytest.raises(Exception):
            separate_audio(
                input_path=str(non_existent),
                output_path=str(output),
            )


# ─────────────────────────────────────────────────────────────────────
# TESTS - Main Function
# ─────────────────────────────────────────────────────────────────────

class TestMain:
    """Tests cho main function."""

    def test_main_missing_input(self, monkeypatch):
        """Test main exits with error if input missing."""
        from cli.demucs_audio import main
        
        monkeypatch.setattr("sys.argv", ["demucs_audio.py"])
        
        with pytest.raises(SystemExit):
            main()

    def test_main_nonexistent_input(self, monkeypatch, tmp_path):
        """Test main exits with error if input file doesn't exist."""
        from cli.demucs_audio import main
        
        non_existent = tmp_path / "non_existent.wav"
        monkeypatch.setattr("sys.argv", ["demucs_audio.py", "--input", str(non_existent)])
        
        with pytest.raises(SystemExit):
            main()


# ─────────────────────────────────────────────────────────────────────
# TESTS - Output Path Logic
# ─────────────────────────────────────────────────────────────────────

class TestOutputPath:
    """Tests cho output path logic."""

    def test_default_output_bgm(self):
        """Test default output path for bgm."""
        from cli.demucs_audio import build_parser
        parser = build_parser()
        
        args = parser.parse_args(["--input", "audio.wav"])
        
        # Output should be None (will be generated)
        assert args.output is None

    def test_custom_output(self):
        """Test custom output path."""
        from cli.demucs_audio import build_parser
        parser = build_parser()
        
        args = parser.parse_args(["--input", "audio.wav", "--output", "custom.wav"])
        
        assert args.output == "custom.wav"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])