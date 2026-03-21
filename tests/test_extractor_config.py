import os
import argparse
import pytest
from unittest.mock import patch, MagicMock
from cli.video_ocr import parse_args, load_config
import sys
import yaml

@pytest.fixture
def mock_yaml_config(tmp_path):
    config = {
        "video": {"frame_interval": 60},
        "roi": {"boxes": [{"name": "test", "x": 1, "y": 2, "w": 3, "h": 4}]},
        "scene_detection": {"threshold": 25.0, "min_scene_frames": 5},
        "ocr": {"model": "test-model", "device": "cpu", "batch_size": 16},
        "chinese_filter": {"enabled": True, "keep_punctuation": False, "min_char_count": 3},
        "output": {"format": "txt", "include_timestamp": False, "deduplicate": False, "default_duration": 5.0, "min_duration": 2.0, "max_duration": 10.0}
    }
    
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
        
    return str(config_file), config

def test_config_precedence(mock_yaml_config):
    config_path, yaml_config = mock_yaml_config
    
    # Giả lập sys.argv với một số CLI params
    test_args = [
        "cli/video_ocr.py", "video.mp4",
        "--config", config_path,
        "--frame-interval", "45",  # override YAML 60
        "--device", "cuda",        # override YAML cpu
        "--no-scene-detection"     # override YAML threshold 25.0 -> 0
    ]
    
    with patch.object(sys, 'argv', test_args):
        args = parse_args()
        
    config = load_config(args.config)
    
    # Helper get_param copy từ cli/video_ocr.py
    def get_param(cli_name: str, yaml_path: tuple, default_val):
        if hasattr(args, cli_name):
            return getattr(args, cli_name)
        val = config
        for key in yaml_path:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                val = None
                break
        if val is not None:
            return val
        return default_val
        
    # 1. CLI Override YAML
    assert get_param("frame_interval", ("video", "frame_interval"), 30) == 45
    assert get_param("device", ("ocr", "device"), "cuda") == "cuda"
    
    # 2. YAML Override Default (không có CLI)
    assert get_param("min_scene_frames", ("scene_detection", "min_scene_frames"), 10) == 5
    assert get_param("ocr_model", ("ocr", "model"), "default") == "test-model"
    assert get_param("batch_size", ("ocr", "batch_size"), 8) == 16
    assert get_param("format", ("output", "format"), "srt") == "txt"
    assert get_param("default_duration", ("output", "default_duration"), 3.0) == 5.0
    
    # 3. Default fallback (không CLI, không YAML)
    assert get_param("missing_val", ("non", "existent"), "fallback") == "fallback"
    
    # Check flags logic
    scene_thresh = 0 if hasattr(args, "no_scene_detection") else get_param("scene_threshold", ("scene_detection", "threshold"), 30.0)
    assert scene_thresh == 0

def test_boolean_flags_precedence(mock_yaml_config):
    config_path, yaml_config = mock_yaml_config
    
    test_args = [
        "cli/video_ocr.py", "video.mp4",
        "--config", config_path,
        "--no-punctuation",
        "--no-timestamp"
    ]
    
    with patch.object(sys, 'argv', test_args):
        args = parse_args()
        
    config = load_config(args.config)
    
    def get_param(cli_name: str, yaml_path: tuple, default_val):
        if hasattr(args, cli_name):
            return getattr(args, cli_name)
        val = config
        for key in yaml_path:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                val = None
                break
        if val is not None:
            return val
        return default_val
        
    # YAML says True for deduplicate, CLI doesn't suppress -> True
    deduplicate_output = not hasattr(args, "no_deduplicate") if hasattr(args, "no_deduplicate") else get_param("deduplicate", ("output", "deduplicate"), True)
    assert deduplicate_output == False # from yaml
    
    # YAML says keep_punctuation=False, CLI says --no-punctuation -> False (CLI win)
    keep_punct = not hasattr(args, "no_punctuation") if hasattr(args, "no_punctuation") else get_param("keep_punctuation", ("chinese_filter", "keep_punctuation"), True)
    assert keep_punct == False
    
    # CLI --no-timestamp -> False (overrides any default/yaml)
    include_timestamp = not hasattr(args, "no_timestamp") if hasattr(args, "no_timestamp") else get_param("include_timestamp", ("output", "include_timestamp"), True)
    assert include_timestamp == False