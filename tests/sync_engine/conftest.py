#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/sync_engine/conftest.py - Fixtures dùng chung cho sync_engine tests

Chứa các fixtures cho:
- Real video path (cho test_concat_demuxer Layer 3)
- Concat workers config
"""

import os
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def real_video_path(request) -> Path:
    """
    Trả về Path đến video thật từ --video-path hoặc env var TEST_REAL_VIDEO_PATH.
    Skip test nếu không cung cấp.
    """
    path_str = request.config.getoption("--video-path", None) or os.getenv("TEST_REAL_VIDEO_PATH", "")
    if not path_str:
        pytest.skip("Cần cung cấp --video-path=<đường_dẫn_video> hoặc biến môi trường TEST_REAL_VIDEO_PATH")

    path = Path(path_str)
    if not path.exists():
        pytest.skip(f"Video không tồn tại: {path}")

    return path


@pytest.fixture(scope="session")
def concat_workers(request) -> int:
    """Trả về số worker chạy song song từ --workers hoặc env var TEST_CONCAT_WORKERS."""
    workers_str = request.config.getoption("--workers", None) or os.getenv("TEST_CONCAT_WORKERS", "4")
    try:
        return int(workers_str)
    except ValueError:
        return 4
