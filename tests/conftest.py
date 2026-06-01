#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/conftest.py - Pytest fixtures dùng chung

Chứa các fixtures cho:
- Hardware check (GPU, CPU cores)
- FFmpeg availability
- Rubberband availability
"""

import multiprocessing
import shutil

import pytest

from utils.ffmpeg_probe import detect_hevc_nvenc


def pytest_addoption(parser):
    """Đăng ký custom CLI options."""
    parser.addoption(
        "--video-path",
        action="store",
        default=None,
        help="Đường dẫn đến video thật cho test Concat Demuxer desync",
    )
    parser.addoption(
        "--workers",
        action="store",
        default=4,
        type=int,
        help="Số lượng worker chạy song song (mặc định: 4)",
    )


@pytest.fixture(scope="session")
def use_gpu() -> bool:
    """Tự động phát hiện HEVC NVENC usable ở runtime.

    Fixture giữ tên ``use_gpu`` để tương thích các test cũ, nhưng SSOT phát hiện
    encoder hiện nằm ở ``utils.ffmpeg_probe.detect_hevc_nvenc``. Hàm dùng chung này
    không chỉ đọc ``ffmpeg -encoders`` mà còn dummy encode 1 frame bằng đúng tham số
    production ``hevc_nvenc -preset p4 -tune hq -cq 28``.
    """
    return detect_hevc_nvenc()


# ─────────────────────────────────────────────────────────────────────
# HARDWARE CHECK FIXTURES
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def skip_if_weak_hardware():
    """
    Skip tests nếu hardware yếu (không GPU, CPU < 4 cores).
    Dùng cho các tests cần FFmpeg/rubberband xử lý audio/video.
    
    Returns:
        True nếu hardware đủ mạnh
        
    Raises:
        pytest.skip: Nếu hardware yếu
    """
    # Kiểm tra GPU
    has_gpu = False
    gpu_name = None
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    
    # Kiểm tra CPU cores
    cpu_cores = multiprocessing.cpu_count()
    
    # Quyết định
    if has_gpu:
        return True  # GPU available
    
    if cpu_cores < 4:
        pytest.skip(f"Hardware yếu: CPU {cpu_cores} cores < 4 minimum. Cần GPU hoặc CPU >= 4 cores.")
    
    return True


@pytest.fixture(scope="module")
def skip_if_no_gpu():
    """Skip nếu không có GPU (bắt buộc cho các OCR model lớn)."""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("GPU không khả dụng. OCR model cần CUDA GPU.")
    except ImportError:
        pytest.skip("torch không được cài đặt.")
    return True


@pytest.fixture(scope="module")
def check_ffmpeg_available():
    """
    Kiểm tra FFmpeg có sẵn trong PATH.
    
    Returns:
        True nếu FFmpeg available
        
    Raises:
        pytest.skip: Nếu FFmpeg không có
    """
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg không có trong PATH. Cài đặt: apt-get install ffmpeg hoặc download từ https://ffmpeg.org")
    return True

