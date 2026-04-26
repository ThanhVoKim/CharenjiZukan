#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/conftest.py - Pytest fixtures dùng chung

Chứa các fixtures cho:
- Hardware check (GPU, CPU cores)
- FFmpeg availability
- Rubberband availability
"""

import os
import subprocess
import pytest
import multiprocessing
import shutil
from pathlib import Path


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
    """Tự động phát hiện NVIDIA NVENC (h264_nvenc) có sẵn hay không.

    Phát hiện 2 bước:
      1. ``torch.cuda.is_available()`` — kiểm tra nhanh GPU hardware + driver.
         PyTorch đã có sẵn trong project nên không cần thêm dependency.
      2. Dummy encode test — thực sự gọi FFmpeg encode 1 frame bằng h264_nvenc
         để xác nhận encoder hoạt động (tránh trường hợp driver lỗi / libcuda thiếu).

    Trả về True chỉ khi CẢ HAI bước đều pass.
    """
    # ── Bước 1: PyTorch CUDA check (nhanh, ~50ms) ──────────────────────
    try:
        import torch
        if not torch.cuda.is_available():
            return False
    except ImportError:
        # Không có PyTorch → không thể xác nhận GPU → fallback qua bước 2
        pass

    # ── Bước 2: Dummy encode test (chậm hơn, ~2-5s) ────────────────────
    if not shutil.which("ffmpeg"):
        return False
    try:
        # Tạo 1 frame đen 64x64, encode bằng h264_nvenc, pipe ra null
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.04",
                "-c:v", "h264_nvenc",
                "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


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

