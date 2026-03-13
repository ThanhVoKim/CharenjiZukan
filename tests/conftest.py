#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/conftest.py - Pytest fixtures dùng chung

Chứa các fixtures cho:
- Hardware check (GPU, CPU cores)
- FFmpeg availability
- Rubberband availability
"""

import pytest
import multiprocessing
import shutil


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


@pytest.fixture(scope="module")
def check_rubberband_available():
    """
    Kiểm tra rubberband binary có sẵn trong PATH.
    
    Returns:
        True nếu rubberband available
        
    Raises:
        pytest.skip: Nếu rubberband không có
    """
    if not shutil.which("rubberband"):
        pytest.skip("rubberband-cli không có trong PATH. Cài đặt: apt-get install rubberband-cli")
    return True


@pytest.fixture(scope="module")
def check_pyrubberband_installed():
    """
    Kiểm tra pyrubberband Python library đã cài đặt.
    
    Returns:
        True nếu pyrubberband installed
        
    Raises:
        pytest.skip: Nếu pyrubberband không có
    """
    try:
        import pyrubberband  # noqa: F401
        return True
    except ImportError:
        pytest.skip("pyrubberband chưa cài đặt. Cài đặt: pip install pyrubberband")


@pytest.fixture(scope="module")
def check_soundfile_installed():
    """
    Kiểm tra soundfile Python library đã cài đặt.
    
    Returns:
        True nếu soundfile installed
        
    Raises:
        pytest.skip: Nếu soundfile không có
    """
    try:
        import soundfile  # noqa: F401
        return True
    except ImportError:
        pytest.skip("soundfile chưa cài đặt. Cài đặt: pip install soundfile")


# ─────────────────────────────────────────────────────────────────────
# COMBINED CHECK FIXTURES
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def check_audio_stretch_dependencies():
    """
    Kiểm tra tất cả dependencies cần thiết cho audio stretching.
    - FFmpeg (bắt buộc)
    - rubberband binary (tùy chọn, fallback to atempo)
    - pyrubberband (tùy chọn, fallback to atempo)
    - soundfile (tùy chọn, fallback to atempo)
    
    Returns:
        dict với các keys: has_ffmpeg, has_rubberband, has_pyrubberband, has_soundfile
    """
    result = {
        'has_ffmpeg': shutil.which("ffmpeg") is not None,
        'has_rubberband': shutil.which("rubberband") is not None,
        'has_pyrubberband': False,
        'has_soundfile': False,
    }
    
    try:
        import pyrubberband  # noqa: F401
        result['has_pyrubberband'] = True
    except ImportError:
        pass
    
    try:
        import soundfile  # noqa: F401
        result['has_soundfile'] = True
    except ImportError:
        pass
    
    # FFmpeg là bắt buộc
    if not result['has_ffmpeg']:
        pytest.skip("FFmpeg không có trong PATH. Cài đặt: apt-get install ffmpeg")
    
    return result