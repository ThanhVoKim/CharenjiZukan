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


@pytest.fixture(scope="session")
def concat_workers(request) -> int:
    """Trả về số worker chạy song song từ --workers hoặc env var TEST_CONCAT_WORKERS."""
    workers_str = request.config.getoption("--workers", None) or os.getenv("TEST_CONCAT_WORKERS", "4")
    try:
        return int(workers_str)
    except ValueError:
        return 4


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
def skip_if_insufficient_vram(min_vram_gb: float = 10.0):
    """
    Skip nếu VRAM không đủ cho model OCR.
    - DeepSeek-OCR-2: ~8GB VRAM
    - Qwen3-VL-8B: ~16GB VRAM
    """
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("Không có GPU.")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb < min_vram_gb:
            pytest.skip(
                f"VRAM không đủ: {vram_gb:.1f}GB < {min_vram_gb}GB yêu cầu."
            )
    except ImportError:
        pytest.skip("torch không được cài đặt.")
    return True


@pytest.fixture(scope="module")
def native_video_gpu_preflight():
    """
    Preflight GPU cho test Native Video OCR.

    Điều kiện:
    - torch đã cài đặt
    - CUDA khả dụng
    - VRAM >= ngưỡng tối thiểu (mặc định 10GB, có thể override qua env)

    Environment variable:
        NATIVE_OCR_MIN_VRAM_GB (default: 10)
    """
    min_vram_gb = float(os.getenv("NATIVE_OCR_MIN_VRAM_GB", "10"))

    try:
        import torch
    except ImportError:
        pytest.skip("torch không được cài đặt.")

    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU không khả dụng. Test Native Video OCR yêu cầu GPU.")

    device_index = torch.cuda.current_device() if torch.cuda.device_count() > 0 else 0
    device_name = torch.cuda.get_device_name(device_index)
    total_vram_gb = torch.cuda.get_device_properties(device_index).total_memory / (1024**3)

    if total_vram_gb < min_vram_gb:
        pytest.skip(
            f"VRAM không đủ cho preflight Native Video OCR: "
            f"{total_vram_gb:.1f}GB < {min_vram_gb:.1f}GB."
        )

    return {
        "cuda_available": True,
        "device_index": device_index,
        "device_name": device_name,
        "total_vram_gb": total_vram_gb,
        "min_vram_gb": min_vram_gb,
    }


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