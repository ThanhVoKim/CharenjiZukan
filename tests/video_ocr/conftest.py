#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/video_ocr/conftest.py - Fixtures dùng chung cho video_ocr tests

Chứa các fixtures cho:
- GPU/VRAM preflight checks (cho Native Video OCR Layer 4)
"""

import os
import pytest


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
