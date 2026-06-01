#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/ffmpeg_probe.py — Shared FFmpeg encoder/runtime capability probes.

Module này cố ý độc lập với sync_engine để CLI/utilities như pre_cut_video có thể dùng
chung một nguồn logic phát hiện encoder FFmpeg.
"""

import logging
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

HEVC_NVENC_VIDEO_ARGS = ["-c:v", "hevc_nvenc", "-preset", "p4", "-tune", "hq", "-cq", "28"]

_HEVC_NVENC_USABLE: Optional[bool] = None
_HEVC_NVENC_LAST_ERROR = ""


def _set_hevc_nvenc_probe_result(usable: bool, reason: str = "") -> bool:
    """Cache kết quả probe NVENC để không chạy dummy encode nhiều lần trong cùng process."""
    global _HEVC_NVENC_USABLE, _HEVC_NVENC_LAST_ERROR
    _HEVC_NVENC_USABLE = usable
    _HEVC_NVENC_LAST_ERROR = reason
    if not usable and reason:
        logger.debug("hevc_nvenc runtime probe failed: %s", reason)
    return usable


def get_hevc_nvenc_unavailable_reason() -> str:
    """Trả về lý do gần nhất khiến runtime probe hevc_nvenc fail."""
    return _HEVC_NVENC_LAST_ERROR


def reset_hevc_nvenc_probe_cache() -> None:
    """Reset cache probe; chủ yếu dùng cho tests cần tách biệt từng case."""
    global _HEVC_NVENC_USABLE, _HEVC_NVENC_LAST_ERROR
    _HEVC_NVENC_USABLE = None
    _HEVC_NVENC_LAST_ERROR = ""


def detect_hevc_nvenc(force_refresh: bool = False) -> bool:
    """Kiểm tra `hevc_nvenc` có chạy được thật ở runtime hay không.

    `ffmpeg -encoders` chỉ chứng minh binary FFmpeg được build kèm encoder NVENC.
    Trên Colab/Linux, trường hợp false-positive rất phổ biến: encoder có trong list
    nhưng khi encode thật lại fail vì thiếu NVIDIA driver/CUDA runtime (`libcuda.so.1`).

    Vì vậy hàm này kiểm tra 2 bước:
    1. FFmpeg có advertise encoder `hevc_nvenc`.
    2. Dummy encode 1 frame bằng đúng tham số production `-preset p4 -tune hq -cq 28`.
    """
    if _HEVC_NVENC_USABLE is not None and not force_refresh:
        return _HEVC_NVENC_USABLE

    if not shutil.which("ffmpeg"):
        return _set_hevc_nvenc_probe_result(False, "Không tìm thấy ffmpeg trong PATH")

    try:
        encoders = subprocess.run(
            ["ffmpeg", "-encoders", "-v", "quiet"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        encoders_output = f"{encoders.stdout}\n{encoders.stderr}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _set_hevc_nvenc_probe_result(False, f"Không chạy được ffmpeg -encoders: {exc}")

    if "hevc_nvenc" not in encoders_output:
        return _set_hevc_nvenc_probe_result(False, "FFmpeg không advertise encoder hevc_nvenc")

    try:
        probe = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=256x256:r=30:d=0.04",
                "-frames:v",
                "1",
                *HEVC_NVENC_VIDEO_ARGS,
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _set_hevc_nvenc_probe_result(False, f"Dummy encode hevc_nvenc không chạy được: {exc}")

    if probe.returncode != 0:
        reason = (probe.stderr or probe.stdout or "Dummy encode hevc_nvenc thất bại").strip()
        return _set_hevc_nvenc_probe_result(False, reason[-1000:])

    return _set_hevc_nvenc_probe_result(True, "")
