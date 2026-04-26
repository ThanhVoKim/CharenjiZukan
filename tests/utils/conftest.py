#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/utils/conftest.py - Fixtures dùng chung cho utils tests

Chứa các fixtures cho:
- Rubberband availability check
- pyrubberband/soundfile installed check
- Audio stretch dependencies combined check
"""

import shutil
import pytest


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
