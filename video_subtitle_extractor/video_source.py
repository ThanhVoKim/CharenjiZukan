# -*- coding: utf-8 -*-
"""
video_subtitle_extractor/video_source.py — AV1 / unsupported-codec fallback.

opencv-python-headless đi kèm FFmpeg nội bộ chỉ có AV1 hardware decoder; trên
Colab (và hầu hết máy tính thường) không có AV1 hardware decode nên cap.read()
trả rỗng. Module này dò codec qua ffprobe; nếu là AV1 (hoặc codec được cấu hình)
thì transcode sang file tạm H.264/HEVC qua system ffmpeg (có libdav1d), sau đó
trả về path để cv2.VideoCapture đọc bình thường.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Codec mặc định cần transcode qua ffmpeg trước khi cv2 có thể đọc
DEFAULT_TRANSCODE_CODECS = ("av1",)


def probe_video_codec(video_path: str) -> str:
    """Trả về tên codec video stream đầu tiên (ví dụ 'av1', 'h264', 'hevc').

    Dùng ffprobe hệ thống. Trả về "" nếu ffprobe không có hoặc lỗi (fail-open:
    caller sẽ coi như không cần transcode).
    """
    if not shutil.which("ffprobe"):
        return ""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=nk=1:nw=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        return result.stdout.strip().lower()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def prepare_opencv_source(
    video_path: str,
    transcode_codecs: Tuple[str, ...] = DEFAULT_TRANSCODE_CODECS,
) -> Tuple[str, Optional[str]]:
    """Chuẩn bị đường dẫn video có thể đọc bởi cv2.VideoCapture.

    Nếu codec thuộc `transcode_codecs` (mặc định: av1), transcode sang file tạm
    H.264 hoặc HEVC qua system ffmpeg. Caller phải xoá file tạm sau khi dùng xong
    (dùng phần tử thứ 2 của tuple trả về).

    Returns:
        (readable_path, temp_path_or_None)
        - temp_path_or_None: None nếu không cần transcode; ngược lại là đường dẫn
          file tạm cần xoá sau khi cv2.VideoCapture đã release.

    Fail-open: nếu ffprobe không có, codec trống, hoặc transcode thất bại → trả
    về (video_path, None) để OpenCV tự thử (hành vi cũ, không tệ hơn).
    """
    if not shutil.which("ffmpeg"):
        return video_path, None

    codec = probe_video_codec(video_path)
    if not codec or codec not in transcode_codecs:
        return video_path, None

    logger.info(
        "AV1/unsupported codec detected (%s) → transcoding via system ffmpeg: %s",
        codec, video_path,
    )

    # Chọn encoder
    video_args = _choose_encoder()
    suffix = ".mp4"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="ocr_transcode_")
    os.close(tmp_fd)

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-an",                # bỏ audio (không cần cho OCR)
        *video_args,
        "-pix_fmt", "yuv420p",
        tmp_path,
    ]

    logger.info("Transcode command: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=3600)
        if result.returncode != 0:
            err = (result.stderr or "").strip()[-800:]
            logger.error("Transcode failed (rc=%d): %s — falling back to direct OpenCV read", result.returncode, err)
            _safe_remove(tmp_path)
            return video_path, None
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.error("Transcode error: %s — falling back to direct OpenCV read", exc)
        _safe_remove(tmp_path)
        return video_path, None

    logger.info("Transcode complete → %s", tmp_path)
    return tmp_path, tmp_path


# ── internal helpers ──────────────────────────────────────────────────

def _choose_encoder():
    """Chọn encoder cho file tạm: HEVC NVENC (GPU) nếu có, else libx264 (CPU)."""
    try:
        from utils.ffmpeg_probe import detect_hevc_nvenc, HEVC_NVENC_VIDEO_ARGS
        if detect_hevc_nvenc():
            logger.debug("Using hevc_nvenc for transcode temp file")
            return list(HEVC_NVENC_VIDEO_ARGS)
    except ImportError:
        pass
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]


def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
