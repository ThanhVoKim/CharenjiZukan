# -*- coding: utf-8 -*-
"""
Text Isolator — Tách phụ đề lời thoại khỏi watermark/overlay mờ trước khi OCR.

Bối cảnh:
    Watermark/overlay text của creator thường có **opacity < 70%** → pixel bị trộn
    với nền (`hiển_thị = α·text + (1−α)·nền`, α < 0.7). Hệ quả KHÔNG phụ thuộc màu:
      1. Tương phản cục bộ / gradient bị nhân với α → biên mờ, cạnh yếu.
      2. Watermark mờ không có viền (stroke) tối sắc như hardsub.
      3. Màu bị blend về phía nền → lệch khỏi màu phụ đề gốc.

    Phụ đề lời thoại thì đặc (α≈1), tương phản cao, có viền sắc, đúng màu chỉ định.

Thuật toán (deterministic, thuần OpenCV/NumPy — KHÔNG gọi model, không hallucinate):
    Quyết định GIỮ/XÓA ở mức **từng connected component (glyph)**, không phải pixel:
      A. color_mask  — pixel khớp màu phụ đề chỉ định (khoảng cách Lab ≤ tolerance).
                       Nếu không chỉ định màu → fallback theo độ sáng (bright mask).
      B. contrast    — morphological gradient (local max−min) làm proxy cho opacity.
      C. Với mỗi component của color_mask: GIỮ nếu tương phản đỉnh ≥ min_contrast
                       (đặc) VÀ (tùy chọn) có viền tối lân cận VÀ diện tích đủ lớn.
      D. Dựng ảnh sạch: nền đen, giữ nguyên pixel màu của các component được giữ.

Module này độc lập, không phụ thuộc `cli/` hay phần còn lại của extractor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 luôn có ở môi trường chạy thật
    cv2 = None


# ── Bảng màu có tên (RGB) cho parse_color_spec ──────────────────────────────
_NAMED_COLORS: dict[str, Tuple[int, int, int]] = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "yellow": (255, 255, 0),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "gold": (255, 215, 0),
}


@dataclass
class TextIsolationConfig:
    """Cấu hình lọc phụ đề khỏi watermark/overlay mờ.

    Các ngưỡng dưới đây được dò bằng `tools/calibrate_text_isolation.py`.

    Attributes:
        enabled: Bật/tắt toàn bộ tính năng. Mặc định TẮT để không phá luồng cũ.
        subtitle_colors: Danh sách màu phụ đề (RGB) do người dùng chỉ định.
            Rỗng → bỏ color gate, chỉ dựa độ sáng + tương phản.
        color_tolerance: Sai số khoảng cách màu trong không gian Lab.
        min_contrast: Ngưỡng morphological gradient (proxy opacity). Component có
            tương phản đỉnh dưới ngưỡng này bị coi là mờ → xóa.
        stroke_max_luminance: Pixel được coi là "viền tối" nếu độ sáng ≤ giá trị này.
        stroke_search_px: Bán kính tìm viền tối quanh component (pixel).
        min_component_area: Diện tích tối thiểu (pixel) để giữ component — chỉ để
            DIỆT NHIỄU lốm đốm, KHÔNG dùng để loại watermark. Đặt nhỏ.
        require_stroke: Bật kiểm tra viền tối. Tắt nếu phụ đề không có viền.
        bright_luminance: Ngưỡng độ sáng cho fallback bright mask khi không chỉ định màu.
    """

    enabled: bool = False
    subtitle_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    color_tolerance: int = 40
    min_contrast: int = 40
    stroke_max_luminance: int = 80
    stroke_search_px: int = 3
    min_component_area: int = 8
    require_stroke: bool = True
    bright_luminance: int = 200


def parse_color_spec(spec: str) -> List[Tuple[int, int, int]]:
    """Parse chuỗi mô tả màu phụ đề thành danh sách RGB tuple.

    Hỗ trợ nhiều màu, ngăn cách bằng ``;`` hoặc ``,``. Mỗi màu là một trong:
      - Tên: ``white``, ``yellow``, ``gold``, ... (xem ``_NAMED_COLORS``).
      - Hex: ``#FFD700`` hoặc ``#FFF``.
      - Bộ ba RGB: ``255:215:0`` hoặc ``255-215-0`` (dùng ``:`` hoặc ``-``).

    Quy tắc tách: nếu chuỗi chứa ``;`` thì tách theo ``;``. Nếu không và toàn chuỗi
    đúng 3 số nguyên ngăn bằng ``,`` (vd ``255,255,255``) thì coi là MỘT màu RGB.
    Ngược lại tách theo ``,``.

    Example:
        >>> parse_color_spec("white,#FFD700")
        [(255, 255, 255), (255, 215, 0)]
        >>> parse_color_spec("255,255,255")
        [(255, 255, 255)]
        >>> parse_color_spec("white;255:215:0")
        [(255, 255, 255), (255, 215, 0)]
    """
    spec = (spec or "").strip()
    if not spec:
        return []

    if ";" in spec:
        tokens = spec.split(";")
    else:
        parts = spec.split(",")
        if len(parts) == 3 and all(p.strip().lstrip("-").isdigit() for p in parts):
            tokens = [spec]  # nguyên chuỗi là một bộ RGB "r,g,b"
        else:
            tokens = parts

    colors: List[Tuple[int, int, int]] = []
    for token in tokens:
        color = _parse_single_color(token.strip())
        if color is not None:
            colors.append(color)
    return colors


def _parse_single_color(token: str) -> Tuple[int, int, int] | None:
    """Parse một token màu đơn lẻ → RGB tuple, hoặc None nếu không hợp lệ."""
    if not token:
        return None

    low = token.lower()
    if low in _NAMED_COLORS:
        return _NAMED_COLORS[low]

    # Hex: #RRGGBB hoặc #RGB
    if low.startswith("#"):
        hex_str = low[1:]
        if len(hex_str) == 3:
            hex_str = "".join(ch * 2 for ch in hex_str)
        if len(hex_str) == 6:
            try:
                return (
                    int(hex_str[0:2], 16),
                    int(hex_str[2:4], 16),
                    int(hex_str[4:6], 16),
                )
            except ValueError:
                return None
        return None

    # Bộ ba RGB phân tách bằng ':' '-' hoặc ','
    for sep in (":", "-", ","):
        if sep in token:
            parts = token.split(sep)
            if len(parts) == 3 and all(p.strip().isdigit() for p in parts):
                vals = [max(0, min(255, int(p.strip()))) for p in parts]
                return (vals[0], vals[1], vals[2])
            break
    return None


def _build_color_mask(
    roi_bgr: np.ndarray,
    colors: List[Tuple[int, int, int]],
    tolerance: int,
    bright_luminance: int,
) -> np.ndarray:
    """Mask pixel khớp một trong các màu phụ đề (khoảng cách Lab ≤ tolerance).

    Không chỉ định màu → fallback: pixel sáng (độ sáng ≥ bright_luminance).
    """
    height, width = roi_bgr.shape[:2]

    if not colors:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        return gray >= bright_luminance

    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    mask = np.zeros((height, width), dtype=bool)

    for rgb in colors:
        # RGB → BGR (1x1) → Lab để khoảng cách màu nhất quán với roi.
        bgr_px = np.array([[[rgb[2], rgb[1], rgb[0]]]], dtype=np.uint8)
        target_lab = cv2.cvtColor(bgr_px, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
        dist = np.sqrt(np.sum((lab - target_lab) ** 2, axis=2))
        mask |= dist <= float(tolerance)

    return mask


def _local_contrast(roi_bgr: np.ndarray) -> np.ndarray:
    """Bản đồ tương phản cục bộ = morphological gradient (dilate − erode) trên gray.

    Cao tại biên glyph đặc, thấp ở vùng mờ/phẳng → proxy cho opacity.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    return grad


def isolate_subtitle_text(
    roi_bgr: np.ndarray, config: TextIsolationConfig
) -> np.ndarray:
    """Trả về ROI đã làm sạch: giữ pixel phụ đề, xóa watermark/overlay mờ.

    Quyết định GIỮ/XÓA ở mức từng connected component (glyph). Output có nền đen,
    giữ nguyên màu pixel phụ đề — dùng làm input cho OCR.

    Args:
        roi_bgr: Ảnh ROI (BGR, từ OpenCV).
        config: Cấu hình ``TextIsolationConfig``.

    Returns:
        Ảnh BGR cùng kích thước, chỉ còn pixel phụ đề trên nền đen. Nếu
        ``config.enabled`` False hoặc ảnh rỗng → trả về ``roi_bgr`` nguyên trạng.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) chưa được cài đặt — không thể chạy text_isolator.")

    if not config.enabled or roi_bgr is None or roi_bgr.size == 0:
        return roi_bgr
    if roi_bgr.ndim != 3 or roi_bgr.shape[2] != 3:
        return roi_bgr

    # A. Color gate (hoặc bright fallback)
    color_mask = _build_color_mask(
        roi_bgr, config.subtitle_colors, config.color_tolerance, config.bright_luminance
    )
    if not np.any(color_mask):
        return np.zeros_like(roi_bgr)

    # B. Bản đồ tương phản (proxy opacity)
    contrast = _local_contrast(roi_bgr)

    # Viền tối: pixel đủ tối + (đã giãn nở) để kiểm tra lân cận component.
    near_dark = None
    if config.require_stroke:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        dark_mask = (gray <= config.stroke_max_luminance).astype(np.uint8)
        radius = max(1, int(config.stroke_search_px))
        ksize = 2 * radius + 1
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        near_dark = cv2.dilate(dark_mask, dilate_kernel) > 0

    # C. Duyệt từng connected component của color_mask, quyết định GIỮ/XÓA.
    color_u8 = color_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(color_u8, connectivity=8)

    keep_mask = np.zeros(roi_bgr.shape[:2], dtype=bool)
    for label in range(1, num_labels):  # 0 là nền
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < config.min_component_area:
            continue  # nhiễu lốm đốm

        component = labels == label

        # Opacity gate: tương phản đỉnh trong component phải đủ cao (đặc).
        if int(contrast[component].max()) < config.min_contrast:
            continue  # mờ → watermark/overlay

        # Stroke gate: phải có viền tối lân cận.
        if near_dark is not None and not np.any(near_dark & component):
            continue

        keep_mask |= component

    # D. Dựng ảnh sạch: nền đen, giữ nguyên pixel màu phụ đề.
    clean = np.zeros_like(roi_bgr)
    clean[keep_mask] = roi_bgr[keep_mask]
    return clean
