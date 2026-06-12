#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/calibrate_text_isolation.py — Dò ngưỡng cho text_isolator từ mẫu thật.

Chế độ GIÁM SÁT 2 thư mục: bạn tự cắt vài crop phụ đề và vài crop watermark, script
đo "chữ ký pixel" của từng lớp, đề xuất ngưỡng cho `TextIsolationConfig`, và xuất ảnh
preview trước/sau để bạn xác nhận bằng mắt.

QUY TẮC CẮT MẪU (bắt buộc để ngưỡng khớp lúc chạy thật):
  1. Cắt ở ĐỘ PHÂN GIẢI GỐC của video (1:1 pixel) — không zoom/resize/screenshot phóng to.
  2. Ôm sát chữ nhưng CHỪA vài pixel nền xung quanh (để đo tương phản/viền so với nền).
  3. Dùng PNG (lossless), tránh JPG.
  4. ~5–10 ảnh mỗi thư mục.
  - Crop watermark-ĐÈ-phụ-đề: bỏ vào --subtitle-samples (đầu ra mong muốn = phụ đề sống).
  - Crop watermark đứng một mình: bỏ vào --watermark-samples.

Cách dùng:
    uv run python tools/calibrate_text_isolation.py \
        --subtitle-samples ./samples/subtitle/ \
        --watermark-samples ./samples/watermark/ \
        --subtitle-colors "white,#FFD700" \
        --out ./samples/text_isolation_config.json

Kết quả:
  - <out>.json: đủ tham số TextIsolationConfig (xem độ tin cậy ở docs/text-isolation-guide.md).
  - preview/<tên>_before_after.png: ảnh trước/sau mask từng mẫu.
  - Bảng tóm tắt in ra console.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import cv2
except ImportError:
    print("❌ Cần OpenCV: pip install opencv-python", file=sys.stderr)
    sys.exit(1)

from video_subtitle_extractor.text_isolator import (  # noqa: E402
    TextIsolationConfig,
    _build_color_mask,
    _local_contrast,
    isolate_subtitle_text,
    parse_color_spec,
)

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _load_images(folder: Path) -> List[Tuple[str, np.ndarray]]:
    """Đọc tất cả ảnh trong thư mục → list (tên, BGR ndarray)."""
    images: List[Tuple[str, np.ndarray]] = []
    if not folder.is_dir():
        print(f"⚠️  Thư mục không tồn tại: {folder}", file=sys.stderr)
        return images
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in _IMG_EXTS:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            print(f"⚠️  Bỏ qua ảnh không đọc được: {path.name}", file=sys.stderr)
            continue
        images.append((path.name, img))
    return images


# ── Đo đặc trưng từng crop ──────────────────────────────────────────────────

def _contrast_p90(roi: np.ndarray) -> float:
    """Tương phản đỉnh (P90 morphological gradient) — proxy opacity của crop."""
    return float(np.percentile(_local_contrast(roi), 90))


def _color_distances(roi: np.ndarray, colors: List[Tuple[int, int, int]]) -> np.ndarray:
    """Khoảng cách Lab tới màu phụ đề gần nhất, lấy 40% pixel gần nhất (≈ pixel chữ)."""
    if not colors:
        return np.array([])
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    best = None
    for rgb in colors:
        bgr_px = np.array([[[rgb[2], rgb[1], rgb[0]]]], dtype=np.uint8)
        target = cv2.cvtColor(bgr_px, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
        dist = np.sqrt(np.sum((lab - target) ** 2, axis=2)).ravel()
        best = dist if best is None else np.minimum(best, dist)
    best.sort()
    keep = max(1, int(len(best) * 0.40))
    return best[:keep]


def _otsu_luminance(roi: np.ndarray) -> float:
    """Ngưỡng Otsu trên độ sáng — tách cụm tối (viền/nền) khỏi cụm sáng (chữ)."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thresh)


def _summ(values: List[float]) -> Dict[str, float]:
    """Tóm tắt phân bố: min / mean / max / P10 / P90."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "p10": 0.0, "p90": 0.0}
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def calibrate(
    subtitle_imgs: List[Tuple[str, np.ndarray]],
    watermark_imgs: List[Tuple[str, np.ndarray]],
    colors: List[Tuple[int, int, int]],
) -> Tuple[TextIsolationConfig, Dict]:
    """Đo 2 lớp mẫu → đề xuất TextIsolationConfig + báo cáo độ tách bạch."""
    sub_contrast = [_contrast_p90(img) for _, img in subtitle_imgs]
    wm_contrast = [_contrast_p90(img) for _, img in watermark_imgs]
    sub_c, wm_c = _summ(sub_contrast), _summ(wm_contrast)

    # min_contrast: nằm giữa cụm phụ đề (cao) và watermark (thấp).
    # Dùng trung điểm giữa P10 phụ đề và P90 watermark; fallback nếu thiếu mẫu.
    if sub_contrast and wm_contrast:
        min_contrast = int(round((sub_c["p10"] + wm_c["p90"]) / 2))
    elif sub_contrast:
        min_contrast = int(round(sub_c["p10"] * 0.6))
    else:
        min_contrast = 40
    min_contrast = max(10, min(min_contrast, 250))

    # color_tolerance: bao ~P90 khoảng cách màu của pixel chữ phụ đề.
    color_tolerance = 40
    sub_color_p90 = wm_color_p90 = 0.0
    if colors and subtitle_imgs:
        sub_dists = np.concatenate(
            [_color_distances(img, colors) for _, img in subtitle_imgs]
        )
        if sub_dists.size:
            sub_color_p90 = float(np.percentile(sub_dists, 90))
            color_tolerance = int(round(sub_color_p90))
        if watermark_imgs:
            wm_dists = np.concatenate(
                [_color_distances(img, colors) for _, img in watermark_imgs]
            )
            if wm_dists.size:
                wm_color_p90 = float(np.percentile(wm_dists, 90))
    color_tolerance = max(10, min(color_tolerance, 120))

    # stroke_max_luminance: dưới ngưỡng Otsu của phụ đề (tách cụm tối là viền/nền).
    if subtitle_imgs:
        otsu_vals = [_otsu_luminance(img) for _, img in subtitle_imgs]
        stroke_max_luminance = int(round(float(np.mean(otsu_vals)) * 0.6))
    else:
        stroke_max_luminance = 80
    stroke_max_luminance = max(30, min(stroke_max_luminance, 150))

    # require_stroke: bật nếu phụ đề thật sự có pixel tối (viền) đáng kể.
    require_stroke = True
    if subtitle_imgs:
        dark_fracs = []
        for _, img in subtitle_imgs:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dark_fracs.append(float(np.mean(gray <= stroke_max_luminance)))
        require_stroke = bool(np.mean(dark_fracs) >= 0.10)

    # Dựng config sơ bộ để đo min_component_area (P5 diện tích component phụ đề).
    base_cfg = TextIsolationConfig(
        enabled=True,
        subtitle_colors=colors,
        color_tolerance=color_tolerance,
        min_contrast=min_contrast,
        stroke_max_luminance=stroke_max_luminance,
        require_stroke=require_stroke,
        min_component_area=1,
    )
    areas: List[int] = []
    for _, img in subtitle_imgs:
        color_mask = _build_color_mask(
            img, colors, color_tolerance, base_cfg.bright_luminance
        ).astype(np.uint8)
        num, _, stats, _ = cv2.connectedComponentsWithStats(color_mask, connectivity=8)
        areas.extend(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num))
    if areas:
        p5_area = float(np.percentile(np.asarray(areas), 5))
        min_component_area = max(2, int(round(p5_area * 0.5)))
    else:
        min_component_area = 8

    config = TextIsolationConfig(
        enabled=True,
        subtitle_colors=colors,
        color_tolerance=color_tolerance,
        min_contrast=min_contrast,
        stroke_max_luminance=stroke_max_luminance,
        min_component_area=min_component_area,
        require_stroke=require_stroke,
    )

    report = {
        "subtitle_contrast": sub_c,
        "watermark_contrast": wm_c,
        "subtitle_color_p90": round(sub_color_p90, 1),
        "watermark_color_p90": round(wm_color_p90, 1),
        "num_subtitle_samples": len(subtitle_imgs),
        "num_watermark_samples": len(watermark_imgs),
    }
    return config, report


def _write_previews(
    images: List[Tuple[str, np.ndarray]],
    config: TextIsolationConfig,
    out_dir: Path,
    tag: str,
) -> None:
    """Ghi ảnh ghép [gốc | đã mask] cho mỗi mẫu để kiểm tra bằng mắt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, img in images:
        cleaned = isolate_subtitle_text(img, config)
        gap = np.full((img.shape[0], 8, 3), 60, dtype=np.uint8)
        combo = np.hstack([img, gap, cleaned])
        stem = Path(name).stem
        cv2.imwrite(str(out_dir / f"{tag}_{stem}_before_after.png"), combo)


def _print_report(config: TextIsolationConfig, report: Dict) -> None:
    sc = report["subtitle_contrast"]
    wc = report["watermark_contrast"]
    print("\n" + "=" * 64)
    print("📊 BÁO CÁO HIỆU CHỈNH text_isolator")
    print("=" * 64)
    print(f"  Mẫu phụ đề: {report['num_subtitle_samples']}  |  "
          f"Mẫu watermark: {report['num_watermark_samples']}")
    print("\n  Tương phản (P90 gradient) — proxy opacity:")
    print(f"    Phụ đề   : mean={sc['mean']:.0f}  P10={sc['p10']:.0f}  min={sc['min']:.0f}")
    print(f"    Watermark: mean={wc['mean']:.0f}  P90={wc['p90']:.0f}  max={wc['max']:.0f}")
    gap = sc["p10"] - wc["p90"]
    sep = "TỐT ✅" if gap > 20 else ("HẸP ⚠️" if gap > 0 else "CHỒNG LẤN ❌")
    print(f"    → Độ tách bạch: {sep} (P10_phụđề − P90_watermark = {gap:.0f})")
    if report["subtitle_color_p90"] or report["watermark_color_p90"]:
        print("\n  Khoảng cách màu (Lab, pixel chữ):")
        print(f"    Phụ đề P90   : {report['subtitle_color_p90']:.0f}")
        print(f"    Watermark P90: {report['watermark_color_p90']:.0f}")
    print("\n  📌 Tham số đề xuất (xác nhận lại bằng ảnh preview!):")
    print(f"    min_contrast        = {config.min_contrast}   (mạnh — feature lõi)")
    print(f"    color_tolerance     = {config.color_tolerance}   (mạnh)")
    print(f"    stroke_max_luminance= {config.stroke_max_luminance}   (khá)")
    print(f"    require_stroke      = {config.require_stroke}   (khuyến nghị)")
    print(f"    min_component_area  = {config.min_component_area}   (yếu — chỉ diệt nhiễu, đặt nhỏ)")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="calibrate_text_isolation",
        description="Dò ngưỡng text_isolator từ 2 thư mục mẫu (giám sát).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subtitle-samples", required=True,
                        help="Thư mục crop phụ đề (gồm cả crop watermark-đè-phụ-đề)")
    parser.add_argument("--watermark-samples", required=True,
                        help="Thư mục crop chỉ chứa watermark/overlay")
    parser.add_argument("--subtitle-colors", default="",
                        help='Màu phụ đề: "white,#FFD700" hoặc "255,255,255"')
    parser.add_argument("--out", default="text_isolation_config.json",
                        help="Đường dẫn file JSON config xuất ra")
    parser.add_argument("--preview-dir", default=None,
                        help="Thư mục ảnh preview (mặc định: <out_dir>/preview)")
    args = parser.parse_args()

    colors = parse_color_spec(args.subtitle_colors)
    if args.subtitle_colors and not colors:
        print(f"⚠️  Không parse được màu từ: {args.subtitle_colors!r}", file=sys.stderr)

    subtitle_imgs = _load_images(Path(args.subtitle_samples))
    watermark_imgs = _load_images(Path(args.watermark_samples))
    if not subtitle_imgs:
        print("❌ Không có mẫu phụ đề hợp lệ.", file=sys.stderr)
        sys.exit(1)

    config, report = calibrate(subtitle_imgs, watermark_imgs, colors)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "enabled": config.enabled,
        "subtitle_colors": config.subtitle_colors,
        "color_tolerance": config.color_tolerance,
        "min_contrast": config.min_contrast,
        "stroke_max_luminance": config.stroke_max_luminance,
        "stroke_search_px": config.stroke_search_px,
        "min_component_area": config.min_component_area,
        "require_stroke": config.require_stroke,
        "bright_luminance": config.bright_luminance,
        "_report": report,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    preview_dir = Path(args.preview_dir) if args.preview_dir else out_path.parent / "preview"
    _write_previews(subtitle_imgs, config, preview_dir, "subtitle")
    _write_previews(watermark_imgs, config, preview_dir, "watermark")

    _print_report(config, report)
    print(f"\n✅ Config:  {out_path}")
    print(f"✅ Preview: {preview_dir}/  (soi xem phụ đề còn nguyên, watermark đã đen)")


if __name__ == "__main__":
    main()
