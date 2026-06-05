"""
sync_engine/tuber_prerender.py
===============================
Pre-render toàn bộ body×mouth combinations bằng Python/PIL (V2-P2).

Thay thế Remotion: bake sẵn mọi tổ hợp body_frame × mouth_state → runtime chỉ
cần lookup frame đúng → FFmpeg overlay. Port thuật toán 2-triangle affine warp
từ remotion_tuber/src/mouthWarp.ts.

Output:
  assets/pngtuber/<assetId>/prerendered/
    frame-{trackIdx:03d}_{state}.png
    prerender_manifest.json
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("sync_video")

# ── Affine math ───────────────────────────────────────────────────────────

Point = Tuple[float, float]
Triangle = Tuple[Point, Point, Point]


def compute_affine(
    src_a: Point, src_b: Point, src_c: Point,
    dst_a: Point, dst_b: Point, dst_c: Point,
) -> Optional[Tuple[float, float, float, float, float, float]]:
    """6-parameter affine: maps source triangle → destination triangle.

    Port mouthsWarp.ts:computeAffine(). Giải system:
      dst_x = a * src_x + c * src_y + e
      dst_y = b * src_x + d * src_y + f

    Return (a, b, c, d, e, f) hoặc None nếu denominator = 0 (degenerate).
    """
    sx0, sy0 = src_a
    sx1, sy1 = src_b
    sx2, sy2 = src_c
    dx0, dy0 = dst_a
    dx1, dy1 = dst_b
    dx2, dy2 = dst_c

    denominator = sx0 * (sy1 - sy2) + sx1 * (sy2 - sy0) + sx2 * (sy0 - sy1)
    if denominator == 0:
        return None

    a = (dx0 * (sy1 - sy2) + dx1 * (sy2 - sy0) + dx2 * (sy0 - sy1)) / denominator
    b = (dy0 * (sy1 - sy2) + dy1 * (sy2 - sy0) + dy2 * (sy0 - sy1)) / denominator
    c = (dx0 * (sx2 - sx1) + dx1 * (sx0 - sx2) + dx2 * (sx1 - sx0)) / denominator
    d = (dy0 * (sx2 - sx1) + dy1 * (sx0 - sx2) + dy2 * (sx1 - sx0)) / denominator
    e = (
        dx0 * (sx1 * sy2 - sx2 * sy1)
        + dx1 * (sx2 * sy0 - sx0 * sy2)
        + dx2 * (sx0 * sy1 - sx1 * sy0)
    ) / denominator
    f = (
        dy0 * (sx1 * sy2 - sx2 * sy1)
        + dy1 * (sx2 * sy0 - sx0 * sy2)
        + dy2 * (sx0 * sy1 - sx1 * sy0)
    ) / denominator

    return (a, b, c, d, e, f)


def _invert_affine(
    a: float, b: float, c: float, d: float, e: float, f: float,
) -> Optional[Tuple[float, float, float, float, float, float]]:
    """Invert 2D affine matrix.

    Canvas matrix: [a c e; b d f; 0 0 1] maps source→destination.
    Inverse maps destination→source (cần cho PIL AFFINE).
    """
    det = a * d - c * b
    if abs(det) < 1e-10:
        return None
    inv_det = 1.0 / det
    # [d/det, -c/det, (c*f-d*e)/det; -b/det, a/det, (b*e-a*f)/det; 0, 0, 1]
    pil_a = d * inv_det
    pil_b = -c * inv_det
    pil_c = (c * f - d * e) * inv_det
    pil_d = -b * inv_det
    pil_e = a * inv_det
    pil_f = (b * e - a * f) * inv_det
    return (pil_a, pil_b, pil_c, pil_d, pil_e, pil_f)


# ── PIL warp ──────────────────────────────────────────────────────────────

def _make_triangle_mask(
    size: Tuple[int, int],
    tri: Triangle,
) -> 'Image':
    """Tạo PIL Image mask RGBA: vùng trong tam giác = opaque, ngoài = transparent."""
    from PIL import Image, ImageDraw

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon([(int(x), int(y)) for x, y in tri], fill=255)
    return mask


def warp_sprite_to_quad(
    body: 'Image',
    sprite: 'Image',
    quad: List[Point],
    *,
    canvas_size: Tuple[int, int] = (1920, 1080),
    supersample: int = 1,
) -> 'Image':
    """Two-triangle affine warp: warp mouth sprite → quad, composite lên body.

    Port mouthsWarp.ts:drawWarpedSprite().

    Sprite được warp lên 1 overlay trong suốt riêng, sau đó alpha_composite
    lên body. Paste-with-mask KHÔNG phải alpha compositing — nó thay thế
    pixel canvas bằng pixel source (kể cả transparent) trong vùng mask.
    Dùng alpha_composite để giữ nguyên body bên dưới vùng sprite trong suốt.

    Args:
        body: Body frame RGBA Image (canvas_size).
        sprite: Mouth sprite RGBA Image (thường 128x85).
        quad: 4 điểm (TL, TR, BR, BL) của quad trên canvas.
        canvas_size: Kích thước canvas (width, height).
        supersample: Mẫu > 1 cho anti-aliasing (mặc định 1 = tắt).

    Returns:
        Canvas RGBA Image với body + warped mouth (alpha composited).
    """
    from PIL import Image

    sw, sh = sprite.size
    if sw <= 0 or sh <= 0:
        return body.copy()

    # Warp sprite lên overlay trong suốt riêng (KHÔNG paste trực tiếp lên body)
    mouth_overlay = Image.new("RGBA", canvas_size, (0, 0, 0, 0))

    # Sprite corners (TL, TR, BR, BL)
    s0 = (0.0, 0.0)
    s1 = (float(sw), 0.0)
    s2 = (float(sw), float(sh))
    s3 = (0.0, float(sh))

    q0, q1, q2, q3 = [tuple(map(float, p)) for p in quad]

    # Triangle 1: (s0,s1,s2) → (q0,q1,q2)
    mouth_overlay = _apply_triangle_warp(mouth_overlay, sprite, s0, s1, s2, q0, q1, q2, supersample)

    # Triangle 2: (s0,s2,s3) → (q0,q2,q3)
    mouth_overlay = _apply_triangle_warp(mouth_overlay, sprite, s0, s2, s3, q0, q2, q3, supersample)

    # Alpha-composite overlay lên body (giữ body bên dưới vùng sprite trong suốt)
    canvas = body.copy()
    canvas = Image.alpha_composite(canvas, mouth_overlay)
    return canvas


def _apply_triangle_warp(
    canvas: 'Image',
    sprite: 'Image',
    s0: Point, s1: Point, s2: Point,
    d0: Point, d1: Point, d2: Point,
    supersample: int,
) -> 'Image':
    """Warp 1 tam giác của sprite vào 1 tam giác trên canvas."""
    from PIL import Image

    # Forward affine: sprite → destination
    fwd = compute_affine(s0, s1, s2, d0, d1, d2)
    if fwd is None:
        return canvas

    # Inverse affine: destination → sprite (cho PIL)
    inv = _invert_affine(*fwd)
    if inv is None:
        return canvas

    # Bounding box của destination triangle
    xs = [d0[0], d1[0], d2[0]]
    ys = [d0[1], d1[1], d2[1]]
    min_x = max(0, int(min(xs)))
    min_y = max(0, int(min(ys)))
    max_x = min(canvas.width - 1, int(math.ceil(max(xs))))
    max_y = min(canvas.height - 1, int(math.ceil(max(ys))))

    box_w = max_x - min_x + 1
    box_h = max_y - min_y + 1
    if box_w <= 0 or box_h <= 0:
        return canvas

    # Tạo mask cho vùng destination triangle (để chỉ warp phần trong tam giác)
    # Dùng supersample nếu cần
    ss = supersample
    mask_size = (box_w * ss, box_h * ss) if ss > 1 else (box_w, box_h)
    warped_mask = _make_triangle_mask(mask_size, (
        ((d0[0] - min_x) * ss, (d0[1] - min_y) * ss),
        ((d1[0] - min_x) * ss, (d1[1] - min_y) * ss),
        ((d2[0] - min_x) * ss, (d2[1] - min_y) * ss),
    ) if ss > 1 else (
        (d0[0] - min_x, d0[1] - min_y),
        (d1[0] - min_x, d1[1] - min_y),
        (d2[0] - min_x, d2[1] - min_y),
    ))

    # PIL AFFINE: src_x = pil_a * dst_x + pil_b * dst_y + pil_c
    #            src_y = pil_d * dst_x + pil_e * dst_y + pil_f
    # Điều chỉnh offset vì chúng ta làm việc trên bounding box
    pil_a, pil_b, pil_c, pil_d, pil_e, pil_f = inv
    # Adjust for box offset: the transform expects coordinates in
    # destination (canvas) space. Our warped image starts at (min_x, min_y),
    # so we adjust c and f.
    c_adjusted = pil_a * min_x + pil_b * min_y + pil_c
    f_adjusted = pil_d * min_x + pil_e * min_y + pil_f

    if ss > 1:
        # Supersample: scale mask + transform accordingly
        c_ss = pil_a * min_x * ss + pil_b * min_y * ss + pil_c * ss
        f_ss = pil_d * min_x * ss + pil_e * min_y * ss + pil_f * ss
        # PIL transform: map output → input, but we need to account for
        # the supersampled mask. Each mask pixel at (mx, my) corresponds
        # to canvas point (min_x + mx/ss, min_y + my/ss).
        # So src_x = pil_a*(min_x + mx/ss) + pil_b*(min_y + my/ss) + pil_c
        #           = (pil_a*min_x + pil_b*min_y + pil_c) + (pil_a*mx + pil_b*my)/ss
        # Similarly for src_y.
        pil_ss_data = (
            pil_a / ss, pil_b / ss, c_ss,
            pil_d / ss, pil_e / ss, f_ss,
        )
    else:
        pil_data = (pil_a, pil_b, c_adjusted, pil_d, pil_e, f_adjusted)

    # Sprite transform (crop to bounding box)
    if ss > 1:
        warped_sprite = sprite.transform(
            (box_w * ss, box_h * ss),
            Image.AFFINE,
            pil_ss_data,
            Image.BILINEAR,
        )
    else:
        warped_sprite = sprite.transform(
            (box_w, box_h),
            Image.AFFINE,
            pil_data,
            Image.BILINEAR,
        )

    # Downsample if supersampled
    if ss > 1:
        warped_sprite = warped_sprite.resize((box_w, box_h), Image.LANCZOS)
        warped_mask = warped_mask.resize((box_w, box_h), Image.LANCZOS)

    # Paste warped mouth sprite onto canvas using mask
    canvas.paste(warped_sprite, (min_x, min_y), warped_mask)

    return canvas


# ── Pre-render orchestrration ─────────────────────────────────────────────

def apply_mouth_calibration(
    quad: List[Point],
    track: Optional[Dict[str, Any]] = None,
) -> List[Point]:
    """Port mouthsWarp.ts:applyMouthCalibration().

    Nếu calibrationApplied=False → return quad không đổi.
    """
    if track is None:
        return quad
    calibration = track.get("calibration") or {"offset": [0, 0], "scale": 1, "rotation": 0}
    if not track.get("calibrationApplied", False):
        return quad

    offset = calibration.get("offset", [0, 0])
    scale = calibration.get("scale", 1)
    rotation = math.radians(calibration.get("rotation", 0))

    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    cx = sum(xs) / len(quad)
    cy = sum(ys) / len(quad)
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)

    result = []
    for x, y in quad:
        dx = (x - cx) * scale
        dy = (y - cy) * scale
        rx = dx * cos_r - dy * sin_r + cx + offset[0]
        ry = dx * sin_r + dy * cos_r + cy + offset[1]
        result.append((rx, ry))

    return result


def compute_track_frame_index(
    global_frame: int,
    composition_fps: float,
    track_fps: float,
    track_frames: int,
    loop_frames: Optional[int] = None,
) -> int:
    """Port mouthsMotionPngTuberCharacter.tsx: trackFrameIndex.

    global_frame = frame tuyệt đối trên timeline (sau stretch).
    composition_fps = fps của output video.
    track_fps = fps của mouth_track (thường 30).
    track_frames = tổng số frame trong mouth_track.
    loop_frames = số composition frame cho 1 chu kỳ loop (tự tính nếu None).
    """
    if loop_frames is None:
        loop_frames = round((track_frames / track_fps) * composition_fps)
    if loop_frames <= 0:
        loop_frames = track_frames

    loop_frame = global_frame % loop_frames
    track_idx = math.floor((loop_frame / composition_fps) * track_fps) % track_frames
    return track_idx


def compute_character_box(
    character: Dict[str, Any],
    width: int,
    height: int,
    track_aspect: float,
) -> Dict[str, int]:
    """Tính character box từ config — mirror mouthsTuberOverlay.tsx:resolveCharacterBox().

    Returns:
        {compWidth, compHeight, compOffsetX, compOffsetY}
        Trong đó compWidth/compHeight là kích thước composition (character box).
        Nếu cả width/height đều 0 → trả về box hợp lệ (default 60%).
    """
    w_src: Optional[int] = character.get("width")
    if w_src is None and "widthRatio" in character:
        w_src = round(character["widthRatio"] * width)

    h_src: Optional[int] = character.get("height")
    if h_src is None and "heightRatio" in character:
        h_src = round(character["heightRatio"] * height)

    if w_src is not None:
        box_w = w_src
        box_h = round(w_src / track_aspect)
    elif h_src is not None:
        box_h = h_src
        box_w = round(h_src * track_aspect)
    else:
        box_h = round(0.6 * height)
        box_w = round(box_h * track_aspect)

    box_l = character.get("left")
    if box_l is None:
        box_l = round(character.get("leftRatio", 0.6) * width)

    box_t = character.get("top")
    if box_t is None:
        box_t = round(character.get("topRatio", 0.3) * height)

    return {
        "compWidth": box_w,
        "compHeight": box_h,
        "compOffsetX": int(box_l),
        "compOffsetY": int(box_t),
    }


# ── Chromakey extract ─────────────────────────────────────────────────────

def _auto_detect_chroma_color(video_path: Path) -> str:
    """Lấy màu nền từ 4 góc frame đầu tiên (median RGB) → '0xRRGGBB'.

    Port từ prepare-assets.ts auto-detect logic.
    Dùng ffmpeg crop 80x80 từ 4 góc → lấy pixel trung bình → median 4 góc.
    Fallback '0x00FF00' nếu không probe được.
    """
    import subprocess
    import struct

    corners = []
    # 4 góc: top-left, top-right, bottom-left, bottom-right
    crops = ["crop=80:80:0:0", "crop=80:80:iw-80:0", "crop=80:80:0:ih-80", "crop=80:80:iw-80:ih-80"]
    for crop in crops:
        try:
            cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vf", f"{crop},scale=1:1:flags=area",
                "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=15)
            if result.returncode == 0 and len(result.stdout) >= 3:
                r, g, b = result.stdout[0], result.stdout[1], result.stdout[2]
                corners.append((r, g, b))
        except Exception:
            pass

    if not corners:
        logger.warning("Auto-detect chroma color thất bại → fallback 0x00FF00")
        return "0x00FF00"

    # Median từng channel
    r = sorted(c[0] for c in corners)[len(corners) // 2]
    g = sorted(c[1] for c in corners)[len(corners) // 2]
    b = sorted(c[2] for c in corners)[len(corners) // 2]
    color = f"0x{r:02X}{g:02X}{b:02X}"
    logger.info("Auto-detect chroma color: %s", color)
    return color


def extract_body_transparent(
    body_source: Path,
    out_dir: Path,
    *,
    chroma_color: Optional[str] = None,
    similarity: float = 0.10,
    blend: float = 0.10,
) -> Path:
    """Extract body frames từ video source với chromakey → RGBA PNG sequence.

    Port từ remotion_tuber/scripts/prepare-assets.ts.

    Args:
        body_source: Path tới video body loop (nền màu đặc, vd green).
        out_dir: Output dir cho frame-NNN.png (tạo tự động nếu chưa có).
        chroma_color: Màu nền dạng '0xRRGGBB'. None → auto-detect từ 4 góc.
        similarity: FFmpeg chromakey similarity (0–1). Default 0.10.
        blend: FFmpeg chromakey blend (0–1). Default 0.10.

    Returns:
        out_dir path.

    Raises:
        FileNotFoundError: body_source không tồn tại.
        RuntimeError: FFmpeg thất bại.
    """
    import subprocess

    if not body_source.exists():
        raise FileNotFoundError(f"bodySource không tồn tại: {body_source}")

    out_dir.mkdir(parents=True, exist_ok=True)

    if chroma_color is None:
        chroma_color = _auto_detect_chroma_color(body_source)

    # Normalize: '0xRRGGBB' → '0xRRGGBB' (FFmpeg dùng 0x prefix)
    color_ffmpeg = chroma_color.replace("#", "0x")

    vf = f"chromakey={color_ffmpeg}:{similarity}:{blend},format=rgba"
    out_pattern = str(out_dir / "frame-%03d.png")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(body_source),
        "-vf", vf,
        "-vsync", "0",
        "-start_number", "0",
        out_pattern,
    ]

    logger.info(
        "extract_body_transparent: chromakey %s sim=%.2f blend=%.2f → %s",
        color_ffmpeg, similarity, blend, out_dir,
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg chromakey thất bại (code {result.returncode}):\n{result.stderr[-1000:]}"
        )

    frames = list(out_dir.glob("frame-*.png"))
    logger.info("extract_body_transparent: %d frames xuất ra %s", len(frames), out_dir)
    if not frames:
        raise RuntimeError(f"FFmpeg chạy xong nhưng không có frame nào trong {out_dir}")

    return out_dir


# ── Pre-render all body×mouth ─────────────────────────────────────────────

def prerender_character(
    body_dir: Path,
    mouth_dir: Path,
    mouth_track_path: Path,
    mouth_states: List[str],
    out_dir: Path,
    *,
    sprite_dir: Optional[Path] = None,  # fallback mouth sprite dir
    character_box: Optional[Dict[str, int]] = None,
    supersample: int = 1,
    max_workers: int = 1,  # NEW: parallel workers cho loop body×state
) -> Dict[str, Any]:
    """Pre-render toàn bộ body×mouth combinations → PNG files + manifest.

    Args:
        body_dir: Thư mục body-transparent (frame-*.png).
        mouth_dir: Thư mục mouth sprites.
        mouth_track_path: Path đến mouth_track.json.
        mouth_states: List các mouth state IDs (vd ['closed', 'half', 'open']).
        out_dir: Thư mục output (tự tạo nếu chưa có).
        character_box: Nếu có → crop body/output về kích thước này.
        supersample: Anti-aliasing (1=tắt, 2=2x).

    Returns:
        prerender_manifest dict.
    """
    from PIL import Image

    # Validate inputs
    if not body_dir.is_dir():
        raise FileNotFoundError(f"body_dir không tồn tại: {body_dir}")
    if not mouth_dir.is_dir():
        raise FileNotFoundError(f"mouth_dir không tồn tại: {mouth_dir}")
    if not mouth_track_path.exists():
        raise FileNotFoundError(f"mouth_track không tồn tại: {mouth_track_path}")

    # Load mouth track
    track = json.loads(mouth_track_path.read_text(encoding="utf-8"))
    track_frames = len(track["frames"])
    track_fps = float(track["fps"])
    track_width = int(track.get("width", 1920))
    track_height = int(track.get("height", 1080))

    # Load mouth sprites
    mouth_sprite_paths: Dict[str, Path] = {}
    for state in mouth_states:
        sp = mouth_dir / f"{state}.png"
        if not sp.exists():
            # Try sprite_dir fallback
            if sprite_dir:
                sp = sprite_dir / f"{state}.png"
            if not sp.exists():
                raise FileNotFoundError(f"Mouth sprite không tồn tại: {state} ({sp})")
        mouth_sprite_paths[state] = sp

    # Load body frame files
    body_files = sorted(body_dir.glob("frame-*.png"))
    if not body_files:
        raise FileNotFoundError(f"Không tìm thấy body frame trong {body_dir}")
    num_body_frames = len(body_files)

    # Prepare output dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute character box (fallback: full canvas)
    track_aspect = track_width / track_height if track_height > 0 else 16 / 9
    final_box = character_box or {"compWidth": track_width, "compHeight": track_height,
                                   "compOffsetX": 0, "compOffsetY": 0}
    out_w = final_box["compWidth"]
    out_h = final_box["compHeight"]

    logger.info("Pre-render %d body × %d mouth states = %d frames → %s",
                num_body_frames, len(mouth_states), num_body_frames * len(mouth_states),
                out_dir)

    # Pre-render loop — eager load body + sprite caches, then parallel or sequential
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rendered = 0
    body_cache: Dict[int, Image] = {
        i: Image.open(body_files[i]).convert("RGBA") for i in range(num_body_frames)
    }
    sprites_cache: Dict[str, Image] = {
        s: Image.open(mouth_sprite_paths[s]).convert("RGBA") for s in mouth_states
    }

    total = num_body_frames * len(mouth_states)

    def _render_one(body_idx: int, state: str) -> None:
        """Worker: warp mouth lên body, scale, save PNG."""
        body_img = body_cache[body_idx]
        sprite = sprites_cache[state]
        quad = track["frames"][body_idx]["quad"]
        calibrated = apply_mouth_calibration(quad, track)
        canvas = warp_sprite_to_quad(
            body_img, sprite, calibrated,
            canvas_size=(track_width, track_height),
            supersample=supersample,
        )
        if character_box:
            canvas = canvas.resize(
                (character_box["compWidth"], character_box["compHeight"]), Image.LANCZOS
            )
        out_name = f"frame-{body_idx:03d}_{state}.png"
        canvas.save(str(out_dir / out_name), "PNG")

    tasks = [(bi, s) for bi in range(num_body_frames) for s in mouth_states]
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_render_one, bi, s): (bi, s) for bi, s in tasks}
            for _f in as_completed(futures):
                rendered += 1
                if rendered % 100 == 0:
                    logger.info("  pre-rendered %d / %d ...", rendered, total)
    else:
        for bi, s in tasks:
            _render_one(bi, s)
            rendered += 1
            if rendered % 100 == 0:
                logger.info("  pre-rendered %d / %d ...", rendered, total)

    # Clean caches
    body_cache.clear()
    sprites_cache.clear()

    # Build manifest
    manifest = {
        "schemaVersion": 1,
        "renderedAt": __import__("datetime").datetime.now().isoformat(),
        "assetId": mouth_track_path.parent.name,
        "trackFrames": track_frames,
        "trackFps": track_fps,
        "trackWidth": track_width,
        "trackHeight": track_height,
        "mouthStates": list(mouth_states),
        "bodyFrameCount": num_body_frames,
        "outputCount": rendered,
        "framePattern": "frame-{trackIdx:03d}_{state}.png",
        "outputWidth": out_w,
        "outputHeight": out_h,
        "characterBox": final_box,
    }

    manifest_path = out_dir / "prerender_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Pre-render hoàn tất: %d frames → %s", rendered, out_dir)
    return manifest


def get_prerender_frame(
    track_frame_index: int,
    mouth_state: str,
    prerender_dir: Path,
    manifest: Dict[str, Any],
) -> Path:
    """Trả về path tới pre-rendered frame.

    Args:
        track_frame_index: Body frame index (0..trackFrames-1).
        mouth_state: State string.
        prerender_dir: Thư mục prerendered.
        manifest: prerender_manifest.json dict.

    Returns:
        Path tới PNG file.
    """
    fn = f"frame-{track_frame_index:03d}_{mouth_state}.png"
    return prerender_dir / fn


def build_prerender_frame_sequence(
    group_start_frame: int,
    group_end_frame: int,
    fps: float,
    track_fps: float,
    track_frames: int,
    prerender_dir: Path,
    prerender_manifest: Dict[str, Any],
    mouth_events_map: Dict[str, Any],
) -> List[Path]:
    """Tạo list pre-rendered frame paths cho toàn bộ group.

    Với mỗi global_frame từ group_start → group_end:
      - track_idx = compute_track_frame_index(...)
      - mouth_state = lookup từ mouthEvents tại global_frame
      - frame path = prerender_dir / frame-{track_idx:03d}_{state}.png

    Returns:
        List[Path] các pre-rendered frame theo thứ tự timeline.
    """
    frame_list: List[Path] = []
    for gf in range(group_start_frame, group_end_frame):
        track_idx = compute_track_frame_index(gf, fps, track_fps, track_frames)
        mouth_state = _lookup_mouth_state(gf, mouth_events_map)
        fp = get_prerender_frame(track_idx, mouth_state, prerender_dir, prerender_manifest)
        if not fp.exists():
            # Fallback: dùng closed (nếu thiếu)
            fp = get_prerender_frame(track_idx, "closed", prerender_dir, prerender_manifest)
        frame_list.append(fp)
    return frame_list


def _lookup_mouth_state(
    global_frame: int,
    mouth_events_map: Dict[str, Any],
) -> str:
    """Binary search mouthEvents gần nhất ≤ global_frame.

    mouthEvents dict: {segment_idx: [{"frame": int, "state": str}, ...]}
    """
    # Hiện tại đơn giản: segment đầu tiên chứa frame này
    # Có thể optimize sau
    for seg_idx, events in mouth_events_map.items():
        if not events:
            continue
        # Kiểm tra frame có nằm trong segment này không
        first_ev = events[0]
        last_ev = events[-1]
        # Không biết segment bounds ở đây → binary search event
        pass

    # Binary search trên events
    best_state = "closed"
    for seg_idx, events in mouth_events_map.items():
        if not events:
            continue
        low, high = 0, len(events) - 1
        while low <= high:
            mid = (low + high) // 2
            ev = events[mid]
            if ev["frame"] <= global_frame:
                best_state = ev["state"]
                low = mid + 1
            else:
                high = mid - 1

    return best_state
