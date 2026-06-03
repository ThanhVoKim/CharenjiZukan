#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/sync_engine/test_tuber_remotion_validation.py
===================================================
Validation cho subproject MotionPNGTuber Remotion (`remotion_tuber/`) — phần
overlay alpha của flow tuber trong sync_video.

Gom các bước verify đã chạy tay khi dựng subproject:
  - cấu trúc project + schema manifest/asset đúng (Layer 1)
  - asset-prep: chromakey green-screen → body-transparent PNG sequence, giữ
    kích thước + số frame khớp mouth_track (Layer 2, chỉ cần ffmpeg)
  - Remotion render thật: overlay PNG alpha cho 1 group synthetic, kiểm tra
    fps/độ phân giải ĐỘNG, số frame, alpha trong suốt, composite không lộ nền
    (Layer 4, cần Node + opt-in)

Cấu trúc layers:
  Layer 1 — Unit/Structure      (file + JSON, không subprocess)
  Layer 2 — Component (ffmpeg)   (chromakey body → transparent frames)
  Layer 4 — Real Remotion        (Node + Chromium; opt-in REMOTION_TUBER_E2E=1)

Cách chạy:
    pytest tests/sync_engine/test_tuber_remotion_validation.py -v -k "Layer1"
    pytest tests/sync_engine/test_tuber_remotion_validation.py -v -k "Layer2"
    # Layer 4 cần: (cd remotion_tuber && npm install) rồi:
    REMOTION_TUBER_E2E=1 pytest tests/sync_engine/test_tuber_remotion_validation.py -v -k "Layer4"
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# ── Project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

REMOTION_DIR = PROJECT_ROOT / "remotion_tuber"
ASSET_DIR = PROJECT_ROOT / "assets" / "pngtuber" / "nike_loop_fix"
SYNTH_MANIFEST = REMOTION_DIR / "test-manifests" / "group_synthetic.json"
BODY_SOURCE = ASSET_DIR / "loop_mouthless_h264.mp4"

REMOTION_PKGS = ("remotion", "@remotion/bundler", "@remotion/renderer")


# ═════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ffprobe_stream(path: Path) -> dict:
    """Trả về {width, height, pix_fmt} của 1 ảnh/video qua ffprobe."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,pix_fmt",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    st = json.loads(out)["streams"][0]
    return {"width": st["width"], "height": st["height"], "pix_fmt": st["pix_fmt"]}


def _avg_alpha(path: Path, crop: str) -> int:
    """Alpha trung bình (0..255) của vùng crop ('w:h:x:y') qua ffmpeg alphaextract."""
    raw = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-vf", f"alphaextract,crop={crop},scale=1:1",
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ],
        capture_output=True, check=True,
    ).stdout
    return raw[0] if raw else -1


def _avg_rgb(path: Path, crop: str) -> tuple:
    """RGB trung bình của vùng crop qua ffmpeg (đọc 3 byte)."""
    raw = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-vf", f"crop={crop},scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        capture_output=True, check=True,
    ).stdout
    return (raw[0], raw[1], raw[2]) if len(raw) >= 3 else (-1, -1, -1)


def _detect_bg_color(src: Path) -> str:
    """Median RGB của 4 góc frame đầu → '0xRRGGBB' (mirror prepare-assets.ts detectBackgroundColor)."""
    corners = [
        _avg_rgb(src, c)
        for c in ("80:80:0:0", "80:80:iw-80:0", "80:80:0:ih-80", "80:80:iw-80:ih-80")
    ]
    chan = lambda ch: sorted(s[ch] for s in corners)[1:3]  # 2 giá trị giữa
    r, g, b = (round(sum(chan(ch)) / 2) for ch in range(3))
    return "0x" + "".join(f"{n:02X}" for n in (r, g, b))


def _npm_bin() -> str | None:
    return shutil.which("npm")


def _run_npm(script: str, *args: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    """Chạy `npm run <script> -- <args>` trong remotion_tuber (cross-platform)."""
    npm = _npm_bin()
    assert npm, "npm không có trong PATH"
    parts = [npm, "run", script]
    if args:
        parts += ["--", *args]
    # encoding/errors: tránh UnicodeDecodeError khi npm/ffmpeg in ký tự ngoài cp1252 (Windows).
    if os.name == "nt":
        # .cmd cần shell trên Windows
        cmd = subprocess.list2cmdline(parts)
        return subprocess.run(
            cmd, cwd=str(REMOTION_DIR), shell=True,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )
    return subprocess.run(
        parts, cwd=str(REMOTION_DIR),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout,
    )


def _character_box(manifest: dict, track: dict) -> tuple:
    """Suy ô character ra px, mirror resolveCharacterBox (width ưu tiên) trong TuberOverlay.tsx."""
    W, H = manifest["width"], manifest["height"]
    c = manifest.get("character", {})
    aspect = track["width"] / track["height"]

    if "width" in c:
        w_src = c["width"]
    elif "widthRatio" in c:
        w_src = round(c["widthRatio"] * W)
    else:
        w_src = None
    if "height" in c:
        h_src = c["height"]
    elif "heightRatio" in c:
        h_src = round(c["heightRatio"] * H)
    else:
        h_src = None

    if w_src is not None:          # width ưu tiên → suy height giữ tỉ lệ
        box_w = w_src
        box_h = round(w_src / aspect)
    elif h_src is not None:        # chỉ có height → suy width
        box_h = h_src
        box_w = round(h_src * aspect)
    else:                         # default theo heightRatio 0.6
        box_h = round(0.6 * H)
        box_w = round(box_h * aspect)

    box_l = c.get("left", round(c.get("leftRatio", 0.6) * W))
    box_t = c.get("top", round(c.get("topRatio", 0.3) * H))
    return box_l, box_t, box_w, box_h


def _parse_render_result(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("__TUBER_RENDER_RESULT__="):
            return json.loads(line.split("=", 1)[1])
    raise AssertionError("Không thấy dòng __TUBER_RENDER_RESULT__ trong stdout")


# Skip helpers (collection-time)
_FFMPEG_OK = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
_E2E_OPT_IN = os.getenv("REMOTION_TUBER_E2E", "") == "1"
_NODE_OK = bool(_npm_bin() and shutil.which("node"))
_NODE_MODULES = (REMOTION_DIR / "node_modules").is_dir()


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — STRUCTURE & SCHEMA (không subprocess)
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_TuberProjectAndSchema:
    """Cấu trúc subproject + schema manifest/asset đúng hợp đồng với code TS."""

    def test_project_files_exist(self):
        for rel in [
            "package.json", "tsconfig.json", "remotion.config.ts",
            "src/index.ts", "src/Root.tsx", "src/manifest.ts",
            "src/mouthWarp.ts", "src/mouthState.ts",
            "src/MotionPngTuberMouthCanvas.tsx",
            "src/MotionPngTuberCharacter.tsx", "src/TuberOverlay.tsx",
            "scripts/prepare-assets.ts", "scripts/render-groups.ts",
            "test-manifests/group_synthetic.json",
        ]:
            assert (REMOTION_DIR / rel).is_file(), f"Thiếu {rel}"

    def test_remotion_deps_versions_in_sync(self):
        """Remotion yêu cầu remotion + @remotion/* cùng version spec."""
        pkg = _load_json(REMOTION_DIR / "package.json")
        deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
        specs = {p: deps.get(p) for p in REMOTION_PKGS}
        assert all(specs.values()), f"Thiếu remotion dep: {specs}"
        assert len(set(specs.values())) == 1, f"Version remotion lệch nhau: {specs}"

    def test_asset_present(self):
        assert BODY_SOURCE.is_file(), f"Thiếu body source: {BODY_SOURCE}"
        for s in ("closed", "half", "open"):
            assert (ASSET_DIR / "mouth" / f"{s}.png").is_file(), f"Thiếu sprite {s}"

    def test_mouth_track_schema(self):
        track = _load_json(ASSET_DIR / "mouth_track.json")
        for k in ("fps", "width", "height", "frames"):
            assert k in track, f"mouth_track thiếu key {k}"
        assert isinstance(track["frames"], list) and len(track["frames"]) > 0
        f0 = track["frames"][0]
        assert "quad" in f0 and "valid" in f0
        assert len(f0["quad"]) == 4 and all(len(p) == 2 for p in f0["quad"])

    def test_synthetic_manifest_schema(self):
        m = _load_json(SYNTH_MANIFEST)
        for k in ("groupId", "fps", "width", "height",
                  "groupStartFrame", "groupEndFrame", "segments", "character"):
            assert k in m, f"manifest thiếu key {k}"

        # renderDurationFrames khớp độ dài group (M2: padding 0 ở V1)
        expected = m["groupEndFrame"] - m["groupStartFrame"]
        assert m.get("renderDurationFrames", expected) == expected

        # segments hợp lệ + nằm trong [groupStart, groupEnd], hasTts là bool
        valid_types = {"tts", "mute", "gap", "tail"}
        for seg in m["segments"]:
            assert seg["blockType"] in valid_types
            assert isinstance(seg["hasTts"], bool)
            assert m["groupStartFrame"] <= seg["startFrame"] < seg["endFrame"] <= m["groupEndFrame"]


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — ASSET PREP (ffmpeg chromakey, không cần Node)
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _FFMPEG_OK, reason="Cần ffmpeg + ffprobe trong PATH")
class TestLayer2_BodyChromakeyExtraction:
    """B1: body nền MÀU ĐẶC → transparent PNG sequence (giữ kích thước + số frame + alpha thật)."""

    def test_chromakey_extracts_rgba_frames_matching_track(self, tmp_path: Path):
        track = _load_json(ASSET_DIR / "mouth_track.json")
        n_track = len(track["frames"])
        # Auto-dò màu nền như prepare-assets (asset này nền green ~0x08A702, KHÔNG phải 0x00FF00).
        key = _detect_bg_color(BODY_SOURCE)

        out_dir = tmp_path / "body-transparent"
        out_dir.mkdir()
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(BODY_SOURCE),
                "-vf", f"chromakey={key}:0.10:0.10,format=rgba",
                "-vsync", "0", "-start_number", "0",
                str(out_dir / "frame-%03d.png"),
            ],
            capture_output=True, check=True,
        )

        frames = sorted(out_dir.glob("frame-*.png"))
        # body loop dùng modulo theo mouth_track.frames.length → cần khớp đúng số frame
        assert len(frames) == n_track, (
            f"body frames {len(frames)} != mouth_track frames {n_track}"
        )

        st = _ffprobe_stream(frames[0])
        assert st["pix_fmt"] == "rgba", f"frame body không có alpha: {st['pix_fmt']}"
        assert (st["width"], st["height"]) == (track["width"], track["height"]), (
            "asset-prep phải giữ nguyên kích thước frame để khớp toạ độ mouth_track"
        )

        # B1 cốt lõi: nền màu đặc bị key → góc TRONG SUỐT, tâm nhân vật còn ĐẶC.
        mid = frames[len(frames) // 2]
        w, h = st["width"], st["height"]
        corner = _avg_alpha(mid, "200:200:0:0")
        center = _avg_alpha(mid, f"120:120:{w // 2 - 60}:{h // 2 - 60}")
        assert corner <= 8, f"Góc nền chưa trong suốt sau key (alpha={corner}); key={key}"
        assert center >= 200, f"Tâm nhân vật bị key nhầm (alpha={center}); key={key}"


# ═════════════════════════════════════════════════════════════════════
# LAYER 4 — REAL REMOTION RENDER (Node + Chromium; opt-in)
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _E2E_OPT_IN, reason="Đặt REMOTION_TUBER_E2E=1 để chạy render Remotion thật")
@pytest.mark.skipif(not _NODE_OK, reason="Cần Node + npm trong PATH")
@pytest.mark.skipif(not _NODE_MODULES, reason="Chạy `cd remotion_tuber && npm install` trước")
@pytest.mark.skipif(not _FFMPEG_OK, reason="Cần ffmpeg + ffprobe trong PATH")
class TestLayer4_RemotionOverlayRender:
    """Render overlay alpha thật cho 1 group; kiểm tra res/fps động + alpha + composite."""

    @pytest.fixture(scope="class")
    def prepared_assets(self):
        res = _run_npm("prepare-assets", timeout=600)
        assert res.returncode == 0, f"prepare-assets fail:\n{res.stdout}\n{res.stderr}"
        return REMOTION_DIR / "public" / "pngtuber" / "nike_loop_fix"

    @pytest.fixture(scope="class")
    def rendered(self, prepared_assets, tmp_path_factory):
        """Render 1 group synthetic ở ĐỘ PHÂN GIẢI/ FPS bất kỳ (chứng minh B3/B4)."""
        out_base = tmp_path_factory.mktemp("tuber_out")
        manifest = {
            "schemaVersion": 1, "groupId": "group_test",
            "fps": 25, "width": 960, "height": 540,          # khác 1920x1080@30 → test general
            "groupStartFrame": 0, "groupEndFrame": 50,
            "renderStartFrame": 0, "renderDurationFrames": 50,
            "assetId": "nike_loop_fix",
            "character": {"leftRatio": 0.45, "topRatio": 0.35, "heightRatio": 0.5, "clipInset": "0px"},
            "mouth": {"mode": "cue"},
            "segments": [
                {"segmentIndex": 0, "startFrame": 0, "endFrame": 35, "blockType": "tts", "hasTts": True},
                {"segmentIndex": 1, "startFrame": 35, "endFrame": 50, "blockType": "mute", "hasTts": False},
            ],
        }
        mpath = out_base / "group_test.json"
        mpath.write_text(json.dumps(manifest), encoding="utf-8")

        res = _run_npm("render-groups", str(mpath), "--out-dir", str(out_base), timeout=1800)
        assert res.returncode == 0, f"render-groups fail:\n{res.stdout}\n{res.stderr}"
        result = _parse_render_result(res.stdout)
        assert result["ok"] is True, f"render result not ok: {result}"

        out_dir = out_base / "group_test" / "overlay_frames"
        return {"manifest": manifest, "result": result, "out_dir": out_dir}

    @pytest.fixture(scope="class")
    def rendered_aspect(self, prepared_assets, tmp_path_factory):
        """Render group có character width & height tường minh LỆCH aspect.
        width=400, height=500 trên comp 960x540, aspect mouth_track 16:9 →
        width ưu tiên → ô thực = 400x225 (height=500 BỊ BỎ QUA)."""
        out_base = tmp_path_factory.mktemp("tuber_aspect")
        manifest = {
            "schemaVersion": 1, "groupId": "group_aspect",
            "fps": 25, "width": 960, "height": 540,
            "groupStartFrame": 0, "groupEndFrame": 30,
            "renderStartFrame": 0, "renderDurationFrames": 30,
            "assetId": "nike_loop_fix",
            # left/top tường minh để ô nằm gọn trong khung; height=500 cố tình lệch.
            "character": {"left": 200, "top": 100, "width": 400, "height": 500},
            "mouth": {"mode": "cue"},
            "segments": [
                {"segmentIndex": 0, "startFrame": 0, "endFrame": 30, "blockType": "tts", "hasTts": True},
            ],
        }
        mpath = out_base / "group_aspect.json"
        mpath.write_text(json.dumps(manifest), encoding="utf-8")

        res = _run_npm("render-groups", str(mpath), "--out-dir", str(out_base), timeout=1800)
        assert res.returncode == 0, f"render-groups fail:\n{res.stdout}\n{res.stderr}"
        result = _parse_render_result(res.stdout)
        assert result["ok"] is True, f"render result not ok: {result}"

        out_dir = out_base / "group_aspect" / "overlay_frames"
        return {"manifest": manifest, "result": result, "out_dir": out_dir}

    def test_frame_count_matches_duration(self, rendered):
        frames = sorted(rendered["out_dir"].glob("*.png"))
        assert len(frames) == rendered["manifest"]["renderDurationFrames"] == 50
        assert rendered["result"]["frames"] == 50

    def test_dynamic_resolution_and_alpha(self, rendered):
        """B3/B4: composition đúng 960x540, PNG rgba (alpha)."""
        frame = sorted(rendered["out_dir"].glob("*.png"))[10]
        st = _ffprobe_stream(frame)
        assert (st["width"], st["height"]) == (960, 540)
        assert st["pix_fmt"] == "rgba"

    def test_background_transparent_character_opaque(self, rendered):
        """Góc khung trong suốt (alpha≈0); TÂM ô character có nội dung (alpha cao)."""
        frame = sorted(rendered["out_dir"].glob("*.png"))[10]
        track = _load_json(ASSET_DIR / "mouth_track.json")
        bl, bt, bw, bh = _character_box(rendered["manifest"], track)
        cx, cy = bl + bw // 2, bt + bh // 2  # tâm ô → map vào tâm nhân vật (opaque)

        corner = _avg_alpha(frame, "40:40:0:0")
        char = _avg_alpha(frame, f"60:60:{cx - 30}:{cy - 30}")
        assert corner <= 8, f"Góc khung phải trong suốt, alpha={corner}"
        assert char >= 200, f"Tâm ô character phải có nội dung, alpha={char}"

    def test_body_animates_between_frames(self, rendered):
        """Body không đứng hình: 2 frame cách nhau khác nhau."""
        frames = sorted(rendered["out_dir"].glob("*.png"))
        a = frames[0].read_bytes()
        b = frames[20].read_bytes()
        assert a != b, "Body có vẻ đứng hình (2 frame giống hệt)"

    def test_composite_over_magenta_no_bleed(self, rendered, tmp_path: Path):
        """Composite overlay lên nền magenta: góc lộ magenta (không nền đen/xanh)."""
        frame = sorted(rendered["out_dir"].glob("*.png"))[10]
        comp = tmp_path / "composite.png"
        subprocess.run(
            [
                "ffmpeg", "-v", "error",
                "-f", "lavfi", "-i", "color=c=magenta:s=960x540",
                "-i", str(frame),
                "-filter_complex", "[0][1]overlay=format=auto",
                "-frames:v", "1", "-y", str(comp),
            ],
            capture_output=True, check=True,
        )
        r, g, b = _avg_rgb(comp, "40:40:0:0")
        # magenta ≈ (255, 0, 255): R cao, G thấp, B cao
        assert r > 200 and b > 200 and g < 60, f"Góc composite không phải magenta: {(r, g, b)}"

    def test_width_drives_height_preserves_aspect(self, rendered_aspect):
        """width ưu tiên: character width=400/height=500 → ô thực 400x225 (suy từ width).
        Vùng dưới ô thực (đáng lẽ thuộc box nếu dùng height=500) phải TRONG SUỐT."""
        frame = sorted(rendered_aspect["out_dir"].glob("*.png"))[5]
        track = _load_json(ASSET_DIR / "mouth_track.json")
        bl, bt, bw, bh = _character_box(rendered_aspect["manifest"], track)
        assert (bw, bh) == (400, 225), f"ô character phải là 400x225 (width ưu tiên), got {(bw, bh)}"

        cx, cy = bl + bw // 2, bt + bh // 2      # tâm ô thực → có nội dung
        center = _avg_alpha(frame, f"60:60:{cx - 30}:{cy - 30}")
        # Điểm nằm DƯỚI ô thực nhưng trong vùng height=500 (bt + 350) → phải trong suốt
        below_y = bt + 350
        below = _avg_alpha(frame, f"40:40:{cx - 20}:{below_y}")
        assert center >= 200, f"Tâm ô character phải có nội dung, alpha={center}"
        assert below <= 8, (
            f"Vùng dưới ô thực phải trong suốt (height suy từ width=225, không phải 500), alpha={below}"
        )
