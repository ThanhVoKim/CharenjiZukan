"""Dynamic ASS note overlay layout expansion.

This module converts a remapped note ASS file into a final ASS overlay where each
source dialogue is expanded into two events:

- Layer 0: an ASS drawing rectangle used as the note background.
- Layer 1: pixel-wrapped text positioned inside that rectangle.

Layout selection is per-dialogue via the ASS ``Name``/Actor field. The selected
key is resolved against ``render_config['note_overlay']['layouts']`` with a
``default_layout`` fallback for legacy input whose ``Name`` is empty.
"""

from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.ass_utils import parse_ass_file, wrap_text_pixel, write_ass_file

try:  # Pillow is optional at import-time; runtime falls back to a heuristic font.
    from PIL import ImageFont
except ImportError:  # pragma: no cover - exercised only when Pillow is absent.
    ImageFont = None

logger = logging.getLogger(__name__)

ALLOWED_ANCHORS = {"top_left", "top_right", "bottom_left", "bottom_right", "center"}
ASS_COLOR_RE = re.compile(r"^&H([0-9A-Fa-f]{8})$")

HARDCODED_DEFAULT_PRESET = {
    "anchor": "top_left",
    "margin_x": 60,
    "margin_y": 60,
    "width": 640,
    "height": 0,
    "padding_left": 28,
    "padding_right": 28,
    "padding_top": 24,
    "padding_bottom": 36,
    "height_safety_margin": 10,
    "background_color": "&HCC000000",
    "text_align": "left",
}

DEFAULT_FONT_CONFIG = {
    "fontname": "Noto Sans CJK JP",
    "font_path": "assets/NotoSansCJKsc-VF.ttf",
    "font_size": 42,
    "bold": False,
    "line_spacing": 1.25,
    "primary_color": "&H00FFFFFF",
}


class _HeuristicFont:
    """Small fallback object compatible with ``ImageFont.getlength``."""

    def __init__(self, font_size: int) -> None:
        self.font_size = font_size

    def getlength(self, text: str) -> float:
        width = 0.0
        for char in text:
            width += self.font_size if _is_cjk_char(char) else self.font_size * 0.55
        return width


class FontState:
    """Lazy font loader/cache for pixel measurement."""

    def __init__(self, layout_cfg: dict, project_root: Optional[Path]):
        self.default_font_cfg = layout_cfg.get("font", {})
        self.project_root = project_root
        self._font_cache = {}

    def get_font_for_preset(self, preset: dict):
        font_path = preset.get("font_path", self.default_font_cfg.get("font_path"))
        font_size = int(preset.get("font_size", self.default_font_cfg.get("font_size", 42)))
        cache_key = (str(font_path), font_size)
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        font = None
        if ImageFont is not None and font_path:
            full_path = Path(font_path)
            if self.project_root and not full_path.is_absolute():
                full_path = self.project_root / full_path

            if full_path.exists():
                try:
                    font = ImageFont.truetype(str(full_path), size=font_size)
                except Exception as exc:  # pragma: no cover - depends on local font runtime.
                    logger.warning("Không thể load font %s: %s", full_path, exc)
            else:
                logger.warning("Font path không tồn tại: %s", full_path)

        if font is None and ImageFont is not None:
            try:
                font = ImageFont.load_default()
            except Exception:  # pragma: no cover - very defensive.
                font = None

        if font is None:
            font = _HeuristicFont(font_size)

        self._font_cache[cache_key] = font
        return font


def _is_cjk_char(char: str) -> bool:
    return (
        "\u4e00" <= char <= "\u9fff"
        or "\u3400" <= char <= "\u4dbf"
        or "\u3040" <= char <= "\u30ff"
        or "\uac00" <= char <= "\ud7af"
        or "\uf900" <= char <= "\ufaff"
    )


def _load_layout_config(render_config: dict) -> dict:
    """Validate and normalize ``render_config['note_overlay']``."""

    note_cfg = copy.deepcopy(render_config.get("note_overlay", {}) or {})

    mode = note_cfg.get("mode", "dynamic_ass_box")
    if mode == "png_legacy":
        logger.warning(
            "DeprecationWarning: note_overlay.mode=png_legacy is no longer supported; "
            "using dynamic_ass_box."
        )
        mode = "dynamic_ass_box"
    if mode != "dynamic_ass_box":
        raise ValueError(f"Invalid note_overlay.mode: {mode}")
    note_cfg["mode"] = mode

    if "png_path" in note_cfg:
        logger.warning(
            "DeprecationWarning: note_overlay.png_path is no longer supported; "
            "using dynamic_ass_box."
        )

    font_cfg = {**DEFAULT_FONT_CONFIG, **(note_cfg.get("font") or {})}
    font_cfg["font_size"] = int(font_cfg.get("font_size", DEFAULT_FONT_CONFIG["font_size"]))
    font_cfg["line_spacing"] = float(font_cfg.get("line_spacing", DEFAULT_FONT_CONFIG["line_spacing"]))
    note_cfg["font"] = font_cfg

    raw_layouts = note_cfg.get("layouts") or {}
    default_key = str(note_cfg.get("default_layout") or "top_left")

    if not raw_layouts:
        legacy_position = note_cfg.get("position") or {}
        preset = dict(HARDCODED_DEFAULT_PRESET)
        if isinstance(legacy_position, dict) and legacy_position:
            preset.update(
                {
                    "width": legacy_position.get("width", preset["width"]),
                    "height": legacy_position.get("height", preset["height"]),
                }
            )
            if legacy_position.get("x") is not None and legacy_position.get("y") is not None:
                preset["x"] = legacy_position.get("x")
                preset["y"] = legacy_position.get("y")
        raw_layouts = {default_key: preset}

    if default_key not in raw_layouts:
        fallback_key = next(iter(raw_layouts))
        logger.warning(
            "note_overlay.default_layout=%s không tồn tại; dùng layout đầu tiên: %s",
            default_key,
            fallback_key,
        )
        default_key = fallback_key

    normalized_layouts = {}
    for key, preset in raw_layouts.items():
        merged = {**HARDCODED_DEFAULT_PRESET, **(preset or {})}
        anchor = merged.get("anchor", "top_left")
        if anchor not in ALLOWED_ANCHORS:
            raise ValueError(f"Invalid anchor: {anchor}")

        merged["width"] = int(merged.get("width", HARDCODED_DEFAULT_PRESET["width"]))
        merged["height"] = int(merged.get("height", HARDCODED_DEFAULT_PRESET["height"]))
        if merged["width"] <= 0:
            raise ValueError(f"Invalid note_overlay.layouts.{key}.width: {merged['width']}")
        if merged["height"] < 0:
            raise ValueError(f"Invalid note_overlay.layouts.{key}.height: {merged['height']}")

        for pad_key in ("padding_left", "padding_right", "padding_top", "padding_bottom"):
            merged[pad_key] = int(merged.get(pad_key, HARDCODED_DEFAULT_PRESET[pad_key]))
            if merged[pad_key] < 0:
                raise ValueError(f"Invalid note_overlay.layouts.{key}.{pad_key}: {merged[pad_key]}")

        merged["height_safety_margin"] = int(merged.get("height_safety_margin", 10))
        merged["margin_x"] = int(merged.get("margin_x", HARDCODED_DEFAULT_PRESET["margin_x"]))
        merged["margin_y"] = int(merged.get("margin_y", HARDCODED_DEFAULT_PRESET["margin_y"]))
        if merged["width"] - merged["padding_left"] - merged["padding_right"] <= 0:
            raise ValueError(f"Invalid note_overlay.layouts.{key}: padding exceeds width")

        merged["font_size"] = int(merged.get("font_size", font_cfg["font_size"]))
        merged["line_spacing"] = float(merged.get("line_spacing", font_cfg["line_spacing"]))
        merged["bold"] = bool(merged.get("bold", font_cfg.get("bold", False)))
        merged["text_color"] = merged.get("text_color", font_cfg.get("primary_color", "&H00FFFFFF"))
        merged["fontname"] = merged.get("fontname", font_cfg.get("fontname", DEFAULT_FONT_CONFIG["fontname"]))
        merged["font_path"] = merged.get("font_path", font_cfg.get("font_path"))
        merged["layout_key"] = key
        normalized_layouts[key] = merged

    note_cfg["default_layout"] = default_key
    note_cfg["layouts"] = normalized_layouts
    note_cfg["enabled"] = bool(note_cfg.get("enabled", False))
    return note_cfg


def _split_dialogue_fields(line: str) -> List[str]:
    """Split a Dialogue line into 10 ASS event fields."""

    if not line.startswith("Dialogue:"):
        raise ValueError(f"Not a Dialogue line: {line}")
    return [part.strip() for part in line.split(":", 1)[1].split(",", 9)]


def _resolve_layout(name: str, layout_cfg: dict) -> Tuple[str, dict, bool]:
    layouts = layout_cfg.get("layouts", {})
    requested_key = (name or "").strip()

    if requested_key in layouts:
        return requested_key, layouts[requested_key], False

    default_key = layout_cfg.get("default_layout") or next(iter(layouts))
    if default_key not in layouts:
        default_key = next(iter(layouts))
    return default_key, layouts[default_key], True


def _wrap_paragraphs(text: str, font, max_width_px: int) -> List[str]:
    return wrap_text_pixel(text, max(1, max_width_px), font)


def _measure_line_width(font, line: str) -> int:
    if hasattr(font, "getlength"):
        return int(font.getlength(line))
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(line)
        return int(bbox[2] - bbox[0])
    return int(font.getsize(line)[0])


def _measure_text_block(lines: List[str], font, preset: dict, default_line_spacing: float) -> Tuple[int, int]:
    text_width_px = 0
    font_size = int(preset.get("font_size", 42))
    line_spacing = float(preset.get("line_spacing", default_line_spacing))
    line_height_px = round(font_size * line_spacing)

    for line in lines or [""]:
        text_width_px = max(text_width_px, _measure_line_width(font, line))

    text_height_px = max(1, len(lines or [""])) * line_height_px
    return text_width_px, text_height_px


def _compute_box_geometry(preset: dict, text_height: int, video_w: int, video_h: int) -> Tuple[int, int, int, int]:
    h_safety = int(preset.get("height_safety_margin", 10))
    required_h = (
        int(preset.get("padding_top", 24))
        + int(text_height)
        + int(preset.get("padding_bottom", 36))
        + h_safety
    )
    final_box_height = max(int(preset.get("height", 0)), required_h)
    h = min(final_box_height, max(1, video_h))
    w = min(int(preset.get("width", 640)), max(1, video_w))

    if preset.get("x") is not None and preset.get("y") is not None:
        x = int(preset["x"])
        y = int(preset["y"])
    else:
        anchor = preset.get("anchor", "top_left")
        mx = int(preset.get("margin_x", 60))
        my = int(preset.get("margin_y", 60))

        if anchor == "top_left":
            x, y = mx, my
        elif anchor == "top_right":
            x, y = video_w - mx - w, my
        elif anchor == "bottom_left":
            x, y = mx, video_h - my - h
        elif anchor == "bottom_right":
            x, y = video_w - mx - w, video_h - my - h
        elif anchor == "center":
            x, y = (video_w - w) // 2, (video_h - h) // 2
        else:
            raise ValueError(f"Invalid anchor: {anchor}")

    max_x = max(0, video_w - w)
    max_y = max(0, video_h - h)
    x = max(0, min(int(x), max_x))
    y = max(0, min(int(y), max_y))
    return x, y, w, h


def _split_ass_color(color: str) -> Tuple[str, str]:
    match = ASS_COLOR_RE.match(str(color or ""))
    if not match:
        logger.warning("ASS color không hợp lệ (%s), fallback &H00FFFFFF", color)
        return "00", "FFFFFF"
    value = match.group(1).upper()
    return value[:2], value[2:]


def _ass_color_tags(color: str, channel: str = "c") -> str:
    alpha, bbggrr = _split_ass_color(color)
    tags = f"\\{channel}&H{bbggrr}&"
    if alpha != "00":
        tags += f"\\alpha&H{alpha}&"
    return tags


def _escape_ass_text(text: str) -> str:
    return text.replace("{", r"\{").replace("}", r"\}")


def _build_background_dialogue(
    layer: str,
    start: str,
    end: str,
    name: str,
    x: int,
    y: int,
    w: int,
    h: int,
    color: str,
) -> str:
    color_tags = _ass_color_tags(color, channel="1c")
    drawing = rf"{{\pos({x},{y}){color_tags}\p1}}m 0 0 l {w} 0 l {w} {h} l 0 {h}{{\p0}}"
    return f"Dialogue: {layer},{start},{end},NoteBox,{name},0,0,0,,{drawing}"


def _build_text_dialogue(
    layer: str,
    start: str,
    end: str,
    name: str,
    x: int,
    y: int,
    lines: List[str],
    preset: dict,
    default_font_size: int,
    default_color: str,
) -> str:
    text_content = _escape_ass_text("\\N".join(lines or [""]))

    override = rf"\pos({x},{y})"
    font_size = int(preset.get("font_size", default_font_size))
    if font_size != int(default_font_size):
        override += rf"\fs{font_size}"

    color = preset.get("text_color", default_color)
    if color != default_color:
        override += _ass_color_tags(color, channel="c")

    final_text = f"{{{override}}}{text_content}"
    return f"Dialogue: {layer},{start},{end},NoteText,{name},0,0,0,,{final_text}"


def _build_output_styles(layout_cfg: dict) -> List[str]:
    font_cfg = layout_cfg.get("font", {})
    bold_flag = -1 if font_cfg.get("bold", False) else 0
    bg_color = "&HCC000000"
    text_color = font_cfg.get("primary_color", "&H00FFFFFF")
    fontname = font_cfg.get("fontname", "Noto Sans CJK JP")
    font_size = int(font_cfg.get("font_size", 42))
    return [
        f"Style: NoteBox,Arial,20,{bg_color},&H000000FF,{bg_color},{bg_color},0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1",
        f"Style: NoteText,{fontname},{font_size},{text_color},&H000000FF,&H00000000,&H00000000,{bold_flag},0,0,0,100,100,0,0,1,0,0,7,0,0,0,1",
    ]


def _write_empty_output(output_ass_path: str, video_width: int, video_height: int, layout_cfg: dict) -> None:
    write_ass_file(
        output_path=output_ass_path,
        script_info={
            "ScriptType": "v4.00+",
            "PlayResX": str(video_width),
            "PlayResY": str(video_height),
            "WrapStyle": "1",
            "Collisions": "Normal",
        },
        styles=_build_output_styles(layout_cfg),
        events_format="Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        dialogues=[],
    )


def expand_note_overlay_ass(
    input_ass_path: str,
    output_ass_path: str,
    render_config: dict,
    video_width: int,
    video_height: int,
    project_root: Optional[Path] = None,
) -> dict:
    """Expand a remapped note ASS file into dynamic background + text ASS."""

    layout_cfg = _load_layout_config(render_config)
    stats = {
        "n_dialogues_in": 0,
        "n_dialogues_out": 0,
        "unknown_layout_keys": [],
        "fallback_count": 0,
    }

    if not layout_cfg.get("enabled", False):
        _write_empty_output(output_ass_path, video_width, video_height, layout_cfg)
        return stats

    font_state = FontState(layout_cfg, project_root)
    ass_data = parse_ass_file(input_ass_path)
    default_line_sp = float(layout_cfg.get("font", {}).get("line_spacing", 1.25))
    default_font_size = int(layout_cfg.get("font", {}).get("font_size", 42))
    default_color = layout_cfg.get("font", {}).get("primary_color", "&H00FFFFFF")

    new_dialogues: List[str] = []

    for line in ass_data.get("dialogues", []):
        parts = _split_dialogue_fields(line)
        if len(parts) != 10:
            logger.warning("Bỏ qua Dialogue không đúng format: %s", line)
            continue

        _layer, start, end, _style, name, _margin_l, _margin_r, _margin_v, _effect, text = parts
        stats["n_dialogues_in"] += 1

        layout_key, preset, used_fallback = _resolve_layout(name, layout_cfg)
        if used_fallback:
            stats["fallback_count"] += 1
            if name and name not in stats["unknown_layout_keys"]:
                stats["unknown_layout_keys"].append(name)
                logger.warning("Unknown layout key: %s, using fallback layout: %s", name, layout_key)

        font = font_state.get_font_for_preset(preset)
        max_width_px = int(preset["width"]) - int(preset["padding_left"]) - int(preset["padding_right"])
        lines = _wrap_paragraphs(text, font, max_width_px)
        _text_w, text_h = _measure_text_block(lines, font, preset, default_line_sp)

        x, y, w, h = _compute_box_geometry(preset, text_h, video_width, video_height)
        bg_dialogue = _build_background_dialogue(
            "0",
            start,
            end,
            layout_key,
            x,
            y,
            w,
            h,
            preset.get("background_color", "&HCC000000"),
        )
        text_dialogue = _build_text_dialogue(
            "1",
            start,
            end,
            layout_key,
            x + int(preset["padding_left"]),
            y + int(preset["padding_top"]),
            lines,
            preset,
            default_font_size,
            default_color,
        )

        new_dialogues.extend([bg_dialogue, text_dialogue])
        stats["n_dialogues_out"] += 2

    script_info = dict(ass_data.get("script_info") or {})
    script_info.setdefault("ScriptType", "v4.00+")
    script_info["PlayResX"] = str(video_width)
    script_info["PlayResY"] = str(video_height)
    script_info.setdefault("WrapStyle", "1")
    script_info.setdefault("Collisions", "Normal")

    write_ass_file(
        output_path=output_ass_path,
        script_info=script_info,
        styles=_build_output_styles(layout_cfg),
        events_format="Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        dialogues=new_dialogues,
    )

    return stats
