import logging
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

from sync_engine.image_overlay import ImageOverlayEvent
from utils.ffmpeg_probe import (
    HEVC_NVENC_VIDEO_ARGS as _HEVC_NVENC_VIDEO_ARGS,
    detect_hevc_nvenc,
    get_hevc_nvenc_unavailable_reason,
)

logger = logging.getLogger("sync_video")


def _get_video_duration(video_path: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def _ffmpeg_path(path: str | Path) -> str:
    """Chuẩn hóa path cho FFmpeg command/filter trên Windows và Unix."""
    return str(path).replace("\\", "/")


def _safe_label(value: str, fallback: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_]", "_", value)
    label = re.sub(r"_+", "_", label).strip("_")
    return label or fallback


def _clamp_opacity(value) -> float:
    try:
        opacity = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, opacity))


def _estimate_command_length(cmd_parts: Sequence[str]) -> int:
    # Windows command line limit is approximately 32k chars. Quoting overhead matters,
    # so this estimate intentionally mirrors the logged command shape.
    return len(" ".join(str(part) for part in cmd_parts))


def _choose_filter_strategy(
    *,
    requested_strategy: str,
    event_count: int,
    direct_overlay_max_events: int,
    command_line_max_chars: int,
    estimated_direct_length: int,
) -> str:
    strategy = (requested_strategy or "auto").strip().lower()
    if strategy not in {"auto", "direct", "script", "intermediate"}:
        logger.warning("render_strategy image_overlay không hợp lệ: %s. Fallback auto.", requested_strategy)
        strategy = "auto"

    if strategy == "intermediate":
        raise NotImplementedError(
            "render_strategy='intermediate' chưa được triển khai. "
            "Phase hiện tại chỉ hỗ trợ direct và filter_complex_script."
        )

    if strategy == "direct":
        if estimated_direct_length > command_line_max_chars:
            logger.warning(
                "FFmpeg command dài %s ký tự, vượt ngưỡng an toàn %s, "
                "nhưng render_strategy='direct' nên vẫn dùng -filter_complex.",
                estimated_direct_length,
                command_line_max_chars,
            )
        return "direct"

    if strategy == "script":
        return "script"

    if event_count > direct_overlay_max_events:
        logger.info(
            "Image overlay có %s events, vượt direct_overlay_max_events=%s. "
            "Chuyển sang -filter_complex_script.",
            event_count,
            direct_overlay_max_events,
        )
        return "script"

    if estimated_direct_length > command_line_max_chars:
        logger.info(
            "FFmpeg command dự kiến dài %s ký tự, vượt command_line_max_chars=%s. "
            "Chuyển sang -filter_complex_script.",
            estimated_direct_length,
            command_line_max_chars,
        )
        return "script"

    return "direct"


def _append_image_overlay_filters(
    *,
    cmd: list[str],
    filter_cx: list[str],
    current_v: str,
    image_overlay_events: Sequence[ImageOverlayEvent],
    image_cfg: dict,
    input_idx: int,
    output_width: int,
    output_height: int,
) -> tuple[str, int]:
    if not image_overlay_events:
        return current_v, input_idx

    fit = (image_cfg.get("fit") or "stretch_to_output").strip().lower()
    if fit != "stretch_to_output":
        logger.warning(
            "image_overlay.fit=%s chưa được hỗ trợ; fallback stretch_to_output.",
            image_cfg.get("fit"),
        )

    x = image_cfg.get("x", "0")
    y = image_cfg.get("y", "0")
    opacity = _clamp_opacity(image_cfg.get("opacity", 1.0))

    path_counts = Counter(str(Path(event.image_path).resolve()) for event in image_overlay_events)
    image_input_map: dict[str, int] = {}

    for image_path in path_counts:
        cmd.extend(["-loop", "1", "-i", _ffmpeg_path(image_path)])
        image_input_map[image_path] = input_idx
        input_idx += 1

    logger.info("Image overlay unique static image inputs: %s", len(image_input_map))

    event_label_queue: dict[str, list[str]] = {}
    for asset_id, (image_path, use_count) in enumerate(path_counts.items()):
        source_idx = image_input_map[image_path]
        base_label = f"img_{asset_id}_base"
        preprocess = f"[{source_idx}:v]scale={output_width}:{output_height},format=rgba"
        if opacity < 1.0:
            preprocess += f",colorchannelmixer=aa={opacity:.3f}"
        preprocess += f"[{base_label}]"
        filter_cx.append(preprocess)

        if use_count == 1:
            event_label_queue[image_path] = [f"[{base_label}]"]
        else:
            split_labels = [f"img_{asset_id}_{idx}" for idx in range(use_count)]
            split_outputs = "".join(f"[{label}]" for label in split_labels)
            filter_cx.append(f"[{base_label}]split={use_count}{split_outputs}")
            event_label_queue[image_path] = [f"[{label}]" for label in split_labels]

    for event_idx, event in enumerate(image_overlay_events):
        image_path = str(Path(event.image_path).resolve())
        image_label = event_label_queue[image_path].pop(0)
        out_label = f"v_img_{event_idx}"
        start_sec = max(0.0, float(event.start_time) / 1000.0)
        end_sec = max(start_sec, float(event.end_time) / 1000.0)
        key_label = _safe_label(event.key, f"event_{event_idx}")
        filter_cx.append(
            f"{current_v}{image_label}overlay=x={x}:y={y}:shortest=1:"
            f"enable='between(t,{start_sec:.3f},{end_sec:.3f})'[{out_label}]"
        )
        logger.debug(
            "Image overlay event #%s key=%s label=%s start=%.3fs end=%.3fs",
            event_idx,
            key_label,
            image_label,
            start_sec,
            end_sec,
        )
        current_v = f"[{out_label}]"

    return current_v, input_idx


def _write_filter_complex_script(
    filter_cx_str: str,
    output_path: str,
    filter_complex_script_dir: Optional[str] = None,
) -> Path:
    if filter_complex_script_dir:
        script_dir = Path(filter_complex_script_dir)
    else:
        script_dir = Path(output_path).resolve().parent / "tmp"
    script_dir.mkdir(parents=True, exist_ok=True)

    script_path = script_dir / f"{Path(output_path).stem}_image_overlay_filter_complex.txt"
    script_path.write_text(filter_cx_str, encoding="utf-8")
    return script_path


def render_final_video(
    stretched_video: str,
    mixed_audio: str,
    subtitle_synced_srt: str,
    output_path: str,
    note_overlay_synced_ass: Optional[str] = None,
    render_config: dict = None,
    use_gpu: bool = True,
    image_overlay_events: Optional[Sequence[ImageOverlayEvent]] = None,
    filter_complex_script_dir: Optional[str] = None,
    keep_filter_complex_script: bool = False,
) -> None:
    if render_config is None:
        render_config = {}

    subtitle_synced_srt_esc = _ffmpeg_path(subtitle_synced_srt)

    if not detect_hevc_nvenc():
        reason = get_hevc_nvenc_unavailable_reason()
        detail = f" Chi tiết probe: {reason}" if reason else ""
        raise RuntimeError(
            "hevc_nvenc không khả dụng ở runtime. Flow sync_video hiện yêu cầu NVIDIA HEVC NVENC "
            "với tham số -c:v hevc_nvenc -preset p4 -tune hq -cq 28."
            f"{detail}"
        )

    if not use_gpu:
        logger.warning(
            "use_gpu=False/--no-gpu bị bỏ qua: final render dùng "
            "hevc_nvenc -preset p4 -tune hq -cq 28."
        )

    project_root = Path(__file__).resolve().parent.parent

    cmd = [
        "ffmpeg", "-y",
        "-i", stretched_video,
        "-i", mixed_audio,
    ]

    input_idx = 2
    filter_cx: list[str] = []
    image_overlay_events = list(image_overlay_events or [])

    # 0. Base scale
    current_v = "[0:v]"
    res_cfg = render_config.get("resolution", {})
    output_width = int(res_cfg.get("width", 1920))
    output_height = int(res_cfg.get("height", 1080))
    if not res_cfg.get("bypass_scale", False):
        filter_cx.append(f"{current_v}scale={output_width}:{output_height}[v_base]")
        current_v = "[v_base]"

    # 1. Image Overlay static image full-screen
    image_cfg = render_config.get("image_overlay", {}) or {}
    if image_overlay_events:
        current_v, input_idx = _append_image_overlay_filters(
            cmd=cmd,
            filter_cx=filter_cx,
            current_v=current_v,
            image_overlay_events=image_overlay_events,
            image_cfg=image_cfg,
            input_idx=input_idx,
            output_width=output_width,
            output_height=output_height,
        )

    # 2. Note Overlay (Dynamic ASS Box)
    note_cfg = render_config.get("note_overlay", {})
    has_note = False
    if note_cfg.get("enabled", False) and note_overlay_synced_ass and Path(note_overlay_synced_ass).exists():
        has_note = True
        ass_esc = _ffmpeg_path(note_overlay_synced_ass)
        filter_cx.append(f"{current_v}ass='{ass_esc}'[v_note]")
        current_v = "[v_note]"

    # 3. Black Strip
    strip_cfg = render_config.get("black_strip", {})
    has_strip = False
    if strip_cfg.get("enabled", False) and strip_cfg.get("path"):
        strip_path = Path(strip_cfg["path"])
        if not strip_path.is_absolute():
            strip_path = project_root / strip_path

        if strip_path.exists():
            strip_path_esc = _ffmpeg_path(strip_path)
            cmd.extend(["-loop", "1", "-i", strip_path_esc])
            strip_idx = input_idx
            input_idx += 1
            has_strip = True

            sw = strip_cfg.get("scale_width")
            sh = strip_cfg.get("scale_height")
            if sw and sh:
                filter_cx.append(f"[{strip_idx}:v]scale={sw}:{sh}[bg_scaled]")
                strip_layer = "[bg_scaled]"
            else:
                strip_layer = f"[{strip_idx}:v]"

            x = strip_cfg.get("x", "(main_w-overlay_w)/2")
            y = strip_cfg.get("y", "968")

            filter_cx.append(f"{current_v}{strip_layer}overlay=x={x}:y={y}:shortest=1[v_strip]")
            current_v = "[v_strip]"

    # 4. Watermark Image
    wm_img_cfg = render_config.get("watermark_img", {})
    if wm_img_cfg.get("enabled", False) and wm_img_cfg.get("path"):
        wm_path = Path(wm_img_cfg["path"])
        if not wm_path.is_absolute():
            wm_path = project_root / wm_path

        if wm_path.exists():
            wm_path_esc = _ffmpeg_path(wm_path)
            cmd.extend(["-i", wm_path_esc])
            wm_idx = input_idx
            input_idx += 1

            x = wm_img_cfg.get("x", "W-w-40")
            y = wm_img_cfg.get("y", "40")

            filter_cx.append(f"{current_v}[{wm_idx}:v]overlay=x={x}:y={y}[v_wm_img]")
            current_v = "[v_wm_img]"

    # 5. Watermark Text
    wm_txt_cfg = render_config.get("watermark_text", {})
    if wm_txt_cfg.get("enabled", False) and wm_txt_cfg.get("text"):
        font_path = Path(wm_txt_cfg.get("font_path", ""))
        if font_path and not font_path.is_absolute():
            font_path = project_root / font_path

        font_path_esc = _ffmpeg_path(font_path) if font_path.exists() else ""

        text = wm_txt_cfg.get("text", "")
        fontsize = wm_txt_cfg.get("fontsize", 25)
        color = wm_txt_cfg.get("color", "white")
        alpha = wm_txt_cfg.get("alpha", 0.7)
        x = wm_txt_cfg.get("x", "w-text_w-30")
        y = wm_txt_cfg.get("y", "8")

        drawtext_parts = [f"text='{text}'", f"fontsize={fontsize}", f"fontcolor={color}", f"alpha={alpha}", f"x={x}", f"y={y}"]
        if font_path_esc:
            drawtext_parts.insert(0, f"fontfile='{font_path_esc}'")

        drawtext_str = ":".join(drawtext_parts)
        filter_cx.append(f"{current_v}drawtext={drawtext_str}[v_wm_txt]")
        current_v = "[v_wm_txt]"

    # 6. Subtitles (SRT)
    sub_cfg = render_config.get("subtitles", {})
    if sub_cfg.get("enabled", False) and sub_cfg.get("burn_hardsub", True):
        style_dict = sub_cfg.get("style", {})
        if style_dict:
            custom_style = ",".join([f"\\,{k}={v}" if i > 0 else f"{k}={v}" for i, (k, v) in enumerate(style_dict.items())])
        else:
            custom_style = ""

        if custom_style:
            filter_cx.append(f"{current_v}subtitles='{subtitle_synced_srt_esc}':force_style='{custom_style}'[v_sub]")
        else:
            filter_cx.append(f"{current_v}subtitles='{subtitle_synced_srt_esc}'[v_sub]")
        current_v = "[v_sub]"

    filter_complex_script_path: Optional[Path] = None
    if filter_cx:
        filter_cx_str = ";".join(filter_cx)
        map_v = current_v
    else:
        filter_cx_str = ""
        map_v = "0:v"

    if filter_cx_str:
        direct_cmd_preview = [*cmd, "-filter_complex", filter_cx_str]
        image_cfg = render_config.get("image_overlay", {}) or {}
        if image_overlay_events:
            direct_overlay_max_events = int(image_cfg.get("direct_overlay_max_events", 200) or 200)
            command_line_max_chars = int(image_cfg.get("command_line_max_chars", 25000) or 25000)
            strategy = _choose_filter_strategy(
                requested_strategy=image_cfg.get("render_strategy", "auto"),
                event_count=len(image_overlay_events),
                direct_overlay_max_events=direct_overlay_max_events,
                command_line_max_chars=command_line_max_chars,
                estimated_direct_length=_estimate_command_length(direct_cmd_preview),
            )
        else:
            strategy = "direct"

        if strategy == "script":
            filter_complex_script_path = _write_filter_complex_script(
                filter_cx_str,
                output_path=output_path,
                filter_complex_script_dir=filter_complex_script_dir,
            )
            cmd.extend(["-filter_complex_script", str(filter_complex_script_path)])
            logger.info("Dùng filter_complex_script: %s", filter_complex_script_path)
        else:
            cmd.extend(["-filter_complex", filter_cx_str])

    cmd.extend([
        "-map", map_v,
        "-map", "1:a",
        *_HEVC_NVENC_VIDEO_ARGS,
        "-c:a", "aac", "-b:a", "192k",
    ])

    if has_strip:
        cmd.append("-shortest")

    cmd.append(output_path)

    logger.info(f"Lệnh FFmpeg render final video:\n{' '.join(cmd)}")

    total_duration = _get_video_duration(stretched_video)

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    try:
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace'
        )

        if has_tqdm and total_duration > 0:
            pbar = tqdm(total=total_duration, desc="Rendering Final Video", unit="s", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            time_pattern = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")

            for line in process.stderr:
                match = time_pattern.search(line)
                if match:
                    h, m, s = match.groups()
                    current_time = int(h) * 3600 + int(m) * 60 + float(s)
                    pbar.n = min(current_time, total_duration)
                    pbar.refresh()

            pbar.close()
        else:
            for line in process.stderr:
                pass

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
    finally:
        if filter_complex_script_path and not keep_filter_complex_script:
            try:
                os.remove(filter_complex_script_path)
                logger.info("Đã xoá filter_complex_script tạm: %s", filter_complex_script_path)
            except OSError as exc:
                logger.warning("Không thể xoá filter_complex_script %s: %s", filter_complex_script_path, exc)
