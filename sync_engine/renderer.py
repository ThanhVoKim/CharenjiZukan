import subprocess
from pathlib import Path
from typing import Tuple, List, Optional
import os
import re
import logging

logger = logging.getLogger("sync_video")

def detect_gpu_encoder() -> Tuple[bool, str, str]:
    """Returns (has_gpu, encoder, preset)."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                       capture_output=True, text=True)
    if "h264_nvenc" in r.stdout:
        return True, "h264_nvenc", "p5"
    return False, "libx264", "fast"

def _get_video_duration(video_path: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0

def render_final_video(
    stretched_video: str,
    mixed_audio: str,
    subtitle_synced_srt: str,
    output_path: str,
    note_overlay_synced_ass: Optional[str] = None,
    render_config: dict = None,
    use_gpu: bool = True,
) -> None:
    if render_config is None:
        render_config = {}
        
    # Path format for ffmpeg filters on Windows needs escaping or forward slashes
    subtitle_synced_srt_esc = subtitle_synced_srt.replace('\\', '/')
    
    enc_cfg = render_config.get("video_encoding", {})
    quality_override = enc_cfg.get("quality")
    preset_override = enc_cfg.get("preset")
    
    has_gpu, auto_encoder, auto_preset = detect_gpu_encoder()
    if use_gpu and has_gpu:
        encoder = auto_encoder
        preset = preset_override if preset_override else auto_preset
        quality = quality_override if quality_override else ["-cq", "23"]
    else:
        encoder = "libx264"
        preset = preset_override if preset_override else "fast"
        quality = quality_override if quality_override else ["-crf", "23"]

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    cmd = [
        "ffmpeg", "-y",
        "-i", stretched_video,
        "-i", mixed_audio,
    ]

    input_idx = 2
    filter_cx = []
    
    # 0. Base scale
    current_v = "[0:v]"
    res_cfg = render_config.get("resolution", {})
    if not res_cfg.get("bypass_scale", False):
        w = res_cfg.get("width", 1920)
        h = res_cfg.get("height", 1080)
        filter_cx.append(f"{current_v}scale={w}:{h}[v_base]")
        current_v = "[v_base]"

    # 1. Watermark Image
    wm_img_cfg = render_config.get("watermark_img", {})
    if wm_img_cfg.get("enabled", False) and wm_img_cfg.get("path"):
        wm_path = Path(wm_img_cfg["path"])
        if not wm_path.is_absolute():
            wm_path = PROJECT_ROOT / wm_path
        
        if wm_path.exists():
            wm_path_esc = str(wm_path).replace('\\', '/')
            cmd.extend(["-i", wm_path_esc])
            wm_idx = input_idx
            input_idx += 1
            
            x = wm_img_cfg.get("x", "W-w-40")
            y = wm_img_cfg.get("y", "40")
            
            filter_cx.append(f"{current_v}[{wm_idx}:v]overlay=x={x}:y={y}[v_wm_img]")
            current_v = "[v_wm_img]"

    # 2. Watermark Text
    wm_txt_cfg = render_config.get("watermark_text", {})
    if wm_txt_cfg.get("enabled", False) and wm_txt_cfg.get("text"):
        font_path = Path(wm_txt_cfg.get("font_path", ""))
        if font_path and not font_path.is_absolute():
            font_path = PROJECT_ROOT / font_path
        
        font_path_esc = str(font_path).replace('\\', '/') if font_path.exists() else ""
        
        text = wm_txt_cfg.get("text", "")
        fontsize = wm_txt_cfg.get("fontsize", 25)
        color = wm_txt_cfg.get("color", "white")
        alpha = wm_txt_cfg.get("alpha", 0.7)
        x = wm_txt_cfg.get("x", "w-text_w-30")
        y = wm_txt_cfg.get("y", "8")
        
        drawtext_parts = [f"text='{text}'", f"fontsize={fontsize}", f"fontcolor={color}", f"alpha={alpha}", f"x={x}", f"y={y}"]
        if font_path_esc:
            # Sửa lỗi fontfile path bằng cách bọc nháy đơn và escape an toàn hơn nếu cần
            drawtext_parts.insert(0, f"fontfile='{font_path_esc}'")
            
        drawtext_str = ":".join(drawtext_parts)
        filter_cx.append(f"{current_v}drawtext={drawtext_str}[v_wm_txt]")
        current_v = "[v_wm_txt]"

    # 3. Note Overlay (Dynamic ASS Box)
    note_cfg = render_config.get("note_overlay", {})
    has_note = False
    if note_cfg.get("enabled", False) and note_overlay_synced_ass and Path(note_overlay_synced_ass).exists():
        has_note = True

    # 4. Black Strip
    strip_cfg = render_config.get("black_strip", {})
    has_strip = False
    if strip_cfg.get("enabled", False) and strip_cfg.get("path"):
        strip_path = Path(strip_cfg["path"])
        if not strip_path.is_absolute():
            strip_path = PROJECT_ROOT / strip_path
            
        if strip_path.exists():
            strip_path_esc = str(strip_path).replace('\\', '/')
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

    # 5. Subtitles (SRT)
    sub_cfg = render_config.get("subtitles", {})
    if sub_cfg.get("enabled", False) and sub_cfg.get("burn_hardsub", True):
        style_dict = sub_cfg.get("style", {})
        if style_dict:
            # Escape dấu phẩy trong style FFmpeg, r"\,Bold=1" -> "\,Bold=1"
            custom_style = ",".join([f"\\,{k}={v}" if i > 0 else f"{k}={v}" for i, (k, v) in enumerate(style_dict.items())])
        else:
            custom_style = ""

        if custom_style:
            filter_cx.append(f"{current_v}subtitles='{subtitle_synced_srt_esc}':force_style='{custom_style}'[v_sub]")
        else:
            filter_cx.append(f"{current_v}subtitles='{subtitle_synced_srt_esc}'[v_sub]")
        current_v = "[v_sub]"

    # 6. ASS Note Overlay (background + text via ASS drawing)
    if has_note and note_overlay_synced_ass:
        ass_esc = note_overlay_synced_ass.replace('\\', '/')
        filter_cx.append(f"{current_v}ass='{ass_esc}'[v_note]")
        current_v = "[v_note]"

    if filter_cx:
        filter_cx_str = ";".join(filter_cx)
        map_v = current_v
    else:
        filter_cx_str = ""
        map_v = "0:v"

    if filter_cx_str:
        cmd.extend(["-filter_complex", filter_cx_str])
        
    cmd.extend([
        "-map", map_v, 
        "-map", "1:a",
        "-c:v", encoder, "-preset", preset, *quality,
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
