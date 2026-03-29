import subprocess
from pathlib import Path
from typing import Tuple, List, Optional
import os

DEFAULT_SUBTITLE_STYLE = (
    "Fontname=Noto Sans CJK JP"
    r"\,Bold=1"
    r"\,FontSize=22"
    r"\,PrimaryColour=&H00EEF5FF"
    r"\,OutlineColour=&H00FFFFFF"
    r"\,Outline=0"
    r"\,Shadow=0"
    r"\,BackColour=0xE6000000"
    r"\,Alignment=2"
    r"\,MarginV=6"
)

def detect_gpu_encoder() -> Tuple[bool, str, str]:
    """Returns (has_gpu, encoder, preset)."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                       capture_output=True, text=True)
    if "h264_nvenc" in r.stdout:
        return True, "h264_nvenc", "p7"
    return False, "libx264", "fast"

def ensure_black_bg(path: str, w: int = 1920, h: int = 1080, alpha: int = 255) -> str:
    """
    Tạo ảnh nền đen có độ trong suốt (Alpha).
    alpha: 0 (trong suốt hoàn toàn) đến 255 (đen đặc).
    """
    if Path(path).exists():
        return path
        
    out_path = path if path.endswith(".png") else str(Path(path).with_suffix(".png"))
    from PIL import Image
    # Đổi "RGB" thành "RGBA" và truyền tuple 4 giá trị (R, G, B, A)
    Image.new("RGBA", (w, h), (0, 0, 0, alpha)).save(out_path, format="PNG")
    return out_path

def _build_ass_enable_expr(ass_path: str) -> str:
    """Đọc file ASS và tạo biểu thức enable cho filter overlay của FFmpeg."""
    from utils.ass_utils import parse_ass_timestamp_to_ms
    
    with open(ass_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    intervals = []
    for line in lines:
        if line.startswith("Dialogue:"):
            parts = line.rstrip("\n").split(",", 9)
            if len(parts) >= 10:
                start_ms = parse_ass_timestamp_to_ms(parts[1].strip())
                end_ms = parse_ass_timestamp_to_ms(parts[2].strip())
                # Chuyển đổi ms thành giây thập phân
                start_s = start_ms / 1000.0
                end_s = end_ms / 1000.0
                intervals.append(f"between(t,{start_s:.3f},{end_s:.3f})")
                
    if not intervals:
        return "0" # Không hiển thị bao giờ
        
    # Nối bằng phép cộng (+ là logical OR trong FFmpeg enable expression)
    return "+".join(intervals)

def render_final_video(
    stretched_video: str,
    mixed_audio: str,
    subtitle_synced_srt: str,
    output_path: str,
    note_overlay_png: Optional[str] = None,
    note_overlay_synced_ass: Optional[str] = None,
    black_bg_path: Optional[str] = None,
    subtitle_style: str = DEFAULT_SUBTITLE_STYLE,
    use_gpu: bool = True,
) -> None:
    
    # Path format for ffmpeg filters on Windows needs escaping or forward slashes
    subtitle_synced_srt_esc = subtitle_synced_srt.replace('\\', '/')
    
    has_gpu, auto_encoder, auto_preset = detect_gpu_encoder()
    if use_gpu and has_gpu:
        encoder = auto_encoder
        preset = auto_preset
        quality = ["-cq", "21"]
    else:
        encoder = "libx264"
        preset = "fast"
        quality = ["-crf", "23"]

    watermark_path = "assets/CharenjiZukan-watermark.png"
    watermark_path_esc = watermark_path.replace('\\', '/')
    TITLE_FONT_PATH = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"

    cmd = [
        "ffmpeg", "-y",
        "-i", stretched_video,
        "-i", mixed_audio,
        "-i", watermark_path_esc
    ]

    has_note = note_overlay_png and Path(note_overlay_png).exists() and \
               note_overlay_synced_ass and Path(note_overlay_synced_ass).exists()

    if has_note:
        bg = ensure_black_bg(black_bg_path or "tmp_black_bg.png")
        note_overlay_synced_ass_esc = note_overlay_synced_ass.replace('\\', '/')
        bg_esc = bg.replace('\\', '/')
        png_esc = note_overlay_png.replace('\\', '/')
        enable_expr = _build_ass_enable_expr(note_overlay_synced_ass)
            
        cmd.extend([
            "-loop", "1", "-i", bg_esc,      # 3: dải đen 1920×h
            "-loop", "1", "-i", png_esc,     # 4: PNG note tĩnh
        ])
        
        filter_cx = "".join([
            # Scale dải đen 1920×h
            "[3:v]scale=1920:94[bg_scaled];",
            # 1. Overlay watermark_img
            "[0:v][2:v]overlay=x=1680:y=39[v_wm_img];",
            # 2. Draw watermark_text
            f"[v_wm_img]drawtext=fontfile='{TITLE_FONT_PATH}':text='@CharenjiZukan':fontsize=25:fontcolor=white:alpha=0.7:x=w-text_w-30:y=8[v_wm_txt];",
            # 3. Overlay note overlay (PNG tĩnh)
            f"[v_wm_txt][4:v]overlay=x=(main_w-overlay_w)/2:y=980:shortest=1:enable='{enable_expr}'[v_note];",
            # 4. Overlay black strip
            "[v_note][bg_scaled]overlay=x=0:y=968:shortest=1[v_strip];",
            # 5. ASS text (đè lên black strip)
            f"[v_strip]ass='{note_overlay_synced_ass_esc}'[v_ass];",
            # 6. Subtitle SRT (burn hardsub)
            f"[v_ass]subtitles='{subtitle_synced_srt_esc}':force_style='{subtitle_style}'[v_out]"
        ])
    elif black_bg_path and Path(black_bg_path).exists():
        bg = ensure_black_bg(black_bg_path)
        bg_esc = bg.replace('\\', '/')
            
        cmd.extend([
            "-loop", "1", "-i", bg_esc,      # 3: dải đen 1920×h
        ])
        
        filter_cx = "".join([
            # Scale dải đen 1920×h
            "[3:v]scale=1920:80[bg_scaled];",
            # 1. Overlay watermark_img
            "[0:v][2:v]overlay=x=1680:y=39[v_wm_img];",
            # 2. Draw watermark_text
            f"[v_wm_img]drawtext=fontfile='{TITLE_FONT_PATH}':text='@CharenjiZukan':fontsize=25:fontcolor=white:alpha=0.7:x=w-text_w-30:y=8[v_wm_txt];",
            # 3. Overlay black strip
            "[v_wm_txt][bg_scaled]overlay=x=0:y=980:shortest=1[v_strip];",
            # 4. Subtitle SRT (burn hardsub)
            f"[v_strip]subtitles='{subtitle_synced_srt_esc}':force_style='{subtitle_style}'[v_out]"
        ])
    else:
        filter_cx = "".join([
            # 1. Overlay watermark_img
            "[0:v][2:v]overlay=x=1680:y=39[v_wm_img];",
            # 2. Draw watermark_text
            f"[v_wm_img]drawtext=fontfile='{TITLE_FONT_PATH}':text='@CharenjiZukan':fontsize=25:fontcolor=white:alpha=0.7:x=w-text_w-30:y=8[v_wm_txt];",
            # 3. Subtitle SRT
            f"[v_wm_txt]subtitles='{subtitle_synced_srt_esc}':force_style='{subtitle_style}'[v_out]"
        ])

    cmd.extend([
        "-filter_complex", filter_cx,
        "-map", "[v_out]", "-map", "1:a",
        "-c:v", encoder, "-preset", preset, *quality,
        "-c:a", "aac", "-b:a", "192k",
    ])
    
    if has_note or (black_bg_path and Path(black_bg_path).exists()):
        cmd.append("-shortest")
        
    cmd.append(output_path)
    
    subprocess.run(cmd, check=True)
