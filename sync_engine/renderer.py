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
        return True, "h264_nvenc", "p4"
    return False, "libx264", "fast"

def ensure_black_bg(path: str, w: int = 1920, h: int = 80, alpha: int = 255) -> str:
    """
    Tạo ảnh nền đen có độ trong suốt (Alpha).
    alpha: 0 (trong suốt hoàn toàn) đến 255 (đen đặc).
    """
    if not Path(path).exists():
        from PIL import Image
        # Đổi "RGB" thành "RGBA" và truyền tuple 4 giá trị (R, G, B, A)
        Image.new("RGBA", (w, h), (0, 0, 0, alpha)).save(path, format="PNG")
    return path

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
        quality = ["-cq", "23"]
    else:
        encoder = "libx264"
        preset = "fast"
        quality = ["-crf", "23"]

    cmd = [
        "ffmpeg", "-y",
        "-i", stretched_video,
        "-i", mixed_audio
    ]

    has_note = note_overlay_png and Path(note_overlay_png).exists() and \
               note_overlay_synced_ass and Path(note_overlay_synced_ass).exists()

    if has_note:
        if not black_bg_path:
            black_bg_path = ensure_black_bg("black_bg.png")
            
        note_overlay_synced_ass_esc = note_overlay_synced_ass.replace('\\', '/')
            
        cmd.extend([
            "-loop", "1", "-i", black_bg_path,      # 2: dải đen 1920×80
            "-loop", "1", "-i", note_overlay_png,   # 3: PNG note tĩnh
        ])
        
        filter_cx = "".join([
            # Scale dải đen 1920×80
            "[2:v]scale=1920:80[bg_scaled];",
            # Overlay PNG note tại y=980
            "[0:v][3:v]overlay=x=(main_w-overlay_w)/2:y=980:shortest=1[v_png];",
            # Overlay dải đen (tạo nền tối cho text ASS)
            "[v_png][bg_scaled]overlay=x=(main_w-overlay_w)/2:y=980:shortest=1[v_strip];",
            # ASS text (timing đã recalculate, hiển thị trên vùng note PNG)
            f"[v_strip]ass='{note_overlay_synced_ass_esc}'[v_ass];",
            # Subtitle SRT (burn hardsub)
            f"[v_ass]subtitles='{subtitle_synced_srt_esc}':force_style='{subtitle_style}'[v_out]"
        ])
    else:
        filter_cx = (
            f"[0:v]subtitles='{subtitle_synced_srt_esc}'"
            f":force_style='{subtitle_style}'[v_out]"
        )

    cmd.extend([
        "-filter_complex", filter_cx,
        "-map", "[v_out]", "-map", "1:a",
        "-c:v", encoder, "-preset", preset, *quality,
        "-c:a", "aac", "-b:a", "192k",
    ])
    
    if has_note:
        cmd.append("-shortest")
        
    cmd.append(output_path)
    
    subprocess.run(cmd, check=True)
