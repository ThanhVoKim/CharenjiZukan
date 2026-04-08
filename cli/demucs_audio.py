#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demucs_audio.py — CLI: Tách voice/background từ audio sử dụng Demucs

Tách audio thành vocals hoặc thành background music sử dụng AI model Demucs.
Hỗ trợ 2 modes: 2-stems (vocals+bgm) hoặc 4-stems (drums/bass/other/vocals).

Ví dụ nhanh:
    uv run demucs_audio.py --input audio_muted.wav
    uv run demucs_audio.py --input audio.wav --keep vocals
    uv run demucs_audio.py --input audio.wav --stems 4 --keep bgm
"""

import sys
import argparse
import multiprocessing
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import setup_logging, get_logger  # noqa: E402

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# HARDWARE CHECK
# ─────────────────────────────────────────────────────────────────────

def check_hardware_requirements():
    """
    Kiểm tra hardware requirements cho Demucs.

    Returns:
        tuple: (can_run, reason)
            - can_run: True nếu có thể chạy
            - reason: Lý do nếu không thể chạy
    """
    # 1. Kiểm tra GPU
    has_gpu = False
    gpu_name = None
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {gpu_name}")
    except ImportError:
        pass

    # 2. Kiểm tra CPU cores
    cpu_cores = multiprocessing.cpu_count()
    logger.info(f"CPU cores: {cpu_cores}")

    # 3. Quyết định
    if has_gpu:
        return True, f"GPU available: {gpu_name}"

    if cpu_cores < 4:
        reason = f"CPU quá yếu ({cpu_cores} cores < 4 cores minimum). Cần GPU hoặc CPU >= 4 cores."
        logger.warning(f"⚠️ {reason}")
        return False, reason

    logger.warning("⚠️ Không có GPU - xử lý sẽ chậm trên CPU")
    return True, f"CPU {cpu_cores} cores (không có GPU)"


# ─────────────────────────────────────────────────────────────────────
# DEMUCS PROCESSING
# ─────────────────────────────────────────────────────────────────────

def check_demucs_installed():
    """
    Kiểm tra demucs đã được cài đặt chưa.

    Returns:
        bool: True nếu đã cài đặt
    """
    try:
        import demucs  # noqa: F401
        return True
    except ImportError:
        logger.error("demucs chưa được cài đặt. Chạy: pip install demucs")
        return False


def get_device():
    """
    Tự động detect device (cuda/cpu).

    Returns:
        str: 'cuda' hoặc 'cpu'
    """
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA GPU detected - sử dụng GPU")
            return "cuda"
        else:
            logger.warning("Không tìm thấy CUDA GPU - sử dụng CPU (chậm hơn)")
            return "cpu"
    except ImportError:
        return "cpu"


def separate_audio_batch(input_paths, output_paths, model="htdemucs", keep="bgm", bitrate="192k", device=None, segment=7):
    """
    Tách audio thành các sources sử dụng Demucs cho một danh sách file.
    Tái sử dụng model để tiết kiệm thời gian tải model.
    """
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    if len(input_paths) != len(output_paths):
        raise ValueError("Số lượng input_paths phải bằng số lượng output_paths")

    if not input_paths:
        return []

    # Auto-detect device nếu không chỉ định
    if device is None:
        device = get_device()

    # 1. Load model ONCE
    logger.info(f"Loading model: {model} for batch processing ({len(input_paths)} files)")
    model_obj = get_model(model)
    model_obj.to(device)
    model_obj.eval()

    if segment is not None:
        logger.info(f"Thiết lập segment size = {segment}s để tiết kiệm VRAM")
        model_obj.segment = float(segment)

    results = []

    # 2. Xử lý từng file
    for input_path, output_path in zip(input_paths, output_paths):
        logger.debug(f"Processing: {input_path}")
        wav, sr = torchaudio.load(input_path)
        
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
            
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)

        with torch.no_grad():
            sources = apply_model(model_obj, wav, device=device, split=True, overlap=0.25)

        num_sources = sources.shape[1]
        source_names = ["drums", "bass", "other", "vocals"]

        if num_sources == 4:
            if keep.isdigit() or (keep.replace(",", "").replace("-", "").isdigit()):
                if "-" in keep:
                    start, end = map(int, keep.split("-"))
                    indices = list(range(start, end + 1))
                else:
                    indices = [int(x.strip()) for x in keep.split(",")]
                
                output_audio = sources[0, indices[0]]
                for idx in indices[1:]:
                    output_audio = output_audio + sources[0, idx]
            else:
                keep_presets = {
                    "bgm": ([0, 1, 2], "background music"),
                    "vocals": ([3], "vocals"),
                    "drums": ([0], "drums"),
                    "bass": ([1], "bass"),
                    "other": ([2], "other"),
                    "all": ([0, 1, 2, 3], "all sources"),
                }
                indices, _ = keep_presets[keep]
                output_audio = sources[0, indices[0]]
                for idx in indices[1:]:
                    output_audio = output_audio + sources[0, idx]
        elif num_sources == 2:
            if keep in ["vocals", "3"]:
                output_audio = sources[0, 1]
            else:
                output_audio = sources[0, 0]

        output_audio = output_audio.cpu()
        
        # Lưu file WAV
        torchaudio.save(output_path, output_audio, sr)
        results.append(output_path)
        
        # Dọn dẹp RAM/VRAM cho từng iteration
        del wav, sources, output_audio

    logger.info(f"✅ Hoàn tất xử lý batch {len(input_paths)} files.")

    # 3. Cleanup VRAM tổng
    logger.info("Giải phóng VRAM...")
    del model_obj
    import gc
    gc.collect()
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.empty_cache()

    return results

def separate_audio(input_path, output_path, model="htdemucs", keep="bgm", bitrate="192k", device=None, segment=7):
    """
    Tách audio thành các sources sử dụng Demucs.

    Args:
        input_path: Đường dẫn file audio input
        output_path: Đường dẫn file output (.wav, .mp3, .m4a, .aac)
        model: Tên model Demucs (htdemucs, htdemucs_ft, mdx, mdx_extra)
        keep: Sources để giữ lại:
            - Presets: "bgm", "vocals", "drums", "bass", "other", "all"
            - Indices: "0,1,2" (drums+bass+other), "2" (other only), "0-2" (drums+bass+other)
            - Index mapping: 0=drums, 1=bass, 2=other, 3=vocals
        bitrate: Bitrate cho MP3/M4A output (mặc định: 192k)
        device: Device để chạy (cuda/cpu), None để auto-detect

    Returns:
        str: Đường dẫn file output
        
    Note:
        Demucs luôn trả về 4 sources: drums, bass, other, vocals
        Index: 0=drums, 1=bass, 2=other, 3=vocals
    """
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    # Auto-detect device nếu không chỉ định
    if device is None:
        device = get_device()

    # 1. Load model
    logger.info(f"Loading model: {model}")
    model_obj = get_model(model)
    model_obj.to(device)
    model_obj.eval()

    if segment is not None:
        logger.info(f"Thiết lập segment size = {segment}s để tiết kiệm VRAM")
        model_obj.segment = float(segment)

    # 2. Load audio
    logger.info(f"Loading audio: {input_path}")
    wav, sr = torchaudio.load(input_path)
    logger.info(f"Audio info: {wav.shape[1]} samples, {sr}Hz, {wav.shape[0]} channels")

    # 2b. Convert mono to stereo if needed (Demucs requires 2 channels)
    if wav.shape[0] == 1:
        logger.info("Converting mono to stereo (Demucs requires 2 channels)...")
        wav = wav.repeat(2, 1)  # Duplicate mono channel to create stereo
        logger.info(f"Audio info after conversion: {wav.shape[1]} samples, {sr}Hz, {wav.shape[0]} channels")

    # 3. Prepare audio for model
    # Demucs expects: [batch, channels, time]
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)

    # KHÔNG đưa toàn bộ wav lên GPU ở đây.
    # Truyền wav ở CPU vào apply_model, Demucs sẽ tự động đưa từng chunk lên GPU
    # xử lý rồi trả về CPU, giúp VRAM không bị tăng tuyến tính.
    # wav = wav.to(device)

    # 4. Apply model
    logger.info("Separating sources...")
    with torch.no_grad():
        sources = apply_model(model_obj, wav, device=device, split=True, overlap=0.25)

    # 5. Extract desired output
    # sources shape: [batch, sources, channels, time]
    # Demucs models always return 4 sources: drums, bass, other, vocals
    # Index: 0=drums, 1=bass, 2=other, 3=vocals
    
    num_sources = sources.shape[1]
    logger.info(f"Model returned {num_sources} sources")
    
    # Parse keep indices (comma-separated: "0,2" means drums + other)
    source_names = ["drums", "bass", "other", "vocals"]
    
    if num_sources == 4:
        # Parse keep parameter
        if keep.isdigit() or (keep.replace(",", "").replace("-", "").isdigit()):
            # Numeric format: "0,2" or "0-2" or "0"
            if "-" in keep:
                # Range format: "0-2" means 0,1,2
                start, end = map(int, keep.split("-"))
                indices = list(range(start, end + 1))
            else:
                # Comma-separated: "0,2"
                indices = [int(x.strip()) for x in keep.split(",")]
            
            # Validate indices
            for idx in indices:
                if idx < 0 or idx >= num_sources:
                    raise ValueError(f"Invalid source index: {idx}. Valid range: 0-{num_sources-1}")
            
            # Sum selected sources
            output_audio = sources[0, indices[0]]
            selected_names = [source_names[indices[0]]]
            for idx in indices[1:]:
                output_audio = output_audio + sources[0, idx]
                selected_names.append(source_names[idx])
            
            logger.info(f"Extracting sources: {', '.join(selected_names)} (indices: {indices})")
        
        else:
            # Named presets for backward compatibility
            keep_presets = {
                "bgm": ([0, 1, 2], "background music (drums + bass + other)"),
                "vocals": ([3], "vocals"),
                "drums": ([0], "drums"),
                "bass": ([1], "bass"),
                "other": ([2], "other"),
                "all": ([0, 1, 2, 3], "all sources"),
            }
            
            if keep not in keep_presets:
                raise ValueError(f"Invalid --keep option: {keep}. Use indices (0,1,2,3) or presets: {list(keep_presets.keys())}")
            
            indices, description = keep_presets[keep]
            output_audio = sources[0, indices[0]]
            for idx in indices[1:]:
                output_audio = output_audio + sources[0, idx]
            logger.info(f"Extracting {description}...")
    
    elif num_sources == 2:
        # Two-stems output: no_vocals, vocals (rare case)
        if keep in ["vocals", "3"]:
            output_audio = sources[0, 1]  # vocals
            logger.info("Extracting vocals...")
        else:
            output_audio = sources[0, 0]  # no_vocals
            logger.info("Extracting background music (no_vocals)...")
    
    else:
        raise ValueError(f"Unexpected number of sources: {num_sources}")

    # 6. Move to CPU for saving
    output_audio = output_audio.cpu()

    # 7. Export with format detection
    logger.info(f"Exporting to: {output_path}")
    output_path_obj = Path(output_path)
    suffix = output_path_obj.suffix.lower()
    
    if suffix in [".mp3", ".m4a", ".aac"]:
        # Use pydub for compressed formats
        from pydub import AudioSegment
        import tempfile
        
        # First save to temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        torchaudio.save(tmp_wav, output_audio, sr)
        
        # Convert to target format
        audio_seg = AudioSegment.from_wav(tmp_wav)
        
        if suffix == ".mp3":
            audio_seg.export(output_path, format="mp3", bitrate=bitrate)
        elif suffix == ".m4a":
            audio_seg.export(output_path, format="ipod", bitrate=bitrate)
        elif suffix == ".aac":
            audio_seg.export(output_path, format="adts", bitrate=bitrate)
        
        # Cleanup temp file
        Path(tmp_wav).unlink()
        logger.info(f"Converted to {suffix.upper()} with bitrate: {bitrate}")
    else:
        # Default WAV format
        torchaudio.save(output_path, output_audio, sr)

    logger.info(f"✅ Done! Output saved to: {output_path}")

    # 8. Cleanup VRAM
    logger.info("Giải phóng VRAM...")
    del model_obj
    del wav
    del sources
    del output_audio
    import gc
    gc.collect()
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.empty_cache()

    return output_path


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Tạo argument parser."""
    parser = argparse.ArgumentParser(
        prog="demucs_audio",
        description="Tách voice/background từ audio sử dụng Demucs AI model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Mặc định: BGM (drums + bass + other)
  python demucs_audio.py --input audio.wav

  # Chỉ lấy vocals
  python demucs_audio.py --input audio.wav --keep vocals

  # Chỉ lấy source "other" (index 2)
  python demucs_audio.py --input audio.wav --keep 2

  # Lấy drums + other (index 0,2)
  python demucs_audio.py --input audio.wav --keep 0,2

  # Lấy drums + bass + other (index 0-2)
  python demucs_audio.py --input audio.wav --keep 0-2

  # Output MP3 với bitrate 128k
  python demucs_audio.py --input audio.wav --output bgm.mp3 --bitrate 128k

Source indices:
  0 = drums
  1 = bass
  2 = other
  3 = vocals
        """,
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Đường dẫn file audio input (wav, mp3, mp4, mkv...)",
    )

    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Đường dẫn file output (.wav, .mp3, .m4a, .aac). Mặc định: <input>_bgm.wav",
    )

    parser.add_argument(
        "--model", "-m",
        default="htdemucs",
        choices=["htdemucs", "htdemucs_ft", "mdx", "mdx_extra"],
        help="Model Demucs sử dụng (mặc định: htdemucs)",
    )

    parser.add_argument(
        "--keep", "-k",
        default="bgm",
        help="Sources để giữ lại. Presets: bgm, vocals, drums, bass, other, all. "
             "Hoặc indices: '0,2' (drums+other), '0-2' (drums+bass+other). "
             "Index: 0=drums, 1=bass, 2=other, 3=vocals. Mặc định: bgm",
    )

    parser.add_argument(
        "--bitrate", "-b",
        default="192k",
        help="Bitrate cho MP3/M4A output (mặc định: 192k)",
    )

    parser.add_argument(
        "--device", "-d",
        default=None,
        help="Device để chạy model (cuda, cuda:0, cpu). Mặc định: auto-detect",
    )

    parser.add_argument(
        "--segment", type=float, default=7,
        help="Độ dài chunk (giây) để xử lý. Dùng '--segment 4' hoặc nhỏ hơn nếu GPU bị lỗi Out of Memory (OOM). Mặc định: 7",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Hiển thị log chi tiết",
    )

    return parser


def main():
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)

    # Check hardware
    can_run, reason = check_hardware_requirements()
    if not can_run:
        logger.error(f"❌ Hardware không đủ yêu cầu: {reason}")
        sys.exit(1)

    # Check demucs installed
    if not check_demucs_installed():
        logger.error("❌ Demucs chưa được cài đặt")
        sys.exit(1)

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"❌ File không tồn tại: {input_path}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Default output based on keep parameter
        if args.keep in ["bgm", "0,1,2", "0-2"]:
            suffix = "_bgm.wav"
        elif args.keep == "vocals" or args.keep == "3":
            suffix = "_vocals.wav"
        else:
            suffix = f"_{args.keep}.wav"
        output_path = input_path.with_suffix(suffix)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Print info
    print(f"\n{'=' * 55}")
    print(f"  🎵 Demucs Audio Separation")
    print(f"{'=' * 55}")
    print(f"  Input   : {input_path}")
    print(f"  Output  : {output_path}")
    print(f"  Model   : {args.model}")
    print(f"  Keep    : {args.keep}")
    print(f"  Bitrate : {args.bitrate}")
    print(f"  Device  : {args.device or 'auto-detect'}")
    print(f"  Segment : {args.segment}")
    print(f"{'=' * 55}\n")

    # Run separation
    try:
        separate_audio(
            input_path=str(input_path),
            output_path=str(output_path),
            model=args.model,
            keep=args.keep,
            bitrate=args.bitrate,
            device=args.device,
            segment=args.segment,
        )
    except Exception as e:
        logger.error(f"❌ Lỗi khi xử lý: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()