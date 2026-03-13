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


def separate_audio(input_path, output_path, model="htdemucs", stems=2, keep="bgm", device=None):
    """
    Tách audio thành các sources sử dụng Demucs.

    Args:
        input_path: Đường dẫn file audio input
        output_path: Đường dẫn file output
        model: Tên model Demucs (htdemucs, htdemucs_ft, mdx, mdx_extra)
        stems: Số nguồn tách (2 hoặc 4)
        keep: Giữ lại 'bgm' hoặc 'vocals'
        device: Device để chạy (cuda/cpu), None để auto-detect

    Returns:
        str: Đường dẫn file output
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

    # 2. Load audio
    logger.info(f"Loading audio: {input_path}")
    wav, sr = torchaudio.load(input_path)
    logger.info(f"Audio info: {wav.shape[1]} samples, {sr}Hz, {wav.shape[0]} channels")

    # 3. Prepare audio for model
    # Demucs expects: [batch, channels, time]
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)

    wav = wav.to(device)

    # 4. Apply model
    logger.info(f"Separating sources with {stems}-stems mode...")
    with torch.no_grad():
        sources = apply_model(model_obj, wav, device=device)

    # 5. Extract desired output
    # sources shape: [batch, sources, channels, time]
    # sources order for 4-stems: drums, bass, other, vocals
    # sources order for 2-stems: no_vocals, vocals

    if stems == 2:
        # Two-stems mode
        if keep == "vocals":
            output_audio = sources[0, 1]  # vocals
            logger.info("Extracting vocals...")
        else:
            output_audio = sources[0, 0]  # no_vocals (bgm)
            logger.info("Extracting background music (no vocals)...")
    else:
        # Four-stems mode: drums, bass, other, vocals
        drums = sources[0, 0]
        bass = sources[0, 1]
        other = sources[0, 2]
        vocals = sources[0, 3]

        if keep == "vocals":
            output_audio = vocals
            logger.info("Extracting vocals...")
        else:
            # bgm = drums + bass + other
            output_audio = drums + bass + other
            logger.info("Extracting background music (drums + bass + other)...")

    # 6. Move to CPU for saving
    output_audio = output_audio.cpu()

    # 7. Export
    logger.info(f"Exporting to: {output_path}")
    torchaudio.save(output_path, output_audio, sr)

    logger.info(f"✅ Done! Output saved to: {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Tạo argument parser."""
    parser = argparse.ArgumentParser(
        prog="demucs_audio",
        description="Tách voice/background từ audio sử dụng Demucs AI model. "
                    "Mặc định: 2-stems mode, output background music (remove vocals).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Mặc định: 2-stems, output bgm (remove vocals)
  python demucs_audio.py --input audio_muted.wav

  # 2-stems, output vocals (remove bgm)
  python democs_audio.py --input audio.wav --keep vocals

  # 4-stems, output bgm
  python demucs_audio.py --input audio.wav --stems 4

  # Với model chất lượng cao
  python demucs_audio.py --input audio.wav --model htdemucs_ft

  # Với GPU cụ thể
  python demucs_audio.py --input audio.wav --device cuda:0
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
        help="Đường dẫn file output. Mặc định: <input>_bgm.wav hoặc <input>_vocals.wav",
    )

    parser.add_argument(
        "--model", "-m",
        default="htdemucs",
        choices=["htdemucs", "htdemucs_ft", "mdx", "mdx_extra"],
        help="Model Demucs sử dụng (mặc định: htdemucs)",
    )

    parser.add_argument(
        "--stems", "-s",
        type=int,
        default=2,
        choices=[2, 4],
        help="Số nguồn tách: 2 (vocals+bgm) hoặc 4 (drums/bass/other/vocals). Mặc định: 2",
    )

    parser.add_argument(
        "--keep", "-k",
        default="bgm",
        choices=["bgm", "vocals"],
        help="Giữ lại: 'bgm' (background music) hoặc 'vocals'. Mặc định: bgm",
    )

    parser.add_argument(
        "--device", "-d",
        default=None,
        help="Device để chạy model (cuda, cuda:0, cpu). Mặc định: auto-detect",
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
        suffix = "_bgm.wav" if args.keep == "bgm" else "_vocals.wav"
        output_path = input_path.with_suffix(suffix)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Print info
    print(f"\n{'=' * 55}")
    print(f"  🎵 Demucs Audio Separation")
    print(f"{'=' * 55}")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"  Model  : {args.model}")
    print(f"  Stems  : {args.stems}")
    print(f"  Keep   : {args.keep}")
    print(f"  Device : {args.device or 'auto-detect'}")
    print(f"{'=' * 55}\n")

    # Run separation
    try:
        separate_audio(
            input_path=str(input_path),
            output_path=str(output_path),
            model=args.model,
            stems=args.stems,
            keep=args.keep,
            device=args.device,
        )
    except Exception as e:
        logger.error(f"❌ Lỗi khi xử lý: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()