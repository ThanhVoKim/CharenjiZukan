#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli/qwen3_asr.py — CLI: Transcribe video/audio → .srt dùng Qwen3-ASR (Transformers Backend)
"""

import sys
import os
import re
import json
import argparse
import gc
import subprocess
import string
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# expandable_segments giảm phân mảnh VRAM khi transcribe audio dài; phải set TRƯỚC khi torch khởi tạo CUDA allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger
from utils.media_utils import clear_vram
from utils.task_utils import resolve_cli_tasks, resolve_output_dir_and_stem

# Logic dùng chung đã được tách sang utils/asr_subtitle_utils.py
from utils.asr_subtitle_utils import (
    merge_punctuation,
    format_srt_time,
    segment_words_to_subtitles,
    write_subtitle_srt,
)

logger = get_logger(__name__)


def extract_audio(video_path: str) -> str:
    """Trích xuất âm thanh (WAV, 16kHz, mono)."""
    audio_path = video_path.rsplit('.', 1)[0] + '.wav'
    if not os.path.exists(audio_path):
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        command = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            audio_path
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path


def run_batch_transcribe(
    tasks: List[Dict[str, str]],
    language: str = "Chinese",
    max_chars: int = 15,
    min_chars: int = 8,
    split_on_comma: bool = False,
    batch_size: int = 32,
    max_new_tokens: int = 1024,
    offset_seconds: float = 0.24,
    model_path: str = "Qwen/Qwen3-ASR-1.7B",
    aligner_path: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    device: str = "cuda:0",
) -> List[Dict]:
    """Chạy batch transcribe bằng Qwen3-ASR."""
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError:
        logger.error("\u274c L\u1ed7i: Th\u01b0 vi\u1ec7n 'qwen-asr' ch\u01b0a \u0111\u01b0\u1ee3c c\u00e0i \u0111\u1eb7t.")
        logger.error("\ud83d\udca1 Vui l\u00f2ng c\u00e0i \u0111\u1eb7t Optional Dependency b\u1eb1ng l\u1ec7nh: pip install .[qwen-asr]")
        sys.exit(1)

    logger.info(f"\ud83d\ude80 \u0110ang kh\u1edfi t\u1ea1o m\u00f4 h\u00ecnh Qwen3-ASR...")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Aligner: {aligner_path}")
    logger.info(f"   Device: {device}")
    logger.info(f"   Batch size: {batch_size}")

    asr = None
    try:
        asr = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
            max_inference_batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            forced_aligner=aligner_path,
            forced_aligner_kwargs=dict(
                dtype=torch.bfloat16,
                device_map=device,
                attn_implementation="flash_attention_2",
            ),
        )

        # Chuẩn bị dữ liệu
        audio_paths = []
        for task in tasks:
            input_path = task["input"]
            output_dir, stem = resolve_output_dir_and_stem(task)
            output_dir.mkdir(parents=True, exist_ok=True)

            audio_path = extract_audio(input_path)
            task["audio_path"] = audio_path
            task["srt_path"] = str(output_dir / f"{stem}.srt")
            task["txt_path"] = str(output_dir / f"{stem}.txt")
            task["json_path"] = str(output_dir / f"{stem}.json")

            audio_paths.append(audio_path)

        # Transcribe
        logger.info(f"\ud83c\udf99\ufe0f \u0110ang x\u1eed l\u00fd {len(audio_paths)} file audio...")
        results = asr.transcribe(
            audio=audio_paths,
            language=language,
            return_time_stamps=True,
        )

        final_outputs = []
        for i, result in enumerate(results):
            task = tasks[i]

            if not result.time_stamps:
                logger.warning(f"\u23ed\ufe0f B\u1ecf qua (kh\u00f4ng c\u00f3 voice): {task['input']}")
                continue

            full_text = result.text
            words = result.time_stamps

            # Lưu TXT
            with open(task["txt_path"], "w", encoding="utf-8") as f:
                f.write(full_text)

            # Merge dấu câu (dùng helper chung từ utils/asr_subtitle_utils.py)
            merged_words = merge_punctuation(words, full_text)

            # Lưu JSON
            json_data = {
                "language": result.language,
                "text": full_text,
                "time_stamps": merged_words
            }
            with open(task["json_path"], "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)

            # Xử lý cắt câu Subtitle (dùng helper chung)
            subtitles = segment_words_to_subtitles(
                merged_words,
                max_chars=max_chars,
                min_chars=min_chars,
                split_on_comma=split_on_comma,
            )

            # Lưu SRT (dùng helper chung)
            write_subtitle_srt(subtitles, task["srt_path"], offset_seconds=offset_seconds)

            logger.info(f"\u2705 \u0110\u00e3 ho\u00e0n th\u00e0nh: {os.path.basename(task['input'])}")
            logger.info(f"   -> SRT: {task['srt_path']}")
            logger.info(f"   -> TXT: {task['txt_path']}")
            logger.info(f"   -> JSON: {task['json_path']}")

            final_outputs.append({
                "input": task["input"],
                "srt": task["srt_path"],
                "txt": task["txt_path"],
                "json": task["json_path"],
                "lines": len(subtitles),
            })

        return final_outputs

    finally:
        if asr is not None:
            logger.info("\ud83e\uddf9 Gi\u1ea3i ph\u00f3ng m\u00f4 h\u00ecnh Qwen3-ASR...")
            del asr
        clear_vram()
        logger.info("\ud83e\uddf9 VRAM \u0111\u00e3 \u0111\u01b0\u1ee3c gi\u1ea3i ph\u00f3ng.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen3_asr_srt",
        description="Transcribe video/audio \u2192 .srt d\u00f9ng Qwen3-ASR (Transformers Backend)",
    )

    io = parser.add_argument_group("Input / Output")
    io.add_argument("--input", "-i", default=None, metavar="FILE", help="\u0110\u01b0\u1eddng d\u1eabn 1 file video/audio \u0111\u1ea7u v\u00e0o")
    io.add_argument("--output", "-o", default=None, metavar="FILE_OR_DIR",
                    help="\u0110\u01b0\u1eddng d\u1eabn file .srt ho\u1eb7c th\u01b0 m\u1ee5c \u0111\u1ea7u ra (d\u00f9ng c\u00f9ng --input). "
                         "N\u1ebfu l\u00e0 th\u01b0 m\u1ee5c, s\u1ebd t\u1ea1o [t\u00ean_video].srt, [t\u00ean_video].txt, [t\u00ean_video].json")
    io.add_argument("--task-file", "-t", default=None, metavar="JSON_FILE", help="File JSON ch\u1ee9a danh s\u00e1ch [{'input': '...', 'output': '...'}]")

    mdl = parser.add_argument_group("Model")
    mdl.add_argument("--model-path", default="Qwen/Qwen3-ASR-1.7B", help="\u0110\u01b0\u1eddng d\u1eabn model ASR (m\u1eb7c \u0111\u1ecbnh: Qwen/Qwen3-ASR-1.7B)")
    mdl.add_argument("--aligner-path", default="Qwen/Qwen3-ForcedAligner-0.6B", help="\u0110\u01b0\u1eddng d\u1eabn model Forced Aligner (m\u1eb7c \u0111\u1ecbnh: Qwen/Qwen3-ForcedAligner-0.6B)")

    dev = parser.add_argument_group("Device")
    dev.add_argument("--device", "-d", default="cuda:0", help="Thi\u1ebft b\u1ecb ch\u1ea1y (m\u1eb7c \u0111\u1ecbnh: cuda:0)")

    seg = parser.add_argument_group("Segmentation / Language")
    seg.add_argument("--language", "-l", default="Chinese", help="Ng\u00f4n ng\u1eef audio (m\u1eb7c \u0111\u1ecbnh: Chinese)")
    seg.add_argument("--max-chars", type=int, default=15, help="S\u1ed1 k\u00fd t\u1ef1 t\u1ed1i \u0111a tr\u00ean m\u1ed7i d\u00f2ng ph\u1ee5 \u0111\u1ec1 (m\u1eb7c \u0111\u1ecbnh: 15, \u0111\u1eb7t 0 \u0111\u1ec3 ch\u1ec9 chia theo d\u1ea5u c\u00e2u, kh\u00f4ng \u00e9p \u0111\u1ed9 d\u00e0i)")
    seg.add_argument("--min-chars", type=int, default=8, help="S\u1ed1 k\u00fd t\u1ef1 t\u1ed1i thi\u1ec3u tr\u00ean m\u1ed7i d\u00f2ng ph\u1ee5 \u0111\u1ec1 (m\u1eb7c \u0111\u1ecbnh: 8, \u0111\u1eb7t 0 \u0111\u1ec3 t\u1eaft)")
    seg.add_argument("--split-on-comma", action="store_true", help="D\u00f9ng d\u1ea5u ph\u1ea9y l\u00e0m \u0111i\u1ec3m c\u1eaft block (m\u1eb7c \u0111\u1ecbnh: t\u1eaft)")
    seg.add_argument("--batch-size", type=int, default=32, help="Batch size cho inference (m\u1eb7c \u0111\u1ecbnh: 32)")
    seg.add_argument("--max-new-tokens", type=int, default=1024, help="S\u1ed1 token t\u1ed1i \u0111a sinh ra m\u1ed7i chunk khi inference (m\u1eb7c \u0111\u1ecbnh: 1024; gi\u1ea3m \u0111\u1ec3 ti\u1ebft ki\u1ec7m VRAM, t\u0103ng n\u1ebfu chunk d\u00e0i b\u1ecb c\u1eaft c\u1ee5t)")
    seg.add_argument("--offset-seconds", type=float, default=0.24, help="\u0110\u1ed9 l\u1ec7ch th\u1eddi gian b\u00f9 tr\u1eeb (gi\u00e2y, m\u1eb7c \u0111\u1ecbnh: 0.24)")

    misc = parser.add_argument_group("Misc")
    misc.add_argument("--verbose", action="store_true", help="B\u1eadt logging DEBUG")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel("DEBUG")
    else:
        logger.setLevel("INFO")

    try:
        tasks = resolve_cli_tasks(
            task_file=args.task_file,
            input_file=args.input,
            output_path=args.output,
            default_ext=".srt"
        )
    except ValueError as e:
        parser.error(str(e))

    try:
        run_batch_transcribe(
            tasks=tasks,
            language=args.language,
            max_chars=args.max_chars,
            min_chars=args.min_chars,
            split_on_comma=args.split_on_comma,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            offset_seconds=args.offset_seconds,
            model_path=args.model_path,
            aligner_path=args.aligner_path,
            device=args.device,
        )
        sys.exit(0)
    except KeyboardInterrupt:
        logger.warning("\n\u26a0\ufe0f D\u1eebng b\u1edfi ng\u01b0\u1eddi d\u00f9ng")
        sys.exit(1)


if __name__ == "__main__":
    main()
