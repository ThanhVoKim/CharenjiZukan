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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger
from utils.media_utils import clear_vram

logger = get_logger(__name__)

# Bộ dấu câu
CJK_PUNCT = set("，。！？；：“”‘’（）《》【】、")
ALL_PUNCT_SET = set(string.punctuation) | CJK_PUNCT
SPLIT_PUNCT_SET = set(".,!?:;。，！？：；、")


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


def format_srt_time(seconds: float) -> str:
    """Định dạng thời gian SRT."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def merge_punctuation(words, full_text: str) -> List[Dict]:
    """Gắn dấu câu từ full_text vào mảng words timestamp."""
    merged_words = []
    text_idx = 0
    full_len = len(full_text)

    for word_obj in words:
        clean_word = word_obj.text
        trailing_chars = ""
        word_len = len(clean_word)

        while text_idx < full_len and full_text[text_idx].lower() != clean_word[0].lower():
            text_idx += 1

        if full_text.lower().startswith(clean_word.lower(), text_idx):
            text_idx += word_len

        while text_idx < full_len:
            char = full_text[text_idx]
            if char.isalnum() and char not in ALL_PUNCT_SET:
                break
            trailing_chars += char
            text_idx += 1

        merged_words.append({
            "text": clean_word + trailing_chars,
            "start_time": word_obj.start_time,
            "end_time": word_obj.end_time
        })
    return merged_words




def _resolve_output_paths(task: Dict[str, str]) -> Tuple[Path, str]:
    """Xác định output directory và file stem từ task."""
    inp_path = Path(task["input"])
    out_path = Path(task["output"])

    if out_path.suffix.lower() != ".srt":
        output_dir = out_path
        stem = inp_path.stem
    else:
        output_dir = out_path.parent
        stem = out_path.stem

    return output_dir, stem


def run_batch_transcribe(
    tasks: List[Dict[str, str]],
    language: str = "Chinese",
    max_chars: int = 15,
    batch_size: int = 32,
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
        logger.error("❌ Lỗi: Thư viện 'qwen-asr' chưa được cài đặt.")
        logger.error("💡 Vui lòng cài đặt Optional Dependency bằng lệnh: pip install .[qwen-asr]")
        sys.exit(1)

    logger.info(f"🚀 Đang khởi tạo mô hình Qwen3-ASR...")
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
            max_new_tokens=4096,
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
            output_dir, stem = _resolve_output_paths(task)
            output_dir.mkdir(parents=True, exist_ok=True)

            audio_path = extract_audio(input_path)
            task["audio_path"] = audio_path
            task["srt_path"] = str(output_dir / f"{stem}.srt")
            task["txt_path"] = str(output_dir / f"{stem}.txt")
            task["json_path"] = str(output_dir / f"{stem}.json")

            audio_paths.append(audio_path)

        # Transcribe
        logger.info(f"🎙️ Đang xử lý {len(audio_paths)} file audio...")
        results = asr.transcribe(
            audio=audio_paths,
            language=language,
            return_time_stamps=True,
        )

        final_outputs = []
        for i, result in enumerate(results):
            task = tasks[i]

            if not result.time_stamps:
                logger.warning(f"⏭️ Bỏ qua (không có voice): {task['input']}")
                continue

            full_text = result.text
            words = result.time_stamps

            # Lưu TXT
            with open(task["txt_path"], "w", encoding="utf-8") as f:
                f.write(full_text)

            # Merge dấu câu
            merged_words = merge_punctuation(words, full_text)

            # Lưu JSON
            json_data = {
                "language": result.language,
                "text": full_text,
                "time_stamps": merged_words
            }
            with open(task["json_path"], "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)

            # Xử lý cắt câu Subtitle
            subtitles = []
            current_sentence = []

            for item in merged_words:
                current_sentence.append(item)
                current_text = "".join([w["text"] for w in current_sentence])

                has_split_punct = any(char in SPLIT_PUNCT_SET for char in item["text"])
                is_too_long = len(current_text.strip()) >= max_chars

                if has_split_punct or is_too_long:
                    subtitles.append(current_sentence)
                    current_sentence = []

            if current_sentence:
                subtitles.append(current_sentence)

            # Lưu SRT
            with open(task["srt_path"], "w", encoding="utf-8") as f:
                for idx, sentence in enumerate(subtitles):
                    raw_start = max(0.0, sentence[0]["start_time"] + offset_seconds)
                    raw_end = sentence[-1]["end_time"] + offset_seconds

                    start_time = format_srt_time(raw_start)
                    end_time = format_srt_time(raw_end)
                    text = "".join([w["text"] for w in sentence]).strip()

                    f.write(f"{idx + 1}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")

            logger.info(f"✅ Đã hoàn thành: {os.path.basename(task['input'])}")
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
            logger.info("🧹 Giải phóng mô hình Qwen3-ASR...")
            del asr
        clear_vram()
        logger.info("🧹 VRAM đã được giải phóng.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen3_asr_srt",
        description="Transcribe video/audio → .srt dùng Qwen3-ASR (Transformers Backend)",
    )

    io = parser.add_argument_group("Input / Output")
    io.add_argument("--input", "-i", default=None, metavar="FILE", help="Đường dẫn 1 file video/audio đầu vào")
    io.add_argument("--output", "-o", default=None, metavar="FILE_OR_DIR",
                    help="Đường dẫn file .srt hoặc thư mục đầu ra (dùng cùng --input). "
                         "Nếu là thư mục, sẽ tạo [tên_video].srt, [tên_video].txt, [tên_video].json")
    io.add_argument("--task-file", "-t", default=None, metavar="JSON_FILE", help="File JSON chứa danh sách [{'input': '...', 'output': '...'}]")

    mdl = parser.add_argument_group("Model")
    mdl.add_argument("--model-path", default="Qwen/Qwen3-ASR-1.7B", help="Đường dẫn model ASR (mặc định: Qwen/Qwen3-ASR-1.7B)")
    mdl.add_argument("--aligner-path", default="Qwen/Qwen3-ForcedAligner-0.6B", help="Đường dẫn model Forced Aligner (mặc định: Qwen/Qwen3-ForcedAligner-0.6B)")

    dev = parser.add_argument_group("Device")
    dev.add_argument("--device", "-d", default="cuda:0", help="Thiết bị chạy (mặc định: cuda:0)")

    seg = parser.add_argument_group("Segmentation / Language")
    seg.add_argument("--language", "-l", default="Chinese", help="Ngôn ngữ audio (mặc định: Chinese)")
    seg.add_argument("--max-chars", type=int, default=15, help="Số ký tự tối đa trên mỗi dòng phụ đề (mặc định: 15)")
    seg.add_argument("--batch-size", type=int, default=32, help="Batch size cho inference (mặc định: 32)")
    seg.add_argument("--offset-seconds", type=float, default=0.24, help="Độ lệch thời gian bù trừ (giây, mặc định: 0.24)")

    misc = parser.add_argument_group("Misc")
    misc.add_argument("--verbose", action="store_true", help="Bật logging DEBUG")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel("DEBUG")
    else:
        logger.setLevel("INFO")

    tasks = []

    if args.task_file:
        try:
            with open(args.task_file, "r", encoding="utf-8") as f:
                tasks = json.load(f)
            logger.info(f"Loaded {len(tasks)} tasks từ {args.task_file}")
        except Exception as e:
            logger.error(f"Lỗi đọc task file JSON: {e}")
            sys.exit(1)
    elif args.input:
        inp = Path(args.input)
        if not inp.exists():
            logger.error(f"File input không tồn tại: {inp}")
            sys.exit(1)

        if args.output:
            out = Path(args.output)
            if out.suffix.lower() != ".srt":
                out = out / f"{inp.stem}.srt"
        else:
            out = inp.parent / f"{inp.stem}.srt"

        tasks.append({"input": str(inp), "output": str(out)})
    else:
        parser.error("Phải cung cấp --input hoặc --task-file")

    try:
        run_batch_transcribe(
            tasks=tasks,
            language=args.language,
            max_chars=args.max_chars,
            batch_size=args.batch_size,
            offset_seconds=args.offset_seconds,
            model_path=args.model_path,
            aligner_path=args.aligner_path,
            device=args.device,
        )
        sys.exit(0)
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Dừng bởi người dùng")
        sys.exit(1)


if __name__ == "__main__":
    main()
