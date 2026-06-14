#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
punctuate_srt.py — CLI: Phục hồi dấu câu cho file .srt (text OCR) bằng LLM

Đứng riêng khỏi `video-ocr` để dễ retry/debug bước phục hồi dấu câu. Mọi tham số
LLM (provider, prompt, language, batch_size, use_full_context, response_parser) lấy
DUY NHẤT từ task YAML (`--task-config`, mặc định
config/llm_tasks/punctuation_restoration.yaml) — đây là SSOT, đúng pattern translate-srt.

LLM chỉ được THÊM dấu câu, không đổi chữ; batch nào vi phạm/parse lỗi sau hết retry
sẽ GIỮ NGUYÊN text gốc (validator `_validate_content_preserved` chống ảo giác). Hạ tầng
batch (vòng lặp lô + context cache + integrity retry) dùng chung với `translate-srt`
qua `llm_ai/srt_batch/`; file này chỉ thêm validator + flatten đặc thù punctuation.

Output mỗi input: `<stem>_punct.srt` và (nếu --flatten) `<stem>_flat.txt` (1 dòng cho
forced aligner align-srt).

Ví dụ:
    punctuate-srt --input ocr.srt
    punctuate-srt --input ocr.srt --output ocr_punct.srt --no-flatten
    punctuate-srt --task-file tasks.json --lang Chinese
"""

import argparse
import logging
import sys
import unicodedata
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm_ai.base import BaseLLMProvider  # noqa: E402
from llm_ai.srt_batch import BatchIntegrityError, run_srt_batch_task  # noqa: E402
from llm_ai.task_runner import create_task_provider, resolve_project_path  # noqa: E402
from llm_ai.tasks.generic_text_task import load_task_config  # noqa: E402
from utils.logger import setup_logging, get_logger  # noqa: E402
from utils.srt_parser import parse_srt, write_segments_to_flat_text  # noqa: E402
from utils.task_utils import resolve_cli_tasks  # noqa: E402

PUNCT_TAG = "PUNCT_TEXT"


# ═════════════════════════════════════════════════════════════════════
# Core: phục hồi dấu câu (gộp từ punctuation/srt_punctuator.py cũ)
# ═════════════════════════════════════════════════════════════════════

def _content_signature(text: str) -> str:
    """Trả về phần 'nội dung chữ' của text: bỏ dấu câu (Unicode P*) + whitespace.

    Dùng để xác nhận LLM chỉ chèn dấu mà không thêm/bớt/đổi ký tự chữ.
    Trung lập ngôn ngữ — không phụ thuộc bộ dấu CJK hardcode.
    """
    return "".join(
        ch
        for ch in text
        if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    )


def _validate_content_preserved(punctuated_batch: list[dict], original_batch: list[dict]) -> None:
    """Raise BatchIntegrityError nếu bất kỳ dòng nào bị đổi nội dung chữ."""
    for idx, (new_item, old_item) in enumerate(zip(punctuated_batch, original_batch)):
        if _content_signature(new_item["text"]) != _content_signature(old_item["text"]):
            raise BatchIntegrityError(
                f"Dòng {old_item.get('line', idx)} bị đổi nội dung chữ "
                f"(không chỉ thêm dấu): gốc={old_item['text']!r} → out={new_item['text']!r}"
            )


def restore_punctuation_srt(
    input_srt: str,
    output_srt: str,
    prompt_file: str,
    provider: BaseLLMProvider,
    language: str = "Chinese",
    batch_size: int = 30,
    use_full_context: bool = True,
    wait_sec: float = 0.0,
) -> dict[str, Any]:
    """Phục hồi dấu câu cho SRT bằng LLM, ghi `output_srt`.

    Wrapper mỏng quanh `run_srt_batch_task` — chỉ đặc tả phần riêng của punctuation:
    response_tag = "PUNCT_TEXT" và validator chống ảo giác (chỉ thêm dấu, không đổi chữ).

    Returns:
        Dict stats: total_blocks, total_batches, success, failed, reverted_blocks, output.
    """
    _, stats = run_srt_batch_task(
        input_srt=input_srt,
        output_srt=output_srt,
        prompt_file=prompt_file,
        provider=provider,
        language=language,
        response_tag=PUNCT_TAG,
        batch_size=batch_size,
        use_full_context=use_full_context,
        wait_sec=wait_sec,
        validator=_validate_content_preserved,
        write_srt=True,
    )
    return stats


def flatten_srt_to_text(input_srt: str, output_txt: str) -> str:
    """Nối text các block SRT thành MỘT dòng liên tục cho forced aligner.

    Aligner (`merge_punctuation`) duyệt full_text theo ký tự và sẽ nuốt `\\n`
    vào phụ đề, nên transcript phải phẳng 1 dòng. Logic flatten (CJK nối không
    khoảng trắng, Latin nối bằng space) nằm trong `write_segments_to_flat_text`.

    Returns:
        Chuỗi flat text đã ghi.
    """
    raw_content = Path(input_srt).read_text(encoding="utf-8", errors="ignore")
    segments = parse_srt(raw_content)
    path = write_segments_to_flat_text(segments, output_txt, cjk_aware=True)
    return path.read_text(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="punctuate-srt",
        description="Phục hồi dấu câu cho .srt (text OCR) bằng LLM theo lô.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input", "-i",
        default=None,
        metavar="FILE",
        help="File SRT nguồn (text OCR chưa dấu). Bắt buộc nếu không dùng --task-file.",
    )
    parser.add_argument(
        "--task-file", "-t",
        default=None,
        metavar="JSON_FILE",
        help="File JSON chứa danh sách [{'input': '...', 'output': '...'}].",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE_OR_DIR",
        help="File/thư mục output (mặc định: <input>_punct.srt cạnh input).",
    )

    parser.add_argument(
        "--task-config",
        default=str(PROJECT_ROOT / "config/llm_tasks/punctuation_restoration.yaml"),
        metavar="YAML",
        help="Task YAML — SSOT mọi tham số LLM punctuation.",
    )
    parser.add_argument(
        "--provider", "-p",
        default=None,
        choices=["gemini", "openai", "vertexai"],
        help="Override provider (mặc định lấy từ task config / provider_chain).",
    )
    parser.add_argument(
        "--provider-config",
        default=None,
        metavar="YAML",
        help="Override provider config YAML (dùng single-provider mode).",
    )
    parser.add_argument(
        "--keys", "-k",
        default=None,
        metavar="KEY[,KEY2]",
        help="API key(s). Có thể set qua GEMINI_API_KEY/OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        metavar="MODEL",
        help="Override model name.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        metavar="FILE",
        help="Override prompt_file trong task config.",
    )
    parser.add_argument(
        "--lang", "-l",
        default=None,
        metavar="LANGUAGE",
        help="Ngôn ngữ nguồn (mặc định lấy từ task config, fallback Chinese).",
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=None,
        metavar="N",
        help="Số SRT block mỗi batch (ưu tiên: CLI > task config > 30).",
    )
    parser.add_argument(
        "--context",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Bật/tắt full-context (ưu tiên: CLI > task config > bật). Dùng --no-context để tắt.",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=None,
        metavar="SEC",
        help="Giây chờ giữa mỗi batch (ưu tiên: CLI > task config > 0).",
    )
    parser.add_argument(
        "--flatten",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sinh thêm <stem>_flat.txt (1 dòng) cho forced aligner. Mặc định bật.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Bật logging debug chi tiết.",
    )

    return parser


def _resolve_by_priority(cli_value, config: dict, config_keys: list[str], default_value):
    """Ưu tiên giá trị: CLI > task config > default."""
    if cli_value is not None:
        return cli_value
    for key in config_keys:
        if key in config and config.get(key) is not None:
            return config.get(key)
    return default_value


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = get_logger(__name__)

    try:
        task_config_path = resolve_project_path(args.task_config)
        task_cfg = load_task_config(str(task_config_path))
    except Exception as exc:
        parser.error(f"Không đọc được task config {args.task_config}: {exc}")
        return

    # Prompt: CLI override > task config.
    if args.prompt:
        prompt_path = resolve_project_path(args.prompt)
    else:
        prompt_path = resolve_project_path(task_cfg.get("prompt_file"))
    if not prompt_path or not Path(prompt_path).exists():
        parser.error(
            f"Prompt file không tồn tại: {prompt_path}. "
            "Dùng --prompt <path> hoặc cập nhật task config."
        )
    prompt_file = str(prompt_path)

    # create_task_provider đọc các attr override này; punctuate-srt không expose chúng
    # (mọi tham số LLM lấy từ task_cfg là SSOT) → set None để không override.
    for _attr in ("base_url", "system_prompt", "temperature", "max_tokens", "request_timeout"):
        if not hasattr(args, _attr):
            setattr(args, _attr, None)

    try:
        provider = create_task_provider(args, task_cfg)
    except Exception as exc:
        parser.error(str(exc))
        return

    language = _resolve_by_priority(args.lang, task_cfg, ["language"], "Chinese")
    batch_size = int(_resolve_by_priority(args.batch, task_cfg, ["batch", "batch_size"], 30))
    wait_sec = float(_resolve_by_priority(args.wait, task_cfg, ["wait", "wait_sec"], 0.0))
    use_full_context = bool(
        _resolve_by_priority(args.context, task_cfg, ["use_full_context", "full_context"], True)
    )

    try:
        tasks = resolve_cli_tasks(
            task_file=args.task_file,
            input_file=args.input,
            output_path=args.output,
            default_ext="_punct.srt",
        )
    except ValueError as exc:
        parser.error(str(exc))
        return

    for task in tasks:
        if Path(task["input"]).suffix.lower() != ".srt":
            parser.error(f"File phải có đuôi .srt: {task['input']}")

    ok_tasks = 0
    for task in tasks:
        input_file = task["input"]
        output_file = task["output"]

        print("=" * 55)
        print("  ✍️   SRT Punctuation Restoration — llm_ai")
        print("=" * 55)
        print(f"  Input    : {input_file}")
        print(f"  Output   : {output_file}")
        print(f"  Provider : {provider.name}")
        print(f"  Lang     : {language}")
        print(f"  Batch    : {batch_size}")
        print(f"  Context  : {'ON' if use_full_context else 'OFF'}")
        print("=" * 55)

        try:
            stats = restore_punctuation_srt(
                input_srt=input_file,
                output_srt=output_file,
                prompt_file=prompt_file,
                provider=provider,
                language=language,
                batch_size=batch_size,
                use_full_context=use_full_context,
                wait_sec=wait_sec,
            )

            flat_msg = ""
            if args.flatten:
                flat_txt = str(Path(output_file).with_name(f"{Path(output_file).stem}_flat.txt"))
                flatten_srt_to_text(output_file, flat_txt)
                flat_msg = f" → {flat_txt}"

            ok_tasks += 1
            print(
                f"✅ {stats['success']}/{stats['total_batches']} batch | "
                f"{stats['reverted_blocks']} block giữ gốc{flat_msg}"
            )
        except KeyboardInterrupt:
            print("\n⚠️  Đã dừng bởi người dùng")
            sys.exit(1)
        except Exception as exc:
            print(f"\n❌ Lỗi nghiêm trọng khi xử lý {input_file}: {exc}")
            logging.exception(exc)

    print(f"\n{'='*55}")
    print(f"  Tổng kết: {ok_tasks}/{len(tasks)} task thành công")
    print(f"{'='*55}")

    sys.exit(0 if ok_tasks == len(tasks) else 2)


if __name__ == "__main__":
    main()
