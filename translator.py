# -*- coding: utf-8 -*-
"""
translator.py — Core logic dịch SRT bằng Gemini API
Trung thành với quy trình của pyvideotrans/translator/_gemini.py
"""

import re
import copy
import time
import logging
from pathlib import Path
from typing import List

from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type,
    before_log,
    after_log,
)

logger = logging.getLogger("srt_translator")

# ─────────────────────────────────────────────
# Template context toàn cục (giống _base.py)
# ─────────────────────────────────────────────
_GLOBAL_CONTEXT_TEMPLATE = """
# Global Context (Read-Only Reference)
The user has provided the FULL subtitle file below for context (plot, terms, gender).
**INSTRUCTIONS for Context:**
1. Read this to understand the story.
2. **DO NOT translate this section.**
3. Use this ONLY to improve the accuracy of the batch translation below.

<FULL_SOURCE_CONTEXT>
{COMPLETE_SRT_TEXT}
</FULL_SOURCE_CONTEXT>
"""


# ─────────────────────────────────────────────
# SRT PARSER  (giống help_srt.py)
# ─────────────────────────────────────────────
def parse_srt(content: str) -> List[dict]:
    """Chuyển chuỗi SRT thành list[dict]: {line, time, text}."""
    blocks = re.split(r'\n\s*\n', content.strip())
    srt_list = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue
        time_line = lines[1].strip()
        if '-->' not in time_line:
            continue
        text = "\n".join(lines[2:]).strip()
        # Loại bỏ HTML tags (giống help_srt.py)
        text = re.sub(r'</?[a-zA-Z]+>', '', text, flags=re.I | re.S)
        text = re.sub(r'\n{2,}', '\n', text).strip()
        srt_list.append({"line": index, "time": time_line, "text": text})
    return srt_list


def srt_list_to_string(srt_list: List[dict]) -> str:
    """Chuyển list[dict] về chuỗi SRT hợp lệ để ghi file."""
    parts = []
    for i, it in enumerate(srt_list, 1):
        text = it.get('text', '').strip()
        if not text:
            continue
        parts.append(f"{i}\n{it['time']}\n{text}")
    return "\n\n".join(parts) + "\n"


# ─────────────────────────────────────────────
# PROMPT LOADER  (giống tools.get_prompt)
# ─────────────────────────────────────────────
def load_prompt(prompt_file: str, target_language: str) -> str:
    """
    Đọc file gemini.txt và thay {lang} bằng ngôn ngữ đích.
    prompt_file: đường dẫn tuyệt đối hoặc tương đối tới gemini.txt
    """
    content = Path(prompt_file).read_text(encoding='utf-8', errors='ignore')
    return content.replace('{lang}', target_language)


# ─────────────────────────────────────────────
# PARSE KẾT QUẢ GEMINI  (giống _gemini.py)
# ─────────────────────────────────────────────
def parse_gemini_response(result: str) -> str:
    """
    Trích xuất nội dung trong <TRANSLATE_TEXT>...</TRANSLATE_TEXT>.
    Loại bỏ <think>...</think> trước khi parse (Gemini thinking mode).
    """
    cleaned = re.sub(r'<think>.*?</think>', '', result, flags=re.I | re.S)
    match = re.search(
        r'<TRANSLATE_TEXT>(.*?)(?:</TRANSLATE_TEXT>|$)',
        cleaned,
        re.S | re.I,
    )
    if match:
        return match.group(1).strip()
    raise RuntimeError(
        f"Không tìm thấy <TRANSLATE_TEXT> trong response:\n{result[:300]}..."
    )


def merge_translated_batch(translated_str: str, original_batch: List[dict]) -> List[dict]:
    """
    Ghép text đã dịch (SRT string) vào đúng vị trí của original_batch.
    Fallback: giữ nguyên bản gốc nếu số block không khớp.
    """
    result = copy.deepcopy(original_batch)
    try:
        translated_blocks = parse_srt(translated_str)
        if len(translated_blocks) == len(original_batch):
            for i, block in enumerate(translated_blocks):
                result[i]['text'] = block['text']
            return result
        logger.warning(
            f"Block mismatch: gốc={len(original_batch)}, dịch={len(translated_blocks)} "
            "→ giữ nguyên batch này"
        )
    except Exception as e:
        logger.warning(f"Lỗi parse translated batch: {e} → giữ nguyên batch này")
    return result


# ─────────────────────────────────────────────
# GEMINI API CALLER  (giống _gemini.py._item_task)
# ─────────────────────────────────────────────
class GeminiCaller:
    """
    Quản lý round-robin API keys và gọi Gemini streaming.
    Giống cơ chế pop(0)/append() trong pyvideotrans.
    """

    def __init__(self, api_keys: List[str], model: str, thinking_budget: int = 8192):
        if not api_keys:
            raise ValueError("Cần ít nhất 1 Gemini API key")
        self.api_keys = api_keys[:]          # copy để tránh mutate ngoài
        self.model = model
        self.thinking_budget = thinking_budget

    def _next_key(self) -> str:
        """Round-robin: lấy key đầu, đẩy xuống cuối."""
        key = self.api_keys.pop(0)
        self.api_keys.append(key)
        return key

    @retry(
        retry=retry_if_exception_type(RuntimeError),
        stop=stop_after_attempt(3),
        wait=wait_fixed(5),
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.DEBUG),
    )
    def call(self, message: str) -> str:
        """Gọi Gemini API với streaming, trả về full text response."""
        from google import genai
        from google.genai import types

        api_key = self._next_key()
        client = genai.Client(api_key=api_key)
        model = self.model

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=message)],
            )
        ]

        # Config thinking (Gemini 2.5+)
        gen_config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=65530,
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
            ],
            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
            system_instruction=[
                types.Part.from_text(text="You are a top-tier Subtitle Translation Engine.")
            ],
        )

        # Gemini 1.x / 2.0 không hỗ trợ thinking_config
        if model.startswith("gemini-1.") or model.startswith("gemini-2.0"):
            gen_config = types.GenerateContentConfig(temperature=0.2, max_output_tokens=65530)

        # Streaming response
        result = ""
        for chunk in client.models.generate_content_stream(
            model=model, contents=contents, config=gen_config
        ):
            result += chunk.text if chunk.text else ""

        if not result:
            raise RuntimeError("[Gemini] Response rỗng — sẽ retry")
        return result


# ─────────────────────────────────────────────
# HÀM DỊCH CHÍNH
# ─────────────────────────────────────────────
def translate_srt_file(
    input_file: str,
    output_file: str,
    prompt_file: str,
    api_keys: List[str],
    model: str = "gemini-2.5-flash",
    target_language: str = "Vietnamese",
    batch_size: int = 30,
    thinking_budget: int = 8192,
    use_full_context: bool = True,
    wait_sec: float = 0.0,
) -> dict:
    """
    Pipeline dịch SRT hoàn chỉnh.
    Trả về dict thống kê: {total, success, failed, elapsed}
    """
    # 1. Đọc và parse SRT gốc
    raw_content = Path(input_file).read_text(encoding="utf-8", errors="ignore")
    srt_list = parse_srt(raw_content)
    total = len(srt_list)

    if total == 0:
        raise ValueError(f"Không đọc được block SRT nào từ: {input_file}")

    logger.info(f"📄 Đọc: {total} block SRT | Model: {model} | Batch: {batch_size}")
    print(f"\n📄 Tổng: {total} block SRT")
    print(f"🤖 Model  : {model}")
    print(f"🌐 Ngôn ngữ: {target_language}")
    print(f"📦 Batch  : {batch_size} block/lần")
    print(f"🔑 API keys: {len(api_keys)} key(s)\n")

    # 2. Load prompt từ file gemini.txt
    base_prompt = load_prompt(prompt_file, target_language)

    # 3. Tạo context toàn cục (giống BaseTrans.__post_init__)
    context_block = ""
    if use_full_context:
        full_text = "\n".join([it["text"] for it in srt_list])
        context_block = _GLOBAL_CONTEXT_TEMPLATE.replace("{COMPLETE_SRT_TEXT}", full_text)
        logger.info(f"📝 Full context: {len(full_text)} ký tự")

    # 4. Khởi tạo Gemini caller
    caller = GeminiCaller(api_keys=api_keys, model=model, thinking_budget=thinking_budget)

    # 5. Chia batch và dịch  (giống BaseTrans._run_srt)
    translated_srt = copy.deepcopy(srt_list)
    batches = [srt_list[i : i + batch_size] for i in range(0, total, batch_size)]
    total_batches = len(batches)
    success_count = 0
    failed_count = 0
    t_start = time.time()

    for idx, batch in enumerate(batches):
        # Tạo SRT string cho batch (giống _run_srt)
        batch_srt_str = "\n\n".join(
            f"{item['line']}\n{item['time']}\n{item['text'].strip()}"
            for item in batch
        )

        # Build prompt hoàn chỉnh (giống _item_task)
        prompt_message = (
            base_prompt
            .replace("{batch_input}", batch_srt_str)
            .replace("{context_block}", context_block)
        )

        b_start = batch[0]["line"]
        b_end = batch[-1]["line"]
        print(
            f"🔄 [{idx + 1:>3}/{total_batches}] block {b_start:>5} → {b_end:<5}",
            end="  ",
            flush=True,
        )

        try:
            t0 = time.time()
            raw_result = caller.call(prompt_message)
            translated_text = parse_gemini_response(raw_result)
            elapsed_batch = time.time() - t0

            # Ghép kết quả vào đúng offset
            offset = idx * batch_size
            translated_batch = merge_translated_batch(translated_text, batch)
            for j, item in enumerate(translated_batch):
                translated_srt[offset + j]["text"] = item["text"]

            success_count += 1
            print(f"✅  {elapsed_batch:.1f}s")

        except Exception as e:
            failed_count += 1
            print(f"❌  {e}")
            logger.error(f"Batch {idx + 1} thất bại: {e}")

        if wait_sec > 0:
            time.sleep(wait_sec)

    # 6. Ghi file output
    output_content = srt_list_to_string(translated_srt)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(output_content, encoding="utf-8")

    elapsed_total = time.time() - t_start
    stats = {
        "total_blocks": total,
        "total_batches": total_batches,
        "success": success_count,
        "failed": failed_count,
        "elapsed": round(elapsed_total, 1),
        "output": output_file,
    }

    print(f"\n{'─'*50}")
    print(f"✅ Hoàn thành: {output_file}")
    print(f"   Batch: {success_count}/{total_batches} thành công | {failed_count} lỗi")
    print(f"   Thời gian: {elapsed_total:.1f}s")

    return stats
