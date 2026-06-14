from pathlib import Path
from typing import Iterable


GLOBAL_CONTEXT_TEMPLATE = """
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


def load_prompt(prompt_file: str, language: str) -> str:
    """Đọc prompt template, thay cả `{lang}` lẫn `{language}` bằng ngôn ngữ.

    Hợp nhất 2 loader cũ (translation dùng `{lang}`, punctuation dùng `{language}`).
    Prompt chỉ chứa 1 trong 2 placeholder → thay placeholder còn lại là no-op.
    """
    content = Path(prompt_file).read_text(encoding="utf-8", errors="ignore")
    return content.replace("{language}", language).replace("{lang}", language)


def build_global_context(srt_list: Iterable[dict]) -> tuple[str, str]:
    full_text = "\n".join([it["text"] for it in srt_list])
    context_block = GLOBAL_CONTEXT_TEMPLATE.replace("{COMPLETE_SRT_TEXT}", full_text)
    return context_block, full_text


def render_batch_prompt(
    *,
    base_prompt: str,
    batch_srt_str: str,
    context_block: str,
) -> str:
    return (
        base_prompt
        .replace("{batch_input}", batch_srt_str)
        .replace("{context_block}", context_block)
    )
