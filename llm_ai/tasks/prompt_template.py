from pathlib import Path
from typing import Mapping


def load_prompt_template(prompt_file: str) -> str:
    return Path(prompt_file).read_text(encoding="utf-8", errors="ignore")


def render_prompt_template(
    template: str,
    replacements: Mapping[str, str],
    *,
    strict: bool = True,
) -> str:
    """
    Render prompt bằng cách thay các placeholder literal.

    Ví dụ:
        replacements={"[Video Content]": "..."}
    """
    rendered = template
    for placeholder, value in replacements.items():
        if strict and placeholder not in rendered:
            raise ValueError(f"Không tìm thấy placeholder trong prompt template: {placeholder}")
        rendered = rendered.replace(placeholder, value)
    return rendered


def render_single_input_prompt(
    template: str,
    input_text: str,
    input_placeholder: str,
    *,
    strict: bool = True,
) -> str:
    return render_prompt_template(
        template,
        {input_placeholder: input_text},
        strict=strict,
    )
