from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_ai.base import BaseLLMProvider
from llm_ai.tasks.output_writer import write_output_file
from llm_ai.tasks.prompt_template import load_prompt_template, render_single_input_prompt
from llm_ai.tasks.response_parser import parse_task_response


@dataclass
class GenericTextTaskConfig:
    task_name: str
    prompt_file: str
    input_placeholder: str
    default_ext: str = ".md"
    output_parser: str | dict[str, Any] | None = "raw"
    prompt_strict: bool = True
    input_encoding: str = "utf-8"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenericTextTaskConfig":
        return cls(
            task_name=str(data.get("task_name", "generic_text")),
            prompt_file=str(data.get("prompt_file", "")),
            input_placeholder=str(data.get("input_placeholder", data.get("placeholder", "[Content]"))),
            default_ext=str(data.get("default_ext", ".md")),
            output_parser=data.get("output_parser", data.get("parser", "raw")),
            prompt_strict=bool(data.get("prompt_strict", True)),
            input_encoding=str(data.get("input_encoding", "utf-8")),
        )


def load_task_config(config_path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("pyyaml chưa cài. Chạy: pip install PyYAML>=6.0") from exc

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Task config không tồn tại: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_generic_text_task(
    *,
    input_file: str,
    output_file: str,
    provider: BaseLLMProvider,
    task_config: GenericTextTaskConfig | dict[str, Any],
    prompt_file: str | None = None,
    input_placeholder: str | None = None,
) -> dict[str, Any]:
    cfg = (
        task_config
        if isinstance(task_config, GenericTextTaskConfig)
        else GenericTextTaskConfig.from_dict(task_config)
    )

    resolved_prompt_file = prompt_file or cfg.prompt_file
    resolved_placeholder = input_placeholder or cfg.input_placeholder

    input_text = Path(input_file).read_text(encoding=cfg.input_encoding, errors="ignore")
    template = load_prompt_template(resolved_prompt_file)
    prompt = render_single_input_prompt(
        template,
        input_text,
        resolved_placeholder,
        strict=cfg.prompt_strict,
    )

    raw_response = provider.call(prompt)
    parsed_response = parse_task_response(raw_response, cfg.output_parser)
    write_output_file(output_file, parsed_response)

    return {
        "task_name": cfg.task_name,
        "input": input_file,
        "output": output_file,
        "provider": provider.name,
        "input_chars": len(input_text),
        "prompt_chars": len(prompt),
        "output_chars": len(parsed_response),
    }
