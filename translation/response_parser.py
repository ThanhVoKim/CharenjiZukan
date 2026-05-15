from llm_ai.tasks.response_parser import parse_tag_response


def parse_translation_response(result: str) -> str:
    """
    Trích xuất nội dung trong <TRANSLATE_TEXT>...</TRANSLATE_TEXT>.
    Bỏ <think>...</think> trước khi parse.
    Nếu không tìm thấy tag → raise RuntimeError (batch sẽ được giữ nguyên).
    """
    return parse_tag_response(result, "TRANSLATE_TEXT")
