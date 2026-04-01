import re

def parse_translation_response(result: str) -> str:
    """
    Trích xuất nội dung trong <TRANSLATE_TEXT>...</TRANSLATE_TEXT>.
    Bỏ <think>...</think> trước khi parse.
    Nếu không tìm thấy tag → raise RuntimeError (batch sẽ được giữ nguyên).
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
        f"Không tìm thấy <TRANSLATE_TEXT> trong response. "
        f"Model có thể không follow prompt format. "
        f"Preview: {result[:200]}..."
    )
