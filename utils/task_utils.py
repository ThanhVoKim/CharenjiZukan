import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

def resolve_cli_tasks(
    task_file: Optional[str],
    input_file: Optional[str],
    output_path: Optional[str],
    default_ext: str,
    default_out_dir: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """
    Xử lý chung cho --task-file và --input/--output của các CLI.

    Quy tắc xử lý output:
    - Nếu output có phần mở rộng (extension, vd: .srt, .wav, .mp3), coi nó là đường dẫn tới 1 file.
    - Nếu output KHÔNG có phần mở rộng, hoặc nó là một thư mục đã tồn tại, coi nó là thư mục.
      File kết quả sẽ được sinh ra bên trong thư mục này với tên `[tên_file_input_gốc]{default_ext}`.
    - Nếu không có output, mặc định dùng default_out_dir (nếu được cấp) hoặc thư mục chứa input_file.

    Args:
        task_file: Đường dẫn file JSON (vd: `[{"input": "in.mp4", "output": "out.srt"}]`).
        input_file: Đường dẫn file input (nếu không dùng task_file).
        output_path: Đường dẫn file hoặc thư mục output (nếu không dùng task_file).
        default_ext: Phần mở rộng mặc định bắt buộc, bao gồm cả dấu chấm (vd: ".srt", ".wav").
        default_out_dir: Thư mục output mặc định (nếu không truyền output_path).

    Returns:
        Danh sách các task đã được chuẩn hóa đường dẫn output:
        [{"input": "/full/path/in.mp4", "output": "/full/path/out.srt"}]
    """
    tasks = []

    def _normalize_output(inp_path: Path, out_str: Optional[str]) -> str:
        if not out_str:
            if default_out_dir:
                out_dir = Path(default_out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                return str(out_dir / f"{inp_path.stem}{default_ext}")
            else:
                return str(inp_path.parent / f"{inp_path.stem}{default_ext}")

        out_p = Path(out_str)
        # Nếu là thư mục tồn tại, HOẶC không có suffix (như "my_folder")
        if out_p.is_dir() or not out_p.suffix:
            return str(out_p / f"{inp_path.stem}{default_ext}")
        return str(out_p)

    if task_file:
        try:
            with open(task_file, "r", encoding="utf-8") as f:
                raw_tasks = json.load(f)
            
            if not isinstance(raw_tasks, list):
                logger.error("Task file JSON phải chứa một mảng (list) các task.")
                sys.exit(1)
                
            for i, task in enumerate(raw_tasks):
                if not isinstance(task, dict) or "input" not in task:
                    logger.error(f"Task thứ {i} không hợp lệ hoặc thiếu 'input': {task}")
                    sys.exit(1)
                
                inp = Path(task["input"])
                # Lưu ý file json của --task-file input có thể nhận bất cứ file nào (không kiểm tra extension)
                # Normalize output
                task["output"] = _normalize_output(inp, task.get("output"))
                tasks.append(task)

            logger.info(f"Loaded {len(tasks)} tasks từ {task_file}")
            
        except Exception as e:
            logger.error(f"Lỗi đọc task file JSON: {e}")
            sys.exit(1)
            
    elif input_file:
        inp = Path(input_file)
        if not inp.exists():
            logger.error(f"File input không tồn tại: {inp}")
            sys.exit(1)

        out_str = _normalize_output(inp, output_path)
        tasks.append({"input": str(inp), "output": out_str})
    else:
        raise ValueError("Phải cung cấp --input hoặc --task-file")

    return tasks


def resolve_output_dir_and_stem(task: Dict[str, str]) -> Tuple[Path, str]:
    """
    Xác định thư mục đầu ra và file stem gốc từ một task đã được chuẩn hóa bởi `resolve_cli_tasks`.
    
    Vì `task["output"]` sau khi qua `resolve_cli_tasks` luôn là một đường dẫn file đầy đủ
    (có extension), hàm này chỉ đơn giản lấy `parent` làm thư mục và `stem` làm tên file.

    Returns:
        (output_dir, stem)
    """
    out_path = Path(task["output"])
    return out_path.parent, out_path.stem
