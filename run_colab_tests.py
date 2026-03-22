#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_colab_tests.py — Test Runner cho CharenjiZukan trên Google Colab

Cách dùng:
    # Chạy toàn bộ test được enable trong matrix
    python run_colab_tests.py

    # Chỉ chạy test thuộc tag cụ thể
    python run_colab_tests.py --tags unit
    python run_colab_tests.py --tags integration
    python run_colab_tests.py --tags gpu

    # Chỉ chạy 1 test theo tên (substring match)
    python run_colab_tests.py --name "SRT Parser"
    python run_colab_tests.py --name "Native Video"

    # Xem danh sách test mà không chạy
    python run_colab_tests.py --list

    # Dùng file matrix khác
    python run_colab_tests.py --matrix path/to/other_matrix.yaml
"""

import argparse
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
except ImportError:
    print("❌ PyYAML chưa cài. Chạy: pip install pyyaml")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────

DEFAULT_MATRIX_PATH = "tests/test_matrix.yaml"
DEFAULT_REPORTS_DIR = "tests/test_reports"
DEFAULT_TIMEOUT_SEC = 120
SEPARATOR = "─" * 60
PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_path(path_value: str, project_root: Path = PROJECT_ROOT) -> Path:
    """
    Resolve đường dẫn từ CLI theo nguyên tắc:
    - Nếu absolute path: giữ nguyên (ví dụ: /content/test_reports).
    - Nếu relative path: resolve theo thư mục project chứa run_colab_tests.py.
    """
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


# ─────────────────────────────────────────────────────────────────────
# STREAMING SUBPROCESS — Hiển thị output realtime trên Colab
# ─────────────────────────────────────────────────────────────────────

def _stream_process(
    cmd: List[str],
    env: Dict[str, str],
    timeout_sec: int,
) -> tuple:
    """
    Chạy subprocess, stream stdout/stderr realtime ra console,
    đồng thời capture để ghi vào report.

    Returns:
        (returncode, stdout_captured, stderr_captured)
    """
    stdout_lines: List[str] = []
    stderr_lines: List[str] = []

    def _reader(pipe, collector: List[str], prefix: str = "") -> None:
        for line in iter(pipe.readline, ""):
            collector.append(line)
            print(f"{prefix}{line}", end="", flush=True)
        pipe.close()

    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line-buffered
        )

        stdout_thread = threading.Thread(target=_reader, args=(process.stdout, stdout_lines))
        stderr_thread = threading.Thread(target=_reader, args=(process.stderr, stderr_lines, "│ "))

        stdout_thread.start()
        stderr_thread.start()

        try:
            process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            return (
                -1,
                "".join(stdout_lines),
                f"⏰ TIMEOUT sau {timeout_sec}s — process bị kill.\n" + "".join(stderr_lines),
            )

        stdout_thread.join()
        stderr_thread.join()

        return (
            process.returncode,
            "".join(stdout_lines),
            "".join(stderr_lines),
        )

    except FileNotFoundError:
        msg = f"❌ Không tìm thấy lệnh '{cmd[0]}'. Kiểm tra pytest đã được cài chưa."
        return (127, "", msg)


# ─────────────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────

def _classify_result(returncode: int, stdout: str, stderr: str) -> str:
    """
    Phân loại kết quả chi tiết hơn returncode đơn giản.

    pytest exit codes:
        0  = all passed
        1  = some tests failed
        2  = interrupted (Ctrl+C, error before tests ran)
        3  = internal pytest error
        4  = command line usage error
        5  = no tests collected (all skipped hoặc wrong filter)
       -1  = timeout (custom)
    """
    if returncode == 0:
        if "no tests ran" in stdout.lower() or "no tests ran" in stderr.lower():
            return "NO_TESTS"
        if stdout.count(" passed") == 0 and "passed" not in stdout:
            return "ALL_SKIPPED"
        return "PASSED"
    if returncode == -1:
        return "TIMEOUT"
    if returncode == 5:
        return "NO_COLLECTION"
    if returncode == 1:
        # Phân biệt fail thật vs toàn skip
        if "failed" in stdout.lower():
            return "FAILED"
        if "error" in stdout.lower():
            return "ERROR"
        return "FAILED"
    return f"EXIT_{returncode}"


def _extract_pytest_summary(stdout: str) -> str:
    """Trích dòng summary cuối của pytest (dòng có ===)."""
    lines = stdout.strip().splitlines()
    summary_lines = [
        line for line in lines
        if line.strip().startswith("=") and ("passed" in line or "failed" in line
                                             or "error" in line or "warning" in line
                                             or "skipped" in line or "no tests ran" in line)
    ]
    return "\n".join(summary_lines) if summary_lines else "(Không tìm thấy dòng summary)"


def generate_markdown_report(
    test_config: Dict,
    cmd: List[str],
    returncode: int,
    stdout: str,
    stderr: str,
    result_type: str,
    duration_sec: float,
    report_path: Path,
) -> None:
    """Tạo file Markdown chi tiết để nạp cho AI Agent phân tích."""

    env_display = {k: v for k, v in test_config.get("env", {}).items()}
    pytest_summary = _extract_pytest_summary(stdout)

    # Chọn tiêu đề và màu theo kết quả
    status_icon = {
        "FAILED": "❌",
        "ERROR": "💥",
        "TIMEOUT": "⏰",
        "NO_COLLECTION": "🔍",
        "ALL_SKIPPED": "⏭️",
    }.get(result_type, "❓")

    md_content = f"""\
# {status_icon} Test Report: {test_config['name']}

| Thông tin      | Giá trị                                      |
| -------------- | -------------------------------------------- |
| **Thời gian**  | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |
| **File test**  | `{test_config['file']}`                       |
| **Kết quả**    | `{result_type}` (exit code: `{returncode}`)   |
| **Thời lượng** | `{duration_sec:.1f}s`                         |
| **Tags**       | `{', '.join(test_config.get('tags', []))}`    |

---

## 1. Lệnh đã chạy

```bash
{' '.join(cmd)}
```

## 2. Biến môi trường bổ sung

```json
{env_display if env_display else "{}"}
```

## 3. Pytest Summary

```
{pytest_summary}
```

## 4. Full Output (stdout)

```text
{stdout.strip() if stdout.strip() else "(Không có output)"}
```

## 5. Stderr / Traceback

```text
{stderr.strip() if stderr.strip() else "(Không có stderr)"}
```

---

## 6. Gợi ý cho AI Agent

Phân tích lỗi trong file test `{test_config['file']}`.

**Kết quả:** `{result_type}`

**Câu hỏi:**
- Traceback ở mục 5 chỉ ra lỗi ở dòng nào / hàm nào?
- Lỗi xuất phát từ code test hay từ code production (`utils/`, `cli/`, `video_subtitle_extractor/`)?
- Biến môi trường trong mục 2 có ảnh hưởng đến kết quả không?
- Đề xuất fix cụ thể với code snippet.
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md_content, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────
# BUILD PYTEST COMMAND
# ─────────────────────────────────────────────────────────────────────

def _build_pytest_cmd(test_config: Dict) -> List[str]:
    """Tổng hợp lệnh pytest từ config."""
    cmd = ["python", "-m", "pytest", test_config["file"]]

    # Thêm keyword filter nếu có
    keyword = test_config.get("keyword", "").strip()
    if keyword:
        cmd += ["-k", keyword]

    # Thêm marker filter nếu có
    markers = test_config.get("markers", [])
    if markers:
        cmd += ["-m", " and ".join(markers)]

    # Thêm các pytest args tùy chỉnh
    cmd += test_config.get("pytest_args", [])

    # Luôn thêm --tb=short nếu chưa có --tb flag
    if not any(arg.startswith("--tb") for arg in cmd):
        cmd += ["--tb=short"]

    # Luôn thêm --no-header để output gọn hơn
    if "--no-header" not in cmd:
        cmd += ["--no-header"]

    return cmd


# ─────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────

def load_matrix(matrix_path: str) -> List[Dict]:
    """Load và validate test matrix từ YAML."""
    path = Path(matrix_path)
    if not path.exists():
        print(f"❌ Không tìm thấy file matrix: {matrix_path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("tests", [])


def filter_tests(
    tests: List[Dict],
    tags: Optional[List[str]],
    name_filter: Optional[str],
) -> List[Dict]:
    """Lọc test theo tags và/hoặc name substring."""
    result = [t for t in tests if t.get("enabled", True)]

    if tags:
        result = [
            t for t in result
            if any(tag in t.get("tags", []) for tag in tags)
        ]

    if name_filter:
        result = [
            t for t in result
            if name_filter.lower() in t.get("name", "").lower()
        ]

    return result


def print_test_list(tests: List[Dict]) -> None:
    """In danh sách test không chạy."""
    print(f"\n📋 Danh sách test ({len(tests)} entries):\n")
    for i, t in enumerate(tests, 1):
        status = "✅" if t.get("enabled", True) else "⏭️ (disabled)"
        tags = ", ".join(t.get("tags", []))
        print(f"  {i:>2}. {status} [{tags}] {t['name']}")
        print(f"       → {t['file']}")
    print()


def run_all(
    tests: List[Dict],
    reports_dir: Path,
) -> Dict[str, List[str]]:
    """
    Chạy toàn bộ test đã được lọc.

    Returns:
        dict với keys: "passed", "failed", "skipped", "no_collection"
    """
    results: Dict[str, List[str]] = {
        "passed": [],
        "failed": [],
        "all_skipped": [],
        "no_collection": [],
        "timeout": [],
        "error": [],
    }

    print(f"\n🚀 CHẠY {len(tests)} TEST(S) — {datetime.now().strftime('%H:%M:%S')}")
    print(SEPARATOR)

    for i, test_config in enumerate(tests, 1):
        name = test_config["name"]
        print(f"\n[{i}/{len(tests)}] ⏳ {name}")
        print(f"  File   : {test_config['file']}")

        keyword = test_config.get("keyword", "")
        if keyword:
            print(f"  Filter : -k \"{keyword}\"")
        tags = test_config.get("tags", [])
        if tags:
            print(f"  Tags   : {', '.join(tags)}")
        print(SEPARATOR)

        # Merge env vars
        current_env = os.environ.copy()
        current_env.update(test_config.get("env", {}))

        # Build command
        cmd = _build_pytest_cmd(test_config)
        timeout = test_config.get("timeout_sec", DEFAULT_TIMEOUT_SEC)

        # Chạy với stream realtime
        t_start = datetime.now()
        returncode, stdout, stderr = _stream_process(cmd, current_env, timeout)
        duration = (datetime.now() - t_start).total_seconds()

        # Phân loại kết quả
        result_type = _classify_result(returncode, stdout, stderr)

        print(f"\n{SEPARATOR}")

        if result_type == "PASSED":
            print(f"✅ [PASSED] {name} ({duration:.1f}s)")
            results["passed"].append(name)

        elif result_type == "ALL_SKIPPED":
            print(f"⏭️ [ALL SKIPPED] {name} ({duration:.1f}s)")
            results["all_skipped"].append(name)

        elif result_type == "NO_COLLECTION":
            print(f"🔍 [NO TEST COLLECTED] {name} — kiểm tra lại --keyword/-k filter")
            results["no_collection"].append(name)

        else:
            # Tất cả trường hợp fail: FAILED, ERROR, TIMEOUT, EXIT_*
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            report_path = reports_dir / f"failed_{safe_name}.md"

            generate_markdown_report(
                test_config=test_config,
                cmd=cmd,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                result_type=result_type,
                duration_sec=duration,
                report_path=report_path,
            )

            icon = {"FAILED": "❌", "ERROR": "💥", "TIMEOUT": "⏰"}.get(result_type, "❓")
            print(f"{icon} [{result_type}] {name} ({duration:.1f}s)")
            print(f"   📄 Report: {report_path}")
            results["failed"].append(name)

    return results


def print_summary(results: Dict[str, List[str]], reports_dir: Path) -> None:
    """In tổng kết sau khi chạy xong."""
    total = sum(len(v) for v in results.values())
    passed = len(results["passed"])
    failed = len(results["failed"])
    skipped = len(results["all_skipped"])
    no_col = len(results["no_collection"])

    print(f"\n{'═' * 60}")
    print(f"  TỔNG KẾT: {total} tests")
    print(f"  ✅ Passed       : {passed}")
    print(f"  ❌ Failed/Error : {failed}")
    print(f"  ⏭️  All Skipped  : {skipped}")
    print(f"  🔍 No Collection: {no_col}")
    print(f"{'═' * 60}")

    if failed > 0:
        reports_dir_display = reports_dir.as_posix().rstrip("/") + "/"
        print(f"\n📁 File report lỗi tại thư mục: `{reports_dir_display}`")
        print("   Mở file .md tương ứng và gửi cho AI Agent để phân tích.\n")
        for name in results["failed"]:
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            print(f"   → failed_{safe_name}.md")


# ─────────────────────────────────────────────────────────────────────
# CLI ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_colab_tests",
        description="Test Runner cho CharenjiZukan trên Google Colab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python run_colab_tests.py                        # Chạy tất cả enabled tests
  python run_colab_tests.py --tags unit            # Chỉ unit tests
  python run_colab_tests.py --tags integration     # Chỉ integration tests
  python run_colab_tests.py --tags gpu             # Chỉ GPU tests
  python run_colab_tests.py --name "SRT Parser"   # Tìm theo tên
  python run_colab_tests.py --list                 # Xem danh sách không chạy
        """,
    )
    parser.add_argument(
        "--matrix",
        default=DEFAULT_MATRIX_PATH,
        help=f"Đường dẫn file test_matrix.yaml (mặc định: {DEFAULT_MATRIX_PATH})",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        metavar="TAG",
        help="Lọc test theo tag (unit / integration / ffmpeg / gpu / native_ocr)",
    )
    parser.add_argument(
        "--name",
        metavar="SUBSTR",
        help="Lọc test theo tên (substring, không phân biệt hoa thường)",
    )
    parser.add_argument(
        "--reports-dir",
        default=DEFAULT_REPORTS_DIR,
        help=(
            f"Thư mục lưu báo cáo lỗi (mặc định: {DEFAULT_REPORTS_DIR}; "
            "đường dẫn tương đối được resolve theo thư mục project chứa run_colab_tests.py)"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="In danh sách test mà không chạy",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    matrix_path = _resolve_path(args.matrix)
    reports_dir = _resolve_path(args.reports_dir)

    all_tests = load_matrix(str(matrix_path))
    tests_to_run = filter_tests(
        all_tests,
        tags=args.tags,
        name_filter=args.name,
    )

    if args.list:
        print_test_list(tests_to_run)
        return

    if not tests_to_run:
        print("⚠️  Không có test nào khớp với filter. Kiểm tra --tags hoặc --name.")
        print_test_list(all_tests)
        return

    results = run_all(tests_to_run, reports_dir=reports_dir)
    print_summary(results, reports_dir=reports_dir)

    # Exit code phản ánh kết quả để có thể dùng trong CI
    failed_count = len(results["failed"])
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
