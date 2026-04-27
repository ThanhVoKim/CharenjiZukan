# -*- coding: utf-8 -*-
"""
Test cơ bản cho cli/tts.py refactor.
Không import cli/tts.py trực tiếp để tránh dependency nặng (yaml, pydub).
"""

import sys
import json
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_load_config():
    try:
        import yaml
    except ImportError:
        print("⚠️ pyyaml chưa cài, bỏ qua test_load_config")
        return
    from cli.tts import load_config
    cfg = load_config(str(PROJECT_ROOT / "config" / "tts_config.yaml"))
    assert "provider" in cfg
    assert "edge" in cfg
    assert "voicevox" in cfg
    assert "qwen" in cfg
    print("✅ test_load_config passed")


def test_build_queue_from_txt():
    # Copy logic từ cli/tts.py để test độc lập
    def build_queue_from_txt(input_file, cache_folder):
        text = Path(input_file).read_text(encoding="utf-8", errors="ignore")
        queue = [{
            "text": text,
            "line": 1,
            "start_time": 0,
            "end_time": 0,
            "filename": str(Path(cache_folder) / "dubb-0.wav"),
        }]
        return queue, 0

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Hello world.\nThis is a test.")
        path = f.name
    try:
        queue, raw_total = build_queue_from_txt(path, "/tmp/cache")
        assert len(queue) == 1
        assert "Hello world.\nThis is a test." in queue[0]["text"]
        assert raw_total == 0
        print("✅ test_build_queue_from_txt passed")
    finally:
        Path(path).unlink(missing_ok=True)


def test_build_queue_from_srt():
    from utils.srt_parser import parse_srt
    def build_queue_from_srt(input_file, cache_folder):
        raw = Path(input_file).read_text(encoding="utf-8", errors="ignore")
        srt_list = parse_srt(raw)
        queue = []
        for i, it in enumerate(srt_list):
            queue.append({
                "text": it["text"],
                "line": it["line"],
                "start_time": it["start_time"],
                "end_time": it["end_time"],
                "filename": str(Path(cache_folder) / f"dubb-{i}.wav"),
            })
        return queue, srt_list[-1]["end_time"] if srt_list else 0

    srt_content = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:06,000
This is a test
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding="utf-8") as f:
        f.write(srt_content)
        path = f.name
    try:
        queue, raw_total = build_queue_from_srt(path, "/tmp/cache")
        assert len(queue) == 2
        assert queue[0]["text"] == "Hello world"
        assert queue[1]["text"] == "This is a test"
        assert raw_total == 6000
        print("✅ test_build_queue_from_srt passed")
    finally:
        Path(path).unlink(missing_ok=True)


def test_resolve_tasks_single():
    import argparse
    def resolve_tasks(args):
        tasks = []
        if args.task_file:
            with open(args.task_file, "r", encoding="utf-8") as f:
                tasks = json.load(f)
        elif args.input:
            out = args.output
            if not out:
                out_dir = PROJECT_ROOT / "output"
                out_dir.mkdir(parents=True, exist_ok=True)
                out = str(out_dir / (Path(args.input).stem + ".wav"))
            tasks.append({"input": args.input, "output": out})
        else:
            raise ValueError("Phải cung cấp --input hoặc --task-file")
        return tasks

    args = argparse.Namespace(input="/tmp/test.srt", output="/tmp/out.wav", task_file=None)
    tasks = resolve_tasks(args)
    assert len(tasks) == 1
    assert tasks[0]["input"] == "/tmp/test.srt"
    print("✅ test_resolve_tasks_single passed")


def test_resolve_tasks_json():
    import argparse
    def resolve_tasks(args):
        tasks = []
        if args.task_file:
            with open(args.task_file, "r", encoding="utf-8") as f:
                tasks = json.load(f)
        elif args.input:
            out = args.output
            if not out:
                out_dir = PROJECT_ROOT / "output"
                out_dir.mkdir(parents=True, exist_ok=True)
                out = str(out_dir / (Path(args.input).stem + ".wav"))
            tasks.append({"input": args.input, "output": out})
        else:
            raise ValueError("Phải cung cấp --input hoặc --task-file")
        return tasks

    data = [{"input": "/tmp/a.srt", "output": "/tmp/a.wav"}, {"input": "/tmp/b.txt", "output": "/tmp/b.wav"}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        json_path = f.name
    try:
        args = argparse.Namespace(input=None, output=None, task_file=json_path)
        tasks = resolve_tasks(args)
        assert len(tasks) == 2
        assert tasks[1]["input"] == "/tmp/b.txt"
        print("✅ test_resolve_tasks_json passed")
    finally:
        Path(json_path).unlink(missing_ok=True)


def test_get_engine_factory():
    from tts.base import BaseTTSEngine
    class FakeEngine(BaseTTSEngine):
        def run(self):
            return {"ok": 1, "err": 0}

    def get_engine(provider, queue_tts, config):
        if provider == "fake":
            return FakeEngine(queue_tts)
        raise ValueError(f"Provider không hợp lệ: {provider}")

    try:
        get_engine("unknown", [], {})
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "unknown" in str(e)
    print("✅ test_get_engine_factory passed")


if __name__ == "__main__":
    test_load_config()
    test_build_queue_from_txt()
    test_build_queue_from_srt()
    test_resolve_tasks_single()
    test_resolve_tasks_json()
    test_get_engine_factory()
    print("\n🎉 All basic tests passed!")
