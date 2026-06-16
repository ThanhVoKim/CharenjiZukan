#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/cli/test_blend_overlay_parallel.py
=========================================
Test cho cli/blend_overlay_parallel.py — phủ blend video song song, frame-accurate.

Cấu trúc layers:
  Layer 1 — Unit: argument parser & default; plan_segments (chia đoạn theo bội số
            độ dài blend, dồn dư vào đoạn cuối, edge-cases); resolve_tasks
            (--video/--output vs --task-file JSON); build_segment_cmd (kết cấu lệnh
            ffmpeg: -frames:v chốt cứng, -an, giữ nguyên W×H, blend restart mỗi đoạn).
  Layer 2 — Component: concat_and_mux dựng list-file + lệnh mux audio gốc (mock
            subprocess, không gọi ffmpeg thật).

Cách chạy:
    pytest tests/cli/test_blend_overlay_parallel.py -v -k "Layer1"
    pytest tests/cli/test_blend_overlay_parallel.py -v -k "Layer2"

Không cần GPU/FFmpeg: chỉ kiểm tra logic Python thuần và cấu trúc lệnh (không exec).
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cli.blend_overlay_parallel as bop


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_Parser:
    """Argument parser: cờ mới sau khi bỏ width/height/tmp-dir/dry-run."""

    def test_parser_exists(self):
        assert bop.build_parser() is not None

    def test_has_expected_args(self):
        actions = {a.dest for a in bop.build_parser()._actions}
        assert {"video", "output", "task_file", "blend",
                "workers", "mode", "opacity", "keep_tmp", "verbose"} <= actions

    def test_removed_args_gone(self):
        """width / height / tmp_dir / dry_run đã bị xoá khỏi parser."""
        actions = {a.dest for a in bop.build_parser()._actions}
        assert "width" not in actions
        assert "height" not in actions
        assert "tmp_dir" not in actions
        assert "dry_run" not in actions

    def test_defaults(self):
        args = bop.build_parser().parse_args(
            ["--video", "in.mp4", "--output", "out.mp4", "--blend", "b.mp4"]
        )
        assert args.workers == 4
        assert args.mode == "subtract"
        assert args.opacity == 0.9
        assert args.keep_tmp is False
        assert args.verbose is False

    def test_blend_required(self):
        with pytest.raises(SystemExit):
            bop.build_parser().parse_args(["--video", "in.mp4", "--output", "out.mp4"])


class TestLayer1_PlanSegments:
    """Chia đoạn theo bội số nguyên của độ dài blend (L)."""

    def test_example_10_5_loops_4_workers(self):
        """Ví dụ thiết kế: video 10.5·L, 4 worker → [3,3,2,2.5] loops."""
        L = 1000
        total = round(10.5 * L)  # 10500
        segs = bop.plan_segments(total, L, 4)
        assert [n for _, n in segs] == [3000, 3000, 2000, 2500]

    def test_segments_cover_all_frames_exactly(self):
        L = 1000
        total = 10500
        segs = bop.plan_segments(total, L, 4)
        assert sum(n for _, n in segs) == total
        # Mốc start liền mạch, không chồng/hở
        cur = 0
        for s, n in segs:
            assert s == cur
            cur += n
        assert cur == total

    def test_internal_joints_are_multiples_of_L(self):
        """Mọi mốc nối nội bộ (start của đoạn != đoạn đầu) chia hết cho L → blend liền mạch."""
        L = 1000
        segs = bop.plan_segments(10500, L, 4)
        for s, _ in segs:
            assert s % L == 0

    def test_remainder_goes_to_last_segment(self):
        """Phần dư lẻ (0.5L) nằm ở đoạn cuối, các đoạn trước là bội số nguyên của L."""
        L = 1000
        segs = bop.plan_segments(10500, L, 4)
        for _, n in segs[:-1]:
            assert n % L == 0
        assert segs[-1][1] % L != 0  # đoạn cuối chứa phần dư

    def test_blend_longer_than_video_single_segment(self):
        assert bop.plan_segments(500, 1000, 4) == [(0, 500)]

    def test_workers_one_single_segment(self):
        assert bop.plan_segments(10500, 1000, 1) == [(0, 10500)]

    def test_fewer_loops_than_workers(self):
        """3 loop / 4 worker → chỉ 3 đoạn, mỗi đoạn 1 loop."""
        segs = bop.plan_segments(3000, 1000, 4)
        assert segs == [(0, 1000), (1000, 1000), (2000, 1000)]

    def test_exact_multiple_no_remainder(self):
        """Video đúng 8·L / 4 worker → [2,2,2,2] loops, không dư."""
        segs = bop.plan_segments(8000, 1000, 4)
        assert [n for _, n in segs] == [2000, 2000, 2000, 2000]
        assert sum(n for _, n in segs) == 8000


class TestLayer1_ResolveTasks:
    """resolve_tasks: --video/--output trực tiếp hoặc --task-file JSON."""

    def _args(self, **kw):
        base = dict(video=None, output=None, task_file=None)
        base.update(kw)
        return type("A", (), base)()

    def test_single_video_output(self):
        tasks = bop.resolve_tasks(self._args(video="in.mp4", output="out.mp4"))
        assert tasks == [("in.mp4", "out.mp4")]

    def test_missing_video_or_output_raises(self):
        with pytest.raises(ValueError):
            bop.resolve_tasks(self._args(video="in.mp4"))

    def test_task_file_list(self, tmp_path: Path):
        tf = tmp_path / "tasks.json"
        tf.write_text(json.dumps([
            {"input": "a.mp4", "output": "a_out.mp4"},
            {"input": "b.mp4", "output": "b_out.mp4"},
        ]), encoding="utf-8")
        tasks = bop.resolve_tasks(self._args(task_file=str(tf)))
        assert tasks == [("a.mp4", "a_out.mp4"), ("b.mp4", "b_out.mp4")]

    def test_task_file_missing_keys_raises(self, tmp_path: Path):
        tf = tmp_path / "bad.json"
        tf.write_text(json.dumps([{"input": "a.mp4"}]), encoding="utf-8")
        with pytest.raises(ValueError):
            bop.resolve_tasks(self._args(task_file=str(tf)))

    def test_task_file_not_array_raises(self, tmp_path: Path):
        tf = tmp_path / "obj.json"
        tf.write_text(json.dumps({"input": "a.mp4", "output": "b.mp4"}), encoding="utf-8")
        with pytest.raises(ValueError):
            bop.resolve_tasks(self._args(task_file=str(tf)))

    def test_task_file_empty_raises(self, tmp_path: Path):
        tf = tmp_path / "empty.json"
        tf.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError):
            bop.resolve_tasks(self._args(task_file=str(tf)))


class TestLayer1_BuildSegmentCmd:
    """Kết cấu lệnh ffmpeg cho 1 đoạn — bảo chứng các tính chất an toàn timeline."""

    def _cmd(self, start_frame, num_frames, is_last=False, w=1920, h=1080):
        return bop.build_segment_cmd(
            "in.mp4", "blend.mp4", "seg.mp4",
            start_frame=start_frame, num_frames=num_frames, is_last=is_last,
            fps_str="30/1", fps_float=30.0, width=w, height=h,
            blend_mode="subtract", opacity=0.9,
        )

    def test_frames_v_hard_caps_count(self):
        """-frames:v = num_frames → chặn cứng, không phụ thuộc EOF blend (không render vô hạn)."""
        cmd = self._cmd(3000, 3000)
        assert "-frames:v" in cmd
        assert cmd[cmd.index("-frames:v") + 1] == "3000"

    def test_audio_disabled_per_segment(self):
        """Mỗi đoạn -an (audio ghép sau ở concat_and_mux)."""
        assert "-an" in self._cmd(0, 1000)

    def test_blend_stream_loop_infinite(self):
        cmd = self._cmd(0, 1000)
        assert "-stream_loop" in cmd
        assert cmd[cmd.index("-stream_loop") + 1] == "-1"

    def test_hevc_nvenc_used(self):
        cmd = self._cmd(0, 1000)
        assert "hevc_nvenc" in cmd
        assert "-video_track_timescale" in cmd

    def test_keeps_original_resolution(self):
        """Không có scale ép 1920x1080 lên MAIN; main giữ nguyên W×H (chỉ setsar/format)."""
        fc = self._cmd(0, 1000, w=1280, h=720)[
            self._cmd(0, 1000, w=1280, h=720).index("-filter_complex") + 1
        ]
        # main chain không chứa scale; blend chain mới scale về W×H gốc
        main_part = fc.split(";")[0]
        assert "scale=" not in main_part
        blend_part = fc.split(";")[1]
        assert "scale=1280:720" in blend_part

    def test_last_segment_has_tail_pad(self):
        fc_last = self._cmd(8000, 2500, is_last=True)[
            self._cmd(8000, 2500, is_last=True).index("-filter_complex") + 1
        ]
        assert "tpad=stop_mode=clone" in fc_last

    def test_non_last_segment_no_tail_pad(self):
        fc_mid = self._cmd(3000, 3000, is_last=False)[
            self._cmd(3000, 3000, is_last=False).index("-filter_complex") + 1
        ]
        assert "tpad=" not in fc_mid

    def test_blend_mode_and_opacity_applied(self):
        fc = self._cmd(0, 1000)[self._cmd(0, 1000).index("-filter_complex") + 1]
        assert "all_mode='subtract'" in fc
        assert "all_opacity=0.9" in fc

    def test_fast_seek_before_input(self):
        """-ss đặt TRƯỚC -i main (fast seek), lùi ~2s so với điểm cắt."""
        cmd = self._cmd(3000, 3000)  # start = 100s
        ss_idx = cmd.index("-ss")
        i_idx = cmd.index("-i")
        assert ss_idx < i_idx
        assert float(cmd[ss_idx + 1]) == pytest.approx(98.0, abs=0.01)


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — COMPONENT TESTS (mock subprocess, không exec ffmpeg)
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_FmtDuration:
    """_fmt_duration: chuyển giây → HH:MM:SS."""

    def test_zero(self):
        assert bop._fmt_duration(0) == "00:00:00"

    def test_seconds_only(self):
        assert bop._fmt_duration(45) == "00:00:45"

    def test_minutes(self):
        assert bop._fmt_duration(90) == "00:01:30"

    def test_hours(self):
        assert bop._fmt_duration(3661) == "01:01:01"

    def test_float_truncated(self):
        assert bop._fmt_duration(3599.9) == "00:59:59"


class TestLayer2_ConcatAndMux:
    """concat_and_mux: dựng list-file + lệnh ghép audio gốc copy (audio zero-drift)."""

    def test_builds_concat_list_and_copy_cmd(self, tmp_path: Path, monkeypatch):
        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            # list-file phải còn tồn tại lúc gọi ffmpeg
            li = cmd[cmd.index("-i") + 1]
            captured["list_content"] = Path(li).read_text(encoding="utf-8")
            class R:  # noqa: N801
                returncode = 0
                stdout = ""
                stderr = ""
            return R()

        monkeypatch.setattr(bop.subprocess, "run", fake_run)

        segs = [str(tmp_path / "seg_000.mp4"), str(tmp_path / "seg_001.mp4")]
        for s in segs:
            Path(s).write_bytes(b"x")
        out = str(tmp_path / "final.mp4")

        bop.concat_and_mux(segs, "orig.mp4", out)

        cmd = captured["cmd"]
        # Audio copy từ input thứ 2 (video gốc), video copy từ concat
        assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "copy"
        assert "-c:a" in cmd and cmd[cmd.index("-c:a") + 1] == "copy"
        assert "0:v" in cmd  # video từ concat demuxer
        assert "1:a?" in cmd  # audio gốc (optional)
        # list-file liệt kê đủ 2 segment
        assert "seg_000.mp4" in captured["list_content"]
        assert "seg_001.mp4" in captured["list_content"]

    def test_list_file_cleaned_up(self, tmp_path: Path, monkeypatch):
        """File .concat.txt tạm bị xoá sau khi mux xong."""
        monkeypatch.setattr(bop.subprocess, "run",
                            lambda cmd, **kw: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
        segs = [str(tmp_path / "seg_000.mp4")]
        Path(segs[0]).write_bytes(b"x")
        out = str(tmp_path / "final.mp4")
        bop.concat_and_mux(segs, "orig.mp4", out)
        assert not Path(out).with_suffix(".concat.txt").exists()
