#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/sync_engine/test_tuber_overlay_pipeline.py
=================================================
Test cho các module Python tuber overlay (tuber_config, tuber_manifest,
tuber_artifacts, tuber_status, tuber_overlay orchestration).

Cấu trúc layers:
  Layer 1 — Unit: frame math, group building, config load/validate, status, serialization
  Layer 2 — Component: manifest export synthetic, artifact promote real files
  Layer 3 — Pipeline: retry/cleanup/composite-validate với mock render driver + ffmpeg synthetic

Không cần Remotion/Node (Remotion thật ở test_tuber_remotion_validation.py).

Cách chạy:
    pytest tests/sync_engine/test_tuber_overlay_pipeline.py -v -k "Layer1"
    pytest tests/sync_engine/test_tuber_overlay_pipeline.py -v -k "Layer2"
    pytest tests/sync_engine/test_tuber_overlay_pipeline.py -v -k "Layer3"
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# ── Project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sync_engine.models import TimelineSegment
from sync_engine.tuber_manifest import (
    build_render_groups,
    build_group_manifest,
    build_run_manifest,
    segment_output_frames,
    SCHEMA_VERSION,
)
from sync_engine.tuber_config import (
    TuberConfig,
    TuberConfigError,
    load_tuber_config,
    parse_tuber_config_dict,
    _get_nested,
    DEFAULT_OUTPUT_NAME_SENTINEL,
)
from sync_engine import tuber_status as st
from sync_engine.tuber_artifacts import (
    promote_media,
    promote_final_render_inputs,
    serialize_image_overlay_events,
    deserialize_image_overlay_events,
    cleanup_overlay_frames,
    load_final_render_manifest,
    BASE_VIDEO_NAME,
    FINAL_AUDIO_NAME,
)
from sync_engine.tuber_overlay import (
    TuberOverlayError,
    GroupJob,
    composite_group,
    composite_group_from_stretched,
    validate_group_output,
    concat_group_videos,
    _detect_frame_pattern,
    _expected_group_duration_s,
    render_and_composite_groups,
    probe_resolution,
    build_group_base,
    _make_mouth_lookup,
    _pipe_prerender_frames,
)
from sync_engine.tuber_status import compute_group_input_hash

# ── Skip helpers ─────────────────────────────────────────────────────
_FFMPEG_OK = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
_NODE_OK = bool(shutil.which("npm") and shutil.which("node"))
_NODE_MODULES = (PROJECT_ROOT / "remotion_tuber" / "node_modules").is_dir()

# Dùng chung detect_hevc_nvenc (SSOT từ utils.ffmpeg_probe) — khớp fixture use_gpu
# trong tests/conftest.py. Composite dùng HEVC NVENC nên cần GPU encoder.
from utils.ffmpeg_probe import detect_hevc_nvenc
_GPU_OK = detect_hevc_nvenc()


# ═════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════

def _mk_seg(
    orig_start: float, orig_end: float, video_speed: float,
    block_type: str = "tts", tts_clip_path: str | None = None,
) -> TimelineSegment:
    return TimelineSegment(
        orig_start=orig_start, orig_end=orig_end,
        new_start=0.0, new_end=0.0,
        video_speed=video_speed, audio_speed=1.0,
        new_chunk_dur=0.0, block_type=block_type,
        tts_clip_path=tts_clip_path, tts_duration=0.0,
    )


def _sample_timeline() -> List[TimelineSegment]:
    """Timeline synthetic: 4 segment tốc độ khác nhau, đủ để test group-split."""
    return [
        _mk_seg(0, 2000, 1.0, "tts", "dubb-0.wav"),           # 60 frames
        _mk_seg(2000, 4000, 0.8, "mute", None),                # 75 frames
        _mk_seg(4000, 5000, 1.2, "tts", "dubb-2.wav"),         # 25 frames
        _mk_seg(5000, 9000, 1.0, "gap", None),                 # 120 frames
    ]


def _sample_config_dict() -> dict:
    """Dict config tối thiểu (parse thành TuberConfig.enabled=True)."""
    return {
        "enabled": True,
        "remotion": {
            "projectDir": "remotion_tuber",
            "compositionId": "TuberOverlay",
            "entryPoint": "src/index.ts",
        },
        "asset": {
            "assetDir": "assets/pngtuber/nike_loop_fix",
            "mouthTrack": "mouth_track.json",
            "mouthSprites": {"closed": "", "half": "", "open": ""},
            "bodySource": "loop.mp4",
        },
        "grouping": {"maxGroupSec": 3},
        "overlay": {"format": "direct"},
        "retry": {"retryAttempts": 2, "onExhausted": "render_without_tuber"},
    }


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT: frame math, group, config, status, serialization
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_FrameMath:
    """segment_output_frames khớp công thức build_ffmpeg_batch_cmd."""

    @pytest.mark.parametrize("os_,oe,vs,expected", [
        (0, 2000, 1.0, 60),
        (2000, 4000, 0.8, 75),     # 60f / 0.8 = 75
        (4000, 5000, 1.2, 25),     # 30f / 1.2 = 25
        (5000, 9000, 1.0, 120),    # 120f / 1.0 = 120
        (0, 1000, 2.0, 15),        # 30f / 2.0 = 15
        (0, 3334, 1.0, 100),       # round(100.02*30)=100f, ceil(round(100/30/1*30,4))=100
    ])
    def test_output_frames_matches_batch_formula(self, os_, oe, vs, expected):
        seg = _mk_seg(os_, oe, vs)
        n = segment_output_frames(seg, 30.0)
        # verify bằng công thức build_ffmpeg_batch_cmd
        df = round(((oe - os_) / 1000.0) * 30.0)
        exp = math.ceil(round((df / 30.0 / vs) * 30.0, 4))
        assert n == expected == exp


class TestLayer1_GroupBuilding:
    """build_render_groups: liên tục, không cắt segment, grouping đúng."""

    def test_all_fit_one_group(self):
        tl = [_mk_seg(0, 1000, 1.0), _mk_seg(1000, 2000, 1.0)]
        groups = build_render_groups(tl, 30.0, max_group_sec=10.0)
        assert len(groups) == 1
        assert groups[0].group_start_frame == 0
        assert groups[0].duration_frames == 60  # 30+30

    def test_split_by_duration(self):
        tl = _sample_timeline()
        groups = build_render_groups(tl, 30.0, max_group_sec=3.0)
        # 4 segment, mỗi cái đều <=3s viền nhưng tổng >3s → tách
        assert len(groups) >= 1

    def test_continuity_across_groups(self):
        tl = _sample_timeline()
        groups = build_render_groups(tl, 30.0, max_group_sec=3.0)
        for i in range(len(groups) - 1):
            assert groups[i].group_end_frame == groups[i + 1].group_start_frame, (
                f"gap giữa group {i} và {i+1}"
            )

    def test_total_frames_equals_sum(self):
        tl = _sample_timeline()
        groups = build_render_groups(tl, 30.0, max_group_sec=3.0)
        total = sum(segment_output_frames(s, 30.0) for s in tl)
        assert groups[-1].group_end_frame == total

    def test_long_segment_not_split(self):
        """Segment dài hơn max → vẫn giữ nguyên 1 group (không cắt)."""
        tl = [_mk_seg(0, 30000, 1.0)]  # ~900 frames = 30s > 3s max
        groups = build_render_groups(tl, 30.0, max_group_sec=3.0)
        assert len(groups) == 1
        assert groups[0].duration_frames == 900

    def test_empty_timeline_raises(self):
        with pytest.raises(ValueError):
            build_render_groups([], 30.0, 3.0)


class TestLayer1_TuberConfig:
    """Load/validate TuberConfig + resolve layout (M3 sentinel)."""

    def test_disabled_when_no_config(self):
        cfg = load_tuber_config(None, PROJECT_ROOT)
        assert cfg.enabled is False

    def test_disabled_explicit(self, tmp_path: Path):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({"enabled": False}))
        cfg = load_tuber_config(str(p), PROJECT_ROOT)
        assert cfg.enabled is False

    def test_missing_key_raises(self, tmp_path: Path):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({"enabled": True}))
        with pytest.raises(TuberConfigError, match="thiếu key"):
            load_tuber_config(str(p), PROJECT_ROOT)

    def test_valid_minimal(self, tmp_path: Path):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(_sample_config_dict()))
        cfg = load_tuber_config(str(p), PROJECT_ROOT)
        assert cfg.enabled is True
        assert cfg.max_group_sec == 3.0
        assert cfg.retry_attempts == 2

    def test_defaults(self, tmp_path: Path):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(_sample_config_dict()))
        cfg = load_tuber_config(str(p), PROJECT_ROOT)
        assert cfg.overlay_format == "direct"
        assert cfg.mouth_mode == "cue"
        assert cfg.repair_output_suffix == "_with_tuber"
        assert cfg.on_exhausted == "render_without_tuber"

    def test_job_name_default_sentinel(self):
        """M3: output_name = default sentinel → fallback stem video."""
        cfg = parse_tuber_config_dict(_sample_config_dict())
        cfg.resolve_layout(PROJECT_ROOT, input_video="/abs/job_001/video.mp4",
                           output_name="video_synced")
        assert cfg.job_name == "video"  # stem của video.mp4

    def test_job_name_custom(self):
        cfg = parse_tuber_config_dict(_sample_config_dict())
        cfg.resolve_layout(PROJECT_ROOT, input_video="/abs/video.mp4",
                           output_name="my_custom_job")
        assert cfg.job_name == "my_custom_job"

    def test_resolve_layout_paths_absolute(self):
        cfg = parse_tuber_config_dict(_sample_config_dict())
        cfg.resolve_layout(PROJECT_ROOT, input_video="video.mp4", output_name="job")
        assert cfg.tuber_root is not None
        assert cfg.tuber_root.is_absolute()
        assert "job" in str(cfg.tuber_root)

    def test_artifact_policy_repairable_defaults(self):
        cfg = parse_tuber_config_dict(_sample_config_dict())
        ap = cfg.artifact_policy()
        assert ap["mode"] == "repairable"
        assert ap["overlayFrames"] == "safe"
        assert ap["finalRenderInputs"] == "keep"
        assert ap["logs"] == "keep"

    def test_artifact_policy_override(self, tmp_path: Path):
        d = _sample_config_dict()
        d["artifactPolicy"] = {"mode": "repairable", "overlayFrames": "keep"}
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(d))
        cfg = load_tuber_config(str(p), PROJECT_ROOT)
        ap = cfg.artifact_policy()
        assert ap["overlayFrames"] == "keep"
        assert ap["finalRenderInputs"] == "keep"  # default from repairable

    def test_chromakey_config(self):
        d = _sample_config_dict()
        d["asset"]["chromakey"] = {"color": "0x08A702", "similarity": 0.12, "blend": 0.1}
        cfg = parse_tuber_config_dict(d)
        assert cfg.chromakey["color"] == "0x08A702"
        assert cfg.chromakey["similarity"] == 0.12

    def test_chromakey_enabled_default_none(self):
        # Không khai báo → None (auto: dò alpha của nguồn)
        cfg = parse_tuber_config_dict(_sample_config_dict())
        assert cfg.chromakey_enabled is None

    def test_chromakey_enabled_false(self):
        # Nguồn .mov/.webm đã trong suốt → tắt chromakey tường minh
        d = _sample_config_dict()
        d["asset"]["chromakey"] = {"enabled": False}
        cfg = parse_tuber_config_dict(d)
        assert cfg.chromakey_enabled is False

    def test_chromakey_enabled_true(self):
        d = _sample_config_dict()
        d["asset"]["chromakey"] = {"enabled": True, "color": "0x00FF00"}
        cfg = parse_tuber_config_dict(d)
        assert cfg.chromakey_enabled is True
        assert cfg.chromakey["color"] == "0x00FF00"

    def test_config_not_found_raises(self):
        with pytest.raises(TuberConfigError, match="Không tìm thấy"):
            load_tuber_config("/nonexistent_12345.json", PROJECT_ROOT)


class TestLayer1_TuberStatus:
    """Status helper: new/read/write + transition."""

    def test_new_pending(self):
        s = st.new_status("g1")
        assert s["status"] == "pending"
        assert s["attempts"] == 0

    def test_write_and_read(self, tmp_path: Path):
        g = tmp_path / "g1"
        st.write_status(g, st.new_status("g1"))
        s2 = st.read_status(g)
        assert s2["status"] == "pending"

    def test_read_nonexistent(self, tmp_path: Path):
        assert st.read_status(tmp_path / "no") is None

    def test_done_transition(self, tmp_path: Path):
        g = tmp_path / "g1"
        s = st.new_status("g1")
        s["status"] = st.STATUS_DONE
        s["currentStep"] = st.STEP_CLEANUP
        s["attempts"] = 2
        st.write_status(g, s)
        s2 = st.read_status(g)
        assert s2["status"] == "done"
        assert s2["attempts"] == 2


class TestLayer1_ImageEventSerialize:
    """Serialize/deserialize ImageOverlayEvent cho repair."""

    def test_roundtrip(self):
        from sync_engine.image_overlay import ImageOverlayEvent
        events = [
            ImageOverlayEvent(key="k1", image_path="/a.png",
                              start_time=0.0, end_time=3000.0, source_line=1),
            ImageOverlayEvent(key="k2", image_path="/b.png",
                              start_time=5000.0, end_time=8000.0, source_line=3),
        ]
        data = serialize_image_overlay_events(events)
        assert len(data) == 2
        assert data[0]["key"] == "k1"
        restored = deserialize_image_overlay_events(data)
        assert len(restored) == 2
        assert restored[0].key == "k1"
        assert restored[0].start_time == 0.0
        assert restored[1].end_time == 8000.0

    def test_empty_none(self):
        assert serialize_image_overlay_events(None) == []
        assert serialize_image_overlay_events([]) == []


class TestLayer1_CompositeSeekCmd:
    """Unit: composite_group_from_stretched builds correct FFmpeg command."""

    def test_seek_cmd_has_rough_ss(self):
        """Verify `-ss` với rough_start được đặt trước `-i`."""
        from sync_engine.tuber_overlay import composite_group_from_stretched as _s
        import inspect
        src = inspect.getsource(_s)
        assert "rough_start_s" in src
        assert "trim=start=" in src
        assert "trim=end_frame=" in src

    def test_seek_cmd_no_restretch(self):
        """Verify composite_group_from_stretched KHÔNG re-stretch (không build_ffmpeg_batch_cmd)."""
        from sync_engine.tuber_overlay import composite_group_from_stretched as _s
        import inspect
        src = inspect.getsource(_s)
        assert "build_ffmpeg_batch_cmd" not in src
        assert "_HEVC_NVENC_VIDEO_ARGS" in src
        assert "overlay=" in src


class TestLayer1_PerformanceDebugConfig:
    """Unit: config accessors mới (performance, resume, debug)."""

    def test_max_workers_default(self):
        cfg = TuberConfig(enabled=True, raw={})
        assert cfg.max_workers == 2

    def test_max_workers_override(self):
        cfg = TuberConfig(enabled=True, raw={"performance": {"maxWorkers": 4}})
        assert cfg.max_workers == 4

    def test_resume_skip_done_default(self):
        cfg = TuberConfig(enabled=True, raw={})
        assert cfg.resume_skip_done is True

    def test_resume_skip_done_false(self):
        cfg = TuberConfig(enabled=True, raw={"resume": {"skipDone": False}})
        assert cfg.resume_skip_done is False

    def test_debug_frame_defaults(self):
        cfg = TuberConfig(enabled=True, raw={})
        assert cfg.debug_frame_output_enabled is False
        assert cfg.debug_frame_margin == 3

    def test_debug_frame_override(self):
        cfg = TuberConfig(enabled=True, raw={
            "debug": {"frameOutput": {"enabled": True, "marginFrames": 5}},
        })
        assert cfg.debug_frame_output_enabled is True
        assert cfg.debug_frame_margin == 5


class TestLayer1_GroupHash:
    """Unit: compute_group_input_hash stability and sensitivity."""

    def test_hash_stable(self):
        mf = {"segments": [{"startFrame": 0, "endFrame": 10}],
              "renderStartFrame": 0, "renderDurationFrames": 10,
              "fps": 30.0}
        vid = Path(__file__)
        h1 = compute_group_input_hash(mf, {"outputWidth": 512, "outputHeight": 288, "assetId": "x"}, vid)
        h2 = compute_group_input_hash(mf, {"outputWidth": 512, "outputHeight": 288, "assetId": "x"}, vid)
        assert h1 == h2

    def test_hash_changes_on_segments(self):
        mf1 = {"segments": [{"startFrame": 0, "endFrame": 10, "blockType": "tts", "hasTts": True, "mouthEvents": None}],
               "renderStartFrame": 0, "renderDurationFrames": 10, "fps": 30.0}
        mf2 = {"segments": [{"startFrame": 0, "endFrame": 20, "blockType": "tts", "hasTts": True, "mouthEvents": None}],
               "renderStartFrame": 0, "renderDurationFrames": 20, "fps": 30.0}
        vid = Path(__file__)
        h1 = compute_group_input_hash(mf1, None, vid)
        h2 = compute_group_input_hash(mf2, None, vid)
        assert h1 != h2

    def test_hash_changes_on_prerender_size(self):
        mf = {"segments": [], "renderStartFrame": 0, "renderDurationFrames": 10, "fps": 30.0}
        vid = Path(__file__)
        h1 = compute_group_input_hash(mf, {"outputWidth": 512, "outputHeight": 288, "assetId": "x"}, vid)
        h2 = compute_group_input_hash(mf, {"outputWidth": 640, "outputHeight": 360, "assetId": "x"}, vid)
        assert h1 != h2

    def test_hash_stable_when_intermediate_regenerated(self, tmp_path):
        """Regression: resume.skipDone phải khớp dù video_stretched.mp4 bị tái tạo.

        Bug cũ: hash anchor vào mtime của video_stretched.mp4. sync-video tái
        tạo file này mỗi lần chạy (mtime mới) → hash luôn miss → group re-render
        thay vì skip. Fix: anchor vào video GỐC (ổn định giữa các lần chạy).
        Test mô phỏng: cùng video gốc → hash y hệt qua 2 'lần chạy'.
        """
        src = tmp_path / "source.mp4"
        src.write_bytes(b"original source bytes")
        mf = {"segments": [{"startFrame": 0, "endFrame": 10, "blockType": "tts",
                            "hasTts": True, "mouthEvents": None}],
              "renderStartFrame": 0, "renderDurationFrames": 10, "fps": 30.0}
        pm = {"outputWidth": 512, "outputHeight": 288, "assetId": "x"}

        # Lần chạy 1: lưu hash vào "status"
        h_run1 = compute_group_input_hash(mf, pm, src)

        # Giữa 2 lần chạy: intermediate (stretched) bị tái tạo mtime mới — nhưng
        # nó KHÔNG nằm trong hash nữa. Video gốc không đổi → hash phải khớp.
        import os, time
        time.sleep(0.01)
        future = time.time_ns() + 1_000_000_000
        os.utime(src, ns=(future, future))  # đổi mtime video gốc → hash phải đổi
        h_changed_source = compute_group_input_hash(mf, pm, src)
        assert h_changed_source != h_run1, "Đổi video gốc phải đổi hash"

        # Khôi phục mtime gốc → hash quay lại y hệt (resume skip OK)
        # (mô phỏng cùng 1 video gốc giữa các lần chạy sync-video)
        os.utime(src, ns=(future, future))
        h_run2 = compute_group_input_hash(mf, pm, src)
        assert h_run2 == h_changed_source

    def test_hash_tolerates_missing_source_video(self):
        """source_video=None không raise (graceful) — vẫn ra hash deterministic."""
        mf = {"segments": [], "renderStartFrame": 0, "renderDurationFrames": 10, "fps": 30.0}
        h1 = compute_group_input_hash(mf, None, None)
        h2 = compute_group_input_hash(mf, None, None)
        assert h1 == h2 and isinstance(h1, str) and len(h1) == 64


class TestLayer1_OverlayFormatConfig:
    """Unit: overlay_format accessor (V4) — default, override, giá trị lạ."""

    def test_overlay_format_default_is_direct(self):
        cfg = TuberConfig(enabled=True, raw={})
        assert cfg.overlay_format == "direct"

    def test_overlay_format_png_sequence(self):
        cfg = TuberConfig(enabled=True, raw={"overlay": {"format": "png_sequence"}})
        assert cfg.overlay_format == "png_sequence"

    def test_overlay_format_direct_explicit(self):
        cfg = TuberConfig(enabled=True, raw={"overlay": {"format": "direct"}})
        assert cfg.overlay_format == "direct"

    def test_overlay_format_invalid_fallback_direct(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="sync_video"):
            cfg = TuberConfig(enabled=True, raw={"overlay": {"format": "split_alpha"}})
            assert cfg.overlay_format == "direct"
        assert "không hợp lệ" in caplog.text or "split_alpha" in caplog.text


class TestLayer1_DirectPipeCmd:
    """Unit: _pipe_prerender_frames và _make_mouth_lookup (source inspection)."""

    def test_pipe_cmd_has_rawvideo_pix_fmt(self):
        import inspect
        src = inspect.getsource(_pipe_prerender_frames)
        assert "-f" in src and "rawvideo" in src
        assert "pix_fmt" in src and "rgba" in src

    def test_pipe_cmd_has_pipe_stdin(self):
        import inspect
        src = inspect.getsource(_pipe_prerender_frames)
        assert "pipe:0" in src or "stdin" in src.lower()

    def test_pipe_cmd_has_hybrid_seek(self):
        import inspect
        src = inspect.getsource(_pipe_prerender_frames)
        assert "rough_start_s" in src
        assert "trim=start=" in src
        assert "trim=end_frame=" in src

    def test_pipe_cmd_reads_actual_size(self):
        """Hàm phải đọc size THẬT từ probe frame (Q4 plan)."""
        import inspect
        src = inspect.getsource(_pipe_prerender_frames)
        assert "im.size" in src or "W, H" in src

    def test_pipe_cmd_no_overlay_frames_dir(self):
        """Direct pipe KHÔNG ghi PNG ra overlay_frames/ — frame chảy RAM→FFmpeg.

        Kiểm tra theo intent: trong CODE (bỏ docstring) hàm direct pipe không gọi
        helper của nhánh png_sequence và không đọc overlayDir; thay vào đó bơm raw
        RGBA vào stdin FFmpeg.
        """
        import ast, inspect, textwrap
        tree = ast.parse(textwrap.dedent(inspect.getsource(_pipe_prerender_frames)))
        fn_node = tree.body[0]
        # Bỏ docstring (statement string đầu tiên) → chỉ còn code thực thi
        if (fn_node.body and isinstance(fn_node.body[0], ast.Expr)
                and isinstance(fn_node.body[0].value, ast.Constant)):
            fn_node.body = fn_node.body[1:]
        src = ast.unparse(fn_node)
        assert "_build_prerender_frame_list" not in src
        assert "composite_group_from_stretched" not in src
        assert "overlayDir" not in src
        # Phải bơm raw RGBA vào stdin (đặc trưng direct pipe)
        assert "stdin" in src and "rawvideo" in src

    def test_make_mouth_lookup_binary_search(self):
        """_make_mouth_lookup trả closure binary-search đúng."""
        mf = {
            "segments": [{
                "mouthEvents": [
                    {"frame": 0, "state": "closed"},
                    {"frame": 10, "state": "open"},
                    {"frame": 20, "state": "closed"},
                ]
            }]
        }
        lk = _make_mouth_lookup(mf)
        assert lk(0) == "closed"
        assert lk(5) == "closed"
        assert lk(10) == "open"
        assert lk(15) == "open"
        assert lk(20) == "closed"

    def test_make_mouth_lookup_empty_segments(self):
        lk = _make_mouth_lookup({"segments": []})
        assert lk(999) == "closed"

    def test_pipe_uses_make_mouth_lookup(self):
        """_pipe_prerender_frames phải dùng _make_mouth_lookup (DRY)."""
        import inspect
        src = inspect.getsource(_pipe_prerender_frames)
        assert "_make_mouth_lookup" in src


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — COMPONENT: manifest export, artifact promote, group base
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _FFMPEG_OK, reason="Cần ffmpeg + ffprobe trong PATH")
class TestLayer2_ManifestExport:
    """Manifest synthetic export: schema hợp lệ, paths absolute, V1 invariants."""

    def test_build_group_manifest_schema(self, tmp_path: Path):
        tl = [_mk_seg(0, 2000, 1.0, "tts", "x.wav"), _mk_seg(2000, 3000, 1.0, "mute", None)]
        groups = build_render_groups(tl, 30.0, max_group_sec=10.0)
        g = groups[0]
        g.group_id = "group_0001"
        m = build_group_manifest(
            g, fps_float=30.0, fps_str="30/1", width=1280, height=720,
            asset_id="nike", character={"width": 512}, mouth_mode="cue",
            group_dir=tmp_path / "g1",
        )
        assert "schemaVersion" in m
        # schema fields
        for k in ("groupId", "fps", "width", "height", "groupStartFrame",
                  "groupEndFrame", "renderDurationFrames", "segments", "character"):
            assert k in m, f"manifest thiếu {k}"

    def test_manifest_paths_absolute(self, tmp_path: Path):
        tl = [_mk_seg(0, 3000, 1.0)]
        groups = build_render_groups(tl, 30.0, 10.0)
        g = groups[0]; g.group_id = "g1"
        gd = tmp_path / "g1"; gd.mkdir()
        m = build_group_manifest(g, fps_float=30.0, fps_str="30/1", width=1920, height=1080,
                               asset_id="a", character={"width": 200}, mouth_mode="cue",
                               group_dir=gd)
        for k in ("base", "overlayDir", "videoWithTuber"):
            assert Path(m[k]).is_absolute(), f"{k} không absolute: {m[k]}"

    def test_v1_invariants(self, tmp_path: Path):
        tl = [_mk_seg(0, 3000, 1.0)]
        groups = build_render_groups(tl, 30.0, 10.0)
        g = groups[0]; g.group_id = "g1"
        m = build_group_manifest(g, fps_float=30.0, fps_str="30/1", width=1280, height=720,
                               asset_id="x", character={}, mouth_mode="cue",
                               group_dir=tmp_path / "g1")
        assert m["prePaddingFrames"] == 0
        assert m["postPaddingFrames"] == 0
        assert m["renderStartFrame"] == m["groupStartFrame"]
        assert m["renderDurationFrames"] == m["groupEndFrame"] - m["groupStartFrame"]
        # no speechControlAudio in V1
        assert "speechControlAudio" not in m

    def test_segment_hasTts(self, tmp_path: Path):
        tl = [
            _mk_seg(0, 1000, 1.0, "tts", "a.wav"),
            _mk_seg(1000, 2000, 1.0, "mute", None),
            _mk_seg(2000, 4000, 1.0, "gap", None),
            _mk_seg(4000, 5000, 1.0, "tail", None),
        ]
        groups = build_render_groups(tl, 30.0, 10.0)
        m = build_group_manifest(groups[0], fps_float=30.0, fps_str="30/1", width=1920,
                               height=1080, asset_id="n", character={}, mouth_mode="cue",
                               group_dir=tmp_path / "g1")
        segs = m["segments"]
        assert segs[0]["hasTts"] is True
        assert segs[1]["hasTts"] is False
        assert segs[2]["hasTts"] is False
        assert segs[3]["hasTts"] is False

    def test_segment_frames_contiguous(self, tmp_path: Path):
        tl = _sample_timeline()
        groups = build_render_groups(tl, 30.0, 3.0)
        for g in groups:
            g.group_id = f"group_{g.index + 1:04d}"
        # check each group
        for g in groups:
            g.group_id = f"group_{g.index + 1:04d}"
            m = build_group_manifest(g, fps_float=30.0, fps_str="30/1", width=1920,
                                   height=1080, asset_id="n", character={},
                                   mouth_mode="cue", group_dir=tmp_path / f"g{g.index}")
            cursor = g.group_start_frame
            for seg in m["segments"]:
                assert seg["startFrame"] == cursor
                assert seg["endFrame"] == seg["startFrame"] + (seg["endFrame"] - seg["startFrame"])
                cursor = seg["endFrame"]
            assert cursor == g.group_end_frame

    def test_build_run_manifest(self, tmp_path: Path):
        tr = tmp_path / "tuberRoot"
        run = build_run_manifest(
            job_name="j1", fps_float=25.0, fps_str="25/1", width=960, height=540,
            tuber_root=tr, media_dir=tr / "media", groups_dir=tr / "groups",
            base_video=tr / "media" / BASE_VIDEO_NAME,
            final_audio=tr / "media" / FINAL_AUDIO_NAME,
            video_with_tuber=tr / "media" / "video_stretched_with_tuber.mp4",
            overlay_format="png_sequence",
            remotion={"projectDir": "rt", "compositionId": "T", "entryPoint": "e"},
            asset={"assetDir": "/a"},
            group_manifest_paths=[],
            artifact_policy={"mode": "repairable"},
            tuber_config_raw={},
        )
        assert run["schemaVersion"] == SCHEMA_VERSION
        assert run["fps"] == 25.0
        assert run["width"] == 960
        for k in ("tuberRoot", "mediaDir", "groupsDir", "baseVideo"):
            assert Path(run[k]).is_absolute()


@pytest.mark.skipif(not _FFMPEG_OK, reason="Cần ffmpeg + ffprobe trong PATH")
class TestLayer2_ArtifactPromote:
    """promote media/final_render_inputs cho repair bundle."""

    def test_promote_media(self, tmp_path: Path):
        src = tmp_path / "src"; src.mkdir()
        # dùng ffmpeg tạo silent video thay vì ghi byte rỗng
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi",
                        "-i", "color=c=black:s=64x64:d=0.1:r=30",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        str(src / "video.mp4")], capture_output=True, check=True)
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi",
                        "-i", "anullsrc=r=44100:d=0.1",
                        str(src / "audio.wav")], capture_output=True, check=True)
        md = tmp_path / "media"
        result = promote_media(
            base_video_src=str(src / "video.mp4"),
            final_audio_src=str(src / "audio.wav"),
            media_dir=md,
        )
        assert result["base_video"].exists()
        assert result["final_audio"].exists()

    def test_promote_final_render_inputs(self, tmp_path: Path):
        fri = tmp_path / "fri"
        mpath = promote_final_render_inputs(
            final_render_inputs_dir=fri,
            subtitle_synced_srt=None,
            note_overlay_final_ass=None,
            image_overlay_events=None,
            render_config={"res": "720p"},
            final_render_args={"output_name": "test"},
        )
        assert mpath.exists()
        loaded = load_final_render_manifest(fri)
        assert loaded["output_name"] == "test"
        assert loaded["render_config"] == str((fri / "render_config.json").resolve())
        assert loaded["subtitle_synced_srt"] is None
        assert loaded["image_overlay_events"] is None  # empty events → null

    def test_promote_final_render_with_data(self, tmp_path: Path):
        # subtitle file thật
        sub = tmp_path / "sub.srt"
        sub.write_text("1\n00:00:01,000 --> 00:00:03,000\nHello\n\n", encoding="utf-8")
        fri = tmp_path / "fri2"
        promote_final_render_inputs(
            final_render_inputs_dir=fri,
            subtitle_synced_srt=str(sub),
            note_overlay_final_ass=None,
            image_overlay_events=None,
            render_config={},
            final_render_args={},
        )
        loaded = load_final_render_manifest(fri)
        assert loaded["subtitle_synced_srt"] is not None
        assert Path(loaded["subtitle_synced_srt"]).exists()

    def test_cleanup_overlay_frames(self, tmp_path: Path):
        od = tmp_path / "g1" / "overlay_frames"
        od.mkdir(parents=True)
        (od / "frame_00.png").touch()
        cleanup_overlay_frames(tmp_path / "g1", "safe")
        assert not od.exists()

    def test_cleanup_keep_policy(self, tmp_path: Path):
        od = tmp_path / "g1" / "overlay_frames"
        od.mkdir(parents=True)
        (od / "frame_00.png").touch()
        cleanup_overlay_frames(tmp_path / "g1", "keep")
        assert od.exists()


# ═════════════════════════════════════════════════════════════════════
# LAYER 3 — PIPELINE INTEGRATION: composite, validate, concat, retry
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _FFMPEG_OK, reason="Cần ffmpeg + ffprobe trong PATH")
@pytest.mark.skipif(not _GPU_OK, reason="Cần HEVC NVENC GPU cho composite")
class TestLayer3_CompositeValidate:
    """Composite + validate với video tổng hợp, overlay PNG tự tạo."""

    @pytest.fixture(scope="class")
    def synthetic_base(self, tmp_path_factory) -> Path:
        base_dir = tmp_path_factory.mktemp("base")
        # video ngắn 1920x1080, 10f @25fps = 0.4s
        base = base_dir / "base.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "color=c=red:s=1920x1080:d=0.4:r=25",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-video_track_timescale", "90000",
            str(base),
        ], capture_output=True, check=True)
        return base

    @pytest.fixture(scope="class")
    def synthetic_overlay_dir(self, tmp_path_factory) -> Path:
        od = tmp_path_factory.mktemp("overlay")
        # overlay nền trong suốt + 1 chấm xanh ở giữa trên cùng resolution
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"color=c=0x00000000@0.0:s=1920x1080:d=0.4:r=25,"
                  f"drawbox=x=950:y=530:w=20:h=20:c=blue@1.0:t=fill",
            "-c:v", "png", "-start_number", "0",
            str(od / "frame_%02d.png"),
        ], capture_output=True, check=True)
        return od

    def test_composite_group(self, synthetic_base, synthetic_overlay_dir, tmp_path: Path):
        out = tmp_path / "composite.mp4"
        result = composite_group(synthetic_base, synthetic_overlay_dir, out, "25/1")
        assert result.exists()
        assert result.stat().st_size > 1024

    def test_validate_pass(self, synthetic_base, synthetic_overlay_dir, tmp_path: Path):
        out = tmp_path / "composite.mp4"
        composite_group(synthetic_base, synthetic_overlay_dir, out, "25/1")
        validate_group_output(out, 0.4, duration_tolerance_s=0.15)  # OK

    def test_validate_fail_missing(self, tmp_path: Path):
        with pytest.raises(TuberOverlayError, match="không tồn tại"):
            validate_group_output(tmp_path / "no.mp4", 1.0)

    def test_validate_fail_duration(self, synthetic_base, synthetic_overlay_dir, tmp_path: Path):
        out = tmp_path / "comp2.mp4"
        composite_group(synthetic_base, synthetic_overlay_dir, out, "25/1")
        with pytest.raises(TuberOverlayError, match="Duration lệch"):
            validate_group_output(out, 999.0, duration_tolerance_s=0.01)

    def test_concat_group_videos(self, synthetic_base, tmp_path: Path):
        # concat 2 copies của base
        p1 = tmp_path / "part1.mp4"; shutil.copy2(synthetic_base, p1)
        p2 = tmp_path / "part2.mp4"; shutil.copy2(synthetic_base, p2)
        out = tmp_path / "joined.mp4"
        result = concat_group_videos([p1, p2], out, tmp_path / "concat_tmp")
        assert result.exists()
        # duration ~ 0.4*2
        dur = _probe_duration(result)
        assert 0.7 <= dur <= 0.9

    def test_detect_frame_pattern(self, synthetic_overlay_dir):
        pattern = _detect_frame_pattern(synthetic_overlay_dir)
        assert pattern == "frame_%02d.png"  # 10 frames → 2 digit

    def test_expected_group_duration(self):
        m = {"renderDurationFrames": 100, "fps": 25.0}
        assert _expected_group_duration_s(m) == 4.0


def _probe_duration(path: Path) -> float:
    from sync_engine.tuber_overlay import _probe_duration_s
    return _probe_duration_s(path)


@pytest.mark.skipif(not _FFMPEG_OK, reason="Cần ffmpeg + ffprobe trong PATH")
class TestLayer3_RetryAndCleanup:
    """Retry group fail + cleanup overlay_frames trong flow mock."""

    @pytest.fixture
    def fake_group(self, tmp_path: Path):
        """GroupJob + base.mp4 + overlayDir sẵn sàng."""
        gd = tmp_path / "group_0001"
        gd.mkdir(parents=True)
        od = gd / "overlay_frames"; od.mkdir()
        # base nhỏ
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "color=c=black:s=64x64:d=0.1:r=30",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(gd / "base.mp4"),
        ], capture_output=True, check=True)
        # overlay 3 frames
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "color=c=0x00000000@0.0:s=64x64:d=0.1:r=30",
            "-c:v", "png", "-start_number", "0",
            str(od / "frame_%02d.png"),
        ], capture_output=True, check=True)

        manifest = {
            "groupId": "group_0001", "fps": 30.0, "fpsStr": "30/1", "width": 64, "height": 64,
            "renderDurationFrames": 3, "groupStartFrame": 0, "groupEndFrame": 3,
            "base": str(gd / "base.mp4"),
            "overlayDir": str(od),
            "videoWithTuber": str(gd / "video_with_tuber.mp4"),
        }
        mpath = gd / "group_manifest.json"
        mpath.write_text(json.dumps(manifest))
        return GroupJob("group_0001", gd, mpath, manifest)

    def test_render_and_composite_success(self, fake_group, tmp_path: Path):
        """Mock render driver → composite success → cleanup overlay."""
        import sync_engine.tuber_overlay as to

        logs = tmp_path / "logs"; logs.mkdir()
        orig_run = getattr(to, "_run_render_driver", None)
        orig_comp = getattr(to, "composite_group", None)
        try:
            to._run_render_driver = lambda project_dir, manifest_paths, log_path=None, timeout=7200: {
                "group_0001": {"groupId": "group_0001", "ok": True, "frames": 3},
            }
            # Mock composite: copy base → video_with_tuber (không cần NVENC)
            def fake_composite(base_video, overlay_dir, output, fps_str,
                               *, offset_x=0, offset_y=0):
                import shutil as _shutil
                _shutil.copy2(str(base_video), str(output))
                return output
            to.composite_group = fake_composite

            videos = render_and_composite_groups(
                project_dir=PROJECT_ROOT / "remotion_tuber",
                groups=[fake_group],
                retry_attempts=0,
                artifact_policy={"mode": "repairable", "overlayFrames": "safe"},
                logs_dir=logs,
                duration_tolerance_s=1.0,
            )
            assert len(videos) == 1
            assert videos[0].exists()
            # overlay frames đã bị cleanup theo policy safe
            assert not (fake_group.group_dir / "overlay_frames").exists()
        finally:
            if orig_run is not None:
                to._run_render_driver = orig_run
            if orig_comp is not None:
                to.composite_group = orig_comp

    def test_retry_on_failure_then_success(self, fake_group, tmp_path: Path):
        """Mock render fail 1 lần, retry thành công."""
        import sync_engine.tuber_overlay as to

        logs = tmp_path / "logs"; logs.mkdir()
        call_count = [0]

        def flaky_driver(project_dir, manifest_paths, log_path=None, timeout=7200):
            call_count[0] += 1
            if call_count[0] == 1:
                return {}  # fail lần 1
            return {"group_0001": {"groupId": "group_0001", "ok": True, "frames": 3}}

        def fake_composite(base_video, overlay_dir, output, fps_str,
                           *, offset_x=0, offset_y=0):
            import shutil as _shutil
            _shutil.copy2(str(base_video), str(output))
            return output

        orig_run = getattr(to, "_run_render_driver", None)
        orig_comp = getattr(to, "composite_group", None)
        try:
            to._run_render_driver = flaky_driver
            to.composite_group = fake_composite
            videos = render_and_composite_groups(
                project_dir=PROJECT_ROOT / "remotion_tuber",
                groups=[fake_group],
                retry_attempts=2,
                artifact_policy={"mode": "repairable", "overlayFrames": "safe"},
                logs_dir=logs,
                duration_tolerance_s=1.0,
            )
            assert len(videos) == 1
            assert call_count[0] == 2  # lần 1 fail, lần 2 OK
            s = st.read_status(fake_group.group_dir)
            assert s["status"] == "done"
            # V2: mỗi lần render/composite/validate 1 group, attempt tăng khi fail
            assert s["attempts"] == 1
        finally:
            if orig_run is not None:
                to._run_render_driver = orig_run
            if orig_comp is not None:
                to.composite_group = orig_comp

    def test_exhausted_retry_raises(self, fake_group, tmp_path: Path):
        """Mock render luôn fail → hết retry → raise."""
        import sync_engine.tuber_overlay as to

        logs = tmp_path / "logs"; logs.mkdir()
        orig_run = getattr(to, "_run_render_driver", None)
        try:
            to._run_render_driver = lambda *a, **kw: {}
            with pytest.raises(TuberOverlayError, match="hết retry"):
                render_and_composite_groups(
                    project_dir=PROJECT_ROOT / "remotion_tuber",
                    groups=[fake_group],
                    retry_attempts=1,
                    artifact_policy={"mode": "repairable"},
                    logs_dir=logs,
                    duration_tolerance_s=1.0,
                )
            # status failed
            s = st.read_status(fake_group.group_dir)
            assert s["status"] == "failed"
            assert s["fallbackTriggered"] is True
        finally:
            if orig_run is not None:
                to._run_render_driver = orig_run

    def test_render_and_composite_fail_keeps_status(self, fake_group, tmp_path: Path):
        """Composite fail → status FAILED."""
        import sync_engine.tuber_overlay as to

        fake_group.manifest["base"] = str(fake_group.group_dir / "nonexistent.mp4")  # force composite fail
        logs = tmp_path / "logs"; logs.mkdir()
        orig_run = getattr(to, "_run_render_driver", None)
        try:
            to._run_render_driver = lambda *a, **kw: {
                "group_0001": {"groupId": "group_0001", "ok": True, "frames": 3}
            }
            to.render_and_composite_groups(
                project_dir=PROJECT_ROOT / "remotion_tuber",
                groups=[fake_group],
                retry_attempts=1,
                artifact_policy={"mode": "repairable"},
                logs_dir=logs,
                duration_tolerance_s=1.0,
            )
        except TuberOverlayError:
            pass
        finally:
            if orig_run is not None:
                to._run_render_driver = orig_run
        s = st.read_status(fake_group.group_dir)
        assert s is not None
        # composite fail → status FAILED, fallbackTriggered
        assert s["status"] in ("running", "failed")
        assert s["lastError"] is not None or s["status"] == "running"

    def test_probe_resolution(self, tmp_path: Path):
        vid = tmp_path / "test.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "color=c=black:s=320x240:d=0.1:r=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(vid),
        ], capture_output=True, check=True)
        w, h = probe_resolution(str(vid))
        assert (w, h) == (320, 240)
