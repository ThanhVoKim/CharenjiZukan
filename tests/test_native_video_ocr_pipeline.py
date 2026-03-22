#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_native_video_ocr_pipeline.py
========================================
Test đầy đủ cho Native Video OCR Pipeline.

Cấu trúc 4 layer độc lập:

  Layer 1 — Unit Tests
      Không cần GPU, FFmpeg, hay video thật.
      Test từng hàm nhỏ riêng lẻ.
      Fixture: chỉ dùng dữ liệu thuần Python.

  Layer 2 — Component Tests (Synthetic Video)
      Không cần GPU hay model.
      Test luồng đọc frame → crop ROI → tạo batch từ video tổng hợp.
      Fixture: `synthetic_video_path` tạo file .mp4 tạm bằng cv2.

  Layer 3 — Pipeline Integration (Mocked Model + Synthetic Video)
      Không cần GPU thật.
      Test toàn bộ pipeline extract() với model và inference bị mock.
      Fixture: `synthetic_video_path` + monkeypatch _load_model/_infer.

  Layer 4 — Real Model Test (GPU Required)
      Cần CUDA GPU + VRAM >= 10GB + model được tải từ HuggingFace.
      Đánh dấu @pytest.mark.gpu — mặc định bị skip trừ khi có GPU đủ mạnh.
      Chạy riêng khi cần xác nhận chất lượng OCR thực tế.

Cách chạy từng layer:
    pytest tests/test_native_video_ocr_pipeline.py -v -k "Layer1"
    pytest tests/test_native_video_ocr_pipeline.py -v -k "Layer2"
    pytest tests/test_native_video_ocr_pipeline.py -v -k "Layer3"
    pytest tests/test_native_video_ocr_pipeline.py -v -k "Layer4" --gpu
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Project root ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Lazy imports (skip nếu thiếu dependency) ────────────────────────
cv2 = pytest.importorskip("cv2", reason="opencv-python chưa cài: pip install opencv-python")
PIL = pytest.importorskip("PIL", reason="Pillow chưa cài: pip install Pillow")
from PIL import Image, ImageDraw, ImageFont


# ═════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ═════════════════════════════════════════════════════════════════════

# ── Subtitle data dùng chung ─────────────────────────────────────────
SYNTHETIC_SUBTITLES: List[Tuple[float, float, str]] = [
    (1.0,  3.5,  "你好世界"),
    (5.0,  8.0,  "这是测试字幕"),
    (10.0, 13.0, "第三段字幕内容"),
    (62.0, 65.0, "跨越第二批次的字幕"),   # Nằm trong batch 2 (sau 60s)
    (67.0, 70.0, "最后一段字幕"),
]

# ── ROI theo assets/boxesOCR.txt ─────────────────────────────────────
ROI_X, ROI_Y, ROI_W, ROI_H = 370, 984, 1180, 80
VIDEO_W, VIDEO_H = 1920, 1080
VIDEO_FPS = 30
FRAME_INTERVAL = 6     # Lấy 1/6 frame → 5 fps effective

# Tham số mặc định theo config/native_video_ocr_config.yaml
SAMPLE_FPS   = 5.0
BATCH_DURATION = 60.0
MAX_NEW_TOKENS = 256   # Nhỏ hơn trong test để nhanh


@pytest.fixture(scope="module")
def synthetic_video_path(tmp_path_factory) -> Path:
    """
    Tạo file video .mp4 tổng hợp chứa chữ trắng trên nền tối
    tại đúng vị trí ROI (370, 984, 1180, 80).

    Video:
    - 1920x1080, 30fps, tổng 75 giây (2250 frames)
    - Mỗi subtitle được vẽ vào đúng khoảng thời gian
    - Ngoài các khoảng đó, frame tối hoàn toàn

    Fixture có scope="module" → tạo 1 lần, dùng lại cho tất cả test trong file.
    """
    tmp_dir = tmp_path_factory.mktemp("video_data")
    video_path = tmp_dir / "synthetic_native_ocr.mp4"

    total_frames = int(75 * VIDEO_FPS)  # 75 giây

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(video_path), fourcc, VIDEO_FPS, (VIDEO_W, VIDEO_H)
    )

    # Khởi tạo font CJK cho Pillow
    font_path = PROJECT_ROOT / "assets" / "NotoSansCJKsc-VF.ttf"
    try:
        font = ImageFont.truetype(str(font_path), 60)
    except IOError:
        print(f"WARNING: Cannot load {font_path}. Using default font (CJK will fail).")
        font = ImageFont.load_default()

    for frame_no in range(total_frames):
        timestamp = frame_no / VIDEO_FPS

        # Tạo frame nền tối
        frame_cv2 = np.full((VIDEO_H, VIDEO_W, 3), 20, dtype=np.uint8)

        # Vẽ chữ nếu timestamp nằm trong khoảng của subtitle nào đó
        for start, end, text in SYNTHETIC_SUBTITLES:
            if start <= timestamp < end:
                # Chuyển BGR (OpenCV) sang RGB (Pillow)
                img_pil = Image.fromarray(cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                
                # Vẽ text với Pillow thay vì cv2.putText
                draw.text(
                    (ROI_X + 10, ROI_Y + 10),
                    text,
                    font=font,
                    fill=(255, 255, 255)
                )
                
                # Chuyển lại RGB sang BGR
                frame_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                break

        writer.write(frame_cv2)

    writer.release()
    assert video_path.exists(), "Tạo video tổng hợp thất bại"
    assert video_path.stat().st_size > 10_000, "File video quá nhỏ, có thể bị lỗi"
    return video_path


@pytest.fixture(scope="module")
def prompt_file_path(tmp_path_factory) -> Path:
    """Tạo file prompt template tạm cho tests."""
    tmp_dir = tmp_path_factory.mktemp("prompts")
    prompt_path = tmp_dir / "test_prompt.txt"
    prompt_path.write_text(
        "You are a subtitle extractor.\n"
        "{previous_context}\n"
        "Now extract all subtitles from the provided video clip:\n",
        encoding="utf-8",
    )
    return prompt_path


@pytest.fixture(scope="module")
def ocr_boxes():
    """Danh sách OcrBox theo boxesOCR.txt."""
    from video_subtitle_extractor.box_manager import OcrBox
    return [OcrBox(name="subtitle", x=ROI_X, y=ROI_Y, w=ROI_W, h=ROI_H)]


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# Không cần GPU, không cần file video.
# Test từng hàm nhỏ hoàn toàn tách biệt.
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_TimestampParser:
    """Unit tests cho _parse_timestamp_to_seconds()."""

    def _parse(self, ts: str) -> float:
        from video_subtitle_extractor.native_video_extractor import (
            NativeVideoSubtitleExtractor,
        )
        return NativeVideoSubtitleExtractor._parse_timestamp_to_seconds(ts)

    def test_mm_ss_dot(self):
        assert self._parse("00:03.20") == pytest.approx(3.20, rel=1e-3)

    def test_mm_ss_colon(self):
        assert self._parse("01:30.50") == pytest.approx(90.50, rel=1e-3)

    def test_hh_mm_ss(self):
        assert self._parse("01:02:03.40") == pytest.approx(3723.40, rel=1e-3)

    def test_zero(self):
        assert self._parse("00:00.00") == 0.0

    def test_invalid_raises(self):
        with pytest.raises((ValueError, IndexError)):
            self._parse("invalid_ts")


class TestLayer1_PromptBuilder:
    """Unit tests cho _load_prompt_template() và _build_prompt()."""

    @pytest.fixture()
    def extractor(self, prompt_file_path, ocr_boxes):
        from video_subtitle_extractor.native_video_extractor import (
            NativeVideoSubtitleExtractor,
        )
        return NativeVideoSubtitleExtractor(
            boxes=ocr_boxes,
            prompt_file=str(prompt_file_path),
        )

    def test_load_template_contains_placeholder(self, extractor):
        assert "{previous_context}" not in extractor._prompt_template
        # Sau khi load, template phải đã có nội dung
        assert len(extractor._prompt_template) > 10

    def test_build_prompt_no_context(self, extractor):
        prompt = extractor._build_prompt(previous_context="")
        assert "previous_context" not in prompt
        assert "extract" in prompt.lower()

    def test_build_prompt_with_context(self, extractor):
        context = "你好世界\n这是测试字幕"
        prompt = extractor._build_prompt(previous_context=context)
        assert context in prompt
        assert "DO NOT repeat" in prompt or "previous" in prompt.lower()

    def test_load_missing_file_raises(self, ocr_boxes):
        from video_subtitle_extractor.native_video_extractor import (
            NativeVideoSubtitleExtractor,
        )
        with pytest.raises(FileNotFoundError):
            NativeVideoSubtitleExtractor(
                boxes=ocr_boxes,
                prompt_file="/nonexistent/path/prompt.txt",
            )


class TestLayer1_BatchSplitter:
    """Unit tests cho _split_into_batches()."""

    @pytest.fixture()
    def extractor(self, prompt_file_path, ocr_boxes):
        from video_subtitle_extractor.native_video_extractor import (
            NativeVideoSubtitleExtractor,
        )
        return NativeVideoSubtitleExtractor(
            boxes=ocr_boxes,
            batch_duration=60.0,
            prompt_file=str(prompt_file_path),
        )

    def _make_frames(self, timestamps: List[float]):
        """Tạo fake frame list từ danh sách timestamp."""
        fake_img = Image.new("RGB", (10, 10), color=(0, 0, 0))
        return [(ts, fake_img) for ts in timestamps]

    def test_single_batch(self, extractor):
        frames = self._make_frames([0.0, 10.0, 30.0, 59.9])
        batches = extractor._split_into_batches(frames)
        assert len(batches) == 1
        assert batches[0][0] == pytest.approx(0.0)   # batch_start
        assert len(batches[0][2]) == 4               # frames in batch

    def test_two_batches(self, extractor):
        frames = self._make_frames([0.0, 30.0, 60.0, 90.0])
        batches = extractor._split_into_batches(frames)
        assert len(batches) == 2
        assert batches[0][0] == pytest.approx(0.0)
        assert batches[1][0] == pytest.approx(60.0)

    def test_empty_input(self, extractor):
        assert extractor._split_into_batches([]) == []

    def test_batch_boundary_exact(self, extractor):
        # Frame đúng tại t=60.0 → bắt đầu batch mới
        frames = self._make_frames(list(range(0, 120, 10)))  # 0,10,...,110
        batches = extractor._split_into_batches(frames)
        assert len(batches) == 2


class TestLayer1_ParseResponse:
    """Unit tests cho _parse_response_to_entries()."""

    @pytest.fixture()
    def extractor(self, prompt_file_path, ocr_boxes):
        from video_subtitle_extractor.native_video_extractor import (
            NativeVideoSubtitleExtractor,
        )
        return NativeVideoSubtitleExtractor(
            boxes=ocr_boxes,
            prompt_file=str(prompt_file_path),
        )

    def test_single_entry(self, extractor):
        raw = "[00:03.20 --> 00:06.80] 你好世界\n"
        entries = extractor._parse_response_to_entries(raw, batch_start_offset=0.0)
        assert len(entries) == 1
        assert entries[0].text == "你好世界"
        assert entries[0].start_time == pytest.approx(3.20, rel=1e-3)
        assert entries[0].end_time == pytest.approx(6.80, rel=1e-3)

    def test_multiple_entries(self, extractor):
        raw = (
            "[00:03.20 --> 00:06.80] 你好世界\n"
            "[00:07.00 --> 00:10.50] 我们今天\n"
            "[00:12.00 --> 00:15.40] 人工智能\n"
        )
        entries = extractor._parse_response_to_entries(raw, batch_start_offset=0.0)
        assert len(entries) == 3

    def test_offset_applied(self, extractor):
        """Offset batch_start được cộng vào mọi timestamp."""
        raw = "[00:03.00 --> 00:06.00] text\n"
        entries = extractor._parse_response_to_entries(raw, batch_start_offset=60.0)
        assert entries[0].start_time == pytest.approx(63.0, rel=1e-3)
        assert entries[0].end_time == pytest.approx(66.0, rel=1e-3)

    def test_invalid_entries_skipped(self, extractor):
        """Entry có end <= start bị bỏ qua."""
        raw = "[00:05.00 --> 00:03.00] bad entry\n"
        entries = extractor._parse_response_to_entries(raw, batch_start_offset=0.0)
        assert len(entries) == 0

    def test_empty_response(self, extractor):
        entries = extractor._parse_response_to_entries("", batch_start_offset=0.0)
        assert len(entries) == 0

    def test_with_thinking_block(self, extractor):
        """Response có <think>...</think> → đã bị strip trước khi parse."""
        from video_subtitle_extractor.ocr.qwen3vl import Qwen3VLOCR
        raw_with_thinking = (
            "<think>Analyzing the video frames...</think>\n"
            "[00:03.20 --> 00:06.80] 你好世界\n"
        )
        cleaned = Qwen3VLOCR.strip_thinking(raw_with_thinking)
        entries = extractor._parse_response_to_entries(cleaned, batch_start_offset=0.0)
        assert len(entries) == 1
        assert entries[0].text == "你好世界"


class TestLayer1_ConversationBuilder:
    """Unit tests cho _build_conversation() và _update_conversation()."""

    @pytest.fixture()
    def extractor(self, prompt_file_path, ocr_boxes):
        from video_subtitle_extractor.native_video_extractor import (
            NativeVideoSubtitleExtractor,
        )
        return NativeVideoSubtitleExtractor(
            boxes=ocr_boxes,
            prompt_file=str(prompt_file_path),
        )

    def _fake_frames(self, n: int = 3) -> List[Image.Image]:
        return [Image.new("RGB", (10, 10)) for _ in range(n)]

    def test_first_batch_single_turn(self, extractor):
        """Batch đầu: chỉ có 1 user turn, không có history."""
        messages = extractor._build_conversation([], self._fake_frames())
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_first_batch_no_previous_context_in_prompt(self, extractor):
        """Batch đầu: prompt không chứa 'DO NOT repeat' (không có context)."""
        messages = extractor._build_conversation([], self._fake_frames())
        text_content = next(
            item["text"] for item in messages[0]["content"] if item.get("type") == "text"
        )
        # Không có context → placeholder đã được thay bằng chuỗi rỗng
        assert "{previous_context}" not in text_content

    def test_second_batch_has_history(self, extractor):
        """Batch 2: có history từ batch 1 → messages có 3 items."""
        # Simulate: sau batch 1, conversation đã được update
        history = [
            {"role": "user", "content": [{"type": "text", "text": "prev prompt"}]},
            {"role": "assistant", "content": "你好世界\n这是测试"},
        ]
        messages = extractor._build_conversation(history, self._fake_frames())
        assert len(messages) == 3
        assert messages[2]["role"] == "user"

    def test_second_batch_context_injected(self, extractor):
        """Batch 2: prompt chứa text của assistant từ batch 1."""
        prev_output = "你好世界\n这是测试字幕"
        history = [
            {"role": "user", "content": [{"type": "text", "text": "prev"}]},
            {"role": "assistant", "content": prev_output},
        ]
        messages = extractor._build_conversation(history, self._fake_frames())
        new_user_text = next(
            item["text"] for item in messages[-1]["content"] if item.get("type") == "text"
        )
        assert prev_output in new_user_text

    def test_update_conversation_strips_video(self, extractor):
        """_update_conversation strip phần video khỏi user turn."""
        fake_video_turn = {
            "role": "user",
            "content": [
                {"type": "video", "video": self._fake_frames()},
                {"type": "text", "text": "extract subtitles"},
            ],
        }
        conversation = [fake_video_turn]
        updated = extractor._update_conversation(conversation, "output text")

        # Kết quả: [stripped_user, assistant]
        assert len(updated) == 2
        assert updated[0]["role"] == "user"
        assert updated[1]["role"] == "assistant"
        assert updated[1]["content"] == "output text"

        # User turn không còn video
        content_types = {item.get("type") for item in updated[0]["content"]}
        assert "video" not in content_types
        assert "text" in content_types

    def test_update_conversation_only_keeps_last_pair(self, extractor):
        """_update_conversation chỉ giữ 1 cặp [user, assistant] gần nhất."""
        # Simulate conversation dài với nhiều turn
        long_conversation = [
            {"role": "user", "content": [{"type": "text", "text": "turn1"}]},
            {"role": "assistant", "content": "response1"},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": []},
                    {"type": "text", "text": "turn2"},
                ],
            },
        ]
        updated = extractor._update_conversation(long_conversation, "response2")
        assert len(updated) == 2


class TestLayer1_VideoMessageFormat:
    """Unit tests cho _build_video_message() — xác nhận format đúng chuẩn Qwen."""

    @pytest.fixture()
    def extractor(self, prompt_file_path, ocr_boxes):
        from video_subtitle_extractor.native_video_extractor import (
            NativeVideoSubtitleExtractor,
        )
        return NativeVideoSubtitleExtractor(
            boxes=ocr_boxes,
            sample_fps=SAMPLE_FPS,
            prompt_file=str(prompt_file_path),
        )

    def test_type_video_present(self, extractor):
        """BẮT BUỘC: content item phải có type='video'."""
        frames = [Image.new("RGB", (10, 10)) for _ in range(5)]
        message = extractor._build_video_message(frames, "test prompt")
        video_items = [c for c in message["content"] if c.get("type") == "video"]
        assert len(video_items) == 1, "Phải có đúng 1 item type='video'"

    def test_sample_fps_not_fps(self, extractor):
        """Khi input là List[PIL.Image], phải dùng 'sample_fps' không phải 'fps'."""
        frames = [Image.new("RGB", (10, 10)) for _ in range(5)]
        message = extractor._build_video_message(frames, "test prompt")
        video_item = next(c for c in message["content"] if c.get("type") == "video")
        assert "sample_fps" in video_item, "Phải có sample_fps cho frame list input"
        assert "fps" not in video_item, "fps chỉ dùng cho file path / URL input"

    def test_sample_fps_value(self, extractor):
        frames = [Image.new("RGB", (10, 10)) for _ in range(3)]
        message = extractor._build_video_message(frames, "test")
        video_item = next(c for c in message["content"] if c.get("type") == "video")
        assert video_item["sample_fps"] == SAMPLE_FPS

    def test_frames_list_assigned(self, extractor):
        frames = [Image.new("RGB", (10, 10)) for _ in range(7)]
        message = extractor._build_video_message(frames, "test")
        video_item = next(c for c in message["content"] if c.get("type") == "video")
        assert video_item["video"] is frames
        assert len(video_item["video"]) == 7


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — COMPONENT TESTS (Synthetic Video)
# Test luồng đọc frame → crop ROI → batch splitting.
# Không cần GPU hay model.
# ═════════════════════════════════════════════════════════════════════

class TestLayer2_FrameSampling:
    """Test iter_sampled_frames() với video tổng hợp."""

    def test_yields_correct_number_of_frames(self, synthetic_video_path):
        from video_subtitle_extractor.frame_processor import iter_sampled_frames

        frames = list(iter_sampled_frames(str(synthetic_video_path), frame_interval=FRAME_INTERVAL))
        # 75s * 30fps = 2250 frames / 6 = 375 frames
        expected = int(75 * VIDEO_FPS / FRAME_INTERVAL)
        assert len(frames) == pytest.approx(expected, rel=0.05)

    def test_tuple_structure(self, synthetic_video_path):
        from video_subtitle_extractor.frame_processor import iter_sampled_frames

        first_frame = next(iter_sampled_frames(str(synthetic_video_path), frame_interval=FRAME_INTERVAL))
        frame_no, timestamp, frame_bgr = first_frame

        assert isinstance(frame_no, int)
        assert isinstance(timestamp, float)
        assert isinstance(frame_bgr, np.ndarray)
        assert frame_bgr.shape == (VIDEO_H, VIDEO_W, 3)

    def test_timestamps_increase_monotonically(self, synthetic_video_path):
        from video_subtitle_extractor.frame_processor import iter_sampled_frames

        frames = list(iter_sampled_frames(str(synthetic_video_path), frame_interval=FRAME_INTERVAL))
        timestamps = [ts for _, ts, _ in frames]
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1]

    def test_nonexistent_video_raises(self):
        from video_subtitle_extractor.frame_processor import iter_sampled_frames

        with pytest.raises(RuntimeError, match="Cannot open video"):
            list(iter_sampled_frames("/nonexistent/video.mp4", frame_interval=6))


class TestLayer2_RoiCrop:
    """Test crop_roi() cho đúng kích thước và vị trí."""

    def test_crop_returns_correct_shape(self, synthetic_video_path, ocr_boxes):
        from video_subtitle_extractor.frame_processor import (
            FrameProcessor,
            iter_sampled_frames,
        )

        fp = FrameProcessor(frame_interval=FRAME_INTERVAL)
        _, _, frame = next(iter_sampled_frames(str(synthetic_video_path), FRAME_INTERVAL))
        roi = fp.crop_roi(frame, ocr_boxes[0])

        assert roi.shape == (ROI_H, ROI_W, 3)

    def test_roi_contains_text_at_subtitle_time(self, synthetic_video_path, ocr_boxes):
        """
        Tại thời điểm subtitle xuất hiện (1.0s-3.5s),
        ROI phải có pixel sáng (chữ trắng).
        """
        from video_subtitle_extractor.frame_processor import (
            FrameProcessor,
            iter_sampled_frames,
        )

        fp = FrameProcessor(frame_interval=FRAME_INTERVAL)

        # Lấy frame tại khoảng 1.2s → subtitle đầu tiên (1.0-3.5s)
        for _, timestamp, frame in iter_sampled_frames(str(synthetic_video_path), FRAME_INTERVAL):
            if 1.0 <= timestamp <= 3.5:
                roi = fp.crop_roi(frame, ocr_boxes[0])
                bright_pixels = int(np.sum(roi > 200))
                assert bright_pixels > 500, (
                    f"ROI tại t={timestamp:.2f}s phải có chữ sáng, "
                    f"nhưng chỉ có {bright_pixels} pixel sáng"
                )
                break

    def test_roi_is_dark_outside_subtitle(self, synthetic_video_path, ocr_boxes):
        """
        Tại thời điểm không có subtitle (ví dụ 4.0s),
        ROI phải gần như tối hoàn toàn.
        """
        from video_subtitle_extractor.frame_processor import (
            FrameProcessor,
            iter_sampled_frames,
        )

        fp = FrameProcessor(frame_interval=FRAME_INTERVAL)

        for _, timestamp, frame in iter_sampled_frames(str(synthetic_video_path), FRAME_INTERVAL):
            if 3.8 <= timestamp <= 4.5:
                roi = fp.crop_roi(frame, ocr_boxes[0])
                bright_pixels = int(np.sum(roi > 200))
                assert bright_pixels < 100, (
                    f"ROI tại t={timestamp:.2f}s không có subtitle, "
                    f"nhưng có {bright_pixels} pixel sáng"
                )
                break


class TestLayer2_BatchSplittingWithRealFrames:
    """Test _split_into_batches() với frames thật từ video tổng hợp."""

    @pytest.fixture()
    def extractor(self, prompt_file_path, ocr_boxes):
        from video_subtitle_extractor.native_video_extractor import (
            NativeVideoSubtitleExtractor,
        )
        return NativeVideoSubtitleExtractor(
            boxes=ocr_boxes,
            frame_interval=FRAME_INTERVAL,
            batch_duration=BATCH_DURATION,
            sample_fps=SAMPLE_FPS,
            prompt_file=str(prompt_file_path),
        )

    def test_75s_video_splits_into_two_batches(self, extractor, synthetic_video_path, ocr_boxes):
        """Video 75s với batch_duration=60s → phải có 2 batches."""
        from video_subtitle_extractor.frame_processor import (
            FrameProcessor,
            iter_sampled_frames,
        )

        fp = FrameProcessor(frame_interval=FRAME_INTERVAL)
        sampled = []
        for _, timestamp, frame_bgr in iter_sampled_frames(str(synthetic_video_path), FRAME_INTERVAL):
            for box in ocr_boxes:
                roi = fp.crop_roi(frame_bgr, box)
                pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                sampled.append((timestamp, pil_img))

        batches = extractor._split_into_batches(sampled)

        assert len(batches) == 2, f"Expected 2 batches, got {len(batches)}"
        assert batches[0][0] == pytest.approx(0.0, abs=0.1)   # batch 1 start
        assert batches[1][0] >= 60.0                            # batch 2 start

    def test_batch_frames_are_pil_images(self, extractor, synthetic_video_path, ocr_boxes):
        """Frames trong batch phải là PIL.Image."""
        from video_subtitle_extractor.frame_processor import (
            FrameProcessor,
            iter_sampled_frames,
        )

        fp = FrameProcessor(frame_interval=FRAME_INTERVAL)
        sampled = []
        for _, ts, frame in iter_sampled_frames(str(synthetic_video_path), FRAME_INTERVAL):
            for box in ocr_boxes:
                roi = fp.crop_roi(frame, box)
                pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                sampled.append((ts, pil_img))

        batches = extractor._split_into_batches(sampled)
        for _, _, frames_in_batch in batches:
            for ts, img in frames_in_batch:
                assert isinstance(img, Image.Image)
                assert img.size == (ROI_W, ROI_H)


# ═════════════════════════════════════════════════════════════════════
# LAYER 3 — PIPELINE INTEGRATION (Mocked Model)
# Test toàn bộ extract() với model bị mock.
# ═════════════════════════════════════════════════════════════════════

# Response giả lập từ model cho mỗi batch
_MOCK_BATCH1_RESPONSE = (
    "[00:01.00 --> 00:03.50] 你好世界\n"
    "[00:05.00 --> 00:08.00] 这是测试字幕\n"
    "[00:10.00 --> 00:13.00] 第三段字幕内容\n"
)
_MOCK_BATCH2_RESPONSE = (
    "[00:02.00 --> 00:05.00] 跨越第二批次的字幕\n"
    "[00:07.00 --> 00:10.00] 最后一段字幕\n"
)


@pytest.fixture()
def mocked_extractor(prompt_file_path, ocr_boxes, monkeypatch):
    """
    Tạo NativeVideoSubtitleExtractor với _load_model và _infer bị mock.
    Không cần GPU, không cần tải model.
    """
    from video_subtitle_extractor.native_video_extractor import (
        NativeVideoSubtitleExtractor,
    )

    extractor = NativeVideoSubtitleExtractor(
        boxes=ocr_boxes,
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        device="cuda",
        frame_interval=FRAME_INTERVAL,
        batch_duration=BATCH_DURATION,
        sample_fps=SAMPLE_FPS,
        max_new_tokens=MAX_NEW_TOKENS,
        prompt_file=str(prompt_file_path),
    )

    # Theo dõi số lần gọi
    load_call_count = {"n": 0}

    def fake_load_model():
        load_call_count["n"] += 1
        extractor._model_loaded = True

    infer_responses = iter([_MOCK_BATCH1_RESPONSE, _MOCK_BATCH2_RESPONSE])

    def fake_infer(messages):
        return next(infer_responses, "")

    monkeypatch.setattr(extractor, "_load_model", fake_load_model)
    monkeypatch.setattr(extractor, "_infer", fake_infer)

    return extractor, load_call_count


class TestLayer3_FullPipeline:
    """Integration test cho toàn bộ pipeline extract()."""

    def test_returns_extraction_result(self, mocked_extractor, synthetic_video_path, tmp_path):
        from video_subtitle_extractor.native_video_extractor import NativeExtractionResult

        extractor, _ = mocked_extractor
        result = extractor.extract(str(synthetic_video_path), output_dir=str(tmp_path))
        assert isinstance(result, NativeExtractionResult)

    def test_model_loaded_exactly_once(self, mocked_extractor, synthetic_video_path, tmp_path):
        """_load_model() chỉ được gọi 1 lần dù có nhiều batches."""
        extractor, load_call_count = mocked_extractor
        extractor.extract(str(synthetic_video_path), output_dir=str(tmp_path))
        assert load_call_count["n"] == 1

    def test_srt_file_created(self, mocked_extractor, synthetic_video_path, tmp_path):
        """File SRT output phải tồn tại sau khi extract."""
        extractor, _ = mocked_extractor
        result = extractor.extract(str(synthetic_video_path), output_dir=str(tmp_path))
        srt_path = Path(result.output_paths["srt"])
        assert srt_path.exists()
        assert srt_path.stat().st_size > 0

    def test_srt_filename_convention(self, mocked_extractor, synthetic_video_path, tmp_path):
        """Tên file SRT: <video_stem>_native.srt."""
        extractor, _ = mocked_extractor
        result = extractor.extract(str(synthetic_video_path), output_dir=str(tmp_path))
        srt_path = Path(result.output_paths["srt"])
        assert srt_path.name == "synthetic_native_ocr_native.srt"

    def test_entries_count(self, mocked_extractor, synthetic_video_path, tmp_path):
        """Tổng số entries phải bằng tổng từ 2 batch responses."""
        extractor, _ = mocked_extractor
        result = extractor.extract(str(synthetic_video_path), output_dir=str(tmp_path))
        # Batch 1: 3 entries, Batch 2: 2 entries → total 5
        assert result.total_entries == 5

    def test_batch2_timestamps_offset_correctly(
        self, mocked_extractor, synthetic_video_path, tmp_path
    ):
        """
        Timestamps trong batch 2 phải có offset = batch2_start (~60s).
        Mock response: [00:02.00 --> 00:05.00] → absolute: 62.0 --> 65.0
        """
        extractor, _ = mocked_extractor
        result = extractor.extract(str(synthetic_video_path), output_dir=str(tmp_path))

        # Tìm entry thuộc batch 2
        batch2_entries = [e for e in result.entries if e.start_time > 60.0]
        assert len(batch2_entries) == 2, f"Batch 2 phải có 2 entries, got {len(batch2_entries)}"

        # Entry đầu batch 2: 00:02.00 + 60.0 offset = 62.0
        first_b2 = min(batch2_entries, key=lambda e: e.start_time)
        assert first_b2.start_time == pytest.approx(62.0, abs=0.5)

    def test_srt_content_valid(self, mocked_extractor, synthetic_video_path, tmp_path):
        """File SRT phải có định dạng hợp lệ với timestamps."""
        extractor, _ = mocked_extractor
        result = extractor.extract(str(synthetic_video_path), output_dir=str(tmp_path))
        content = Path(result.output_paths["srt"]).read_text(encoding="utf-8")

        assert "你好世界" in content
        assert "-->" in content

    def test_metadata_contains_expected_fields(
        self, mocked_extractor, synthetic_video_path, tmp_path
    ):
        """Metadata phải chứa các trường cấu hình."""
        extractor, _ = mocked_extractor
        result = extractor.extract(str(synthetic_video_path), output_dir=str(tmp_path))

        required_keys = {"model", "device", "frame_interval", "batch_duration", "sample_fps"}
        assert required_keys.issubset(set(result.metadata.keys()))


class TestLayer3_OptionalOutputs:
    """Test các output tùy chọn: warn_english, save_minify_txt."""

    @pytest.fixture()
    def extractor_with_options(self, prompt_file_path, ocr_boxes, monkeypatch):
        from video_subtitle_extractor.native_video_extractor import (
            NativeVideoSubtitleExtractor,
        )

        ext = NativeVideoSubtitleExtractor(
            boxes=ocr_boxes,
            frame_interval=FRAME_INTERVAL,
            batch_duration=BATCH_DURATION,
            sample_fps=SAMPLE_FPS,
            warn_english=True,
            save_minify_txt=True,
            prompt_file=str(prompt_file_path),
        )

        def fake_load():
            ext._model_loaded = True

        responses = iter(["[00:01.00 --> 00:03.00] Hello World ABC\n"])

        monkeypatch.setattr(ext, "_load_model", fake_load)
        monkeypatch.setattr(ext, "_infer", lambda msgs: next(responses, ""))
        return ext

    def test_save_minify_txt_creates_file(
        self, extractor_with_options, synthetic_video_path, tmp_path
    ):
        result = extractor_with_options.extract(str(synthetic_video_path), output_dir=str(tmp_path))
        assert "txt" in result.output_paths
        assert Path(result.output_paths["txt"]).exists()

    def test_warn_english_creates_file(
        self, extractor_with_options, synthetic_video_path, tmp_path
    ):
        """English text trong subtitle → file warning được tạo."""
        result = extractor_with_options.extract(str(synthetic_video_path), output_dir=str(tmp_path))
        assert "warnings" in result.output_paths
        assert Path(result.output_paths["warnings"]).exists()


# ═════════════════════════════════════════════════════════════════════
# LAYER 4 — REAL MODEL TEST (GPU Required)
# Chạy với model Qwen3-VL thật.
# Bị skip tự động nếu CUDA không khả dụng hoặc VRAM không đủ.
# ═════════════════════════════════════════════════════════════════════

def _check_gpu_available(min_vram_gb: float = 10.0) -> tuple:
    """Trả về (ok: bool, reason: str)."""
    try:
        import torch
    except ImportError:
        return False, "torch chưa cài đặt"
    if not torch.cuda.is_available():
        return False, "CUDA không khả dụng"
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram < min_vram_gb:
        return False, f"VRAM {vram:.1f}GB < {min_vram_gb}GB yêu cầu"
    return True, f"GPU OK ({vram:.1f}GB VRAM)"


_GPU_OK, _GPU_REASON = _check_gpu_available(
    float(os.getenv("NATIVE_OCR_MIN_VRAM_GB", "10"))
)


@pytest.mark.gpu
@pytest.mark.skipif(not _GPU_OK, reason=f"GPU: {_GPU_REASON}")
class TestLayer4_RealModelOCR:
    """
    Test với model Qwen3-VL thật trên GPU.
    Xác nhận chất lượng OCR thực tế với video tổng hợp đã biết trước nội dung.

    Để chạy layer này:
        NATIVE_OCR_MIN_VRAM_GB=15 pytest tests/test_native_video_ocr_pipeline.py -v -k "Layer4"
    """

    @pytest.fixture(scope="class")
    def real_extractor(self, prompt_file_path, ocr_boxes):
        """
        Khởi tạo extractor với model thật — được cache ở scope="class"
        để không load model lại cho mỗi test method.
        """
        from video_subtitle_extractor.native_video_extractor import (
            NativeVideoSubtitleExtractor,
        )

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

        ext = NativeVideoSubtitleExtractor(
            boxes=ocr_boxes,
            model_name=os.getenv("TEST_OCR_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
            device="cuda",
            hf_token=hf_token,
            frame_interval=FRAME_INTERVAL,
            batch_duration=30.0,   # Batch ngắn hơn trong test để nhanh
            sample_fps=SAMPLE_FPS,
            max_new_tokens=512,
            prompt_file=str(prompt_file_path),
        )
        ext._load_model()  # Load trước để tính thời gian load riêng
        return ext

    def test_model_loads_without_error(self, real_extractor):
        assert real_extractor._model_loaded is True

    def test_pipeline_runs_on_synthetic_video(
        self, real_extractor, synthetic_video_path, tmp_path
    ):
        """Pipeline không raise exception với video tổng hợp."""
        result = real_extractor.extract(str(synthetic_video_path), output_dir=str(tmp_path))
        assert result.total_entries >= 0  # Có thể 0 nếu OCR không nhận ra font

    def test_srt_file_created_by_real_model(
        self, real_extractor, synthetic_video_path, tmp_path
    ):
        result = real_extractor.extract(str(synthetic_video_path), output_dir=str(tmp_path))
        assert Path(result.output_paths["srt"]).exists()

    def test_at_least_one_subtitle_detected(
        self, real_extractor, synthetic_video_path, tmp_path
    ):
        """
        Model thật phải nhận ra ít nhất 1 subtitle từ video tổng hợp.

        Nếu test này fail với real model:
        → Font cv2 không phải CJK → model không nhận ra.
        → Cần dùng Pillow để vẽ chữ CJK (xem fixture synthetic_video_path_cjk bên dưới).
        """
        result = real_extractor.extract(str(synthetic_video_path), output_dir=str(tmp_path))
        assert result.total_entries >= 1, (
            f"Model không detect được subtitle nào. "
            f"Gợi ý: Thay cv2.putText bằng Pillow với font CJK trong fixture tạo video."
        )
