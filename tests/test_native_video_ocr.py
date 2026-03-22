#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_native_video_ocr.py — Integration-style test cho Native Video OCR pipeline
(model được mock nhưng vẫn yêu cầu GPU preflight thật)
"""

from pathlib import Path

import pytest


def test_native_video_pipeline_with_mocked_model_and_gpu_preflight(
    tmp_path,
    monkeypatch,
    native_video_gpu_preflight,
):
    """
    Kiểm thử pipeline Native Video OCR:
    - Có bước preflight GPU thật (CUDA + VRAM tối thiểu)
    - Mock load model và inference để test dataflow ổn định
    - Xác nhận batching, multi-turn context, timestamp offset và output SRT
    """
    # Import deps nặng ngay trong test để có thể skip sạch khi thiếu môi trường
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")

    from video_subtitle_extractor.box_manager import OcrBox
    import video_subtitle_extractor.native_video_extractor as native_module
    from video_subtitle_extractor.native_video_extractor import (
        NativeExtractionResult,
        NativeVideoSubtitleExtractor,
    )

    # 1) GPU preflight phải cung cấp metadata hợp lệ
    assert native_video_gpu_preflight["cuda_available"] is True
    assert native_video_gpu_preflight["device_name"]
    assert native_video_gpu_preflight["total_vram_gb"] > 0

    # 2) Chuẩn bị input tạm
    prompt_file = tmp_path / "native_prompt.txt"
    prompt_file.write_text(
        "You are a subtitle extractor.\n{previous_context}\nNow extract subtitles:\n",
        encoding="utf-8",
    )

    video_path = tmp_path / "sample_video.mp4"
    video_path.write_bytes(b"mock-video-placeholder")

    boxes = [OcrBox(name="subtitle", x=0, y=0, w=200, h=40)]

    extractor = NativeVideoSubtitleExtractor(
        boxes=boxes,
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        device="cuda",
        frame_interval=6,
        batch_duration=60.0,
        sample_fps=5.0,
        max_new_tokens=256,
        warn_english=False,
        save_minify_txt=False,
        prompt_file=str(prompt_file),
    )

    assert extractor.device == "cuda"

    # 3) Mock frame sampling và crop ROI
    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    sampled_frames = [
        (0, 0.0, frame),
        (60, 10.0, frame),
        (366, 61.0, frame),
        (420, 70.0, frame),
    ]

    def fake_iter_sampled_frames(video_path_arg: str, frame_interval_arg: int = 6):
        assert Path(video_path_arg) == video_path
        assert frame_interval_arg == 6
        for item in sampled_frames:
            yield item

    def fake_crop_roi(self, frame_bgr, box):
        return np.full((40, 200, 3), 127, dtype=np.uint8)

    monkeypatch.setattr(native_module, "iter_sampled_frames", fake_iter_sampled_frames)
    monkeypatch.setattr(native_module.FrameProcessor, "crop_roi", fake_crop_roi)

    # 4) Mock model loading + inference
    load_calls = {"count": 0}

    def fake_load_model():
        load_calls["count"] += 1
        assert extractor.device.startswith("cuda")
        extractor._model_loaded = True

    infer_calls = []
    fake_responses = [
        "[00:00.10 --> 00:02.10] alpha zebra\n[00:03.00 --> 00:05.00] quantum moon",
        "[00:00.10 --> 00:01.90] violet river\n[00:03.00 --> 00:04.80] copper stone",
    ]

    def fake_infer(messages):
        infer_calls.append(messages)
        return fake_responses[len(infer_calls) - 1]

    monkeypatch.setattr(extractor, "_load_model", fake_load_model)
    monkeypatch.setattr(extractor, "_infer", fake_infer)

    # 5) Execute pipeline
    result = extractor.extract(str(video_path), output_dir=str(tmp_path))

    # 6) Assert core result
    assert isinstance(result, NativeExtractionResult)
    assert load_calls["count"] == 1
    assert len(infer_calls) == 2

    assert result.total_sampled_frames == 4
    assert result.total_batches == 2
    assert result.total_entries == 4
    assert result.metadata["device"] == "cuda"

    # 7) Assert multi-turn context ở batch 2
    second_call_messages = infer_calls[1]
    assert len(second_call_messages) == 3

    history_user, history_assistant, current_user = second_call_messages

    assert history_user["role"] == "user"
    assert all(item.get("type") == "text" for item in history_user["content"])
    assert all(item.get("type") != "video" for item in history_user["content"])

    assert history_assistant["role"] == "assistant"
    assert "alpha zebra" in history_assistant["content"]

    current_user_types = {item.get("type") for item in current_user["content"]}
    assert current_user_types == {"video", "text"}

    video_item = next(item for item in current_user["content"] if item.get("type") == "video")
    text_item = next(item for item in current_user["content"] if item.get("type") == "text")
    assert len(video_item["video"]) == 2
    assert "alpha zebra" in text_item["text"]

    # 8) Assert timestamp offset của batch 2
    by_text = {entry.text: entry for entry in result.entries}
    assert by_text["alpha zebra"].start_time == pytest.approx(0.10, rel=1e-3)
    assert by_text["quantum moon"].start_time == pytest.approx(3.00, rel=1e-3)
    assert by_text["violet river"].start_time == pytest.approx(61.10, rel=1e-3)
    assert by_text["copper stone"].start_time == pytest.approx(64.00, rel=1e-3)

    # 9) Assert output SRT
    srt_path = Path(result.output_paths["srt"])
    assert srt_path.name == "sample_video_native.srt"
    assert srt_path.exists()

    srt_content = srt_path.read_text(encoding="utf-8")
    assert "alpha zebra" in srt_content
    assert "quantum moon" in srt_content
    assert "violet river" in srt_content
    assert "copper stone" in srt_content
    assert "00:01:01" in srt_content
