# -*- coding: utf-8 -*-
"""
Native Video Subtitle Extractor (Qwen3-VL)

Luồng xử lý:
1) Lấy mẫu frame theo frame_interval
2) Crop ROI subtitle từ boxes
3) Gom frame theo batch_duration (mặc định 60s)
4) Gọi Qwen3-VL theo Native Video input (type="video", video=List[PIL.Image], sample_fps=...)
5) Multi-turn context: giữ text lượt trước, bỏ tham chiếu video ở lịch sử
6) Parse về SubtitleEntry, deduplicate và xuất SRT/TXT/warnings
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from PIL import Image

# Thêm project root vào path để import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

from .box_manager import OcrBox
from .frame_processor import FrameProcessor, iter_sampled_frames
from .ocr.qwen3vl import Qwen3VLOCR
from .subtitle_writer import SubtitleEntry, SubtitleWriter

logger = get_logger(__name__)


@dataclass
class NativeExtractionResult:
    """Kết quả extraction cho Native Video mode."""

    video_path: str
    total_sampled_frames: int
    total_batches: int
    total_entries: int
    output_paths: Dict[str, str]
    entries: List[SubtitleEntry]
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


class NativeVideoSubtitleExtractor:
    """
    Trích xuất subtitle theo Qwen3-VL Native Video mode.

    Điểm khác biệt với extractor cũ:
    - Không OCR từng ảnh độc lập.
    - Mỗi lượt inference nhận một clip frame-list (đã crop ROI) dưới dạng content type="video".
    """

    def __init__(
        self,
        boxes: List[OcrBox],
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        hf_token: Optional[str] = None,
        frame_interval: int = 6,
        batch_duration: float = 60.0,
        sample_fps: float = 5.0,
        max_new_tokens: int = 2048,
        total_pixels: int = 20480 * 32 * 32,
        min_pixels: int = 64 * 32 * 32,
        max_frames: int = 2048,
        warn_english: bool = False,
        save_minify_txt: bool = False,
        prompt_file: str = "prompts/native_video_ocr_prompt.txt",
    ):
        if not boxes:
            raise ValueError("Phải cung cấp ít nhất 1 OcrBox")

        self.boxes = boxes

        self.model_name = model_name
        self.device = device
        self.hf_token = hf_token

        self.frame_interval = frame_interval
        self.batch_duration = batch_duration
        self.sample_fps = sample_fps

        self.max_new_tokens = max_new_tokens
        self.total_pixels = total_pixels
        self.min_pixels = min_pixels
        self.max_frames = max_frames

        self.warn_english = warn_english
        self.save_minify_txt = save_minify_txt

        self.prompt_file = prompt_file
        self._prompt_template = self._load_prompt_template(prompt_file)

        self._processor = None
        self._model = None
        self._model_loaded = False

        logger.info("NativeVideoSubtitleExtractor initialized")
        logger.info("  model=%s", model_name)
        logger.info("  device=%s", device)
        logger.info("  frame_interval=%s", frame_interval)
        logger.info("  batch_duration=%s", batch_duration)
        logger.info("  sample_fps=%s", sample_fps)
        logger.info("  boxes=%s", [b.name for b in boxes])

    def _load_prompt_template(self, prompt_file: str) -> str:
        """Đọc file prompt template từ disk."""
        path = Path(prompt_file)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file không tồn tại: {prompt_file}")
        return path.read_text(encoding="utf-8")

    def _build_prompt(self, previous_context: str = "") -> str:
        """
        Inject previous_context vào template.
        Nếu previous_context rỗng, placeholder được thay bằng chuỗi rỗng.
        """
        context_block = (
            "\nFor reference, here is what was extracted from the PREVIOUS clip "
            f"(DO NOT repeat these):\n{previous_context}\n"
            if previous_context.strip()
            else ""
        )
        return self._prompt_template.format(previous_context=context_block)

    def _load_model(self) -> None:
        """Load Qwen3-VL model + processor cho native video inference."""
        if self._model_loaded:
            return

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            logger.info("Loading Native Video processor from %s...", self.model_name)
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                token=self.hf_token,
                min_pixels=self.min_pixels,
                max_pixels=self.total_pixels,
            )

            if hasattr(self._processor, "image_processor"):
                self._processor.image_processor.do_resize = False

            if hasattr(self._processor, "tokenizer") and self._processor.tokenizer:
                self._processor.tokenizer.padding_side = "left"

            model_kwargs = {
                "torch_dtype": torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
                "token": self.hf_token,
            }

            if self.device.startswith("cuda"):
                try:
                    import flash_attn  # noqa: F401

                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Using flash_attention_2 for optimization.")
                except ImportError:
                    logger.warning("flash_attn not found. Falling back to default attention.")

            logger.info("Loading Native Video model from %s...", self.model_name)
            self._model = AutoModelForImageTextToText.from_pretrained(self.model_name, **model_kwargs)
            self._model = self._model.eval().to(self.device)
            self._model_loaded = True
            logger.info("Native Video model loaded successfully.")

        except Exception as e:
            logger.error("Error loading native video model: %s", e)
            raise

    def _build_video_message(self, pil_frames: List[Image.Image], prompt: str) -> Dict[str, Any]:
        """Xây dựng 1 user turn: video(list frame) + text prompt."""
        return {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": pil_frames,
                    "sample_fps": self.sample_fps,
                    "total_pixels": self.total_pixels,
                    "min_pixels": self.min_pixels,
                    "max_frames": self.max_frames,
                },
                {"type": "text", "text": prompt},
            ],
        }

    def _build_conversation(
        self,
        conversation_history: List[Dict[str, Any]],
        new_video_frames: List[Image.Image],
    ) -> List[Dict[str, Any]]:
        """
        Strategy:
        - Batch đầu tiên: prompt không context
        - Batch tiếp theo: lấy text assistant trước đó làm previous_context
        - Lịch sử chỉ giữ text-only turns
        """
        previous_context = ""
        if conversation_history:
            last_assistant = next(
                (m for m in reversed(conversation_history) if m.get("role") == "assistant"),
                None,
            )
            if last_assistant:
                previous_context = str(last_assistant.get("content", ""))

        prompt = self._build_prompt(previous_context=previous_context)
        new_user_turn = self._build_video_message(new_video_frames, prompt)
        return conversation_history + [new_user_turn]

    def _update_conversation(
        self,
        conversation: List[Dict[str, Any]],
        assistant_response: str,
    ) -> List[Dict[str, Any]]:
        """
        Sau mỗi batch:
        - strip video refs khỏi user turn cuối (giữ text prompt)
        - thêm assistant turn
        - chỉ giữ cặp gần nhất [user_text, assistant]
        """
        if not conversation:
            return [{"role": "assistant", "content": assistant_response}]

        last_user = conversation[-1]
        text_only_content = [
            item
            for item in last_user.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        stripped_user = {"role": "user", "content": text_only_content}
        assistant_turn = {"role": "assistant", "content": assistant_response}
        return [stripped_user, assistant_turn]

    def _split_into_batches(
        self,
        sampled_frames: List[Tuple[float, Image.Image]],
    ) -> List[Tuple[float, float, List[Tuple[float, Image.Image]]]]:
        """Chia sampled frames thành các batch theo self.batch_duration (giây)."""
        if not sampled_frames:
            return []

        batches: List[Tuple[float, float, List[Tuple[float, Image.Image]]]] = []
        batch_start = sampled_frames[0][0]
        current_batch: List[Tuple[float, Image.Image]] = []

        for timestamp, frame in sampled_frames:
            if (timestamp - batch_start) >= self.batch_duration and current_batch:
                batches.append((batch_start, timestamp, current_batch))
                batch_start = timestamp
                current_batch = []
            current_batch.append((timestamp, frame))

        if current_batch:
            batches.append((batch_start, current_batch[-1][0], current_batch))

        return batches

    def _infer(self, messages: List[Dict[str, Any]]) -> str:
        """Chạy 1 lượt native video inference với Qwen3-VL."""
        if not self._model_loaded:
            self._load_model()

        from qwen_vl_utils import process_vision_info

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages],
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True,
        )

        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs = list(video_inputs)
            video_metadatas = list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,
            do_resize=False,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]

        output_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return output_text[0] if output_text else ""

    @staticmethod
    def _parse_timestamp_to_seconds(ts: str) -> float:
        """
        Parse timestamp linh hoạt:
        - MM:SS.ff
        - HH:MM:SS.ff
        - hỗ trợ cả dấu ':' hoặc '.' ở phần sub-second
        """
        ts = ts.strip().replace(",", ".")
        parts = ts.split(":")

        if len(parts) == 2:
            mm = int(parts[0])
            ss = float(parts[1])
            return mm * 60 + ss

        if len(parts) == 3:
            hh = int(parts[0])
            mm = int(parts[1])
            ss = float(parts[2])
            return hh * 3600 + mm * 60 + ss

        raise ValueError(f"Invalid timestamp format: {ts}")

    def _parse_response_to_entries(
        self,
        raw_text: str,
        batch_start_offset: float,
    ) -> List[SubtitleEntry]:
        """
        Parse output format:
            [MM:SS.ff --> MM:SS.ff] text
        hoặc:
            [HH:MM:SS.ff --> HH:MM:SS.ff] text
        """
        entries: List[SubtitleEntry] = []

        # 2 nhóm timestamp, 1 nhóm text đến trước timestamp tiếp theo hoặc hết chuỗi
        pattern = re.compile(
            r"\[(\d{1,2}:\d{2}(?::\d{2})?[\.:]\d{2})\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?[\.:]\d{2})\]\s*(.+?)(?=\n\[|\Z)",
            flags=re.DOTALL,
        )

        for match in pattern.finditer(raw_text):
            start_str, end_str, text = match.groups()
            text = text.strip()
            if not text:
                continue

            try:
                start_sec = self._parse_timestamp_to_seconds(start_str) + batch_start_offset
                end_sec = self._parse_timestamp_to_seconds(end_str) + batch_start_offset
            except Exception:
                continue

            if end_sec <= start_sec:
                continue

            entries.append(
                SubtitleEntry(
                    index=len(entries) + 1,
                    start_time=start_sec,
                    end_time=end_sec,
                    text=text,
                )
            )

        return entries

    def extract(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
    ) -> NativeExtractionResult:
        """Trích xuất subtitle từ video bằng Native Video mode."""
        start_time = time.time()

        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        out_dir = Path(output_dir) if output_dir else video_path_obj.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        video_stem = video_path_obj.stem

        logger.info("=" * 60)
        logger.info("🎬 Native Video Subtitle Extraction")
        logger.info("=" * 60)
        logger.info("Input: %s", video_path_obj)
        logger.info("Output dir: %s", out_dir)

        self._load_model()

        frame_processor = FrameProcessor(frame_interval=self.frame_interval, scene_threshold=30.0)

        sampled: List[Tuple[float, Image.Image]] = []
        sampled_count = 0

        logger.info("📹 Sampling + cropping ROI frames...")
        for _, timestamp, frame_bgr in iter_sampled_frames(str(video_path_obj), self.frame_interval):
            sampled_count += 1
            for box in self.boxes:
                roi_bgr = frame_processor.crop_roi(frame_bgr, box)
                if roi_bgr is None or roi_bgr.size == 0:
                    continue
                roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(roi_rgb)
                sampled.append((timestamp, pil_img))

        batches = self._split_into_batches(sampled)
        logger.info("📦 Total sampled frames: %d", sampled_count)
        logger.info("📦 Total ROI frames: %d", len(sampled))
        logger.info("📦 Total batches: %d", len(batches))

        all_entries: List[SubtitleEntry] = []
        conversation: List[Dict[str, Any]] = []

        for i, (batch_start, batch_end, batch_frames) in enumerate(batches, start=1):
            logger.info(
                "⏳ Processing batch %d/%d: %.2fs - %.2fs",
                i,
                len(batches),
                batch_start,
                batch_end,
            )

            pil_frames = [img for _, img in batch_frames]
            messages = self._build_conversation(conversation, pil_frames)

            raw_response = self._infer(messages)
            clean_response = Qwen3VLOCR.strip_thinking(raw_response)
            clean_response = Qwen3VLOCR.apply_hallucination_filter(clean_response)

            batch_entries = self._parse_response_to_entries(clean_response, batch_start)
            all_entries.extend(batch_entries)

            logger.info("   ✅ Extracted %d entries", len(batch_entries))

            conversation = self._update_conversation(messages, clean_response)

        writer = SubtitleWriter()

        srt_path = out_dir / f"{video_stem}_native.srt"
        writer.write_srt(all_entries, str(srt_path), deduplicate=True)

        output_paths: Dict[str, str] = {"srt": str(srt_path)}

        if self.save_minify_txt:
            txt_path = out_dir / f"{video_stem}_native_script.txt"
            writer.write_txt(all_entries, str(txt_path), include_timestamp=False, deduplicate=True)
            output_paths["txt"] = str(txt_path)

        if self.warn_english:
            warn_path = out_dir / f"{video_stem}_subtitle_english_warnings.txt"
            final_entries = writer.deduplicate(all_entries)
            writer.generate_english_warnings(final_entries, str(warn_path))
            output_paths["warnings"] = str(warn_path)

        processing_time = time.time() - start_time

        logger.info("=" * 60)
        logger.info("✅ Native extraction complete")
        logger.info("   Entries: %d", len(all_entries))
        logger.info("   SRT: %s", output_paths.get("srt"))
        logger.info("   Time: %.2fs", processing_time)
        logger.info("=" * 60)

        return NativeExtractionResult(
            video_path=str(video_path_obj),
            total_sampled_frames=sampled_count,
            total_batches=len(batches),
            total_entries=len(all_entries),
            output_paths=output_paths,
            entries=all_entries,
            metadata={
                "model": self.model_name,
                "device": self.device,
                "frame_interval": self.frame_interval,
                "batch_duration": self.batch_duration,
                "sample_fps": self.sample_fps,
                "max_new_tokens": self.max_new_tokens,
                "total_pixels": self.total_pixels,
                "min_pixels": self.min_pixels,
                "max_frames": self.max_frames,
                "prompt_file": self.prompt_file,
                "warn_english": self.warn_english,
                "save_minify_txt": self.save_minify_txt,
                "boxes": [
                    {"name": b.name, "x": b.x, "y": b.y, "w": b.w, "h": b.h}
                    for b in self.boxes
                ],
            },
            processing_time=processing_time,
        )
