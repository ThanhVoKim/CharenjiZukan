from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import logging

from sync_engine.analyzer import remap_timestamp
from sync_engine.models import TimelineSegment
from utils.srt_parser import parse_srt_file, segments_to_srt

logger = logging.getLogger("sync_video")


@dataclass(frozen=True)
class ImageOverlayRawEvent:
    """Image overlay event đọc từ SRT, timestamp còn theo timeline video gốc."""

    key: str
    start_time: float
    end_time: float
    source_line: int


@dataclass(frozen=True)
class ImageOverlayEvent:
    """Image overlay event đã resolve PNG và sẵn sàng remap/render."""

    key: str
    image_path: str
    start_time: float
    end_time: float
    source_line: int


@dataclass(frozen=True)
class ImageOverlayAsset:
    """Unique PNG asset dùng để renderer deduplicate FFmpeg image inputs."""

    key: str
    image_path: str


def _normalize_missing_policy(missing_policy: str) -> str:
    policy = (missing_policy or "warn").strip().lower()
    if policy not in {"warn", "raise"}:
        logger.warning("missing_policy không hợp lệ cho image_overlay: %s. Fallback warn.", missing_policy)
        return "warn"
    return policy


def _normalize_file_ext(file_ext: str) -> str:
    ext = (file_ext or ".png").strip()
    if not ext:
        ext = ".png"
    if not ext.startswith("."):
        ext = f".{ext}"
    return ext


def normalize_image_overlay_key(text: str, file_ext: str = ".png") -> str:
    """Normalize text block SRT thành basename PNG không có extension."""
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("Image overlay SRT block có text rỗng.")

    key = stripped.splitlines()[0].strip()
    if not key:
        raise ValueError("Image overlay SRT block có dòng key rỗng.")

    if "/" in key or "\\" in key:
        raise ValueError(f"Image overlay key không được chứa path separator: {key!r}")
    if ".." in key:
        raise ValueError(f"Image overlay key không được chứa path traversal '..': {key!r}")

    ext = _normalize_file_ext(file_ext)
    if key.lower().endswith(ext.lower()):
        raise ValueError(
            f"Image overlay key không được chứa extension {ext!r}; "
            f"hãy dùng basename không có đuôi tệp: {key!r}"
        )

    return key


def resolve_image_overlay_path(
    key: str,
    overlay_dir: str | Path,
    file_ext: str = ".png",
    missing_policy: str = "warn",
) -> Optional[Path]:
    """Resolve image key thành PNG path trong overlay_dir."""
    policy = _normalize_missing_policy(missing_policy)
    ext = _normalize_file_ext(file_ext)
    base_dir = Path(overlay_dir)
    image_path = base_dir / f"{key}{ext}"

    if not image_path.exists():
        message = f"Không tìm thấy image overlay asset: {image_path}"
        if policy == "raise":
            raise FileNotFoundError(message)
        logger.warning(message)
        return None

    if not image_path.is_file():
        message = f"Image overlay asset không phải file: {image_path}"
        if policy == "raise":
            raise FileNotFoundError(message)
        logger.warning(message)
        return None

    if image_path.suffix.lower() != ext.lower():
        message = f"Image overlay asset không đúng extension {ext}: {image_path}"
        if policy == "raise":
            raise ValueError(message)
        logger.warning(message)
        return None

    return image_path


def load_image_overlay_events(
    srt_path: str | Path,
    overlay_dir: str | Path,
    file_ext: str = ".png",
    missing_policy: str = "warn",
) -> List[ImageOverlayEvent]:
    """Đọc SRT overlay, normalize key, resolve PNG và trả về events timestamp gốc."""
    policy = _normalize_missing_policy(missing_policy)
    segments = parse_srt_file(str(srt_path))
    events: List[ImageOverlayEvent] = []

    for segment in segments:
        text = segment.get("text", "")
        try:
            key = normalize_image_overlay_key(text, file_ext=file_ext)
        except ValueError as exc:
            if policy == "raise":
                raise
            logger.warning("Bỏ qua image overlay block line=%s: %s", segment.get("line"), exc)
            continue

        lines = [line.strip() for line in str(text).strip().splitlines()]
        extra_lines = [line for line in lines[1:] if line]
        if extra_lines:
            logger.warning(
                "Image overlay block line=%s có nhiều dòng text; chỉ dùng dòng đầu tiên làm key: %s",
                segment.get("line"),
                key,
            )

        image_path = resolve_image_overlay_path(
            key,
            overlay_dir=overlay_dir,
            file_ext=file_ext,
            missing_policy=policy,
        )
        if image_path is None:
            continue

        events.append(
            ImageOverlayEvent(
                key=key,
                image_path=str(image_path),
                start_time=float(segment["start_time"]),
                end_time=float(segment["end_time"]),
                source_line=int(segment.get("line", len(events) + 1)),
            )
        )

    return events


def remap_image_overlay_events(
    events: Iterable[ImageOverlayEvent],
    timeline: List[TimelineSegment],
    fps_float: float = 30.0,
    min_duration_ms: float = 100.0,
) -> List[ImageOverlayEvent]:
    """Remap timestamp image overlay theo timeline video đã stretch."""
    remapped: List[ImageOverlayEvent] = []

    for event in events:
        new_start = remap_timestamp(event.start_time, timeline, fps_float)
        new_end = remap_timestamp(event.end_time, timeline, fps_float)
        if new_end <= new_start:
            new_end = new_start + min_duration_ms

        remapped.append(
            ImageOverlayEvent(
                key=event.key,
                image_path=event.image_path,
                start_time=float(round(new_start)),
                end_time=float(round(new_end)),
                source_line=event.source_line,
            )
        )

    return remapped


def get_unique_image_overlay_assets(events: Iterable[ImageOverlayEvent]) -> List[ImageOverlayAsset]:
    """Trả về danh sách unique PNG assets theo thứ tự xuất hiện đầu tiên."""
    seen: set[str] = set()
    assets: List[ImageOverlayAsset] = []

    for event in events:
        resolved = str(Path(event.image_path).resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        assets.append(ImageOverlayAsset(key=event.key, image_path=resolved))

    return assets


def write_image_overlay_debug_srt(events: Iterable[ImageOverlayEvent], output_path: str | Path) -> Path:
    """Ghi SRT debug với timestamp đã remap và text là image key."""
    segments = []
    for idx, event in enumerate(events, start=1):
        segments.append(
            {
                "line": idx,
                "start_time": int(round(event.start_time)),
                "end_time": int(round(event.end_time)),
                "text": event.key,
            }
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(segments_to_srt(segments), encoding="utf-8")
    return path


def render_intermediate_overlay_track(*args, **kwargs):
    """
    Placeholder cho phase tương lai.

    Mục đích dự kiến: render toàn bộ image overlay events thành một video overlay
    trong suốt có cùng duration và resolution với stretched video, giữ alpha để
    final render chỉ cần overlay một input video phụ thay vì chain hàng trăm hoặc
    hàng nghìn PNG events trong filter graph.

    Phase hiện tại không triển khai logic này, không gọi hàm này trong pipeline,
    và không tạo file video trung gian.
    """
    raise NotImplementedError(
        "Intermediate image overlay track is reserved for a future phase."
    )
