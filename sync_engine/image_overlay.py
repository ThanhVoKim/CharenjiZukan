from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import logging

from sync_engine.analyzer import remap_timestamp
from sync_engine.models import TimelineSegment
from utils.srt_parser import parse_srt_file, segments_to_srt

logger = logging.getLogger("sync_video")

AUTO_IMAGE_FILE_EXT_VALUES = {"", "auto", "all", "*", "static", "image", "images"}
SUPPORTED_STATIC_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".jfif",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".avif",
    ".heic",
    ".heif",
    ".jp2",
    ".j2k",
    ".jxl",
    ".tga",
    ".svg",
    ".ico",
)


@dataclass(frozen=True)
class ImageOverlayRawEvent:
    """Image overlay event đọc từ SRT, timestamp còn theo timeline video gốc."""

    key: str
    start_time: float
    end_time: float
    source_line: int


@dataclass(frozen=True)
class ImageOverlayEvent:
    """Image overlay event đã resolve ảnh tĩnh và sẵn sàng remap/render."""

    key: str
    image_path: str
    start_time: float
    end_time: float
    source_line: int


@dataclass(frozen=True)
class ImageOverlayAsset:
    """Unique static image asset dùng để renderer deduplicate FFmpeg image inputs."""

    key: str
    image_path: str


def _normalize_missing_policy(missing_policy: str) -> str:
    policy = (missing_policy or "warn").strip().lower()
    if policy not in {"warn", "raise"}:
        logger.warning("missing_policy không hợp lệ cho image_overlay: %s. Fallback warn.", missing_policy)
        return "warn"
    return policy


def _normalize_single_file_ext(file_ext: str) -> str:
    ext = (file_ext or "").strip().lower()
    if not ext:
        return ""
    if not ext.startswith("."):
        ext = f".{ext}"
    return ext


def _is_auto_file_ext(file_ext: object) -> bool:
    if file_ext is None:
        return True
    if isinstance(file_ext, str):
        return file_ext.strip().lower() in AUTO_IMAGE_FILE_EXT_VALUES
    return False


def _normalize_file_extensions(file_ext: str | Iterable[str] | None = "auto") -> tuple[str, ...]:
    if _is_auto_file_ext(file_ext):
        return SUPPORTED_STATIC_IMAGE_EXTENSIONS

    if isinstance(file_ext, str):
        ext = _normalize_single_file_ext(file_ext)
        return (ext,) if ext else SUPPORTED_STATIC_IMAGE_EXTENSIONS

    extensions: list[str] = []
    try:
        raw_extensions = list(file_ext or [])
    except TypeError:
        raw_extensions = [str(file_ext)]

    for raw_ext in raw_extensions:
        text = str(raw_ext or "").strip()
        if text.lower() in AUTO_IMAGE_FILE_EXT_VALUES:
            return SUPPORTED_STATIC_IMAGE_EXTENSIONS
        ext = _normalize_single_file_ext(text)
        if ext and ext not in extensions:
            extensions.append(ext)

    return tuple(extensions) or SUPPORTED_STATIC_IMAGE_EXTENSIONS


def _extensions_for_key_validation(file_ext: str | Iterable[str] | None = "auto") -> tuple[str, ...]:
    extensions = list(SUPPORTED_STATIC_IMAGE_EXTENSIONS)
    for ext in _normalize_file_extensions(file_ext):
        if ext not in extensions:
            extensions.append(ext)
    return tuple(sorted(extensions, key=len, reverse=True))


def normalize_image_overlay_key(text: str, file_ext: str | Iterable[str] | None = "auto") -> str:
    """Normalize text block SRT thành basename ảnh tĩnh không có extension."""
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

    lower_key = key.lower()
    for ext in _extensions_for_key_validation(file_ext):
        if lower_key.endswith(ext):
            raise ValueError(
                f"Image overlay key không được chứa extension ảnh {ext!r}; "
                f"hãy dùng basename không có đuôi tệp: {key!r}"
            )

    return key


def _find_image_overlay_candidates(
    key: str,
    overlay_dir: Path,
    extensions: tuple[str, ...],
) -> list[Path]:
    extension_priority = {ext.lower(): idx for idx, ext in enumerate(extensions)}
    candidates: list[Path] = []

    key_folded = key.casefold()
    for entry in overlay_dir.iterdir():
        suffix = entry.suffix.lower()
        if entry.stem.casefold() == key_folded and suffix in extension_priority:
            candidates.append(entry)

    candidates.sort(key=lambda path: (extension_priority[path.suffix.lower()], path.name.lower()))
    return candidates


def resolve_image_overlay_path(
    key: str,
    overlay_dir: str | Path,
    file_ext: str | Iterable[str] | None = "auto",
    missing_policy: str = "warn",
) -> Optional[Path]:
    """Resolve image key thành static image path trong overlay_dir."""
    policy = _normalize_missing_policy(missing_policy)
    extensions = _normalize_file_extensions(file_ext)
    base_dir = Path(overlay_dir)

    if not base_dir.exists() or not base_dir.is_dir():
        message = f"Không tìm thấy image overlay dir: {base_dir}"
        if policy == "raise":
            raise FileNotFoundError(message)
        logger.warning(message)
        return None

    candidates = _find_image_overlay_candidates(key, base_dir, extensions)
    valid_candidates = [path for path in candidates if path.is_file()]

    if not valid_candidates:
        extensions_label = ", ".join(extensions)
        message = (
            f"Không tìm thấy image overlay asset cho key={key!r} trong {base_dir} "
            f"với extension: {extensions_label}"
        )
        if policy == "raise":
            raise FileNotFoundError(message)
        logger.warning(message)
        return None

    if len(valid_candidates) > 1:
        chosen = valid_candidates[0]
        message = (
            f"Có nhiều image overlay asset cùng basename key={key!r}: "
            f"{[str(path) for path in valid_candidates]}. "
            f"Dùng file ưu tiên theo thứ tự extension: {chosen}"
        )
        if policy == "raise":
            raise ValueError(message)
        logger.warning(message)
        return chosen

    return valid_candidates[0]


def load_image_overlay_events(
    srt_path: str | Path,
    overlay_dir: str | Path,
    file_ext: str | Iterable[str] | None = "auto",
    missing_policy: str = "warn",
) -> List[ImageOverlayEvent]:
    """Đọc SRT overlay, normalize key, resolve ảnh tĩnh và trả về events timestamp gốc."""
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
    """Trả về danh sách unique static image assets theo thứ tự xuất hiện đầu tiên."""
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
    hàng nghìn static image events trong filter graph.

    Phase hiện tại không triển khai logic này, không gọi hàm này trong pipeline,
    và không tạo file video trung gian.
    """
    raise NotImplementedError(
        "Intermediate image overlay track is reserved for a future phase."
    )
