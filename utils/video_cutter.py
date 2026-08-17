#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/video_cutter.py — Pre-cut video: remove unwanted segments before transcript/sync.

Core logic for the pre-cut CLI (cli/pre_cut_video.py).
Independent from sync_engine — operates on source video timeline only.
"""

import bisect
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.ffmpeg_probe import detect_hevc_nvenc, get_hevc_nvenc_unavailable_reason
from utils.srt_parser import parse_srt, ts_to_ms

logger = logging.getLogger(__name__)

MIN_KEEP_MS = 50
KEYFRAME_EXPANSION_WARN_THRESHOLD_MS = 1000


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class RemoveRange:
    start_ms: float
    end_ms: float
    line: int = 0
    text: str = ""


@dataclass
class KeepRange:
    start_ms: float
    end_ms: float
    clean_start_ms: float = 0.0
    clean_end_ms: float = 0.0
    part_file: str = ""
    line: int = 0
    text: str = ""
    clip_path: str = ""


@dataclass
class VideoInfo:
    duration_ms: float
    fps: float
    has_audio: bool
    video_bitrate: Optional[int] = None
    video_codec: str = ""


@dataclass
class CutResult:
    output_path: str
    manifest_path: str
    manifest: dict = field(default_factory=dict)


# ── Probe functions ───────────────────────────────────────────────────

def probe_video_info(video_path: str) -> VideoInfo:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    duration_ms = float(data.get("format", {}).get("duration", 0)) * 1000

    has_audio = False
    video_bitrate = None
    fps = 30.0
    video_codec = ""

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_codec = stream.get("codec_name", "")
            r_fps = stream.get("r_frame_rate", "30/1")
            try:
                num, den = r_fps.split("/")
                fps = float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                fps = 30.0
            br = stream.get("bit_rate")
            if br:
                try:
                    video_bitrate = int(br)
                except ValueError:
                    pass
        elif stream.get("codec_type") == "audio":
            has_audio = True

    return VideoInfo(
        duration_ms=duration_ms,
        fps=fps,
        has_audio=has_audio,
        video_bitrate=video_bitrate,
        video_codec=video_codec,
    )


def query_keyframes(video_path: str) -> List[float]:
    """Query all keyframe timestamps (in ms) from video using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "frame=key_frame,best_effort_timestamp_time",
        "-of", "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    kfs: List[float] = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) < 2 or parts[0] != "1":
            continue
        try:
            kfs.append(float(parts[1]) * 1000)
        except ValueError:
            pass
    return sorted(kfs)


def probe_output_duration_ms(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip()) * 1000
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


# ── Range processing ─────────────────────────────────────────────────

def parse_remove_srt(srt_path: str) -> List[RemoveRange]:
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
    segments = parse_srt(content, skip_empty_text=False)
    ranges = []
    for seg in segments:
        ranges.append(RemoveRange(
            start_ms=seg["start_time"],
            end_ms=seg["end_time"],
            line=seg["line"],
            text=seg.get("text", ""),
        ))
    return ranges


def parse_keep_srt(srt_path: str) -> List[KeepRange]:
    """Parse an SRT whose blocks are the source ranges to keep as clips."""
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
    segments = parse_srt(content, skip_empty_text=False)
    ranges = []
    for seg in segments:
        ranges.append(KeepRange(
            start_ms=seg["start_time"],
            end_ms=seg["end_time"],
            line=seg["line"],
            text=seg.get("text", ""),
        ))
    return ranges


def normalize_keep_ranges(
    ranges: List[KeepRange],
    source_duration_ms: float,
    min_keep_ms: float = MIN_KEEP_MS,
) -> List[KeepRange]:
    """Clamp and order keep ranges without merging overlapping SRT blocks."""
    normalized: List[KeepRange] = []
    for r in sorted(ranges, key=lambda item: (item.start_ms, item.end_ms, item.line)):
        start_ms = max(0, r.start_ms)
        end_ms = min(source_duration_ms, r.end_ms)
        if end_ms <= start_ms:
            logger.warning(
                "Skipping invalid keep range line=%d: start=%.0f >= end=%.0f",
                r.line,
                start_ms,
                end_ms,
            )
            continue
        if end_ms - start_ms < min_keep_ms:
            logger.warning(
                "Skipping short keep range line=%d: duration=%.0fms < %.0fms",
                r.line,
                end_ms - start_ms,
                min_keep_ms,
            )
            continue
        normalized.append(KeepRange(
            start_ms=start_ms,
            end_ms=end_ms,
            line=r.line,
            text=r.text,
        ))
    return normalized


def expand_keep_ranges(
    ranges: List[KeepRange],
    margin_ms: float,
    source_duration_ms: float,
) -> List[KeepRange]:
    """Add context around keep ranges while retaining block text and order."""
    if margin_ms <= 0:
        return [KeepRange(**r.__dict__) for r in ranges]

    expanded = []
    for r in ranges:
        expanded.append(KeepRange(
            start_ms=max(0, r.start_ms - margin_ms),
            end_ms=min(source_duration_ms, r.end_ms + margin_ms),
            line=r.line,
            text=r.text,
        ))
    return expanded


def snap_keep_ranges_to_frame_grid(
    ranges: List[KeepRange],
    fps: float,
    source_duration_ms: float,
) -> List[KeepRange]:
    """Snap keep ranges to frame boundaries for re-encoded clip output."""
    if fps <= 0:
        return [KeepRange(**r.__dict__) for r in ranges]

    def _snap(ms: float) -> float:
        frame_idx = round((ms / 1000.0) * fps)
        return (frame_idx / fps) * 1000.0

    snapped = []
    for r in ranges:
        snapped.append(KeepRange(
            start_ms=max(0, _snap(r.start_ms)),
            end_ms=min(source_duration_ms, _snap(r.end_ms)),
            line=r.line,
            text=r.text,
        ))
    return normalize_keep_ranges(snapped, source_duration_ms)


_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def sanitize_clip_stem(text: str, fallback: str = "clip", max_length: int = 120) -> str:
    """Turn subtitle text into a safe, readable cross-platform filename stem."""
    stem = re.sub(r"\s+", " ", str(text or "").replace("\n", " ").replace("\r", " ")).strip()
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', "_", stem)
    stem = stem.rstrip(" .")
    if not stem:
        stem = fallback
    if stem.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        stem = f"_{stem}"
    stem = stem[:max_length].rstrip(" .")
    return stem or fallback


def clip_filename(index: int, text: str) -> str:
    """Build an ordered clip filename from an SRT block's text."""
    return f"{index:04d}_{sanitize_clip_stem(text)}.mp4"


def apply_safe_margin(
    ranges: List[RemoveRange],
    safe_margin_ms: float,
    source_duration_ms: float,
) -> List[RemoveRange]:
    result = []
    for r in ranges:
        result.append(RemoveRange(
            start_ms=max(0, r.start_ms - safe_margin_ms),
            end_ms=min(source_duration_ms, r.end_ms + safe_margin_ms),
            line=r.line,
            text=r.text,
        ))
    return result


def normalize_and_merge(
    ranges: List[RemoveRange],
    source_duration_ms: float,
) -> List[RemoveRange]:
    if not ranges:
        return []

    clamped = []
    for r in ranges:
        s = max(0, r.start_ms)
        e = min(source_duration_ms, r.end_ms)
        if e <= s:
            logger.warning("Skipping invalid remove range line=%d: start=%.0f >= end=%.0f", r.line, s, e)
            continue
        clamped.append(RemoveRange(start_ms=s, end_ms=e, line=r.line, text=r.text))

    clamped.sort(key=lambda x: x.start_ms)

    merged: List[RemoveRange] = []
    for r in clamped:
        if merged and r.start_ms <= merged[-1].end_ms:
            prev = merged[-1]
            prev.end_ms = max(prev.end_ms, r.end_ms)
            prev.text = f"{prev.text}; {r.text}".strip("; ") if r.text else prev.text
        else:
            merged.append(RemoveRange(
                start_ms=r.start_ms, end_ms=r.end_ms, line=r.line, text=r.text,
            ))
    return merged


def expand_to_keyframes(
    ranges: List[RemoveRange],
    keyframes: List[float],
    source_duration_ms: float,
) -> List[RemoveRange]:
    if not keyframes:
        raise RuntimeError("No keyframes found. Cannot perform hybrid-copy safely.")

    expanded = []
    for r in ranges:
        idx_start = bisect.bisect_right(keyframes, r.start_ms) - 1
        exp_start = keyframes[max(0, idx_start)] if idx_start >= 0 else 0

        idx_end = bisect.bisect_left(keyframes, r.end_ms)
        exp_end = keyframes[idx_end] if idx_end < len(keyframes) else source_duration_ms

        exp_start = max(0, exp_start)
        exp_end = min(source_duration_ms, exp_end)

        delta = (r.start_ms - exp_start) + (exp_end - r.end_ms)
        if delta > KEYFRAME_EXPANSION_WARN_THRESHOLD_MS:
            logger.warning(
                "Expanded remove range line=%d from %.3f-%.3f to %.3f-%.3f "
                "due to keyframe boundaries; extra_removed=%.0fms",
                r.line, r.start_ms, r.end_ms, exp_start, exp_end, delta,
            )

        expanded.append(RemoveRange(
            start_ms=exp_start, end_ms=exp_end, line=r.line, text=r.text,
        ))

    return normalize_and_merge(expanded, source_duration_ms)


def snap_to_frame_grid(
    ranges: List[RemoveRange],
    fps: float,
    source_duration_ms: float,
) -> List[RemoveRange]:
    def _snap(ms: float) -> float:
        frame_idx = round((ms / 1000.0) * fps)
        return (frame_idx / fps) * 1000.0

    snapped = []
    for r in ranges:
        snapped.append(RemoveRange(
            start_ms=max(0, _snap(r.start_ms)),
            end_ms=min(source_duration_ms, _snap(r.end_ms)),
            line=r.line,
            text=r.text,
        ))
    return normalize_and_merge(snapped, source_duration_ms)


def invert_to_keep_ranges(
    remove_ranges: List[RemoveRange],
    source_duration_ms: float,
    min_keep_ms: float = MIN_KEEP_MS,
) -> List[KeepRange]:
    keeps: List[KeepRange] = []
    prev_end = 0.0

    for r in remove_ranges:
        if r.start_ms > prev_end:
            keeps.append(KeepRange(start_ms=prev_end, end_ms=r.start_ms))
        prev_end = r.end_ms

    if prev_end < source_duration_ms:
        keeps.append(KeepRange(start_ms=prev_end, end_ms=source_duration_ms))

    filtered = [k for k in keeps if (k.end_ms - k.start_ms) >= min_keep_ms]
    if len(filtered) < len(keeps):
        skipped = len(keeps) - len(filtered)
        logger.warning("Dropped %d keep range(s) shorter than %dms", skipped, min_keep_ms)

    clean_offset = 0.0
    for k in filtered:
        dur = k.end_ms - k.start_ms
        k.clean_start_ms = clean_offset
        k.clean_end_ms = clean_offset + dur
        clean_offset += dur

    return filtered


# ── FFmpeg part commands ──────────────────────────────────────────────

def _ms_to_s(ms: float) -> str:
    return f"{ms / 1000.0:.6f}"


def _build_audio_filter(
    part_duration_ms: float,
    fade_ms: float,
    fade_enabled: bool,
    is_first_part: bool = False,
    is_last_part: bool = False,
) -> Optional[str]:
    if not fade_enabled or fade_ms <= 0:
        return None
    fade_s = fade_ms / 1000.0
    dur_s = part_duration_ms / 1000.0

    filters = []
    if not is_first_part:
        filters.append(f"afade=t=in:st=0:d={fade_s:.4f}")
    if not is_last_part:
        fade_out_start = max(0, dur_s - fade_s)
        filters.append(f"afade=t=out:st={fade_out_start:.4f}:d={fade_s:.4f}")

    return ",".join(filters) if filters else None


def build_hybrid_copy_part_cmd(
    input_path: str,
    output_path: str,
    keep: KeepRange,
    audio_bitrate: str = "256k",
    audio_sample_rate: int = 48000,
    audio_channels: int = 2,
    audio_fade_ms: float = 10,
    audio_fade_enabled: bool = True,
    is_first_part: bool = False,
    is_last_part: bool = False,
) -> List[str]:
    duration_ms = keep.end_ms - keep.start_ms
    af = _build_audio_filter(duration_ms, audio_fade_ms, audio_fade_enabled, is_first_part, is_last_part)

    cmd = [
        "ffmpeg", "-y",
        "-ss", _ms_to_s(keep.start_ms),
        "-t", _ms_to_s(duration_ms),
        "-i", input_path,
        "-c:v", "copy",
    ]
    if af:
        cmd.extend(["-af", af])
    cmd.extend([
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-ar", str(audio_sample_rate),
        "-ac", str(audio_channels),
        output_path,
    ])
    return cmd


def build_reencode_part_cmd(
    input_path: str,
    output_path: str,
    keep: KeepRange,
    cq: int = 28,
    preset: str = "p4",
    maxrate: Optional[int] = None,
    bufsize: Optional[int] = None,
    audio_bitrate: str = "256k",
    audio_sample_rate: int = 48000,
    audio_channels: int = 2,
    audio_fade_ms: float = 10,
    audio_fade_enabled: bool = True,
    is_first_part: bool = False,
    is_last_part: bool = False,
) -> List[str]:
    duration_ms = keep.end_ms - keep.start_ms
    af = _build_audio_filter(duration_ms, audio_fade_ms, audio_fade_enabled, is_first_part, is_last_part)

    cmd = [
        "ffmpeg", "-y",
        "-ss", _ms_to_s(keep.start_ms),
        "-t", _ms_to_s(duration_ms),
        "-i", input_path,
        "-c:v", "hevc_nvenc",
        "-preset", preset,
        "-tune", "hq",
        "-rc", "vbr",
        "-cq", str(cq),
    ]
    if maxrate is not None:
        cmd.extend(["-maxrate", str(maxrate)])
    if bufsize is not None:
        cmd.extend(["-bufsize", str(bufsize)])
    cmd.extend(["-pix_fmt", "yuv420p"])

    if af:
        cmd.extend(["-af", af])
    cmd.extend([
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-ar", str(audio_sample_rate),
        "-ac", str(audio_channels),
        output_path,
    ])
    return cmd


def concat_parts(part_paths: List[str], output_path: str) -> None:
    if not part_paths:
        raise RuntimeError("No parts to concatenate")

    list_file = Path(output_path).with_suffix(".concat.txt")
    try:
        with open(list_file, "w", encoding="utf-8") as f:
            for p in part_paths:
                f.write(f"file '{Path(p).resolve().as_posix()}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            output_path,
        ]
        logger.info("Concat command: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Concatenated %d parts → %s", len(part_paths), output_path)
    except subprocess.CalledProcessError as e:
        logger.error("FFmpeg concat failed: %s", e.stderr[-2000:] if e.stderr else str(e))
        raise
    finally:
        if list_file.exists():
            list_file.unlink(missing_ok=True)


def _build_keep_manifest(
    input_path: str,
    output_dir: str,
    keep_srt_path: str,
    method: str,
    safe_margin_ms: float,
    audio_fade_ms: float,
    audio_fade_enabled: bool,
    info: VideoInfo,
    keep_ranges: List[KeepRange],
    total_clip_duration_ms: float,
    hevc_cq: int,
    hevc_preset: str,
    maxrate: Optional[int],
    bufsize: Optional[int],
    audio_bitrate: str,
    warnings: List[str],
) -> dict:
    """Build manifest for keep-SRT mode, including every named clip."""
    encoder: Dict = {
        "video": "copy" if method == "hybrid-copy" else "hevc_nvenc",
        "audio": "aac",
        "audio_bitrate": audio_bitrate,
    }
    if method == "reencode-smooth":
        encoder.update({
            "hevc_nvenc_available": True,
            "input_video_bitrate": info.video_bitrate,
            "maxrate": maxrate,
            "bufsize": bufsize,
            "cq": hevc_cq,
            "preset": hevc_preset,
            "tune": "hq",
            "maxrate_used": maxrate is not None,
        })
    else:
        encoder.update({
            "hevc_nvenc_available": None,
            "input_video_bitrate": None,
            "maxrate_used": False,
        })

    clips = []
    for r in keep_ranges:
        clips.append({
            "line": r.line,
            "text": r.text,
            "start_ms": r.start_ms,
            "end_ms": r.end_ms,
            "duration_ms": r.end_ms - r.start_ms,
            "clip_path": r.clip_path,
        })

    return {
        "version": 1,
        "mode": "keep-srt",
        "input_video": input_path,
        "output_dir": output_dir,
        "keep_srt": keep_srt_path,
        "method": method,
        "safe_margin_ms": safe_margin_ms,
        "audio_fade_ms": audio_fade_ms,
        "audio_fade_enabled": audio_fade_enabled,
        "source_duration_ms": info.duration_ms,
        "total_clip_duration_ms": total_clip_duration_ms,
        "clip_count": len(clips),
        "fps": info.fps,
        "keep_ranges": clips,
        "clips": clips,
        "encoder": encoder,
        "warnings": warnings,
    }


def run_keep_srt(
    input_path: str,
    output_dir: str,
    keep_srt_path: str,
    manifest_path: Optional[str] = None,
    method: str = "hybrid-copy",
    hevc_cq: int = 28,
    maxrate_ratio: float = 1.15,
    hevc_preset: str = "p4",
    audio_bitrate: str = "256k",
    audio_fade_ms: float = 10,
    safe_margin_ms: float = 100,
    audio_fade_enabled: bool = True,
) -> CutResult:
    """Create one named clip per block in a keep/highlight SRT.

    Each individual clip is written to ``output_dir``. Subtitle text is used as
    a sanitized filename stem, prefixed with the SRT block order to keep
    filenames unique and ordered. No combined video is created.
    """
    logger.info("Probing input video: %s", input_path)
    info = probe_video_info(input_path)

    if not info.has_audio:
        raise RuntimeError("Input video has no audio stream. Keep-SRT requires video with audio.")

    if method == "reencode-smooth":
        if not detect_hevc_nvenc():
            reason = get_hevc_nvenc_unavailable_reason()
            detail = f" Probe detail: {reason}" if reason else ""
            raise RuntimeError(
                "hevc_nvenc not available. Method 'reencode-smooth' requires NVIDIA GPU encoder. "
                "Use --method hybrid-copy or install NVIDIA drivers with NVENC support."
                f"{detail}"
            )

    logger.info("Parsing keep SRT: %s", keep_srt_path)
    source_ranges = parse_keep_srt(keep_srt_path)
    if not source_ranges:
        raise RuntimeError("Keep SRT is empty — no clips to create.")

    normalized = normalize_keep_ranges(source_ranges, info.duration_ms)
    margined = expand_keep_ranges(normalized, safe_margin_ms, info.duration_ms)
    margined = normalize_keep_ranges(margined, info.duration_ms)
    if method == "reencode-smooth":
        final_ranges = snap_keep_ranges_to_frame_grid(margined, info.fps, info.duration_ms)
    else:
        final_ranges = margined

    if not final_ranges:
        raise RuntimeError("No valid keep ranges remain after normalization.")

    clip_dir = Path(output_dir)
    if clip_dir.exists() and not clip_dir.is_dir():
        raise RuntimeError(f"Keep output path exists but is not a directory: {clip_dir}")
    clip_dir.mkdir(parents=True, exist_ok=True)

    maxrate = None
    bufsize = None
    if method == "reencode-smooth" and info.video_bitrate:
        maxrate = int(info.video_bitrate * maxrate_ratio)
        bufsize = int(maxrate * 2.0)
        logger.info("Bitrate probe OK: maxrate=%d, bufsize=%d", maxrate, bufsize)
    elif method == "reencode-smooth":
        logger.info("Could not probe video bitrate — skipping maxrate/bufsize")

    warnings: List[str] = []

    for index, keep in enumerate(final_ranges, start=1):
        clip_path = clip_dir / clip_filename(index, keep.text)
        keep.clip_path = str(clip_path)
        keep.part_file = str(clip_path)

        if method == "hybrid-copy":
            cmd = build_hybrid_copy_part_cmd(
                input_path,
                str(clip_path),
                keep,
                audio_bitrate=audio_bitrate,
                audio_fade_ms=audio_fade_ms,
                audio_fade_enabled=audio_fade_enabled,
                is_first_part=False,
                is_last_part=False,
            )
        else:
            cmd = build_reencode_part_cmd(
                input_path,
                str(clip_path),
                keep,
                cq=hevc_cq,
                preset=hevc_preset,
                maxrate=maxrate,
                bufsize=bufsize,
                audio_bitrate=audio_bitrate,
                audio_fade_ms=audio_fade_ms,
                audio_fade_enabled=audio_fade_enabled,
                is_first_part=False,
                is_last_part=False,
            )

        logger.info(
            "Clip %d/%d [%s]: %.0fms-%.0fms -> %s",
            index,
            len(final_ranges),
            keep.text or "(untitled)",
            keep.start_ms,
            keep.end_ms,
            clip_path,
        )
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1200)
        except subprocess.CalledProcessError as e:
            err = e.stderr[-2000:] if e.stderr else str(e)
            raise RuntimeError(f"FFmpeg failed on keep clip {index}: {err}") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"FFmpeg timeout on keep clip {index} (1200s)") from e

        if not clip_path.exists() or clip_path.stat().st_size == 0:
            raise RuntimeError(f"Keep clip empty or missing: {clip_path}")

    total_clip_duration_ms = sum(r.end_ms - r.start_ms for r in final_ranges)
    logger.info("Created %d individual keep clips in %s", len(final_ranges), clip_dir)

    if not manifest_path:
        manifest_path = str(clip_dir / "keep_manifest.json")
    else:
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)

    manifest = _build_keep_manifest(
        input_path=input_path,
        output_dir=str(clip_dir),
        keep_srt_path=keep_srt_path,
        method=method,
        safe_margin_ms=safe_margin_ms,
        audio_fade_ms=audio_fade_ms,
        audio_fade_enabled=audio_fade_enabled,
        info=info,
        keep_ranges=final_ranges,
        total_clip_duration_ms=total_clip_duration_ms,
        hevc_cq=hevc_cq,
        hevc_preset=hevc_preset,
        maxrate=maxrate,
        bufsize=bufsize,
        audio_bitrate=audio_bitrate,
        warnings=warnings,
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("Keep manifest written: %s", manifest_path)

    return CutResult(
        output_path=str(clip_dir),
        manifest_path=str(manifest_path),
        manifest=manifest,
    )


# ── Main orchestration ────────────────────────────────────────────────

def run_pre_cut(
    input_path: str,
    output_path: str,
    remove_srt_path: str,
    manifest_path: Optional[str] = None,
    method: str = "hybrid-copy",
    hevc_cq: int = 28,
    maxrate_ratio: float = 1.15,
    hevc_preset: str = "p4",
    audio_bitrate: str = "256k",
    audio_fade_ms: float = 10,
    safe_margin_ms: float = 100,
    audio_fade_enabled: bool = True,
    keep_tmp: bool = False,
) -> CutResult:
    # 1. Probe input
    logger.info("Probing input video: %s", input_path)
    info = probe_video_info(input_path)

    if not info.has_audio:
        raise RuntimeError("Input video has no audio stream. Pre-cut requires video with audio.")

    logger.info(
        "Video info: duration=%.0fms, fps=%.2f, codec=%s, bitrate=%s",
        info.duration_ms, info.fps, info.video_codec,
        info.video_bitrate if info.video_bitrate else "unknown",
    )

    # 2. Method-specific checks
    if method == "reencode-smooth":
        if not detect_hevc_nvenc():
            reason = get_hevc_nvenc_unavailable_reason()
            detail = f" Probe detail: {reason}" if reason else ""
            raise RuntimeError(
                "hevc_nvenc not available. Method 'reencode-smooth' requires NVIDIA GPU encoder. "
                "Use --method hybrid-copy or install NVIDIA drivers with NVENC support."
                f"{detail}"
            )

    # 3. Parse remove SRT
    logger.info("Parsing remove SRT: %s", remove_srt_path)
    source_ranges = parse_remove_srt(remove_srt_path)
    if not source_ranges:
        raise RuntimeError("Remove SRT is empty — no ranges to remove.")
    logger.info("Parsed %d remove range(s) from SRT", len(source_ranges))

    # 4. Apply safe margin
    margined_ranges = apply_safe_margin(source_ranges, safe_margin_ms, info.duration_ms)

    # 5. Normalize and merge
    normalized = normalize_and_merge(margined_ranges, info.duration_ms)
    logger.info("After normalize+merge: %d remove range(s)", len(normalized))

    # 6. Method-specific boundary adjustment
    if method == "hybrid-copy":
        logger.info("Querying keyframes for hybrid-copy...")
        keyframes = query_keyframes(input_path)
        if not keyframes:
            raise RuntimeError("Failed to query keyframes. Cannot perform hybrid-copy safely.")
        logger.info("Found %d keyframes", len(keyframes))
        # Do not expand remove ranges to GOP/keyframe boundaries here.
        # Expanding DELETE ranges can make separate remove intervals overlap,
        # which collapses valid keep ranges and silently removes content between
        # SRT blocks. hybrid-copy is a speed-oriented stream-copy path, but the
        # source SRT topology must remain the source of truth for part creation.
        final_remove = normalized
    else:
        final_remove = snap_to_frame_grid(normalized, info.fps, info.duration_ms)
        keyframes = []

    # 7. Invert to keep ranges
    keep_ranges = invert_to_keep_ranges(final_remove, info.duration_ms)
    if not keep_ranges:
        raise RuntimeError("No content remaining after removing all specified ranges.")
    logger.info("Keep ranges: %d segment(s)", len(keep_ranges))

    expected_clean_ms = sum(k.end_ms - k.start_ms for k in keep_ranges)

    # 8. Setup temp directory
    out_path = Path(output_path)
    tmp_dir = out_path.parent / f"{out_path.stem}_precut_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 9. Compute maxrate/bufsize for reencode
    maxrate = None
    bufsize = None
    if method == "reencode-smooth" and info.video_bitrate:
        maxrate = int(info.video_bitrate * maxrate_ratio)
        bufsize = int(maxrate * 2.0)
        logger.info("Bitrate probe OK: maxrate=%d, bufsize=%d", maxrate, bufsize)
    elif method == "reencode-smooth":
        logger.info("Could not probe video bitrate — skipping maxrate/bufsize")

    # 10. Cut each keep part
    part_paths: List[str] = []
    warnings: List[str] = []

    for i, keep in enumerate(keep_ranges):
        part_file = str(tmp_dir / f"keep_{i:04d}.mp4")
        keep.part_file = part_file

        is_first = (keep.start_ms == 0)
        is_last = (keep.end_ms >= info.duration_ms)

        if method == "hybrid-copy":
            cmd = build_hybrid_copy_part_cmd(
                input_path, part_file, keep,
                audio_bitrate=audio_bitrate,
                audio_fade_ms=audio_fade_ms,
                audio_fade_enabled=audio_fade_enabled,
                is_first_part=is_first,
                is_last_part=is_last,
            )
        else:
            cmd = build_reencode_part_cmd(
                input_path, part_file, keep,
                cq=hevc_cq, preset=hevc_preset,
                maxrate=maxrate, bufsize=bufsize,
                audio_bitrate=audio_bitrate,
                audio_fade_ms=audio_fade_ms,
                audio_fade_enabled=audio_fade_enabled,
                is_first_part=is_first,
                is_last_part=is_last,
            )

        logger.info("Part %d/%d: %s", i + 1, len(keep_ranges), " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1200)
        except subprocess.CalledProcessError as e:
            err = e.stderr[-2000:] if e.stderr else str(e)
            raise RuntimeError(f"FFmpeg failed on part {i}: {err}") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"FFmpeg timeout on part {i} (1200s)") from e

        if not Path(part_file).exists() or Path(part_file).stat().st_size == 0:
            raise RuntimeError(f"Part file empty or missing: {part_file}")

        part_paths.append(part_file)

    # 11. Concat
    logger.info("Concatenating %d parts...", len(part_paths))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concat_parts(part_paths, output_path)

    # 12. Verify output duration
    actual_ms = probe_output_duration_ms(output_path)
    drift_ms = actual_ms - expected_clean_ms
    logger.info(
        "Duration check: expected=%.0fms, actual=%.0fms, drift=%.0fms",
        expected_clean_ms, actual_ms, drift_ms,
    )
    if abs(drift_ms) > 500:
        warn_msg = f"Output duration drift {drift_ms:.0f}ms exceeds 500ms tolerance"
        logger.warning(warn_msg)
        warnings.append(warn_msg)

    # 13. Build manifest
    if not manifest_path:
        manifest_path = str(out_path.with_name(f"{out_path.stem}_cut_manifest.json"))

    manifest = _build_manifest(
        input_path=input_path,
        output_path=output_path,
        remove_srt_path=remove_srt_path,
        method=method,
        safe_margin_ms=safe_margin_ms,
        audio_fade_ms=audio_fade_ms,
        audio_fade_enabled=audio_fade_enabled,
        info=info,
        source_ranges=source_ranges,
        normalized_ranges=normalized,
        expanded_ranges=final_remove,
        keep_ranges=keep_ranges,
        expected_clean_ms=expected_clean_ms,
        actual_ms=actual_ms,
        drift_ms=drift_ms,
        hevc_cq=hevc_cq,
        hevc_preset=hevc_preset,
        maxrate=maxrate,
        bufsize=bufsize,
        audio_bitrate=audio_bitrate,
        warnings=warnings,
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("Manifest written: %s", manifest_path)

    # 14. Cleanup
    if not keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Cleaned up temp directory: %s", tmp_dir)
    else:
        logger.info("Keeping temp directory: %s", tmp_dir)

    return CutResult(
        output_path=output_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _build_manifest(
    input_path: str,
    output_path: str,
    remove_srt_path: str,
    method: str,
    safe_margin_ms: float,
    audio_fade_ms: float,
    audio_fade_enabled: bool,
    info: VideoInfo,
    source_ranges: List[RemoveRange],
    normalized_ranges: List[RemoveRange],
    expanded_ranges: List[RemoveRange],
    keep_ranges: List[KeepRange],
    expected_clean_ms: float,
    actual_ms: float,
    drift_ms: float,
    hevc_cq: int,
    hevc_preset: str,
    maxrate: Optional[int],
    bufsize: Optional[int],
    audio_bitrate: str,
    warnings: List[str],
) -> dict:
    encoder: Dict = {
        "video": "copy" if method == "hybrid-copy" else "hevc_nvenc",
        "audio": "aac",
        "audio_bitrate": audio_bitrate,
    }
    if method == "reencode-smooth":
        encoder.update({
            "hevc_nvenc_available": True,
            "input_video_bitrate": info.video_bitrate,
            "maxrate": maxrate,
            "bufsize": bufsize,
            "cq": hevc_cq,
            "preset": hevc_preset,
            "tune": "hq",
            "maxrate_used": maxrate is not None,
        })
    else:
        encoder.update({
            "hevc_nvenc_available": None,
            "input_video_bitrate": None,
            "maxrate_used": False,
        })

    return {
        "version": 1,
        "input_video": input_path,
        "output_video": output_path,
        "remove_srt": remove_srt_path,
        "method": method,
        "safe_margin_ms": safe_margin_ms,
        "audio_fade_ms": audio_fade_ms,
        "audio_fade_enabled": audio_fade_enabled,
        "source_duration_ms": info.duration_ms,
        "expected_clean_duration_ms": expected_clean_ms,
        "actual_output_duration_ms": actual_ms,
        "duration_drift_ms": drift_ms,
        "fps": info.fps,
        "source_remove_ranges": [
            {"line": r.line, "start_ms": r.start_ms, "end_ms": r.end_ms, "text": r.text}
            for r in source_ranges
        ],
        "normalized_remove_ranges": [
            {"start_ms": r.start_ms, "end_ms": r.end_ms}
            for r in normalized_ranges
        ],
        "expanded_remove_ranges": [
            {
                "start_ms": r.start_ms,
                "end_ms": r.end_ms,
                "reason": "keyframe_expansion" if method == "hybrid-copy" else "frame_snap",
            }
            for r in expanded_ranges
        ],
        "keep_ranges": [
            {
                "start_ms": k.start_ms,
                "end_ms": k.end_ms,
                "clean_start_ms": k.clean_start_ms,
                "clean_end_ms": k.clean_end_ms,
                "part_file": k.part_file,
            }
            for k in keep_ranges
        ],
        "encoder": encoder,
        "warnings": warnings,
    }
