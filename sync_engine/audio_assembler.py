"""Audio assembly helpers cho pipeline sync-video.

Review nhanh kiến trúc file này:
- `audio_policies` là lớp cấu hình canonical cho mute chunks, ambient và global BGM.
- Runtime luôn build `main concat` bám theo final timeline trước.
- Ambient và global BGM là các overlay độc lập, được xử lý sau `main concat`.
- Gain cuối cùng của ambient/BGM được quyết định ở final mix để tránh nhân volume hai lần.
- Các key cũ `audio_separator.extract_bgm` / `extract_vocals` chỉ còn được giữ để tương thích ngược.
"""

import concurrent.futures
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from sync_engine.models import TimelineSegment
from utils.media_utils import _build_atempo_filter

_PCM_AUDIO_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".pcm"}

logger = logging.getLogger("sync_video")

_DEFAULT_FADE_MS = 10
_TTS_FADE_MS = 5

# Default an toàn cho pipeline mới:
# - mute giữ audio gốc nếu caller không yêu cầu khác
# - ambient tắt ở các đoạn mute để tránh làm rối phần speech/TTS
# - global BGM mặc định tắt hoàn toàn
_DEFAULT_AUDIO_POLICIES = {
    "global_bgm": "off",
    "mute_audio": "original",
    "ambient": "exclude_mute",
}

_ALLOWED_AUDIO_POLICY_VALUES = {
    "global_bgm": {"off", "whole_video", "exclude_mute"},
    "mute_audio": {"original", "vocals", "instrumental", "silence"},
    "ambient": {"off", "whole_video", "exclude_mute"},
}

# Alias để render_config có thể dùng nhiều cách viết nhưng runtime chỉ còn 1 dạng canonical.
_AUDIO_POLICY_ALIASES = {
    "whole video": "whole_video",
    "whole-video": "whole_video",
    "whole_video": "whole_video",
    "exclude mute": "exclude_mute",
    "exclude-mute": "exclude_mute",
    "exclude_mute": "exclude_mute",
}


def _build_audio_fade_filter(
    part_duration_ms: float,
    fade_ms: float,
    is_first_chunk: bool = False,
    is_last_chunk: bool = False,
) -> Optional[str]:
    if fade_ms <= 0 or part_duration_ms <= 0:
        return None
    fade_s = fade_ms / 1000.0
    dur_s = part_duration_ms / 1000.0
    if dur_s <= fade_s * 2:
        return None
    filters = []
    if not is_first_chunk:
        filters.append(f"afade=t=in:st=0:d={fade_s:.4f}")
    if not is_last_chunk:
        fade_out_start = max(0, dur_s - fade_s)
        filters.append(f"afade=t=out:st={fade_out_start:.4f}:d={fade_s:.4f}")
    return ",".join(filters) if filters else None


def _normalize_audio_policy_value(name: str, value: str) -> str:
    raw_value = _AUDIO_POLICY_ALIASES.get(str(value).strip().lower(), str(value).strip().lower().replace("-", "_").replace(" ", "_"))
    if raw_value not in _ALLOWED_AUDIO_POLICY_VALUES[name]:
        allowed = ", ".join(sorted(_ALLOWED_AUDIO_POLICY_VALUES[name]))
        raise ValueError(f"audio_policies.{name} không hợp lệ: {value}. Giá trị hợp lệ: {allowed}")
    return raw_value


def resolve_audio_policies(render_config: Optional[Mapping[str, object]] = None) -> Dict[str, str]:
    """Chuẩn hóa policy audio từ render_config sang 3 key canonical.

    Thứ tự ưu tiên:
    1. `audio_policies` mới
    2. fallback từ `audio_separator.extract_bgm` / `extract_vocals` cũ

    Hàm này dành cho tầng đọc config/CLI, nên mọi alias/deprecation được xử lý ở đây
    để tầng runtime mix audio chỉ nhận các giá trị đã validate.
    """
    render_config = dict(render_config or {})
    audio_policies_cfg = dict(render_config.get("audio_policies") or {})
    audio_separator_cfg = dict(render_config.get("audio_separator") or {})

    has_legacy_bgm = "extract_bgm" in audio_separator_cfg
    has_legacy_vocals = "extract_vocals" in audio_separator_cfg
    has_legacy_flags = has_legacy_bgm or has_legacy_vocals

    if audio_policies_cfg:
        if has_legacy_flags:
            logger.warning(
                "Phát hiện đồng thời audio_policies và audio_separator.extract_bgm/extract_vocals. "
                "Ưu tiên audio_policies; extract_bgm/extract_vocals được xem là deprecated compatibility keys."
            )
        merged = dict(_DEFAULT_AUDIO_POLICIES)
        merged.update(audio_policies_cfg)
        return {
            "global_bgm": _normalize_audio_policy_value("global_bgm", merged["global_bgm"]),
            "mute_audio": _normalize_audio_policy_value("mute_audio", merged["mute_audio"]),
            "ambient": _normalize_audio_policy_value("ambient", merged["ambient"]),
        }

    if has_legacy_flags:
        logger.warning(
            "Đang dùng audio_separator.extract_bgm/extract_vocals dạng cũ. "
            "Nên migrate sang block audio_policies trong render_config."
        )

    return {
        "global_bgm": "whole_video" if audio_separator_cfg.get("extract_bgm", False) else "off",
        "mute_audio": "vocals" if audio_separator_cfg.get("extract_vocals", False) else "original",
        "ambient": _DEFAULT_AUDIO_POLICIES["ambient"],
    }


def _normalize_audio_policies_for_assembly(
    audio_policies: Optional[Mapping[str, str]],
    *,
    use_vocal_extraction: bool = False,
    has_bgm_path: bool = False,
) -> Dict[str, str]:
    """Chuẩn hóa policy cho tầng runtime audio assembly.

    Nếu caller đã truyền `audio_policies` thì validate trực tiếp.
    Nếu chưa, suy luận fallback từ các cờ runtime cũ để giữ backward compatibility
    cho code path hoặc test chưa migrate hoàn toàn.
    """
    if audio_policies:
        merged = dict(_DEFAULT_AUDIO_POLICIES)
        merged.update(dict(audio_policies))
        return {
            "global_bgm": _normalize_audio_policy_value("global_bgm", merged["global_bgm"]),
            "mute_audio": _normalize_audio_policy_value("mute_audio", merged["mute_audio"]),
            "ambient": _normalize_audio_policy_value("ambient", merged["ambient"]),
        }

    return {
        "global_bgm": "whole_video" if has_bgm_path else "off",
        "mute_audio": "vocals" if use_vocal_extraction else "original",
        "ambient": _DEFAULT_AUDIO_POLICIES["ambient"],
    }


def compress_tts_clip(
    wav_path: str,
    audio_speed: float,
    output_path: str,
    tts_provider: str = "edge",
    target_dur_s: Optional[float] = None,
    fade_ms: float = _TTS_FADE_MS,
    is_first_chunk: bool = False,
    is_last_chunk: bool = False,
) -> None:
    # Voicevox family (Voicevox chính thức và Voicevox Nemo) đã tự tăng volumeScale, không cần filter
    if tts_provider.startswith("voicevox"):
        base_filter = ""
    else:
        # EdgeTTS và các provider khác: chuẩn hóa âm lượng theo chuẩn EBU R128
        base_filter = "volume=1.75"

    if audio_speed > 1.01:
        atempo_str = _build_atempo_filter(audio_speed)
        filter_str = f"{atempo_str},{base_filter}" if base_filter else atempo_str
    else:
        filter_str = base_filter
    # Thêm atrim và apad để đảm bảo duration chính xác tuyệt đối
    if target_dur_s is not None:
        pad_trim_filter = (
            f"atrim=start=0,asetpts=PTS-STARTPTS,apad=whole_dur={target_dur_s:.6f},"
            f"atrim=end={target_dur_s:.6f}"
        )
        filter_str = f"{filter_str},{pad_trim_filter}" if filter_str else pad_trim_filter
    # Fade in/out tại biên chunk để triệt tiêu pop/click khi concat
    fade_dur_ms = (target_dur_s * 1000.0) if target_dur_s else 0
    af_fade = _build_audio_fade_filter(fade_dur_ms, fade_ms, is_first_chunk, is_last_chunk)
    if af_fade:
        filter_str = f"{filter_str},{af_fade}" if filter_str else af_fade

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        wav_path,
        "-ar",
        "48000",
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
    ]
    if filter_str:
        cmd.extend(["-filter:a", filter_str])
    cmd.append(output_path)

    subprocess.run(cmd, check=True, capture_output=True)


def _prepare_synced_audio_chunk(
    index: int,
    seg: TimelineSegment,
    source_path: str,
    output_path: str,
    sample_rate: int,
    fade_ms: float = _DEFAULT_FADE_MS,
    is_first_chunk: bool = False,
    is_last_chunk: bool = False,
) -> Tuple[int, str]:
    target_dur_s = seg.new_chunk_dur / 1000.0
    if target_dur_s <= 0:
        return index, ""

    if Path(output_path).exists():
        return index, output_path

    start_s = seg.orig_start / 1000.0
    orig_dur_s = (seg.orig_end - seg.orig_start) / 1000.0

    filters = []
    if seg.video_speed > 1.01 or seg.video_speed < 0.99:
        filters.append(_build_atempo_filter(seg.video_speed))

    filters.append(
        f"atrim=start=0,asetpts=PTS-STARTPTS,apad=whole_dur={target_dur_s:.6f},atrim=end={target_dur_s:.6f}"
    )

    af_fade = _build_audio_fade_filter(seg.new_chunk_dur, fade_ms, is_first_chunk, is_last_chunk)
    if af_fade:
        filters.append(af_fade)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_s:.6f}",
        "-t",
        f"{orig_dur_s:.6f}",
        "-i",
        source_path,
        "-filter:a",
        ",".join(f for f in filters if f),
        "-ar",
        str(sample_rate),
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return index, output_path


def _prepare_bgm_chunk(
    index: int,
    seg: TimelineSegment,
    bgm_path: str,
    tmp_dir: str,
    sample_rate: int,
    fade_ms: float = _DEFAULT_FADE_MS,
    is_first_chunk: bool = False,
    is_last_chunk: bool = False,
) -> Tuple[int, str]:
    bgm_chunk = str(Path(tmp_dir) / f"bgm_chunk_{index:04d}.wav")
    return _prepare_synced_audio_chunk(
        index=index,
        seg=seg,
        source_path=bgm_path,
        output_path=bgm_chunk,
        sample_rate=sample_rate,
        fade_ms=fade_ms,
        is_first_chunk=is_first_chunk,
        is_last_chunk=is_last_chunk,
    )


def extract_quoted_audio(
    video_path: str,
    orig_start_ms: float,
    orig_end_ms: float,
    output_path: str,
    pad_s: float = 0.0,
) -> float:
    """
    Extract đoạn audio từ nguồn video hoặc WAV.

    - Có hỗ trợ Padding (pad_s) ở 2 đầu để giữ context cho mô hình AI.
    - Video source (MP4/MKV/...): Dùng 2-pass seek (rough + fine) để bù
      encoder delay của codec nén (AAC/MP3). FFmpeg xử lý edit list trong
      container để bỏ priming samples, nên cần chiến lược này.
    - PCM WAV source (Demucs output): Dùng single-pass seek trước -i.
      WAV/PCM không có encoder delay; single-pass seek là sample-accurate
      và KHÔNG bị lệch time reference so với quá trình pre-extract FFmpeg.

    Trả về số giây padding thực tế đã thêm vào phía trước (left pad).
    """
    start_s = orig_start_ms / 1000.0
    end_s = orig_end_ms / 1000.0

    actual_left_pad = min(pad_s, start_s)
    pad_start_s = start_s - actual_left_pad
    pad_end_s = end_s + pad_s

    duration_s = pad_end_s - pad_start_s
    src_ext = Path(video_path).suffix.lower()

    if src_ext in _PCM_AUDIO_EXTENSIONS:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{pad_start_s:.9f}",
            "-i",
            video_path,
            "-t",
            f"{duration_s:.9f}",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            output_path,
        ]
        logger.debug(
            "extract_quoted_audio [PCM] %.3fs–%.3fs → %s",
            pad_start_s,
            pad_end_s,
            output_path,
        )
    else:
        rough_start_s = max(0.0, pad_start_s - 5.0)
        exact_offset_s = pad_start_s - rough_start_s
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{rough_start_s:.6f}",
            "-i",
            video_path,
            "-ss",
            f"{exact_offset_s:.6f}",
            "-t",
            f"{duration_s:.6f}",
            "-vn",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            output_path,
        ]
        logger.debug(
            "extract_quoted_audio [Video] rough=%.3fs offset=%.3fs dur=%.3fs → %s",
            rough_start_s,
            exact_offset_s,
            duration_s,
            output_path,
        )

    subprocess.run(cmd, check=True, capture_output=True)
    return actual_left_pad


def build_mute_ranges(timeline: List[TimelineSegment]) -> List[Tuple[float, float]]:
    """Chuyển các mute segments của final timeline thành giây để dùng cho FFmpeg."""
    return sorted(
        (segment.new_start / 1000.0, segment.new_end / 1000.0)
        for segment in timeline
        if segment.block_type == "mute"
    )


def build_ambient_mask(
    timeline: List[TimelineSegment],
    total_ms: float,
) -> List[Tuple[float, float]]:
    """
    Trả về list khoảng (new_start, new_end) cho phép ambient phát.
    Ambient bị tắt trong khoảng new_start..new_end của mute segments.
    """
    mute_ranges_ms = sorted((s.new_start, s.new_end) for s in timeline if s.block_type == "mute")
    ambient_ranges = []
    cursor = 0.0
    for ms, me in mute_ranges_ms:
        if cursor < ms:
            ambient_ranges.append((cursor, ms))
        cursor = me
    if cursor < total_ms:
        ambient_ranges.append((cursor, total_ms))
    return ambient_ranges


def _build_mute_volume_filter(mute_ranges: List[Tuple[float, float]], active_volume: float) -> str:
    """Sinh volume expression cho FFmpeg dựa trên các đoạn mute.

    Ngoài mute range → giữ `active_volume`.
    Trong mute range → ép volume về 0.

    Hàm này được dùng lại cho cả ambient và global BGM để chỉ có một nguồn logic mask.
    """
    active_volume = float(active_volume)
    if not mute_ranges:
        return f"volume={active_volume:.6f}"

    between_exprs = [f"between(t,{start_s:.3f},{end_s:.3f})" for start_s, end_s in mute_ranges]
    expr = "+".join(between_exprs)
    if len(expr) > 10000:
        logger.warning("Quá nhiều đoạn mute, volume expression có thể chạm giới hạn độ dài lệnh FFmpeg.")
    return f"volume='if({expr}, 0, {active_volume:.6f})':eval=frame"


def _finalize_audio_chunk(
    input_path: str,
    output_path: str,
    sample_rate: int,
    target_dur_s: float,
    *,
    trim_start_s: float = 0.0,
    trim_duration_s: Optional[float] = None,
    fade_ms: float = _DEFAULT_FADE_MS,
    is_first_chunk: bool = False,
    is_last_chunk: bool = False,
) -> None:
    atrim_expr = f"atrim=start={trim_start_s:.6f}"
    if trim_duration_s is not None:
        atrim_expr += f":duration={trim_duration_s:.6f}"

    base_filter = (
        f"{atrim_expr},asetpts=PTS-STARTPTS,apad=whole_dur={target_dur_s:.6f},"
        f"atrim=end={target_dur_s:.6f}"
    )
    af_fade = _build_audio_fade_filter(target_dur_s * 1000.0, fade_ms, is_first_chunk, is_last_chunk)
    filter_str = f"{base_filter},{af_fade}" if af_fade else base_filter

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-filter:a",
        filter_str,
        "-ar",
        str(sample_rate),
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _generate_silence_chunk(output_path: str, sample_rate: int, target_dur_s: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={sample_rate}:cl=stereo",
            "-t",
            f"{target_dur_s:.6f}",
            output_path,
        ],
        check=True,
        capture_output=True,
    )


def _process_ambient_track(
    ambient_path: str,
    timeline: List[TimelineSegment],
    total_ms: float,
    output_path: str,
    sample_rate: int = 48000,
    ambient_policy: str = "exclude_mute",
) -> bool:
    """
    Preprocess ambient track theo timeline final.

    - whole_video: chỉ loop/trim về đúng duration, giữ gain 1.0.
    - exclude_mute: loop/trim và apply mask 0/1 trên các mute ranges.
    - off: caller không nên gọi; trả False để bỏ qua layer.
    """
    if ambient_policy == "off":
        return False
    if not ambient_path or not Path(ambient_path).exists():
        return False

    total_s = total_ms / 1000.0
    mute_ranges = build_mute_ranges(timeline)
    if ambient_policy == "exclude_mute":
        volume_expr = _build_mute_volume_filter(mute_ranges, 1.0)
    else:
        volume_expr = "volume=1.000000"

    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        ambient_path,
        "-t",
        f"{total_s:.3f}",
        "-filter:a",
        volume_expr,
        "-ar",
        str(sample_rate),
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi xử lý ambient track: {e.stderr.decode('utf-8', errors='ignore')}")
        return False


def assemble_audio_track(
    timeline: List[TimelineSegment],
    video_path: str,
    ambient_path: Optional[str],
    output_path: str,
    tmp_dir: str,
    sample_rate: int = 48000,
    use_vocal_extraction: bool = False,
    tts_provider: str = "edge",
    video_duration_override: Optional[float] = None,
    bgm_path: Optional[str] = None,
    audio_mix_config: Optional[dict] = None,
    audio_separator_config: Optional[dict] = None,
    audio_policies: Optional[Mapping[str, str]] = None,
) -> None:
    """Ghép audio cuối cùng cho video đã được phân tích timeline.

    Quy trình review nên đọc theo 3 lớp:
    1. Build `main concat` từ từng segment trong timeline final.
       - `mute`: giữ original / vocals / instrumental / silence theo `mute_audio`
       - `tts`: nén/trim/pad clip TTS về đúng `new_chunk_dur`
       - block còn lại: sinh silence để timeline luôn kín
    2. Preprocess các layer toàn cục độc lập (`ambient`, `global_bgm`) về đúng total duration.
    3. Final mix theo thứ tự đáy → đỉnh: main concat → ambient overlay → global BGM overlay.

    Các quyết định quan trọng:
    - Final mix là nơi duy nhất áp gain cuối cùng cho ambient/BGM.
    - Policy `exclude_mute` dùng volume mask trên final timeline thay vì cắt track thành nhiều mảnh.
    - Audio separator chỉ chạy cho mute chunks khi policy thật sự yêu cầu.
    """
    if not timeline:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=r={sample_rate}:cl=stereo",
                "-t",
                "0.1",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        return

    if video_duration_override is not None:
        total_ms = int(video_duration_override)
    else:
        total_ms = int(timeline[-1].new_end)

    audio_mix_config = audio_mix_config or {}
    has_bgm_source = bool(bgm_path and Path(bgm_path).exists())
    resolved_policies = _normalize_audio_policies_for_assembly(
        audio_policies,
        use_vocal_extraction=use_vocal_extraction,
        has_bgm_path=has_bgm_source,
    )
    global_bgm_policy = resolved_policies["global_bgm"]
    mute_audio_policy = resolved_policies["mute_audio"]
    ambient_policy = resolved_policies["ambient"]

    logger.info(
        "Bắt đầu mix audio (Concat approach), tổng thời lượng: %.2fs | policies: global_bgm=%s, mute_audio=%s, ambient=%s",
        total_ms / 1000.0,
        global_bgm_policy,
        mute_audio_policy,
        ambient_policy,
    )

    # `mute_audio` chỉ ảnh hưởng audio bên trong các mute segments.
    # Ambient và global BGM sẽ được xử lý ở các pass sau như những overlay độc lập.
    if has_bgm_source and global_bgm_policy == "whole_video" and mute_audio_policy in {"original", "instrumental"}:
        logger.warning(
            "Cấu hình có thể gây cảm giác double BGM ở vùng mute: global_bgm=whole_video và mute_audio=%s.",
            mute_audio_policy,
        )

    reuse_bgm_for_mute = mute_audio_policy == "instrumental" and has_bgm_source
    mute_separator_preset = None
    if mute_audio_policy == "vocals":
        mute_separator_preset = "vocal_extraction"
    elif mute_audio_policy == "instrumental" and not reuse_bgm_for_mute:
        mute_separator_preset = "bgm_extraction"

    pad_s = 1.0 if mute_separator_preset else 0.0
    # `quoted_pad_info` lưu metadata của pass-1 extract raw mute audio.
    # Sau khi batch-separate xong, ta dùng metadata này để trim bỏ padding,
    # map kết quả separator về đúng `final_q` và giữ chuẩn duration của timeline cuối.
    quoted_pad_info: Dict[str, Tuple[float, float, str, float, bool, bool]] = {}

    logger.info(f"Đang chuẩn bị {len(timeline)} audio chunks...")
    total_chunks = len(timeline)

    def _prepare_chunk(index: int, seg: TimelineSegment) -> Tuple[int, str]:
        target_dur_s = seg.new_chunk_dur / 1000.0
        is_first = index == 0
        is_last = index == total_chunks - 1

        if target_dur_s <= 0:
            return index, ""

        if seg.block_type == "mute":
            tmp_q = str(Path(tmp_dir) / f"chunk_{index:04d}_mute_raw.wav")
            final_q = str(Path(tmp_dir) / f"chunk_{index:04d}_mute.wav")

            if Path(final_q).exists():
                return index, final_q

            if mute_audio_policy == "silence":
                _generate_silence_chunk(final_q, sample_rate, target_dur_s)
                return index, final_q

            if mute_audio_policy == "instrumental" and reuse_bgm_for_mute and bgm_path:
                return _prepare_synced_audio_chunk(
                    index=index,
                    seg=seg,
                    source_path=bgm_path,
                    output_path=final_q,
                    sample_rate=sample_rate,
                    fade_ms=_DEFAULT_FADE_MS,
                    is_first_chunk=is_first,
                    is_last_chunk=is_last,
                )

            actual_left_pad = extract_quoted_audio(
                video_path,
                seg.orig_start,
                seg.orig_end,
                tmp_q,
                pad_s=pad_s,
            )

            if mute_separator_preset:
                duration_s = (seg.orig_end - seg.orig_start) / 1000.0
                quoted_pad_info[tmp_q] = (
                    actual_left_pad,
                    duration_s,
                    final_q,
                    target_dur_s,
                    is_first,
                    is_last,
                )
            else:
                _finalize_audio_chunk(
                    tmp_q,
                    final_q,
                    sample_rate,
                    target_dur_s,
                    is_first_chunk=is_first,
                    is_last_chunk=is_last,
                )
            return index, final_q

        if seg.block_type == "tts" and seg.tts_clip_path:
            final_c = str(Path(tmp_dir) / f"chunk_{index:04d}_tts.wav")
            if not Path(final_c).exists():
                compress_tts_clip(
                    seg.tts_clip_path,
                    seg.audio_speed,
                    final_c,
                    tts_provider,
                    target_dur_s=target_dur_s,
                    is_first_chunk=is_first,
                    is_last_chunk=is_last,
                )
            return index, final_c

        final_s = str(Path(tmp_dir) / f"chunk_{index:04d}_{seg.block_type}.wav")
        if not Path(final_s).exists():
            _generate_silence_chunk(final_s, sample_rate, target_dur_s)
        return index, final_s

    # Chạy prepare song song
    cpu_count = os.cpu_count() or 2
    max_workers = max(1, int(cpu_count * 0.8))

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_prepare_chunk, i, seg) for i, seg in enumerate(timeline)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda x: x[0])
    ordered_chunk_paths = [path for _, path in results if path]

    if mute_separator_preset and quoted_pad_info:
        raw_quoted_paths = list(quoted_pad_info.keys())
        logger.info(
            "Đang chạy audio-separator trên %s mute chunks (preset=%s, padding %.1fs)...",
            len(raw_quoted_paths),
            mute_separator_preset,
            pad_s,
        )
        from cli.audio_separator import separate_audio_batch

        output_suffix = "vocals" if mute_separator_preset == "vocal_extraction" else "bgm"
        separator_outputs = [str(Path(p).with_name(f"{Path(p).stem}_{output_suffix}.wav")) for p in raw_quoted_paths]

        try:
            separate_audio_batch(
                input_paths=raw_quoted_paths,
                output_paths=separator_outputs,
                preset=mute_separator_preset,
                override_kwargs=audio_separator_config,
            )

            for raw_p, sep_p in zip(raw_quoted_paths, separator_outputs):
                actual_left_pad, dur_s, final_q, target_dur_s, is_first, is_last = quoted_pad_info[raw_p]
                src_to_trim = sep_p if Path(sep_p).exists() else raw_p
                _finalize_audio_chunk(
                    src_to_trim,
                    final_q,
                    sample_rate,
                    target_dur_s,
                    trim_start_s=actual_left_pad,
                    trim_duration_s=dur_s,
                    is_first_chunk=is_first,
                    is_last_chunk=is_last,
                )
            logger.info("Hoàn tất tách audio cho mute chunks và trim padding.")
        except Exception as e:
            logger.error(f"Lỗi khi chạy audio-separator batch, fallback dùng audio gốc cho mute chunks: {e}")
            for raw_p in raw_quoted_paths:
                actual_left_pad, dur_s, final_q, target_dur_s, is_first, is_last = quoted_pad_info[raw_p]
                _finalize_audio_chunk(
                    raw_p,
                    final_q,
                    sample_rate,
                    target_dur_s,
                    trim_start_s=actual_left_pad,
                    trim_duration_s=dur_s,
                    is_first_chunk=is_first,
                    is_last_chunk=is_last,
                )

    ambient_processed_path = str(Path(tmp_dir) / "ambient_processed.wav")
    has_ambient = False
    # Ambient được preprocess trước để bám đúng total duration và mute mask nếu cần.
    # Gain cuối cùng chưa áp ở đây; final mix mới là SSOT của `audio_mix_config.ambient_volume`.
    if ambient_path and Path(ambient_path).exists() and ambient_policy != "off":
        logger.info("Đang xử lý nhạc nền (ambient)...")
        has_ambient = _process_ambient_track(
            ambient_path,
            timeline,
            total_ms,
            ambient_processed_path,
            sample_rate,
            ambient_policy=ambient_policy,
        )

    logger.info(f"Đang nối (concat) {len(ordered_chunk_paths)} audio chunks...")
    concat_list_path = str(Path(tmp_dir) / "concat_list.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for p in ordered_chunk_paths:
            safe_p = Path(p).as_posix().replace("'", "'\\''")
            f.write(f"file '{safe_p}'\n")

    concatenated_audio = str(Path(tmp_dir) / "concatenated_main.wav")
    concat_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_list_path,
        "-c",
        "copy",
        concatenated_audio,
    ]

    try:
        subprocess.run(concat_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi concat audio: {e.stderr.decode('utf-8', errors='ignore')}")
        raise RuntimeError("Không thể concat audio chunks")

    synced_bgm_path = str(Path(tmp_dir) / "synced_bgm.wav")
    has_bgm = False
    # Global BGM luôn được sync theo toàn bộ final timeline trước.
    # Nếu policy là `exclude_mute`, ta vẫn build full synced track rồi mới mute theo volume expression
    # ở final mix; cách này đơn giản hơn việc cắt/nối lại track BGM nhiều lần.
    if has_bgm_source and global_bgm_policy in {"whole_video", "exclude_mute"}:
        logger.info("Đang xử lý BGM track (synced theo timeline)...")
        bgm_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _prepare_bgm_chunk,
                    i,
                    seg,
                    bgm_path,
                    tmp_dir,
                    sample_rate,
                    _DEFAULT_FADE_MS,
                    i == 0,
                    i == total_chunks - 1,
                )
                for i, seg in enumerate(timeline)
            ]
            for future in concurrent.futures.as_completed(futures):
                bgm_results.append(future.result())
        bgm_results.sort(key=lambda x: x[0])
        ordered_bgm_paths = [path for _, path in bgm_results if path]

        if ordered_bgm_paths:
            bgm_concat_list = str(Path(tmp_dir) / "bgm_concat_list.txt")
            with open(bgm_concat_list, "w", encoding="utf-8") as f:
                for p in ordered_bgm_paths:
                    safe_p = Path(p).as_posix().replace("'", "'\\''")
                    f.write(f"file '{safe_p}'\n")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    bgm_concat_list,
                    "-c",
                    "copy",
                    synced_bgm_path,
                ],
                check=True,
                capture_output=True,
            )
            has_bgm = True
            logger.info(f"Đã tạo synced BGM track: {synced_bgm_path}")

    logger.info("Đang thực hiện mix cuối cùng (Final Mix)...")

    if not has_ambient and not has_bgm:
        shutil.copy(concatenated_audio, output_path)
        logger.info("Mix audio hoàn tất (chỉ có track chính).")
        return

    # Final mix là nơi duy nhất áp gain cuối cùng cho các overlay toàn cục.
    # Nhờ vậy ambient/BGM preprocess chỉ cần lo timing + mask, tránh double attenuation.
    mute_ranges = build_mute_ranges(timeline)
    inputs = [concatenated_audio]
    filter_parts = ["[0:a]volume=1.0[a0]"]
    mix_labels = ["[a0]"]
    input_idx = 1

    if has_ambient:
        inputs.append(ambient_processed_path)
        amb_vol = float(audio_mix_config.get("ambient_volume", 0.03))
        filter_parts.append(f"[{input_idx}:a]volume={amb_vol:.6f}[a{input_idx}]")
        mix_labels.append(f"[a{input_idx}]")
        input_idx += 1

    if has_bgm:
        inputs.append(synced_bgm_path)
        bgm_vol = float(audio_mix_config.get("bgm_volume", 1.0))
        bgm_filter = (
            _build_mute_volume_filter(mute_ranges, bgm_vol)
            if global_bgm_policy == "exclude_mute"
            else f"volume={bgm_vol:.6f}"
        )
        filter_parts.append(f"[{input_idx}:a]{bgm_filter}[a{input_idx}]")
        mix_labels.append(f"[a{input_idx}]")
        input_idx += 1

    filter_complex = ";".join(filter_parts)
    filter_complex += ";" + "".join(mix_labels) + f"amix=inputs={len(mix_labels)}:duration=first:dropout_transition=0:normalize=0[out]"

    mix_cmd = [
        "ffmpeg",
        "-y",
        *sum([["-i", p] for p in inputs], []),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-ar",
        str(sample_rate),
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        output_path,
    ]

    try:
        subprocess.run(mix_cmd, check=True, capture_output=True)
        logger.info("Mix audio hoàn tất thành công.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi final mix: {e.stderr.decode('utf-8', errors='ignore')}")
        shutil.copy(concatenated_audio, output_path)
