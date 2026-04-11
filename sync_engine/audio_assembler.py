import concurrent.futures
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
import tempfile
import logging

from sync_engine.models import TimelineSegment
from utils.media_utils import _build_atempo_filter

_PCM_AUDIO_EXTENSIONS = {'.wav', '.flac', '.aiff', '.aif', '.pcm'}

logger = logging.getLogger("sync_video")

def compress_tts_clip(wav_path: str, audio_speed: float, output_path: str, tts_provider: str = "edge", target_dur_s: Optional[float] = None) -> None:
    # Chỉ áp dụng filter tăng âm lượng và limiter cho EdgeTTS
    if tts_provider == "edge":
        base_filter = "volume=1.5,alimiter=limit=0.95:level_in=1:level_out=1"
    else:
        base_filter = "" # Voicevox đã tự tăng volumeScale
    
    if audio_speed > 1.01:
        atempo_str = _build_atempo_filter(audio_speed)  # Reuse từ media_utils.py
        filter_str = f"{atempo_str},{base_filter}" if base_filter else atempo_str
    else:
        filter_str = base_filter
        
    # Thêm atrim và apad để đảm bảo duration chính xác tuyệt đối
    if target_dur_s is not None:
        pad_trim_filter = f"atrim=start=0,asetpts=PTS-STARTPTS,apad=whole_dur={target_dur_s:.6f},atrim=end={target_dur_s:.6f}"
        filter_str = f"{filter_str},{pad_trim_filter}" if filter_str else pad_trim_filter

    cmd = [
        "ffmpeg", "-y", "-i", wav_path,
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
    ]
    if filter_str:
        cmd.extend(["-filter:a", filter_str])
    cmd.append(output_path)
    
    subprocess.run(cmd, check=True, capture_output=True)

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
        # PCM audio: single-pass seek — sample-accurate, không có codec delay.
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{pad_start_s:.9f}",
            "-i", video_path,
            "-t", f"{duration_s:.9f}",
            "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
            output_path,
        ]
        logger.debug(
            "extract_quoted_audio [PCM] %.3fs–%.3fs → %s",
            pad_start_s, pad_end_s, output_path,
        )
    else:
        # Video source: 2-pass seek để xử lý codec delay (AAC priming samples).
        rough_start_s = max(0.0, pad_start_s - 5.0)
        exact_offset_s = pad_start_s - rough_start_s
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{rough_start_s:.6f}",
            "-i", video_path,
            "-ss", f"{exact_offset_s:.6f}",
            "-t", f"{duration_s:.6f}",
            "-vn", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
            output_path,
        ]
        logger.debug(
            "extract_quoted_audio [Video] rough=%.3fs offset=%.3fs dur=%.3fs → %s",
            rough_start_s, exact_offset_s, duration_s, output_path,
        )

    subprocess.run(cmd, check=True, capture_output=True)
    return actual_left_pad

def build_ambient_mask(
    timeline: List[TimelineSegment],
    total_ms: float,
) -> List[Tuple[float, float]]:
    """
    Trả về list khoảng (new_start, new_end) cho phép ambient phát.
    Ambient bị tắt trong khoảng new_start..new_end của mute segments.
    """
    mute_ranges = sorted(
        [(s.new_start, s.new_end) for s in timeline if s.block_type == "mute"]
    )
    ambient = []
    cursor = 0.0
    for ms, me in mute_ranges:
        if cursor < ms:
            ambient.append((cursor, ms))
        cursor = me
    if cursor < total_ms:
        ambient.append((cursor, total_ms))
    return ambient

def _process_ambient_track(
    ambient_path: str,
    timeline: List[TimelineSegment],
    total_ms: float,
    output_path: str,
    sample_rate: int = 48000,
    use_demucs: bool = False
) -> bool:
    """
    Xử lý nhạc nền: loop, giảm âm lượng, và mute tại các đoạn quoted audio.
    """
    if not ambient_path or not Path(ambient_path).exists():
        return False

    # Tạo biểu thức volume để mute tại các đoạn quoted
    mute_ranges = [(s.new_start / 1000.0, s.new_end / 1000.0) for s in timeline if s.block_type == "mute"]
    
    # Base volume: -25dB ~ 0.056
    base_vol = 0.03
    
    if not mute_ranges or use_demucs:
        volume_expr = f"volume={base_vol}"
    else:
        # Xây dựng biểu thức: if(between(t, start1, end1) + between(t, start2, end2) + ..., 0, base_vol)
        between_exprs = [f"between(t,{s:.3f},{e:.3f})" for s, e in mute_ranges]
        # FFmpeg expression có giới hạn độ dài, nếu quá nhiều mute ranges, ta cần cách tiếp cận khác.
        # Tuy nhiên, với số lượng mute ranges thông thường, cách này vẫn ổn định.
        # Để an toàn với số lượng lớn, ta chia nhỏ hoặc dùng file filter.
        # Ở đây, ta dùng cách an toàn: tạo file filter_complex
        
        # Thay vì dùng expression phức tạp, ta dùng filter volume với timeline editing
        # volume=0:enable='between(t,s1,e1)+between(t,s2,e2)...'
        # Nhưng timeline editing cũng có giới hạn độ dài.
        # Cách tốt nhất: dùng `volume` filter nhiều lần, mỗi lần cho 1 khoảng mute.
        # Hoặc đơn giản nhất: tạo một file audio im lặng cùng độ dài, sau đó mix.
        # Cách tối ưu bằng FFmpeg:
        
        # Tạo chuỗi expression. Nếu quá dài, ta sẽ dùng cách khác.
        expr = "+".join(between_exprs)
        if len(expr) > 10000:
             logger.warning("Quá nhiều đoạn mute, việc xử lý ambient có thể gặp lỗi giới hạn độ dài lệnh.")
             
        volume_expr = f"volume='if({expr}, 0, {base_vol})':eval=frame"

    # Lệnh FFmpeg:
    # -stream_loop -1: loop vô hạn
    # -t total_s: cắt đúng độ dài
    total_s = total_ms / 1000.0
    
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", ambient_path,
        "-t", f"{total_s:.3f}",
        "-filter:a", volume_expr,
        "-ar", str(sample_rate), "-ac", "2", "-c:a", "pcm_s16le",
        output_path
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
    use_demucs: bool = False,
    tts_provider: str = "edge",
    video_duration_override: Optional[float] = None,
) -> None:
    """
    Sử dụng FFmpeg để ghép (concat) audio tuần tự.
    Layer order (bottom → top): ambient → (tts/quoted/silence concat)
    """
    if not timeline:
        # Tạo file im lặng nếu timeline trống
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=stereo",
            "-t", "0.1", output_path
        ], check=True, capture_output=True)
        return

    if video_duration_override is not None:
        total_ms = int(video_duration_override)
    else:
        total_ms = int(timeline[-1].new_end)
        
    logger.info(f"Bắt đầu mix audio (Concat approach), tổng thời lượng: {total_ms/1000:.2f}s")

    # Cấu hình padding cho Demucs
    pad_s = 3.5 if use_demucs else 0.0
    quoted_pad_info = {} # dict lưu {path: (actual_left_pad_s, duration_s, final_q, target_dur_s)}
    
    # Danh sách lưu đường dẫn các chunk đã được xử lý xong, theo đúng thứ tự timeline
    ordered_chunk_paths: List[str] = []

    # 1. Chuẩn bị / Extract / Compress tất cả các chunk (chạy đa luồng)
    logger.info(f"Đang chuẩn bị {len(timeline)} audio chunks...")
    
    def _prepare_chunk(index: int, seg: TimelineSegment) -> Tuple[int, str]:
        target_dur_s = seg.new_chunk_dur / 1000.0
        
        if target_dur_s <= 0:
            return index, ""
            
        if seg.block_type == "mute":
            tmp_q = str(Path(tmp_dir) / f"chunk_{index:04d}_mute_raw.wav")
            final_q = str(Path(tmp_dir) / f"chunk_{index:04d}_mute.wav")
            
            if not Path(final_q).exists():
                if not Path(tmp_q).exists():
                    actual_left_pad = extract_quoted_audio(
                        video_path, seg.orig_start, seg.orig_end, tmp_q, pad_s=pad_s
                    )
                    if use_demucs:
                        duration_s = (seg.orig_end - seg.orig_start) / 1000.0
                        quoted_pad_info[tmp_q] = (actual_left_pad, duration_s, final_q, target_dur_s)
                
                # Nếu không dùng demucs, tạo final chunk luôn bằng atrim/apad
                if not use_demucs and Path(tmp_q).exists():
                    cmd = [
                        "ffmpeg", "-y", "-i", tmp_q,
                        "-filter:a", f"atrim=start=0,asetpts=PTS-STARTPTS,apad=whole_dur={target_dur_s:.6f},atrim=end={target_dur_s:.6f}",
                        "-ar", str(sample_rate), "-ac", "2", "-c:a", "pcm_s16le",
                        final_q
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
            return index, final_q
            
        elif seg.block_type == "tts" and seg.tts_clip_path:
            final_c = str(Path(tmp_dir) / f"chunk_{index:04d}_tts.wav")
            if not Path(final_c).exists():
                compress_tts_clip(seg.tts_clip_path, seg.audio_speed, final_c, tts_provider, target_dur_s=target_dur_s)
            return index, final_c
            
        else: # gap hoặc tail
            final_s = str(Path(tmp_dir) / f"chunk_{index:04d}_{seg.block_type}.wav")
            if not Path(final_s).exists():
                cmd = [
                    "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=stereo",
                    "-t", f"{target_dur_s:.6f}",
                    final_s
                ]
                subprocess.run(cmd, check=True, capture_output=True)
            return index, final_s

    # Chạy prepare song song
    cpu_count = os.cpu_count() or 2
    max_workers = max(1, int(cpu_count * 0.8))
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_prepare_chunk, i, seg) for i, seg in enumerate(timeline)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            
    # Sắp xếp lại theo đúng thứ tự
    results.sort(key=lambda x: x[0])
    ordered_chunk_paths = [path for _, path in results if path]

    # Xử lý riêng cho Demucs (chạy batch trên các tmp_q)
    if use_demucs and quoted_pad_info:
        raw_quoted_paths = list(quoted_pad_info.keys())
        logger.info(f"Đang chạy Demucs trên {len(raw_quoted_paths)} đoạn quoted clips (batch, có padding {pad_s}s)...")
        from cli.demucs_audio import separate_audio_batch
        
        demucs_outputs = [str(Path(p).with_name(f"{Path(p).stem}_vocals.wav")) for p in raw_quoted_paths]
        
        try:
            separate_audio_batch(
                input_paths=raw_quoted_paths,
                output_paths=demucs_outputs,
                model="htdemucs",
                keep="vocals",
                device=None,
                segment=7
            )
            
            # Cắt bỏ phần padding (trim) sau khi có kết quả Demucs, và apad/atrim chuẩn hóa
            for raw_p, demucs_p in zip(raw_quoted_paths, demucs_outputs):
                actual_left_pad, dur_s, final_q, target_dur_s = quoted_pad_info[raw_p]
                
                src_to_trim = demucs_p if Path(demucs_p).exists() else raw_p
                
                trim_cmd = [
                    "ffmpeg", "-y",
                    "-i", src_to_trim,
                    "-filter:a", f"atrim=start={actual_left_pad:.6f}:duration={dur_s:.6f},asetpts=PTS-STARTPTS,apad=whole_dur={target_dur_s:.6f},atrim=end={target_dur_s:.6f}",
                    "-ar", str(sample_rate), "-ac", "2", "-c:a", "pcm_s16le",
                    final_q
                ]
                subprocess.run(trim_cmd, check=True, capture_output=True)
            logger.info("Hoàn tất tách lời và trim padding bằng Demucs.")
        except Exception as e:
            logger.error(f"Lỗi khi chạy Demucs batch, fallback dùng audio có nhạc nền: {e}")
            for raw_p in raw_quoted_paths:
                actual_left_pad, dur_s, final_q, target_dur_s = quoted_pad_info[raw_p]
                trim_cmd = [
                    "ffmpeg", "-y",
                    "-i", raw_p,
                    "-filter:a", f"atrim=start={actual_left_pad:.6f}:duration={dur_s:.6f},asetpts=PTS-STARTPTS,apad=whole_dur={target_dur_s:.6f},atrim=end={target_dur_s:.6f}",
                    "-ar", str(sample_rate), "-ac", "2", "-c:a", "pcm_s16le",
                    final_q
                ]
                subprocess.run(trim_cmd, check=True, capture_output=True)

    # 2. Xử lý Ambient Track (Song song với việc nối)
    ambient_processed_path = str(Path(tmp_dir) / "ambient_processed.wav")
    has_ambient = False
    if ambient_path and Path(ambient_path).exists():
        logger.info("Đang xử lý nhạc nền (ambient)...")
        has_ambient = _process_ambient_track(
            ambient_path, timeline, total_ms, ambient_processed_path, sample_rate, use_demucs
        )

    # 3. Concat tất cả các chunks
    logger.info(f"Đang nối (concat) {len(ordered_chunk_paths)} audio chunks...")
    concat_list_path = str(Path(tmp_dir) / "concat_list.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for p in ordered_chunk_paths:
            # Escape path cho ffconcat
            safe_p = Path(p).as_posix().replace("'", "'\\''")
            f.write(f"file '{safe_p}'\n")

    concatenated_audio = str(Path(tmp_dir) / "concatenated_main.wav")
    
    # Dùng concat demuxer, -c copy cực nhanh do tất cả đã cùng format (pcm_s16le 48000Hz stereo)
    concat_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        concatenated_audio
    ]
    
    try:
        subprocess.run(concat_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi concat audio: {e.stderr.decode('utf-8', errors='ignore')}")
        raise RuntimeError("Không thể concat audio chunks")

    # 4. Mix Final (Concatenated Audio + Ambient)
    logger.info("Đang thực hiện mix cuối cùng (Final Mix)...")
    
    if not has_ambient:
        shutil.copy(concatenated_audio, output_path)
        logger.info("Mix audio hoàn tất (chỉ có track chính, không có ambient).")
    else:
        mix_cmd = [
            "ffmpeg", "-y",
            "-i", concatenated_audio,
            "-i", ambient_processed_path,
            "-filter_complex", "amix=inputs=2:duration=first:dropout_transition=0:normalize=0[out]",
            "-map", "[out]",
            "-ar", str(sample_rate), "-ac", "2", "-c:a", "pcm_s16le",
            output_path
        ]
        
        try:
            subprocess.run(mix_cmd, check=True, capture_output=True)
            logger.info("Mix audio hoàn tất thành công (Main Track + Ambient).")
        except subprocess.CalledProcessError as e:
            logger.error(f"Lỗi final mix: {e.stderr.decode('utf-8', errors='ignore')}")
            shutil.copy(concatenated_audio, output_path)
