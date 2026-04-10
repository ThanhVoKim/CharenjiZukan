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

def compress_tts_clip(wav_path: str, audio_speed: float, output_path: str, tts_provider: str = "edge") -> None:
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

def _get_audio_duration_s(audio_path: str) -> float:
    """Đo duration audio bằng ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

def _mix_audio_batch(
    inputs: List[Tuple[str, float]], # List of (file_path, delay_ms)
    output_path: str,
    sample_rate: int = 48000
) -> bool:
    """
    Mix một batch các file audio đã được delay.
    """
    if not inputs:
        return False

    # Tạo file filter_complex
    filter_script_path = output_path + ".filter.txt"
    
    with open(filter_script_path, "w", encoding="utf-8") as f:
        # 1. Delay từng input bằng adelay (hậu tố S cho precision thập phân)
        # - adelay=Xs|Xs: chèn silence X giây ở ĐẦU audio (precision thập phân)
        # - apad=whole_dur=Y: đảm bảo tổng duration = Y (bù silence ở CUỐI nếu cần)
        # - atrim=end=Y: cắt chính xác tại mốc Y
        max_end_s = 0.0
        
        for i, (path, delay_ms) in enumerate(inputs):
            delay_s = delay_ms / 1000.0
            clip_dur_s = _get_audio_duration_s(path)
            total_dur_s = delay_s + clip_dur_s
            
            f.write(
                f"[{i}:a]adelay={delay_s:.6f}S|{delay_s:.6f}S,"
                f"apad=whole_dur={total_dur_s:.6f},"
                f"atrim=end={total_dur_s:.6f}[aud{i}];\n"
            )
            
            if total_dur_s > max_end_s:
                max_end_s = total_dur_s
        
        # 2. Mix tất cả lại
        mix_inputs = "".join([f"[aud{i}]" for i in range(len(inputs))])
        f.write(
            f"{mix_inputs}amix=inputs={len(inputs)}:all=1:"
            f"dropout_transition=0:normalize=0,"
            f"atrim=end={max_end_s:.6f}[out]\n"
        )

    # Xây dựng lệnh FFmpeg
    cmd = ["ffmpeg", "-y"]
    for path, _ in inputs:
        cmd.extend(["-i", path])
    
    cmd.extend([
        "-filter_complex_script", filter_script_path,
        "-map", "[out]",
        "-ar", str(sample_rate), "-ac", "2", "-c:a", "pcm_s16le",
        output_path
    ])
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi mix audio batch: {e.stderr.decode('utf-8', errors='ignore')}")
        return False
    finally:
        # Dọn dẹp file script
        try:
            Path(filter_script_path).unlink(missing_ok=True)
        except Exception:
            pass

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
    Sử dụng FFmpeg để mix audio siêu tốc (thay thế pydub).
    Layer order (bottom → top): ambient → quoted → TTS
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
        
    logger.info(f"Bắt đầu mix audio, tổng thời lượng: {total_ms/1000:.2f}s")

    # 1. Chuẩn bị các file input (Quoted và TTS)
    prepared_inputs: List[Tuple[str, float]] = [] # (path, delay_ms)
    
    # Cấu hình padding cho Demucs (htdemucs segment=7s, overlap=0.25 -> tối thiểu padding 3.5s)
    pad_s = 3.5 if use_demucs else 0.0
    quoted_pad_info = {} # dict lưu {path: (actual_left_pad_s, duration_s)}
    
    logger.info("Đang chuẩn bị các file audio thành phần...")
    for i, seg in enumerate(timeline):
        if seg.block_type == "mute":
            tmp_q = str(Path(tmp_dir) / f"quoted_{int(seg.orig_start)}.wav")
            if not Path(tmp_q).exists():
                actual_left_pad = extract_quoted_audio(
                    video_path, seg.orig_start, seg.orig_end, tmp_q, pad_s=pad_s
                )
                if use_demucs:
                    duration_s = (seg.orig_end - seg.orig_start) / 1000.0
                    quoted_pad_info[tmp_q] = (actual_left_pad, duration_s)
            elif use_demucs:
                # Nếu đã có từ lần chạy trước, tính lại duration
                duration_s = (seg.orig_end - seg.orig_start) / 1000.0
                actual_left_pad = min(pad_s, seg.orig_start / 1000.0)
                quoted_pad_info[tmp_q] = (actual_left_pad, duration_s)
                
            if Path(tmp_q).exists():
                prepared_inputs.append((tmp_q, seg.new_start))
                
        elif seg.block_type == "tts" and seg.tts_clip_path:
            tmp_c = str(Path(tmp_dir) / f"processed_{int(seg.new_start)}.wav")
            if not Path(tmp_c).exists():
                compress_tts_clip(seg.tts_clip_path, seg.audio_speed, tmp_c, tts_provider)
            if Path(tmp_c).exists():
                prepared_inputs.append((tmp_c, seg.new_start))

    if use_demucs:
        quoted_clips_paths = [path for path, delay in prepared_inputs if Path(path).name.startswith("quoted_")]
        if quoted_clips_paths:
            logger.info(f"Đang chạy Demucs trên {len(quoted_clips_paths)} đoạn quoted clips (batch, có padding {pad_s}s)...")
            from cli.demucs_audio import separate_audio_batch
            
            demucs_outputs = [str(Path(p).with_name(f"{Path(p).stem}_vocals.wav")) for p in quoted_clips_paths]
            
            try:
                separate_audio_batch(
                    input_paths=quoted_clips_paths,
                    output_paths=demucs_outputs,
                    model="htdemucs",
                    keep="vocals",
                    device=None,
                    segment=7
                )
                
                # Cắt bỏ phần padding (trim) sau khi có kết quả Demucs, ghi đè lại file `tmp_q` gốc
                for orig_p, demucs_p in zip(quoted_clips_paths, demucs_outputs):
                    if Path(demucs_p).exists():
                        actual_left_pad, dur_s = quoted_pad_info[orig_p]
                        trim_cmd = [
                            "ffmpeg", "-y",
                            "-i", demucs_p,
                            "-ss", f"{actual_left_pad:.6f}",
                            "-t", f"{dur_s:.6f}",
                            "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
                            orig_p
                        ]
                        subprocess.run(trim_cmd, check=True, capture_output=True)
                logger.info("Hoàn tất tách lời và trim padding bằng Demucs.")
            except Exception as e:
                logger.error(f"Lỗi khi chạy Demucs batch, fallback dùng audio có nhạc nền: {e}")
                # Nếu lỗi Demucs, ta vẫn phải trim padding của file gốc (do lúc extract đã thêm padding)
                for orig_p in quoted_clips_paths:
                    if orig_p in quoted_pad_info:
                        actual_left_pad, dur_s = quoted_pad_info[orig_p]
                        if actual_left_pad > 0:
                            # Trim trực tiếp trên file gốc
                            tmp_fallback = str(Path(orig_p).with_name(f"{Path(orig_p).stem}_fallback.wav"))
                            trim_cmd = [
                                "ffmpeg", "-y",
                                "-i", orig_p,
                                "-ss", f"{actual_left_pad:.6f}",
                                "-t", f"{dur_s:.6f}",
                                "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
                                tmp_fallback
                            ]
                            subprocess.run(trim_cmd, check=True, capture_output=True)
                            shutil.move(tmp_fallback, orig_p)

    # 2. Xử lý Ambient Track
    ambient_processed_path = str(Path(tmp_dir) / "ambient_processed.wav")
    has_ambient = False
    if ambient_path and Path(ambient_path).exists():
        logger.info("Đang xử lý nhạc nền (ambient)...")
        has_ambient = _process_ambient_track(
            ambient_path, timeline, total_ms, ambient_processed_path, sample_rate, use_demucs
        )

    # 3. Chia lô (Batching) để mix
    # Giới hạn số lượng file mở đồng thời của OS (thường là 1024 trên Linux)
    # Ta chọn batch size an toàn là 100
    BATCH_SIZE = 100
    batch_outputs: List[str] = []

    if prepared_inputs:
        logger.info(f"Đang mix {len(prepared_inputs)} file audio thành phần (Batch size: {BATCH_SIZE})...")

        batches: List[Tuple[int, List[Tuple[str, float]], str]] = []
        for i in range(0, len(prepared_inputs), BATCH_SIZE):
            batch_index = i // BATCH_SIZE
            batch = prepared_inputs[i:i+BATCH_SIZE]
            batch_out = str(Path(tmp_dir) / f"mix_batch_{batch_index}.wav")
            batches.append((batch_index, batch, batch_out))

        # Tính toán số lượng worker dựa trên CPU thực tế
        cpu_count = os.cpu_count() or 2
        # Dùng khoảng 65% số core hiện có, tối thiểu 1, tối đa không vượt quá số batch
        optimal_workers = max(1, int(cpu_count * 0.65))
        max_workers = min(optimal_workers, len(batches)) or 1
        
        logger.info(f"Chạy song song {len(batches)} batch với tối đa {max_workers} worker (CPU cores: {cpu_count})...")

        def _run_batch(
            batch_index: int,
            batch_data: List[Tuple[str, float]],
            batch_out_path: str,
        ) -> Tuple[int, str, bool]:
            logger.info(f"[Worker] Bắt đầu xử lý batch {batch_index} ({len(batch_data)} files)...")
            success = _mix_audio_batch(batch_data, batch_out_path, sample_rate)
            if success:
                logger.info(f"[Worker] Hoàn thành batch {batch_index}.")
            else:
                logger.error(f"[Worker] Thất bại tại batch {batch_index}.")
            return batch_index, batch_out_path, success

        successful_batches: List[Tuple[int, str]] = []
        failed_batches: List[int] = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_run_batch, batch_index, batch_data, batch_out_path)
                for batch_index, batch_data, batch_out_path in batches
            ]

            for future in concurrent.futures.as_completed(futures):
                batch_index, batch_out_path, success = future.result()
                if success:
                    successful_batches.append((batch_index, batch_out_path))
                else:
                    failed_batches.append(batch_index)
                    logger.error(f"Lỗi khi mix batch {batch_index}")

        if failed_batches:
            failed_batches.sort()
            raise RuntimeError(f"Quá trình mix audio thất bại tại các batch: {failed_batches}. Hủy toàn bộ tiến trình.")

        batch_outputs = [path for _, path in sorted(successful_batches, key=lambda item: item[0])]

    # 4. Mix Final (Các batch outputs + Ambient)
    logger.info("Đang thực hiện mix cuối cùng (Final Mix)...")
    final_inputs = []
    
    # Thêm một track im lặng có độ dài bằng total_ms làm nền tảng (đảm bảo độ dài video chính xác)
    base_silence_path = str(Path(tmp_dir) / "base_silence.wav")
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=stereo",
        "-t", f"{total_ms/1000.0:.3f}", base_silence_path
    ], check=True, capture_output=True)
    final_inputs.append(base_silence_path)

    if has_ambient:
        final_inputs.append(ambient_processed_path)
        
    final_inputs.extend(batch_outputs)

    if len(final_inputs) == 1:
        # Chỉ có base silence
        shutil.copy(base_silence_path, output_path)
    else:
        # Mix tất cả lại
        filter_script_path = str(Path(tmp_dir) / "final_mix_filter.txt")
        with open(filter_script_path, "w", encoding="utf-8") as f:
            mix_inputs = "".join([f"[{i}:a]" for i in range(len(final_inputs))])
            f.write(f"{mix_inputs}amix=inputs={len(final_inputs)}:all=1:dropout_transition=0:normalize=0[out]\n")

        cmd = ["ffmpeg", "-y"]
        for path in final_inputs:
            cmd.extend(["-i", path])
        
        cmd.extend([
            "-filter_complex_script", filter_script_path,
            "-map", "[out]",
            "-ar", str(sample_rate), "-ac", "2", "-c:a", "pcm_s16le",
            output_path
        ])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info("Mix audio hoàn tất thành công.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Lỗi final mix: {e.stderr.decode('utf-8', errors='ignore')}")
            # Fallback: copy base silence nếu lỗi
            shutil.copy(base_silence_path, output_path)
