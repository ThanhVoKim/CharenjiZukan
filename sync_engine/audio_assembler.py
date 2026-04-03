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

logger = logging.getLogger("sync_video")

def compress_tts_clip(wav_path: str, audio_speed: float, output_path: str) -> None:
    # Luôn áp dụng filter tăng âm lượng và limiter cho TTS audio
    base_filter = "volume=1.5,alimiter=limit=0.95:level_in=1:level_out=1"
    
    if audio_speed > 1.01:
        atempo_str = _build_atempo_filter(audio_speed)  # Reuse từ media_utils.py
        filter_str = f"{atempo_str},{base_filter}"
    else:
        filter_str = base_filter
        
    subprocess.run([
        "ffmpeg", "-y", "-i", wav_path,
        "-filter:a", filter_str,
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
        output_path,
    ], check=True, capture_output=True)

def extract_quoted_audio(
    video_path: str,
    orig_start_ms: float,
    orig_end_ms: float,
    output_path: str,
) -> None:
    """
    THAY ĐỔI: Dùng 2 pass -ss (rough seek trước -i, fine seek sau -i) để extract chính xác.
    """
    rough_start_s = max(0.0, orig_start_ms / 1000.0 - 5.0)
    exact_offset_s = orig_start_ms / 1000.0 - rough_start_s
    
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", f"{rough_start_s:.6f}",
        "-i", video_path,
        "-ss", f"{exact_offset_s:.6f}",
        "-t",  f"{(orig_end_ms - orig_start_ms) / 1000:.6f}",
        "-vn", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
        output_path,
    ], check=True, capture_output=True)

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
    sample_rate: int = 48000
) -> bool:
    """
    Xử lý nhạc nền: loop, giảm âm lượng, và mute tại các đoạn quoted audio.
    """
    if not ambient_path or not Path(ambient_path).exists():
        return False

    # Tạo biểu thức volume để mute tại các đoạn quoted
    mute_ranges = [(s.new_start / 1000.0, s.new_end / 1000.0) for s in timeline if s.block_type == "mute"]
    
    # Base volume: -25dB ~ 0.056
    base_vol = 0.056
    
    if not mute_ranges:
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
        # 1. Delay từng input
        for i, (path, delay_ms) in enumerate(inputs):
            delay_ms_int = int(delay_ms)
            f.write(f"[{i}:a]adelay={delay_ms_int}|{delay_ms_int}[aud{i}];\n")
        
        # 2. Mix tất cả lại
        mix_inputs = "".join([f"[aud{i}]" for i in range(len(inputs))])
        f.write(f"{mix_inputs}amix=inputs={len(inputs)}:dropout_transition=0:normalize=0[out]\n")

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

    total_ms = int(timeline[-1].new_end)
    logger.info(f"Bắt đầu mix audio, tổng thời lượng: {total_ms/1000:.2f}s")

    # 1. Chuẩn bị các file input (Quoted và TTS)
    prepared_inputs: List[Tuple[str, float]] = [] # (path, delay_ms)
    
    logger.info("Đang chuẩn bị các file audio thành phần...")
    for i, seg in enumerate(timeline):
        if seg.block_type == "mute":
            tmp_q = str(Path(tmp_dir) / f"quoted_{int(seg.orig_start)}.wav")
            if not Path(tmp_q).exists():
                extract_quoted_audio(video_path, seg.orig_start, seg.orig_end, tmp_q)
            if Path(tmp_q).exists():
                prepared_inputs.append((tmp_q, seg.new_start))
                
        elif seg.block_type == "tts" and seg.tts_clip_path:
            tmp_c = str(Path(tmp_dir) / f"processed_{int(seg.new_start)}.wav")
            if not Path(tmp_c).exists():
                compress_tts_clip(seg.tts_clip_path, seg.audio_speed, tmp_c)
            if Path(tmp_c).exists():
                prepared_inputs.append((tmp_c, seg.new_start))

    # 2. Xử lý Ambient Track
    ambient_processed_path = str(Path(tmp_dir) / "ambient_processed.wav")
    has_ambient = False
    if ambient_path and Path(ambient_path).exists():
        logger.info("Đang xử lý nhạc nền (ambient)...")
        has_ambient = _process_ambient_track(
            ambient_path, timeline, total_ms, ambient_processed_path, sample_rate
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
        # Dùng khoảng 75% số core hiện có, tối thiểu 1, tối đa không vượt quá số batch
        optimal_workers = max(1, int(cpu_count * 0.75))
        max_workers = min(optimal_workers, len(batches)) or 1
        
        logger.info(f"Chạy song song {len(batches)} batch với tối đa {max_workers} worker (CPU cores: {cpu_count})...")

        def _run_batch(
            batch_index: int,
            batch_data: List[Tuple[str, float]],
            batch_out_path: str,
        ) -> Tuple[int, str, bool]:
            success = _mix_audio_batch(batch_data, batch_out_path, sample_rate)
            return batch_index, batch_out_path, success

        successful_batches: List[Tuple[int, str]] = []
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
                    logger.error(f"Lỗi khi mix batch {batch_index}")

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
            f.write(f"{mix_inputs}amix=inputs={len(final_inputs)}:dropout_transition=0:normalize=0[out]\n")

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
