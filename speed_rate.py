# -*- coding: utf-8 -*-
"""
speed_rate.py — Audio alignment & concat engine
Trung thành với pyvideotrans/task/_rate.py (TtsSpeedRate)

Hai chế độ:
  voice_autorate=False → _run_no_rate_mode()
      Ghép các wav segment theo đúng timeline SRT.
      Gap (khoảng im lặng giữa 2 subtitle) được chèn silence đúng độ dài.

  voice_autorate=True  → _run_autorate_mode()
      _prepare_data():   tính source_duration = slot thực tế mỗi sub
                         (từ end_time_source của sub trước → start_time của sub sau)
      _calculate():      nếu dubb_time > source_duration → cần nén audio
      rubberband/atempo: time-stretch audio về đúng source_duration
      _concat_aligned(): ghép theo timeline mới, update start/end_time

Không có remove_silent_mid.
"""

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from pydub import AudioSegment

from utils.logger import get_logger

logger = get_logger(__name__)


def _safe_log(level: str, msg: str):
    """Helper để log an toàn, tránh NameError nếu logger không khả dụng."""
    try:
        if level == "debug":
            logger.debug(msg)
        elif level == "info":
            logger.info(msg)
        elif level == "warning":
            logger.warning(msg)
        elif level == "error":
            logger.error(msg)
    except NameError:
        pass  # Logger không khả dụng, bỏ qua


AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS    = 2
MIN_CLIP_MS       = 40   # ms ngắn nhất chấp nhận


# ─────────────────────────────────────────────────────────────────────
# TIME-STRETCH ENGINE
#
# QUAN TRỌNG về pyrubberband:
#   pyrubberband là Python wrapper GỌI BINARY `rubberband` trên hệ thống.
#   Chỉ `pip install pyrubberband` là KHÔNG ĐỦ — cần thêm:
#       apt-get install -y rubberband-cli   (Ubuntu/Colab)
#   Nếu không có binary → dùng FFmpeg atempo (luôn có sẵn trên Colab).
#
# Priority tự động (detect tại load time):
#   1. pyrubberband  — khi binary `rubberband` tồn tại trên PATH
#                      Không đổi pitch, chất lượng cao nhất
#   2. FFmpeg atempo — mặc định trên Colab, không cần binary ngoài
#                      atempo [0.5, 2.0] → chain filter khi ngoài khoảng
# ─────────────────────────────────────────────────────────────────────

# Detect một lần duy nhất lúc import — tránh thử→fail hàng chục lần
_RUBBERBAND_BIN: str = shutil.which("rubberband") or ""
if _RUBBERBAND_BIN:
    print(f"[SpeedRate] rubberband binary: {_RUBBERBAND_BIN} → pyrubberband ON")
else:
    print("[SpeedRate] Không tìm thấy rubberband binary → FFmpeg atempo")


def _build_atempo_filter(speed_factor: float) -> str:
    """
    Tạo chuỗi atempo filter cho FFmpeg.
    atempo chỉ chấp nhận [0.5, 2.0] → chain nhiều filter khi ngoài khoảng.
    Ví dụ: speed=3.5 → "atempo=2.0,atempo=1.75"
    """
    parts = []
    f = speed_factor
    while f > 2.0:
        parts.append("atempo=2.0")
        f /= 2.0
    while f < 0.5:
        parts.append("atempo=0.5")
        f *= 2.0          # ← đúng: f < 0.5 nên nhân 2 để tiến về 0.5
    parts.append(f"atempo={f:.6f}")
    return ",".join(parts)


def _speedup_with_atempo(wav_path: str, target_ms: int) -> bool:
    """
    Time-stretch bằng FFmpeg atempo chain.
    Luôn hoạt động trên Colab — không cần binary ngoài.
    """
    tmp_path = wav_path + ".atempo.wav"
    try:
        seg = AudioSegment.from_file(wav_path, format="wav")
        current_ms = len(seg)
        if current_ms <= 0 or target_ms <= 0 or target_ms >= current_ms:
            return True  # không cần nén

        speed_factor = current_ms / target_ms
        speed_factor = max(0.1, min(speed_factor, 50.0))
        filter_str = _build_atempo_filter(speed_factor)

        cmd = [
            "ffmpeg", "-y", "-i", wav_path,
            "-filter:a", filter_str,
            "-t", f"{target_ms / 1000.0}",
            "-ar", str(AUDIO_SAMPLE_RATE),
            "-ac", str(AUDIO_CHANNELS),
            "-c:a", "pcm_s16le",
            tmp_path,
        ]
        _safe_log("debug",
            f"[atempo] {Path(wav_path).name} "
            f"{current_ms}ms → {target_ms}ms  filter={filter_str}"
        )
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        if Path(tmp_path).exists() and Path(tmp_path).stat().st_size > 0:
            shutil.move(tmp_path, wav_path)
            return True
        return False

    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="ignore")[-300:]
        _safe_log("error", f"[atempo] FFmpeg lỗi {Path(wav_path).name}:\n{stderr}")
        Path(tmp_path).unlink(missing_ok=True)
        return False
    except Exception as e:
        _safe_log("error", f"[atempo] thất bại {Path(wav_path).name}: {e}")
        Path(tmp_path).unlink(missing_ok=True)
        return False


def _has_rubberband_binary() -> bool:
    """
    Kiểm tra rubberband CLI binary có sẵn không.
    pyrubberband Python lib chỉ là wrapper gọi binary này.
    Dù `pip install pyrubberband` thành công, nếu binary không có
    thì pyrb.time_stretch() vẫn raise RuntimeError/CalledProcessError.

    Cài binary trên Colab/Ubuntu:
        !apt-get install -y rubberband-cli
    """
    return shutil.which("rubberband") is not None


def _speedup_rubberband(wav_path: str, target_ms: int) -> bool:
    """
    Time-stretch không đổi pitch bằng pyrubberband.
    CHỈ gọi khi đã xác nhận binary `rubberband` tồn tại.
    """
    try:
        import numpy as np
        import soundfile as sf
        import pyrubberband as pyrb

        y, sr = sf.read(wav_path, always_2d=True)
        current_ms = int(len(y) / sr * 1000)
        if current_ms <= 0 or target_ms <= 0 or target_ms >= current_ms:
            return True

        stretch_rate = current_ms / target_ms
        stretch_rate = max(0.2, min(stretch_rate, 50.0))

        _safe_log("debug",
            f"[rubberband] {Path(wav_path).name} "
            f"{current_ms}ms → {target_ms}ms (rate={stretch_rate:.3f})"
        )
        y_out = pyrb.time_stretch(y, sr, stretch_rate)
        if y_out.ndim == 1:
            y_out = np.column_stack((y_out, y_out))
        sf.write(wav_path, y_out, sr)
        return True

    except Exception as e:
        _safe_log("warning", f"[rubberband] lỗi {Path(wav_path).name}: {e} → fallback atempo")
        return _speedup_with_atempo(wav_path, target_ms)


def _speedup_ffmpeg(wav_path: str, target_ms: int) -> bool:
    """Alias backward compat → atempo."""
    return _speedup_with_atempo(wav_path, target_ms)


# Cache kiểm tra binary và library một lần duy nhất khi module load
_RUBBERBAND_AVAILABLE = _has_rubberband_binary()
_PYRUBBERBAND_INSTALLED = False

if _RUBBERBAND_AVAILABLE:
    try:
        import pyrubberband  # noqa: F401
        _PYRUBBERBAND_INSTALLED = True
        print("[SpeedRate] ✅ rubberband binary + pyrubberband → dùng pyrubberband (pitch-preserving)")
    except ImportError:
        print("[SpeedRate] ⚠️ rubberband binary có, nhưng pyrubberband lib chưa cài → fallback FFmpeg atempo")
        print("[SpeedRate] 💡 Cài đặt: pip install pyrubberband")
else:
    print("[SpeedRate] ℹ️ rubberband binary NOT found → dùng FFmpeg atempo")
    print("[SpeedRate] 💡 Cài đặt: apt-get install -y rubberband-cli && pip install pyrubberband")


def _speedup_audio(wav_path: str, target_ms: int) -> bool:
    """
    Entry point duy nhất cho time-stretch.
    - Có rubberband binary + pyrubberband lib → dùng rubberband (pitch-preserving)
    - Còn lại → FFmpeg atempo chain (luôn available)
    """
    if _RUBBERBAND_AVAILABLE and _PYRUBBERBAND_INSTALLED:
        return _speedup_rubberband(wav_path, target_ms)
    return _speedup_with_atempo(wav_path, target_ms)



# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def _make_silence(cache_folder: str, name: str, duration_ms: int) -> str:
    """Tạo file WAV im lặng có độ dài duration_ms."""
    duration_ms = max(1, int(duration_ms))
    path = str(Path(cache_folder) / f"silence_{name}.wav")
    (
        AudioSegment.silent(duration=duration_ms, frame_rate=AUDIO_SAMPLE_RATE)
        .set_channels(AUDIO_CHANNELS)
        .export(path, format="wav")
    )
    return path


def _concat_wav_files(file_list: List[str], output_path: str) -> bool:
    """
    Ghép danh sách wav bằng FFmpeg concat demuxer.
    Giống _exec_concat_audio() trong _rate.py
    """
    # Lọc file hợp lệ
    valid = [f for f in file_list if Path(f).exists() and Path(f).stat().st_size > 0]
    if not valid:
        _safe_log("error", "[Concat] Không có file nào hợp lệ")
        return False

    # Ghi file danh sách (dùng tên file tương đối để tránh vấn đề path trên Windows)
    cache_dir = Path(valid[0]).parent
    concat_txt = str(cache_dir / f"_concat_{int(time.time())}.txt")
    with open(concat_txt, "w", encoding="utf-8") as f:
        for fp in valid:
            name = Path(fp).name
            f.write(f"file '{name}'\n")

    tmp_out = str(cache_dir / "_concat_tmp.wav")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_txt,
        "-c:a", "copy",
        tmp_out,
    ]
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, timeout=300, cwd=str(cache_dir)
        )
        if Path(tmp_out).exists():
            shutil.move(tmp_out, output_path)
            Path(concat_txt).unlink(missing_ok=True)
            return True
    except Exception as e:
        _safe_log("error", f"[Concat] FFmpeg thất bại: {e}")
    finally:
        Path(concat_txt).unlink(missing_ok=True)
        Path(tmp_out).unlink(missing_ok=True)
    return False


# ─────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────
class SpeedRate:
    """
    Xây dựng audio track cuối từ các segment wav đã được TTS.
    Giống TtsSpeedRate trong pyvideotrans/task/_rate.py
    (không có video_autorate, không có remove_silent_mid)
    """

    def __init__(
        self,
        *,
        queue_tts: List[Dict[str, Any]],
        target_audio: str,
        cache_folder: str,
        voice_autorate: bool = False,
        raw_total_time: int  = 0,    # ms, tổng thời lượng SRT (end_time của sub cuối)
        max_speed_rate: float = 100.0,  # giới hạn tốc độ tăng tối đa
    ):
        self.queue_tts      = [dict(it) for it in queue_tts]  # deep copy
        self.target_audio   = target_audio
        self.cache_folder   = cache_folder
        self.voice_autorate = voice_autorate
        self.raw_total_time = raw_total_time
        self.max_speed_rate = max_speed_rate

        Path(cache_folder).mkdir(parents=True, exist_ok=True)

    # ── PUBLIC ────────────────────────────────────────────────────────
    def run(self) -> List[Dict[str, Any]]:
        """
        Chạy pipeline và ghi file output.
        Trả về queue_tts đã được cập nhật start/end_time.
        """
        if self.voice_autorate:
            _safe_log("info", "[SpeedRate] Chế độ: voice_autorate=ON")
            self._prepare_data()
            self._calculate()
            self._speedup_all()
            self._concat_aligned()
        else:
            _safe_log("info", "[SpeedRate] Chế độ: voice_autorate=OFF → ghép thẳng theo SRT timeline")
            self._run_no_rate_mode()

        return self.queue_tts

    # ── CHẾ ĐỘ KHÔNG AUTORATE ────────────────────────────────────────
    def _run_no_rate_mode(self):
        """
        Ghép audio theo đúng timeline SRT.
        - Chèn silence trước mỗi sub đúng bằng gap từ SRT
        - Nếu audio sub ngắn hơn sub duration → chèn silence đuôi
        - Nếu audio sub dài hơn → âm thanh tràn vào gap kế tiếp
          (không xử lý thêm, giữ nguyên như pyvideotrans)
        Giống _run_no_rate_change_mode() trong _rate.py phiên bản TtsSpeedRate
        """
        audio_list: List[str] = []
        total_dur = 0

        for i, it in enumerate(self.queue_tts):
            # Vị trí kết thúc audio đã ghép trước đó
            prev_end = 0 if i == 0 else self.queue_tts[i - 1].get("_end_pos", 0)
            start_ms = it["start_time"]
            sub_dur  = it["end_time"] - it["start_time"]   # độ dài chính subtitle

            # Chèn silence trước sub nếu có gap
            gap = start_ms - prev_end
            if gap > 0:
                audio_list.append(_make_silence(self.cache_folder, f"gap_{i}", gap))
                total_dur += gap

            # Thêm audio TTS
            dubb_len = 0
            wav = it.get("filename", "")
            if wav and Path(wav).exists() and Path(wav).stat().st_size > 0:
                audio_list.append(wav)
                dubb_len = len(AudioSegment.from_file(wav))
            else:
                # Không có audio → chèn silence bằng độ dài subtitle
                if sub_dur > 0:
                    sil = _make_silence(self.cache_folder, f"nosub_{i}", sub_dur)
                    audio_list.append(sil)
                    dubb_len = sub_dur

            # Nếu audio ngắn hơn subtitle duration → chèn silence đuôi
            # (giống đoạn "如果真实配音短于字幕区间，末尾添加静音" trong _rate.py)
            if dubb_len < sub_dur:
                tail = sub_dur - dubb_len
                audio_list.append(_make_silence(self.cache_folder, f"tail_{i}", tail))
                dubb_len = sub_dur

            total_dur += dubb_len
            it["_end_pos"] = total_dur

        # Chèn silence cuối nếu video dài hơn audio
        if self.raw_total_time > total_dur:
            leftover = self.raw_total_time - total_dur
            audio_list.append(_make_silence(self.cache_folder, "final_tail", leftover))

        _concat_wav_files(audio_list, self.target_audio)

    # ── CHẾ ĐỘ AUTORATE ──────────────────────────────────────────────
    def _prepare_data(self):
        """
        Tính source_duration cho mỗi sub = slot thực tế trên timeline.
        slot[i].start = slot[i-1].end  (= start_time của sub[i])
        slot[i].end   = start_time của sub[i+1]  (hoặc raw_total_time nếu là sub cuối)

        Đây chính xác là logic _prepare_data() trong _rate.py:
          start_time_source = end_time_source của sub trước
          end_time_source   = start_time của sub tiếp theo
          source_duration   = end_time_source - start_time_source
        """
        n = len(self.queue_tts)
        for i, it in enumerate(self.queue_tts):
            it["original_start"] = it["start_time"]
            it["original_end"]   = it["end_time"]

            # start_time_source
            if i == 0:
                it["start_time_source"] = it["start_time"]
            else:
                it["start_time_source"] = self.queue_tts[i - 1]["end_time_source"]

            # end_time_source = start_time của sub kế tiếp, hoặc raw_total_time
            if i < n - 1:
                next_start = self.queue_tts[i + 1]["start_time"]
                it["end_time_source"] = next_start
                it["end_time"]        = next_start   # cập nhật end_time để slot liền kề
            else:
                it["end_time_source"] = self.raw_total_time
                it["end_time"]        = self.raw_total_time

            it["source_duration"] = it["end_time_source"] - it["start_time_source"]

            # Đo thời lượng audio TTS thực tế
            wav = it.get("filename", "")
            if wav and Path(wav).exists() and Path(wav).stat().st_size > 0:
                it["dubb_time"] = len(AudioSegment.from_file(wav))
            else:
                # Tạo silence placeholder
                sil = _make_silence(
                    self.cache_folder, f"place_{i}", max(1, it["source_duration"])
                )
                it["filename"]  = sil
                it["dubb_time"] = it["source_duration"]

            _safe_log("debug",
                f"[Prepare] sub {it.get('line', i)}: "
                f"slot={it['start_time_source']}→{it['end_time_source']} "
                f"({it['source_duration']}ms)  dubb={it['dubb_time']}ms"
            )

    def _calculate(self):
        """
        Xác định những audio nào cần time-stretch.
        dubb_time > source_duration → thêm vào danh sách cần nén.
        Giống _calculate_adjustments() trong TtsSpeedRate._rate.py
        """
        self._to_speedup: List[Dict] = []

        for it in self.queue_tts:
            source_dur = it["source_duration"]
            dubb_dur   = it["dubb_time"]

            if dubb_dur > source_dur:
                ratio = dubb_dur / source_dur
                if ratio <= self.max_speed_rate:
                    audio_target = source_dur
                else:
                    # Quá xa → nén về max_speed_rate thay vì source_duration
                    audio_target = int(dubb_dur / self.max_speed_rate)

                self._to_speedup.append({
                    "filename":    it["filename"],
                    "dubb_time":   dubb_dur,
                    "target_time": audio_target,
                })
                _safe_log("debug",
                    f"[Calc] sub {it.get('line','?')}: "
                    f"dubb={dubb_dur}ms > slot={source_dur}ms "
                    f"→ nén về {audio_target}ms"
                )
            else:
                _safe_log("debug",
                    f"[Calc] sub {it.get('line','?')}: "
                    f"dubb={dubb_dur}ms ≤ slot={source_dur}ms → không cần nén"
                )

    def _speedup_all(self):
        """Nén tất cả audio cần xử lý."""
        if not self._to_speedup:
            _safe_log("info", "[SpeedRate] Không có audio nào cần time-stretch")
            return
        _safe_log("info", f"[SpeedRate] Time-stretch {len(self._to_speedup)} audio segments...")
        for d in self._to_speedup:
            ok = _speedup_audio(d["filename"], d["target_time"])
            if not ok:
                _safe_log("warning", f"[SpeedRate] Bỏ qua speedup: {d['filename']}")

    def _concat_aligned(self):
        """
        Ghép audio đã stretch theo timeline mới.
        Mỗi slot = source_duration của sub đó.
        - Chèn silence trước (original_start - start_time_source)
        - Thêm audio TTS đã stretch
        - Chèn silence đuôi nếu audio ngắn hơn slot
        - Cắt bớt nếu audio dài hơn slot (tràn biên)
        Giống _concat_audio_aligned() trong _rate.py
        """
        audio_list: List[str] = []
        current_timeline = 0

        for i, it in enumerate(self.queue_tts):
            slot_dur = it["source_duration"]
            if slot_dur <= 0:
                slot_dur = max(1, it.get("dubb_time", 1))

            slot_parts: List[str] = []
            slot_len = 0

            # 1. Silence trước (khoảng từ đầu slot đến original_start của sub)
            pre_offset = it["original_start"] - it["start_time_source"]
            if pre_offset > 0:
                sil = _make_silence(self.cache_folder, f"pre_{i}", pre_offset)
                slot_parts.append(sil)
                slot_len += pre_offset

            # 2. Audio TTS
            wav = it["filename"]
            if Path(wav).exists() and Path(wav).stat().st_size > 0:
                try:
                    seg = AudioSegment.from_file(wav)
                    if seg.channels != AUDIO_CHANNELS:
                        seg = seg.set_channels(AUDIO_CHANNELS)
                        seg.export(wav, format="wav")
                    slot_parts.append(wav)
                    slot_len += len(seg)
                except Exception as e:
                    _safe_log("error", f"[Concat-Align] Đọc audio lỗi {wav}: {e}")

            # 3. Điều chỉnh khớp với slot_dur
            if slot_len > slot_dur:
                # Tràn → merge & cắt bớt
                merged = str(Path(self.cache_folder) / f"merged_slot_{i}.wav")
                combined = AudioSegment.empty()
                for f in slot_parts:
                    combined += AudioSegment.from_file(f)
                cut = combined[:slot_dur]
                cut.export(merged, format="wav")
                audio_list.append(merged)
                _safe_log("debug", f"[Align] sub {it.get('line','?')} cắt {slot_len}→{slot_dur}ms")

            elif slot_len < slot_dur:
                # Thiếu → thêm silence đuôi
                diff = slot_dur - slot_len
                tail = _make_silence(self.cache_folder, f"stail_{i}", diff)
                audio_list.extend(slot_parts)
                audio_list.append(tail)
                _safe_log("debug", f"[Align] sub {it.get('line','?')} thêm {diff}ms silence đuôi")
            else:
                audio_list.extend(slot_parts)

            # Cập nhật timeline
            it["start_time"] = current_timeline
            it["end_time"]   = current_timeline + slot_dur
            current_timeline += slot_dur

        _concat_wav_files(audio_list, self.target_audio)