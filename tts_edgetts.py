# -*- coding: utf-8 -*-
"""
tts_edgetts.py — EdgeTTS per-line dubbing engine
Trung thành với pyvideotrans/tts/_edgetts.py

Flow:
  queue_tts (list[dict]) 
      → async _create_audio_with_retry()   [semaphore, retry, save → .mp3]
      → ThreadPoolExecutor convert_to_wav() [mp3 → wav 48k stereo]
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any

import aiohttp
from edge_tts import Communicate
from edge_tts.exceptions import NoAudioReceived
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

logger = logging.getLogger("srt_translator")

# ── giới hạn rate-limit (giống _edgetts.py) ──────────────────────────
MAX_CONCURRENT_TASKS = 10   # semaphore: tối đa bao nhiêu request song song
RETRY_NUMS           = 3    # số lần retry mỗi dòng
RETRY_DELAY          = 5    # giây chờ giữa các lần retry
SAVE_TIMEOUT         = 30   # giây timeout mỗi request
AUDIO_SAMPLE_RATE    = 48000
AUDIO_CHANNELS       = 2


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def _normalize_rate(rate_str: str) -> str:
    """
    Chuyển rate +10 / -10 / +10% / -10% → EdgeTTS format "+10%".
    Giống logic trong _dubbing._tts()
    """
    raw = str(rate_str).replace('%', '').strip()
    try:
        v = int(raw)
        return f"+{v}%" if v >= 0 else f"{v}%"
    except ValueError:
        return "+0%"


def convert_to_wav(mp3_path: str, wav_path: str) -> bool:
    """
    Convert .mp3 → .wav 48kHz stereo.
    Giống BaseTTS.convert_to_wav() + _edgetts._exec() convert block.
    """
    try:
        seg = AudioSegment.from_file(mp3_path, format="mp3")
        if seg.frame_rate != AUDIO_SAMPLE_RATE:
            seg = seg.set_frame_rate(AUDIO_SAMPLE_RATE)
        if seg.channels != AUDIO_CHANNELS:
            seg = seg.set_channels(AUDIO_CHANNELS)
        seg.export(wav_path, format="wav")
        return True
    except Exception as e:
        logger.error(f"[EdgeTTS] convert_to_wav thất bại: {mp3_path} → {e}")
        return False


def strip_audio_silence(
    wav_path: str,
    silence_thresh_dbfs: int = -50,
    min_silence_len_ms: int = 100,
    keep_padding_ms: int = 30,
) -> int:
    try:
        seg = AudioSegment.from_file(wav_path, format="wav")
        original_len = len(seg)

        non_silent = detect_nonsilent(
            seg,
            min_silence_len=min_silence_len_ms,
            silence_thresh=silence_thresh_dbfs,
        )

        if not non_silent:
            return 0  # Toàn silence → giữ nguyên

        # ✅ Strip CẢ HAI ĐẦU
        start_ms = max(0, non_silent[0][0] - keep_padding_ms)
        end_ms   = min(original_len, non_silent[-1][1] + keep_padding_ms)

        trimmed = seg[start_ms:end_ms]
        trimmed.export(wav_path, format="wav")

        trimmed_ms = original_len - len(trimmed)
        return trimmed_ms

    except Exception as e:
        logger.warning(f"[StripSilence] Bỏ qua {Path(wav_path).name}: {e}")
        return 0


# ─────────────────────────────────────────────────────────────────────
# ASYNC ENGINE
# ─────────────────────────────────────────────────────────────────────
class EdgeTTSEngine:
    """
    Async engine tạo audio từng dòng subtitle.
    Giống EdgeTTS class trong pyvideotrans/tts/_edgetts.py nhưng
    không phụ thuộc vào GUI/QThread, dùng được trong CLI.
    """

    def __init__(
        self,
        queue_tts: List[Dict[str, Any]],
        voice: str,
        rate: str    = "+0%",
        volume: str  = "+0%",
        pitch: str   = "+0Hz",
        proxy: str   = None,
        max_concurrent: int = MAX_CONCURRENT_TASKS,
        # ── Tham số strip silence ─────────────────────────────────
        strip_silence: bool = True,          # Bật mặc định
        silence_thresh_dbfs: int = -50,      # Ngưỡng silence
        min_silence_len_ms: int = 100,       # Silence tối thiểu để detect
        keep_padding_ms: int = 30,           # Padding giữ lại ở viền
    ):
        self.queue_tts      = queue_tts
        self.voice          = voice
        self.rate           = _normalize_rate(rate)
        self.volume         = volume
        self.pitch          = pitch
        self.proxy          = proxy or None
        self.max_concurrent = max_concurrent
        
        self.strip_silence       = strip_silence
        self.silence_thresh_dbfs = silence_thresh_dbfs
        self.min_silence_len_ms  = min_silence_len_ms
        self.keep_padding_ms     = keep_padding_ms

        self._stop_event  = asyncio.Event()
        self._lock        = asyncio.Lock()
        self._done_count  = 0
        self.errors: List[str] = []

    async def _increment(self):
        async with self._lock:
            self._done_count += 1

    async def _create_audio_with_retry(
        self,
        item: Dict[str, Any],
        index: int,
        total: int,
        semaphore: asyncio.Semaphore,
    ):
        """
        Tạo audio cho 1 dòng subtitle.
        Giống _create_audio_with_retry() trong _edgetts.py
        """
        task_id = f"[{index + 1}/{total}]"

        # Bỏ qua nếu text rỗng hoặc file đã tồn tại
        if not item.get('text', '').strip():
            logger.debug(f"{task_id} text rỗng, bỏ qua")
            await self._increment()
            return

        wav_path = item['filename']
        if Path(wav_path).exists() and Path(wav_path).stat().st_size > 0:
            logger.debug(f"{task_id} file đã tồn tại, bỏ qua")
            await self._increment()
            return

        try:
            async with semaphore:
                if self._stop_event.is_set():
                    return

                proxy_to_use = self.proxy
                for attempt in range(RETRY_NUMS + 1):
                    if self._stop_event.is_set():
                        return
                    try:
                        mp3_path = wav_path + ".mp3"
                        communicate = Communicate(
                            text    = item['text'],
                            voice   = item.get('role', self.voice),
                            rate    = self.rate,
                            volume  = self.volume,
                            pitch   = self.pitch,
                            proxy   = proxy_to_use,
                            connect_timeout = 5,
                        )
                        # Timeout cứng tránh WebSocket treo
                        await asyncio.wait_for(
                            communicate.save(mp3_path),
                            timeout=SAVE_TIMEOUT,
                        )
                        logger.debug(f"{task_id} ✓ saved {Path(mp3_path).name}")
                        return  # thành công, thoát

                    except asyncio.TimeoutError:
                        if attempt < RETRY_NUMS:
                            logger.warning(f"{task_id} timeout lần {attempt + 1}, retry sau {RETRY_DELAY}s")
                            await asyncio.sleep(RETRY_DELAY)
                        else:
                            msg = f"{task_id} timeout sau {RETRY_NUMS} lần retry"
                            logger.error(msg)
                            self.errors.append(msg)

                    except (NoAudioReceived, aiohttp.ClientError) as e:
                        # Thử lại không dùng proxy
                        if proxy_to_use:
                            logger.warning(f"{task_id} lỗi proxy ({e}), thử không dùng proxy")
                            proxy_to_use = None
                        elif attempt < RETRY_NUMS:
                            logger.warning(f"{task_id} lỗi {e}, retry sau {RETRY_DELAY}s")
                            await asyncio.sleep(RETRY_DELAY)
                        else:
                            msg = f"{task_id} thất bại: {e}"
                            logger.error(msg)
                            self.errors.append(msg)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"{task_id} lỗi nghiêm trọng: {e}")
            self.errors.append(str(e))
        finally:
            await self._increment()

    async def _run_async(self) -> None:
        """Chạy toàn bộ queue song song với semaphore."""
        if not self.queue_tts:
            return

        total = len(self.queue_tts)
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Gán voice cho từng item (giống _exec() trong _edgetts.py)
        for item in self.queue_tts:
            if 'role' not in item or not item['role']:
                item['role'] = self.voice

        worker_tasks = [
            asyncio.create_task(
                self._create_audio_with_retry(item, i, total, semaphore)
            )
            for i, item in enumerate(self.queue_tts)
        ]

        # Timeout tổng: total * SAVE_TIMEOUT * 2
        total_timeout = total * SAVE_TIMEOUT * 2
        try:
            await asyncio.wait_for(
                asyncio.gather(*worker_tasks, return_exceptions=True),
                timeout=total_timeout,
            )
        except asyncio.TimeoutError:
            logger.error("[EdgeTTS] Tổng timeout! Huỷ tất cả tasks.")
            self._stop_event.set()
            for t in worker_tasks:
                t.cancel()
            await asyncio.gather(*worker_tasks, return_exceptions=True)

    def run(self) -> Dict[str, int]:
        """
        Entry point đồng bộ.
        1. Async tạo .mp3 song song
        2. ThreadPool convert .mp3 → .wav
        Trả về {"ok": N, "err": N}
        """
        total = len(self.queue_tts)
        logger.info(f"[EdgeTTS] Bắt đầu {total} dòng | voice={self.voice} rate={self.rate}")

        # ── Phase 1: async tạo mp3 ───────────────────────────────────
        asyncio.run(self._run_async())

        # ── Phase 2: convert mp3 → wav (thread pool) ─────────────────
        ok_count  = 0
        err_count = 0
        to_convert = []
        for item in self.queue_tts:
            mp3_path = item['filename'] + ".mp3"
            if Path(mp3_path).exists() and Path(mp3_path).stat().st_size > 0:
                to_convert.append((mp3_path, item['filename']))
            else:
                err_count += 1

        if to_convert:
            workers = min(4, len(to_convert), os.cpu_count() or 1)
            logger.info(f"[EdgeTTS] Convert {len(to_convert)} mp3→wav với {workers} threads")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(convert_to_wav, mp3, wav) for mp3, wav in to_convert]
                for fut in futures:
                    if fut.result():
                        ok_count += 1
                    else:
                        err_count += 1
            # Dọn mp3 tạm
            for mp3, _ in to_convert:
                try:
                    Path(mp3).unlink(missing_ok=True)
                except Exception:
                    pass

        # ── Phase 2.5: strip silence từ mỗi wav ─────────────────────
        if self.strip_silence and ok_count > 0:
            wav_paths = [wav for _, wav in to_convert if Path(wav).exists()]
            total_trimmed_ms = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(strip_audio_silence, wav,
                                self.silence_thresh_dbfs,
                                self.min_silence_len_ms,
                                self.keep_padding_ms)
                    for wav in wav_paths
                ]
                for fut in futures:
                    total_trimmed_ms += fut.result()
            logger.info(
                f"[StripSilence] Đã cắt tổng {total_trimmed_ms}ms "
                f"({total_trimmed_ms/1000:.1f}s) silence từ {len(wav_paths)} clips"
            )

        logger.info(f"[EdgeTTS] Xong: {ok_count} thành công, {err_count} lỗi")
        return {"ok": ok_count, "err": err_count}
