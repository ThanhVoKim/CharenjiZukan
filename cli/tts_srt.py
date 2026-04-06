#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tts_srt.py — CLI: Đọc .srt → EdgeTTS → file audio

Chuyển file subtitle .srt thành file audio với giọng EdgeTTS.
Hỗ trợ time-stretch (autorate) để audio khớp với timeline SRT.

Xem hướng dẫn chi tiết tại: docs/colab-usage.md

Ví dụ nhanh:
    uv run tts_srt.py --input video.srt --voice vi-VN-HoaiMyNeural
    uv run tts_srt.py --input video.srt --voice ja-JP-KeitaNeural --autorate
"""


import sys
import asyncio
import logging
import argparse
import shutil
import tempfile
import time
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tts.edgetts import EdgeTTSEngine     # noqa: E402
from tts.voicevox import VoicevoxTTSEngine # noqa: E402
from speed_rate import SpeedRate          # noqa: E402
from utils.logger import setup_logging, get_logger  # noqa: E402

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# SRT PARSER (import từ utils/srt_parser)
# ─────────────────────────────────────────────────────────────────────
from utils.srt_parser import parse_srt, parse_srt_file  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# LIST VOICES
# ─────────────────────────────────────────────────────────────────────
async def _list_voices_async(locale_filter: str = None):
    from edge_tts import VoicesManager
    vm = await VoicesManager.create()
    voices = vm.voices
    if locale_filter:
        voices = [v for v in voices if locale_filter.lower() in v['Locale'].lower()]
    return sorted(voices, key=lambda v: v['Locale'])


def list_voices(locale_filter: str = None):
    voices = asyncio.run(_list_voices_async(locale_filter))
    if not voices:
        print(f"Không tìm thấy giọng nào với filter: '{locale_filter}'")
        return
    print(f"\n{'Locale':<20} {'ShortName':<45} {'Gender'}")
    print("─" * 80)
    for v in voices:
        print(f"{v['Locale']:<20} {v['ShortName']:<45} {v['Gender']}")
    print(f"\nTổng: {len(voices)} giọng\n")


# ─────────────────────────────────────────────────────────────────────
# PIPELINE CHÍNH
# ─────────────────────────────────────────────────────────────────────
def run_tts(
    input_file:    str,
    output_file:   str,
    voice:         str,
    rate:          str  = "+0%",
    volume:        str  = "+0%",
    pitch:         str  = "+0Hz",
    voice_autorate:bool = False,
    proxy:         str  = None,
    cache_folder:  str  = None,
    max_concurrent:int  = 10,
    max_speed_rate:float= 100.0,
    strip_silence: bool = True,
    silence_thresh:int  = -50,
    tts_provider:  str  = "edge",
    voicevox_id:   int  = 10008,
) -> dict:
    """
    Pipeline hoàn chỉnh: SRT → queue_tts → EdgeTTS → SpeedRate → audio.
    Giống luồng dubbing/align trong pyvideotrans.
    """
    t_start = time.time()

    # 1. Đọc và parse SRT
    raw = Path(input_file).read_text(encoding="utf-8", errors="ignore")
    srt_list = parse_srt(raw)
    if not srt_list:
        raise ValueError(f"Không đọc được subtitle nào từ: {input_file}")

    total = len(srt_list)
    raw_total_time = srt_list[-1]["end_time"]  # tổng thời lượng SRT (ms)

    print(f"\n{'='*55}")
    print(f"  🎙  TTS Dubbing ({tts_provider.upper()})")
    print(f"{'='*55}")
    print(f"  Input   : {input_file}")
    print(f"  Output  : {output_file}")
    if tts_provider == "edge":
        print(f"  Voice   : {voice}")
    else:
        print(f"  Voice ID: {voicevox_id}")
    print(f"  Rate    : {rate}  Volume: {volume}  Pitch: {pitch}")
    print(f"  Autorate: {'ON' if voice_autorate else 'OFF'}")
    print(f"  Subs    : {total} dòng  ({raw_total_time/1000:.1f}s)")
    print(f"{'='*55}\n")
 
    # 2. Tạo thư mục cache
    # Mặc định: PROJ/tmp/<stem>_<timestamp>/  (cùng thư mục script)
    _tmp_created = False
    if not cache_folder:
        stem = Path(input_file).stem
        cache_folder = str(PROJECT_ROOT / "tmp" / f"{stem}_{int(time.time())}")
        _tmp_created = True
    Path(cache_folder).mkdir(parents=True, exist_ok=True)

    # 3. Build queue_tts (giống _tts() trong trans_create.py)
    #    Mỗi item: text, line, start_time, end_time, role, filename
    queue_tts = []
    for i, it in enumerate(srt_list):
        queue_tts.append({
            "text":       it["text"],
            "line":       it["line"],
            "start_time": it["start_time"],
            "end_time":   it["end_time"],
            "role":       voice,
            "filename":   str(Path(cache_folder) / f"dubb-{i}.wav"),
        })

    # 4. TTS: tạo audio từng dòng
    print(f"🔊 Phase 1/2 — Tạo audio từng dòng ({total} dòng)...")
    if tts_provider == "edge":
        engine = EdgeTTSEngine(
            queue_tts     = queue_tts,
            voice         = voice,
            rate          = rate,
            volume        = volume,
            pitch         = pitch,
            proxy         = proxy,
            max_concurrent= max_concurrent,
            strip_silence       = strip_silence,
            silence_thresh_dbfs = silence_thresh,
        )
    elif tts_provider == "voicevox":
        engine = VoicevoxTTSEngine(
            queue_tts=queue_tts,
            voice_id=voicevox_id,
            concurrent_requests=max_concurrent,
        )
    else:
        raise ValueError(f"Provider không hợp lệ: {tts_provider}")

    tts_stats = engine.run()
    print(f"   ✅ {tts_provider.upper()}: {tts_stats['ok']} OK | {tts_stats['err']} lỗi\n")

    if tts_stats["ok"] == 0:
        raise RuntimeError(f"{tts_provider.upper()} thất bại hoàn toàn — không có audio nào được tạo")

    # 5. SpeedRate: ghép và align
    print(f"🔗 Phase 2/2 — Ghép audio (autorate={'ON' if voice_autorate else 'OFF'})...")
    target_wav = str(Path(cache_folder) / "_target.wav")
    sr = SpeedRate(
        queue_tts      = queue_tts,
        target_audio   = target_wav,
        cache_folder   = cache_folder,
        voice_autorate = voice_autorate,
        raw_total_time = raw_total_time,
        max_speed_rate = max_speed_rate,
    )
    final_queue = sr.run()

    if not Path(target_wav).exists():
        raise RuntimeError(f"SpeedRate không tạo được file audio: {target_wav}")

    # 6. Convert sang định dạng output cuối (wav / mp3)
    out_ext = Path(output_file).suffix.lower()
    
    # Nếu output không có extension → tự động thêm .wav
    if not out_ext:
        output_file = output_file + ".wav"
        out_ext = ".wav"
        logger.info(f"[Output] Output không có extension → tự động thêm .wav: {output_file}")
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    if out_ext == ".wav":
        shutil.move(target_wav, output_file)
    else:
        # Convert sang mp3 hoặc format khác bằng ffmpeg
        import subprocess
        cmd = ["ffmpeg", "-y", "-i", target_wav, "-b:a", "192k", output_file]
        subprocess.run(cmd, check=True, capture_output=True)

    # 7. Dọn cache tạm
    if _tmp_created:
        shutil.rmtree(cache_folder, ignore_errors=True)

    elapsed = time.time() - t_start
    audio_dur_s = raw_total_time / 1000
    
    print(f"\n{'─'*55}")
    print(f"✅ Hoàn thành: {output_file}")
    print(f"   Subs: {total} | TTS OK: {tts_stats['ok']} | Err: {tts_stats['err']}")
    print(f"   Thời lượng SRT: {audio_dur_s:.1f}s")
    print(f"   Thời gian xử lý: {elapsed:.1f}s  (x{audio_dur_s/elapsed:.1f})")

    return {
        "total":       total,
        "tts_ok":      tts_stats["ok"],
        "tts_err":     tts_stats["err"],
        "elapsed":     round(elapsed, 1),
        "output":      output_file,
    }


# ─────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tts_srt",
        description="EdgeTTS: chuyển file .srt thành audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ nhanh:
  python tts_srt.py --input video_vi.srt --voice vi-VN-HoaiMyNeural

Với autorate (nén giọng khớp subtitle slot):
  python tts_srt.py --input video_vi.srt --voice vi-VN-HoaiMyNeural --autorate

Đầu ra mp3:
  python tts_srt.py --input video_vi.srt --voice vi-VN-HoaiMyNeural --output out.mp3

Xem danh sách giọng tiếng Việt:
  python tts_srt.py --list-voices vi
        """,
    )

    # Bắt buộc (trừ khi --list-voices)
    parser.add_argument("--input",  "-i", metavar="FILE",
                        help="File .srt đầu vào (bắt buộc trừ khi --list-voices)")
    parser.add_argument("--voice",  "-v", metavar="VOICE", default="vi-VN-HoaiMyNeural",
                        help="Tên giọng EdgeTTS, vd: vi-VN-HoaiMyNeural")

    # TTS Provider
    parser.add_argument("--tts-provider", choices=["edge", "voicevox"], default="edge",
                        help="Chọn TTS engine (mặc định: edge)")
    parser.add_argument("--voicevox-id", type=int, default=10008,
                        help="ID nhân vật Voicevox (mặc định: 10008)")

    # Tùy chọn audio
    parser.add_argument("--output", "-o", default=None, metavar="FILE",
                        help="File audio đầu ra .wav/.mp3 (mặc định: PROJ/output/<input_stem>.wav)")
    parser.add_argument("--rate",   default="+0%",   metavar="RATE",
                        help="Tốc độ giọng đọc, vd: +10%% hoặc -5%% (mặc định: +0%%)")
    parser.add_argument("--volume", default="+0%",   metavar="VOL",
                        help="Âm lượng, vd: +20%% (mặc định: +0%%)")
    parser.add_argument("--pitch",  default="+0Hz",  metavar="PITCH",
                        help="Pitch, vd: +50Hz (mặc định: +0Hz)")

    # Tùy chọn xử lý
    parser.add_argument("--autorate", action="store_true",
                        help="Bật voice_autorate: tự động nén audio khớp với slot SRT")
    parser.add_argument("--max-speed", type=float, default=100.0, metavar="X",
                        help="Giới hạn tốc độ tăng tối đa cho autorate (mặc định: 100)")
    parser.add_argument("--concurrent", type=int, default=10, metavar="N",
                        help="Số request EdgeTTS song song (mặc định: 10)")
    parser.add_argument("--proxy", default=None, metavar="URL",
                        help="Proxy URL, vd: http://127.0.0.1:7890")
    parser.add_argument("--cache", default=None, metavar="DIR",
                        help="Thư mục cache audio tạm (mặc định: PROJ/tmp/<stem>_<ts>/, tự xoá khi xong)")

    parser.add_argument(
        "--no-strip-silence",
        action="store_true",
        help="Tắt tính năng tự động cắt silence ở đuôi mỗi clip TTS (mặc định: bật)",
    )
    parser.add_argument(
        "--silence-thresh",
        type=int,
        default=-50,
        metavar="DBFS",
        help="Ngưỡng dBFS coi là silence (mặc định: -50). "
             "Giảm xuống -60 nếu bị cắt quá nhiều.",
    )

    # Tiện ích
    parser.add_argument("--list-voices", metavar="LOCALE",
                        help="Liệt kê giọng EdgeTTS, vd: --list-voices vi")
    parser.add_argument("--verbose", action="store_true",
                        help="Bật logging debug")

    return parser


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = build_parser()
    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    # ── Chế độ liệt kê giọng ─────────────────────────────────────────
    if args.list_voices is not None:
        list_voices(args.list_voices or None)
        return

    # ── Validate đầu vào ─────────────────────────────────────────────
    if not args.input:
        parser.error("--input là bắt buộc (hoặc dùng --list-voices)")
    if args.tts_provider == "edge" and not args.voice:
        parser.error("--voice là bắt buộc khi dùng EdgeTTS (hoặc dùng --list-voices)")

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"File không tồn tại: {args.input}")
    if input_path.suffix.lower() != ".srt":
        parser.error(f"File phải có đuôi .srt: {args.input}")

    # Output mặc định: PROJ/output/<stem>.wav
    if args.output:
        output_file = args.output
    else:
        out_dir = PROJECT_ROOT / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(out_dir / (input_path.stem + ".wav"))

    # ── Chạy pipeline ────────────────────────────────────────────────
    try:
        run_tts(
            input_file     = str(input_path),
            output_file    = output_file,
            voice          = args.voice,
            rate           = args.rate,
            volume         = args.volume,
            pitch          = args.pitch,
            voice_autorate = args.autorate,
            proxy          = args.proxy,
            cache_folder   = args.cache,
            max_concurrent = args.concurrent,
            max_speed_rate = args.max_speed,
            strip_silence  = not args.no_strip_silence,
            silence_thresh = args.silence_thresh,
            tts_provider   = args.tts_provider,
            voicevox_id    = args.voicevox_id,
        )
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n⚠️  Dừng bởi người dùng")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()