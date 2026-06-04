#!/usr/bin/env python3
"""
diag_mouth_onset.py — Chẩn đoán desync miệng/audio.

Mục tiêu: xác định vì sao mouthEvents mở miệng trễ ~1s so với audio nghe được,
xảy ra ở CẢ EdgeTTS lẫn Voicevox. Script tái dùng ĐÚNG các hàm của pipeline để
kết quả mang tính authoritative:
  - analyze_tts_amplitude()  → mouthEvents thật mà pipeline sinh ra
  - _read_wav_rms(), _rms_to_db() → RMS/dB per-frame như pipeline thấy

Đồng thời so sánh với detect_nonsilent của pydub (góc nhìn của strip_silence)
để thấy khoảng cách giữa "ngưỡng strip giữ lại (-50dBFS)" và "ngưỡng mouth mở
miệng (-40dB)".

Cách dùng (trên Colab, trong thư mục repo):

  # A) Phân tích các clip thật đã có sẵn (vì bạn chạy với --keep-tmp):
  uv run python diag_mouth_onset.py --glob "/content/CharenjiZukan/tmp/*/tts_clips/dubb-*.wav"

  # hoặc chỉ 1 file:
  uv run python diag_mouth_onset.py --wav "/content/CharenjiZukan/tmp/sync_video_tmp_xxx/tts_clips/dubb-2.wav"

  # B) Sinh clip EdgeTTS MỚI giống pipeline rồi phân tích (repro sạch):
  uv run python diag_mouth_onset.py --edge

In ra cho mỗi clip:
  - duration, số frame, fps
  - onset (frame đầu tiên có tiếng) tại các ngưỡng -40/-45/-50/-55/-60 dB
  - dB envelope ~90 frame đầu (để thấy rõ phần lead-in)
  - mouthEvents mà analyze_tts_amplitude() sinh ra (chính cái nhúng vào manifest)
  - onset theo pydub detect_nonsilent ở -40 và -50 dBFS (góc nhìn strip)
"""
from __future__ import annotations

import argparse
import glob as _glob
import sys
from pathlib import Path

FPS = 30.0  # mouth track fps mặc định (trùng compute_track_frame_index default)


def _import_pipeline():
    """Import các hàm thật của pipeline. Repo phải đã được cài (uv run)."""
    try:
        from sync_engine.tuber_mouth_events import (  # noqa: WPS433
            analyze_tts_amplitude,
            _read_wav_rms,
            _rms_to_db,
        )
    except Exception as exc:  # pragma: no cover - chỉ chạy thủ công
        print(f"[FATAL] Không import được sync_engine.tuber_mouth_events: {exc}")
        print("        Hãy chạy bằng `uv run python diag_mouth_onset.py ...` trong repo.")
        sys.exit(1)
    return analyze_tts_amplitude, _read_wav_rms, _rms_to_db


def _pydub_onset(wav: Path):
    """Onset theo pydub.detect_nonsilent (góc nhìn của strip_silence)."""
    try:
        from pydub import AudioSegment, silence
    except Exception:
        return None
    seg = AudioSegment.from_file(str(wav))
    out = {}
    for thr in (-40, -50):
        ns = silence.detect_nonsilent(seg, min_silence_len=100, silence_thresh=thr)
        out[thr] = (ns[0][0] if ns else None)  # ms của đoạn non-silent đầu tiên
    return len(seg), out


def analyze_one(wav: Path, fns) -> None:
    analyze_tts_amplitude, _read_wav_rms, _rms_to_db = fns

    rms = _read_wav_rms(wav, FPS)
    dbs = [(_rms_to_db(r) if r > 0 else -99.0) for r in rms]

    print("=" * 72)
    print(f"FILE: {wav}")
    print(f"  frames={len(rms)}  (~{len(rms) / FPS:.2f}s @ {FPS:.0f}fps)")

    # Onset (frame đầu tiên có dB >= ngưỡng) — chính cái quyết định mouth mở.
    for thr in (-40, -45, -50, -55, -60):
        onset = next((i for i, d in enumerate(dbs) if d >= thr), None)
        secs = (onset / FPS) if onset is not None else None
        print(f"  onset@{thr:>3}dB = frame {onset}"
              + (f"  (~{secs:.2f}s vào clip)" if onset is not None else "  (không có)"))

    # dB envelope: thấy rõ lead-in có thật dưới -40 hay không.
    head = [round(d) for d in dbs[:90]]
    print(f"  dB[0:90] = {head}")

    # mouthEvents pipeline thật sinh ra (local frame, chưa cộng startFrame).
    events = analyze_tts_amplitude(wav, FPS)
    print(f"  analyze_tts_amplitude() events (local) = {events}")

    # So sánh với góc nhìn của strip_silence (pydub).
    pd = _pydub_onset(wav)
    if pd:
        dur_ms, onsets = pd
        print(f"  pydub dur={dur_ms}ms  nonsilent_onset@-40dBFS={onsets[-40]}ms"
              f"  @-50dBFS={onsets[-50]}ms")
    print()


SAMPLE_LINES = [
    "こんにちは。今日はとてもいい天気ですね。",
    "これは口の同期をテストするための音声です。",
    "では、始めましょう。よろしくお願いします。",
]


def gen_edge(out_dir: Path) -> list[Path]:
    """Sinh clip EdgeTTS giống pipeline (strip_silence=True, -50dBFS)."""
    from tts.edgetts import EdgeTTSEngine

    out_dir.mkdir(parents=True, exist_ok=True)
    queue = [
        {"text": t, "filename": str(out_dir / f"diag-{i}.wav")}
        for i, t in enumerate(SAMPLE_LINES)
    ]
    eng = EdgeTTSEngine(
        queue_tts=queue,
        voice="ja-JP-KeitaNeural",
        strip_silence=True,          # giống pipeline
        silence_thresh_dbfs=-50,     # giống pipeline
        min_silence_len_ms=300,      # giống sync_video.py
    )
    print("[EdgeTTS] generating sample clips ...")
    print("[EdgeTTS] run() ->", eng.run())
    return [Path(q["filename"]) for q in queue if Path(q["filename"]).exists()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Chẩn đoán onset miệng vs audio.")
    ap.add_argument("--wav", help="1 file WAV cụ thể để phân tích")
    ap.add_argument("--glob", help="Glob pattern các WAV (vd tmp/*/tts_clips/dubb-*.wav)")
    ap.add_argument("--edge", action="store_true",
                    help="Sinh clip EdgeTTS mới rồi phân tích (repro sạch)")
    ap.add_argument("--edge-out", default="/content/_mouth_diag",
                    help="Thư mục output cho clip EdgeTTS sinh ra")
    args = ap.parse_args()

    fns = _import_pipeline()
    targets: list[Path] = []

    if args.edge:
        targets += gen_edge(Path(args.edge_out))
    if args.wav:
        targets.append(Path(args.wav))
    if args.glob:
        targets += [Path(p) for p in sorted(_glob.glob(args.glob))]

    if not targets:
        print("Chưa chọn input. Dùng --edge, --wav <path>, hoặc --glob <pattern>.")
        print("Ví dụ: uv run python diag_mouth_onset.py --edge")
        sys.exit(2)

    missing = [t for t in targets if not t.exists()]
    for m in missing:
        print(f"[WARN] không tồn tại: {m}")
    targets = [t for t in targets if t.exists()]

    print(f"\nĐang phân tích {len(targets)} clip @ fps={FPS:.0f}\n")
    for wav in targets:
        try:
            analyze_one(wav, fns)
        except Exception as exc:  # pragma: no cover
            print(f"[ERR] {wav}: {exc}\n")


if __name__ == "__main__":
    main()
