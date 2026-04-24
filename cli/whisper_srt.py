#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
whisper_srt.py — CLI: Transcribe video/audio → .srt dùng WhisperX (Batch Optimized)
"""

import sys
import re
import argparse
import time
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger
from utils.srt_parser import segments_to_srt
from utils.audio_utils import extract_audio_direct

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# NGÔN NGỮ CJK
# ═══════════════════════════════════════════════════════════════
_CJK_LANGS = {"zh", "ja", "th", "yue", "ko", "zh-tw", "zh-cn"}

def _is_cjk(lang: str) -> bool:
    return lang.lower().split("-")[0] in _CJK_LANGS


# ═══════════════════════════════════════════════════════════════
# PHẦN 1 — TEXT PROCESSING & SEGMENTATION
# ═══════════════════════════════════════════════════════════════

def _resegment(texts: List[Dict], language: str, max_speech_ms: int, pause_thresh_ms: int = 800) -> List[Dict]:
    use_space  = not _is_cjk(language)
    end_punc   = set(".?!。？！\n")
    comma_punc = set(",;:，；：、")

    def has_punc(text: str, punc_set: set) -> bool:
        return bool(text) and text[-1] in punc_set

    def build_text(chunk: List[str]) -> str:
        if use_space:
            s = " ".join(chunk)
            s = re.sub(r"\s+([.,?!:;])", r"\1", s)
        else:
            s = "".join(chunk)
        return s.strip()

    final = []
    for seg in texts:
        t0    = float(seg.get("start", 0)) * 1000
        t1    = float(seg.get("end",   0)) * 1000
        dur   = t1 - t0
        words = seg.get("words", [])

        if dur <= max_speech_ms or not words:
            final.append({"text": seg.get("text","").strip(), "start_time": int(t0), "end_time": int(t1)})
            continue

        chunk:          List[str]       = []
        chunk_start:    Optional[float] = None
        prev_end:       Optional[float] = None
        prev_word_text: str             = ""

        for w in words:
            wt  = w.get("word", "").strip()
            if not wt:
                continue
            ws = float(w.get("start", 0)) * 1000
            we = float(w.get("end",   0)) * 1000
            if chunk_start is None:
                chunk_start = ws

            future_dur = we - chunk_start
            split      = False

            if future_dur > max_speech_ms and chunk:
                split = True
            else:
                pause    = (ws - prev_end) if prev_end is not None else 0
                cur_dur  = (prev_end - chunk_start) if prev_end else 0
                if chunk:
                    if has_punc(prev_word_text, end_punc):
                        split = True
                    elif pause >= pause_thresh_ms:
                        split = True
                    elif pause_thresh_ms >= 400 and has_punc(prev_word_text, comma_punc) and pause >= (pause_thresh_ms * 0.375):
                        split = True
                    elif pause_thresh_ms >= 400 and cur_dur > max_speech_ms * 0.5 and pause >= (pause_thresh_ms * 0.5):
                        split = True

            if split:
                final.append({"text": build_text(chunk), "start_time": int(chunk_start), "end_time": int(prev_end)})
                chunk       = [wt]
                chunk_start = ws
            else:
                chunk.append(wt)

            prev_end       = we
            prev_word_text = wt

        if chunk:
            final.append({"text": build_text(chunk), "start_time": int(chunk_start), "end_time": int(prev_end)})

    out = []
    for i, s in enumerate(final):
        out.append({
            "line":       i + 1,
            "text":       s["text"],
            "start_time": s["start_time"],
            "end_time":   s["end_time"],
        })
    return out


def _merge_short_segments(srt_list: List[Dict], min_dur_ms: int = 1000, use_space: bool = True) -> List[Dict]:
    if not srt_list:
        return srt_list

    sep = "" if not use_space else " "

    def duration(seg: Dict) -> int:
        return seg["end_time"] - seg["start_time"]

    def merge_two(a: Dict, b: Dict) -> Dict:
        new_text = (a["text"] + sep + b["text"]).strip()
        if not use_space:
            new_text = re.sub(r"\s+", "", new_text)
        return {
            "line":       a["line"],
            "text":       new_text,
            "start_time": a["start_time"],
            "end_time":   b["end_time"],
        }

    changed = True
    segs = [dict(s) for s in srt_list]

    while changed:
        changed = False
        new_segs: List[Dict] = []
        i = 0
        while i < len(segs):
            seg = segs[i]
            if duration(seg) >= min_dur_ms:
                new_segs.append(seg)
                i += 1
                continue

            n = len(segs)
            has_prev = len(new_segs) > 0
            has_next = i < n - 1

            if not has_prev and not has_next:
                new_segs.append(seg)
            elif not has_prev:
                segs[i + 1] = merge_two(seg, segs[i + 1])
                changed = True
            elif not has_next:
                new_segs[-1] = merge_two(new_segs[-1], seg)
                changed = True
            else:
                gap_prev = seg["start_time"] - new_segs[-1]["end_time"]
                gap_next = segs[i + 1]["start_time"] - seg["end_time"]
                if gap_prev <= gap_next:
                    new_segs[-1] = merge_two(new_segs[-1], seg)
                else:
                    segs[i + 1] = merge_two(seg, segs[i + 1])
                changed = True
            i += 1
        segs = new_segs

    for idx, s in enumerate(segs):
        s["line"] = idx + 1

    return segs


def split_text_by_maxlen(text: str, maxlen: int) -> str:
    flag = [",", ".", "?", "!", ";", "，", "。", "？", "；", "！", " "]
    text = text.strip()
    if len(text) <= maxlen:
        return text

    groups = []
    for line in re.split(r"\n|\\n", text):
        line = line.strip()
        if not line:
            continue
        if len(line) <= maxlen:
            groups.append(line)
            continue
        cursor, n = 0, len(line)
        while cursor < n:
            if n - cursor <= maxlen:
                groups.append(line[cursor:])
                break
            bp = -1
            for i in range(max(cursor + maxlen - 3, 0), min(cursor + maxlen + 2, n)):
                if line[i] in flag:
                    bp = i + 1
                    break
            if bp < 0:
                bp = cursor + maxlen
            groups.append(line[cursor:bp])
            cursor = bp

    if len(groups) > 1 and len(groups[-1]) < 3:
        groups[-2] += groups[-1]
        groups.pop()

    return "\n".join(groups).strip()


def _split_by_punctuation(srt_list: List[Dict], max_chars: int = 50, min_seg_ms: int = 1000, use_space: bool = False) -> List[Dict]:
    STRONG_PUNC = set("。！？.!?…\n")
    WEAK_PUNC   = set("，,；;、")

    def interpolate_ts(seg: Dict, char_offset: int, total_chars: int) -> int:
        if total_chars <= 0:
            return seg["start_time"]
        ratio = char_offset / total_chars
        return int(seg["start_time"] + ratio * (seg["end_time"] - seg["start_time"]))

    def split_text_into_sentences(text: str) -> List[str]:
        sentences: List[str] = []
        current = ""
        for i, ch in enumerate(text):
            current += ch
            if ch in STRONG_PUNC:
                stripped = current.strip()
                if stripped:
                    sentences.append(stripped)
                current = ""
            elif ch in WEAK_PUNC and len(current.rstrip()) >= max_chars // 2:
                stripped = current.strip()
                if stripped:
                    sentences.append(stripped)
                current = ""
        remaining = current.strip()
        if remaining:
            if sentences and len(remaining) < 3:
                sentences[-1] += ("" if not use_space else " ") + remaining
            else:
                sentences.append(remaining)
        return sentences if sentences else [text]

    result:   List[Dict] = []
    line_num: int = 0

    for seg in srt_list:
        text        = seg["text"].strip()
        total_chars = len(text)
        has_strong_punc = any(ch in STRONG_PUNC for ch in text)
        if total_chars <= max_chars or not has_strong_punc:
            line_num += 1
            entry = dict(seg)
            entry["line"] = line_num
            result.append(entry)
            continue

        sentences = split_text_into_sentences(text)
        if len(sentences) == 1:
            line_num += 1
            entry = dict(seg)
            entry["line"] = line_num
            result.append(entry)
            continue

        char_cursor = 0
        for idx, sent in enumerate(sentences):
            sent_len   = len(sent)
            t_start    = interpolate_ts(seg, char_cursor, total_chars)
            char_cursor += sent_len
            t_end       = interpolate_ts(seg, char_cursor, total_chars)

            if t_end - t_start < min_seg_ms and result and idx > 0:
                prev = result[-1]
                sep  = "" if not use_space else " "
                prev["text"]     = prev["text"] + sep + sent
                prev["end_time"] = t_end
                continue

            line_num += 1
            result.append({
                "line":       line_num,
                "text":       sent,
                "start_time": t_start,
                "end_time":   t_end,
            })

    return result


def _clear_vram():
    """Giải phóng VRAM sau mỗi bước nặng."""
    try:
        import torch, gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════
# PHẦN 2 — WHISPERX BATCH PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_batch_transcribe(
    tasks:           List[Dict[str, str]],
    model_name:      str   = "large-v3",
    language:        str   = None,
    device:          str   = "cuda",
    compute_type:    str   = "float16",
    batch_size:      int   = 16,
    max_speech_ms:   int   = 6000,
    pause_thresh_ms: int   = 800,
    min_seg_ms:      int   = 1000,
    max_chars:       int   = 0,
    vad_chunk_size:  int   = 0,
    maxlen:          int   = 0,
    no_align:        bool  = False,
) -> List[Dict]:
    
    try:
        import torch
        torch.set_num_threads(1)
        import whisperx
    except ImportError:
        logger.error("❌ Lỗi: Thư viện 'whisperx' chưa được cài đặt.")
        logger.error("💡 Vui lòng cài đặt Optional Dependency bằng lệnh: pip install .[whisper]")
        sys.exit(1)

    t0 = time.time()
    
    _vad_opts: Dict = {}
    if vad_chunk_size > 0:
        _vad_opts["chunk_size"] = vad_chunk_size

    # ========================================================
    # PHASE 1: TRANSCRIBE TẤT CẢ FILE
    # ========================================================
    logger.info(f"🚀 [PHASE 1] Load Whisper model: {model_name} ({device}/{compute_type})")
    model = whisperx.load_model(
        model_name,
        device,
        compute_type = compute_type,
        language     = language.split("-")[0] if language and language != "auto" else None,
        vad_options  = _vad_opts if _vad_opts else None,
    )

    raw_results = []
    
    for idx, task in enumerate(tasks):
        inp = task["input"]
        logger.info(f"🎙️ Transcribing ({idx+1}/{len(tasks)}): {Path(inp).name}")
        
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_wav:
            try:
                # Extract audio directly using ffmpeg to save RAM
                extract_audio_direct(inp, tmp_wav.name)
                audio = whisperx.load_audio(tmp_wav.name)
                
                # Transcribe
                result = model.transcribe(audio, batch_size=batch_size)
                
                # Store raw result and keep audio for alignment
                raw_results.append({
                    "task": task,
                    "audio": audio, # Tương đối nhỏ vì là 16kHz mono NumPy array
                    "result": result
                })
            except Exception as e:
                logger.error(f"❌ Lỗi xử lý {inp}: {e}")

    # XÓA MODEL WHISPER VÀ XẢ VRAM
    logger.info("🧹 Giải phóng Model WhisperX để nhường VRAM cho Align...")
    del model
    _clear_vram()

    # ========================================================
    # PHASE 2: ALIGN TẤT CẢ KẾT QUẢ (NẾU CẦN)
    # ========================================================
    final_outputs = []
    
    if no_align:
        logger.warning("⏭️ Bỏ qua Phase 2 (Forced Alignment)")
    else:
        logger.info("🚀 [PHASE 2] Bắt đầu Forced Alignment...")
        # Lấy danh sách các ngôn ngữ cần load align model
        langs_needed = set([res["result"].get("language", language or "en") for res in raw_results])
        
        # Load align model cho từng ngôn ngữ
        align_models = {}
        for lang in langs_needed:
            try:
                logger.info(f"   Load align model cho ngôn ngữ: {lang}")
                align_model, align_meta = whisperx.load_align_model(language_code=lang, device=device)
                align_models[lang] = (align_model, align_meta)
            except Exception as e:
                logger.warning(f"   Không có align model cho {lang} ({e}). Sẽ dùng raw timestamps.")

        # Xử lý align
        for idx, item in enumerate(raw_results):
            task = item["task"]
            lang = item["result"].get("language", language or "en")
            logger.info(f"📏 Aligning ({idx+1}/{len(raw_results)}): {Path(task['input']).name}")
            
            if lang in align_models:
                align_model, align_meta = align_models[lang]
                aligned_res = whisperx.align(
                    item["result"]["segments"],
                    align_model,
                    align_meta,
                    item["audio"],
                    device,
                    return_char_alignments=False,
                )
                item["result"] = aligned_res

        # XÓA MODEL ALIGN VÀ XẢ VRAM
        logger.info("🧹 Giải phóng Model Align...")
        del align_models
        _clear_vram()

    # ========================================================
    # PHASE 3: HẬU XỬ LÝ (POST-PROCESSING) VÀ GHI FILE
    # ========================================================
    logger.info("📝 [PHASE 3] Hậu xử lý và xuất file SRT...")
    
    for item in raw_results:
        task = item["task"]
        aligned_result = item["result"]
        detected_lang = aligned_result.get("language", language or "en")
        
        texts = []
        for seg in aligned_result.get("segments", []):
            raw_words = seg.get("words", [])
            words = []
            for w in raw_words:
                word_text = w.get("word", w.get("text", "")).strip()
                if not word_text: continue
                w_start, w_end = w.get("start"), w.get("end")
                if w_start is None or w_end is None: continue
                words.append({"word": word_text, "start": float(w_start), "end": float(w_end)})

            texts.append({
                "text": seg.get("text", "").strip(),
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "words": words,
            })

        # Resegment
        srt_list = _resegment(texts, detected_lang, max_speech_ms, pause_thresh_ms=pause_thresh_ms)
        
        # Merge short
        srt_list = _merge_short_segments(srt_list, min_dur_ms=min_seg_ms, use_space=not _is_cjk(detected_lang))
        
        # Split punctuation
        mc = max_chars if max_chars > 0 else (35 if _is_cjk(detected_lang) else 80)
        srt_list = _split_by_punctuation(srt_list, max_chars=mc, min_seg_ms=min_seg_ms, use_space=not _is_cjk(detected_lang))
        
        # Split maxlen
        effective_maxlen = maxlen
        if effective_maxlen > 0:
            for seg in srt_list:
                seg["text"] = split_text_by_maxlen(seg["text"], effective_maxlen)

        # Write SRT
        out_path = Path(task["output"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        srt_content = segments_to_srt(srt_list)
        out_path.write_text(srt_content, encoding="utf-8")
        
        final_outputs.append({
            "input": task["input"],
            "output": str(out_path),
            "lines": len(srt_list)
        })
        logger.info(f"   ✅ Saved {len(srt_list)} lines -> {out_path}")

    elapsed = time.time() - t0
    logger.info(f"🎉 Hoàn thành xử lý {len(tasks)} file trong {elapsed:.1f}s")
    
    return final_outputs


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whisper_srt",
        description="Transcribe video/audio → .srt dùng WhisperX (Tối ưu Batch Processing)",
    )

    io = parser.add_argument_group("Input / Output")
    # Cho phép truyền 1 file hoặc truyền 1 file JSON chứa danh sách task
    io.add_argument("--input",  "-i", default=None, metavar="FILE", help="Đường dẫn 1 file video/audio đầu vào")
    io.add_argument("--output", "-o", default=None, metavar="FILE", help="Đường dẫn file .srt đầu ra (dùng cùng --input)")
    io.add_argument("--task-file", "-t", default=None, metavar="JSON_FILE", help="File JSON chứa danh sách [{'input': '...', 'output': '...'}]")

    mdl = parser.add_argument_group("Model")
    mdl.add_argument("--model", "-m", default="large-v3", help="Model Whisper (mặc định: large-v3)")
    mdl.add_argument("--lang",  "-l", default=None, help="Mã ngôn ngữ (vi/en/ja...). Mặc định: auto-detect")

    dev = parser.add_argument_group("Device")
    dev.add_argument("--device", "-d", default="cuda", choices=["cuda", "cpu", "auto"])
    dev.add_argument("--compute-type", default="float16", choices=["float16", "float32", "int8", "int8_float16"])
    dev.add_argument("--batch-size", type=int, default=16, help="Batch size cho transcribe (mặc định: 16)")

    seg = parser.add_argument_group("Segmentation")
    seg.add_argument("--vad-chunk", type=int, default=0, help="VAD chunk_size (0=mặc định 30s)")
    seg.add_argument("--max-speech-ms", type=int, default=6000, help="Cắt câu dài hơn N ms (mặc định: 6000)")
    seg.add_argument("--pause-thresh", type=int, default=800, help="Khoảng lặng tối thiểu để cắt câu (mặc định: 800ms)")
    seg.add_argument("--min-seg-ms", type=int, default=1000, help="Gộp câu ngắn hơn N ms (mặc định: 1000)")
    seg.add_argument("--max-chars", type=int, default=0, help="Tách câu theo độ dài ký tự (0=auto)")
    seg.add_argument("--maxlen", type=int, default=0, help="Ký tự tối đa/dòng (mặc định: 0 = KHÔNG ngắt dòng)")
    seg.add_argument("--no-align", action="store_true", help="Bỏ qua bước forced alignment")

    misc = parser.add_argument_group("Misc")
    misc.add_argument("--verbose", action="store_true", help="Bật logging DEBUG")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.verbose:
        logger.setLevel("DEBUG")
    else:
        logger.setLevel("INFO")

    tasks = []
    
    if args.task_file:
        try:
            with open(args.task_file, "r", encoding="utf-8") as f:
                tasks = json.load(f)
            logger.info(f"Loaded {len(tasks)} tasks từ {args.task_file}")
        except Exception as e:
            logger.error(f"Lỗi đọc task file JSON: {e}")
            sys.exit(1)
    elif args.input:
        inp = Path(args.input)
        if not inp.exists():
            logger.error(f"File input không tồn tại: {inp}")
            sys.exit(1)
        out = args.output or str(inp.parent / f"{inp.stem}.srt")
        tasks.append({"input": str(inp), "output": str(out)})
    else:
        parser.error("Phải cung cấp --input hoặc --task-file")

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    compute_type = args.compute_type
    if device == "cpu" and compute_type == "float16":
        logger.warning("CPU không hỗ trợ float16 → tự chuyển sang int8")
        compute_type = "int8"

    try:
        run_batch_transcribe(
            tasks         = tasks,
            model_name    = args.model,
            language      = args.lang,
            device        = device,
            compute_type  = compute_type,
            batch_size    = args.batch_size,
            max_speech_ms = args.max_speech_ms,
            pause_thresh_ms = args.pause_thresh,
            min_seg_ms    = args.min_seg_ms,
            max_chars     = args.max_chars,
            vad_chunk_size= args.vad_chunk,
            maxlen        = args.maxlen,
            no_align      = args.no_align,
        )
        sys.exit(0)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Dừng bởi người dùng")
        sys.exit(1)


if __name__ == "__main__":
    main()
