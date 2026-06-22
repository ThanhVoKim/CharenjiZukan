# -*- coding: utf-8 -*-
"""
Fine-tune Qwen3-TTS -> giọng tiếng Đức KARLSSON (HUI Audio Corpus German).

Script ĐỘC LẬP, chạy trên Google Colab (GPU). Mục tiêu: full SFT
`Qwen/Qwen3-TTS-12Hz-1.7B-Base` thành giọng tiếng Đức chất lượng cao dựa trên giọng
Karlsson của HUI Audio Corpus German (https://opendata.iisys.de/dataset/hui-audio-corpus-german/).

File này nằm trong repo CharenjiZukan chỉ để TIỆN QUẢN LÝ — nội dung KHÔNG import code dự án,
chỉ gọi script chính thức trong QwenLM/Qwen3-TTS/finetuning. Kết quả (checkpoint) sẽ được mang
về dự án ở bước sau (ngoài phạm vi file này).

NGUYÊN TẮC CHẤT LƯỢNG:
  - Dùng script CHÍNH THỨC `finetuning/` (prepare_data.py + sft_12hz.py), KHÔNG wrapper.
  - Nhãn = transcript ground-truth của HUI (người đọc). TUYỆT ĐỐI không ASR phiên âm lại.
  - Dataset `clean`. Dùng CHUNG 1 ref_audio cho mọi mẫu (khuyến nghị chính thức).
  - Chọn checkpoint bằng WER + speaker-similarity + nghe thử, KHÔNG chỉ theo loss.
  - Mọi state ghi ra Google Drive để sống sót qua reset session.

CÁCH CHẠY TRÊN COLAB:
  Mỗi khối `# %% [N]` là một "cell" — paste/chạy lần lượt trên Colab (hoặc mở bằng
  Jupyter/VSCode "Run Cell"). Các dòng bắt đầu bằng `!` / `%%bash` là lệnh shell Colab.

LƯU Ý: Tên cờ CLI của prepare_data.py / sft_12hz.py có thể đổi theo phiên bản.
       Cell 2 in `--help` của cả hai — ĐỐI CHIẾU trước khi chạy Cell 5/6 và chỉnh nếu cần.
"""

# %% [markdown]
# # 🎙️ Fine-tune Qwen3-TTS → Giọng Đức Karlsson — Hướng dẫn
#
# Full SFT `Qwen3-TTS-12Hz-1.7B-Base` thành giọng Đức (HUI Karlsson, clean). Mỗi khối
# `# %% [N]` là **một cell** — chạy lần lượt trên Colab GPU.
#
# ## 🗺️ Bản đồ các cell
#
# | Cell | Việc | Nặng? | Cache trên Drive? |
# |---|---|---|---|
# | **1** | Kiểm GPU + mount Drive + định nghĩa path + HF cache | nhẹ | — |
# | **2** | Cài package + clone repo Qwen3-TTS + in `--help` | ~vài phút | — |
# | **3.0** | EXPLORE (chạy **1 lần đời đầu**): tải zip → Drive, giải nén, khảo sát dataset | ~tải 7GB | zip backup |
# | **3** | Đảm bảo wav local sẵn (giải nén zip backup vào `/content`) | tuỳ | đọc zip Drive |
# | **4** | Dựng manifest từ transcript HUI + lọc + chọn ref + tách val | ~vài phút | `*.jsonl`, `ref_meta.json` |
# | **5** | Trích audio codes (`prepare_data.py`) | GPU, lâu | `train_codes.jsonl` |
# | **6** | Train full SFT → checkpoint mỗi epoch ra Drive | GPU, **lâu nhất** | `checkpoint-epoch-N` |
# | **7** | Eval WER + speaker-sim + nghe thử, chọn checkpoint | GPU | `eval_out/` |
# | **8** | Export checkpoint tốt nhất ra thư mục Drive cố định | nhẹ | `EXPORT_DIR` |
#
# Cell **3, 4, 5** đều **idempotent**: tự skip phần nặng nếu Drive đã có kết quả → chạy lại an toàn.
# (Ngoại lệ: trims Cell 4 nằm ở `/content` — xem ghi chú ⚠️ phần "Restart" bên dưới.)
#
# ## 💾 Drive (bền) vs Local `/content` (mất khi reset)
#
# - **Drive giữ:** zip backup, manifest `.jsonl`, `ref_meta.json`, checkpoint, HF cache, export.
# - **Local `/content` giữ:** repo clone (+2 patch), wav giải nén, **`Karlsson_trimmed` (bản trim Cell 4)**, `eval_out`, `ckpt_eval`. **Mất khi restart** → tạo lại từ Cell 2/3 (trims: xem ghi chú ⚠️ "Restart").
#
# ## 🔄 Restart session sau này thì chạy lại cell nào?
#
# **Cell 1 + 2 LUÔN phải chạy lại** (dựng path + cài package + clone repo). Sau đó theo mục đích:
#
# | Mục đích | Cell cần chạy |
# |---|---|
# | **Chỉ dùng/nghe model đã export** | 1 → 2 → load thẳng từ `EXPORT_DIR` (bỏ 3–8) |
# | **Eval / nghe lại checkpoint** | 1 → 2 → 3 → 4 → **7** |
# | **Train thêm epoch (resume)** | 1 → 2 → 3 → 5 → **6** với `--init_model_path` = `checkpoint-epoch-<cuối>` |
# | **Mất `/content` khi train CHƯA xong** (đổi GPU/reset) | **1 → 2 → 3 → 4** (Cell 4 tự tái tạo trims từ raw) → **5 → 6** |
#
# > **Resume train** chỉ ở **mức epoch**: script lưu checkpoint mỗi epoch nhưng KHÔNG auto-resume
# > giữa epoch. Đứt giữa chừng → trỏ `--init_model_path` vào epoch gần nhất, giảm `num_epochs` còn lại.
#
# > **⚠️ Trims (silence-trim ở Cell 4) là ephemeral nhưng TỰ TÁI TẠO — KHÔNG lưu wav lên Drive.**
# > Bản wav đã trim nằm ở **local `/content`**; manifest trên Drive trỏ vào chúng. Sau khi mất
# > `/content` (đổi GPU/reset), **Cell 4 tự tái tạo trims đang thiếu TỪ RAW** (Cell 3 giải nén lại) —
# > `trim_wav()` deterministic nên bản mới == bản cũ. Chỉ tái tạo cái cần: **codes đã xong → chỉ
# > `ref_audio`** (1 file, nhanh, cho Cell 6/7); **codes chưa xong → cả train trims** (cho Cell 5).
# > **KHÔNG cần xoá file Drive, KHÔNG build lại manifest.** Điều kiện duy nhất: đã chạy **Cell 3** để có raw.
#
# ### 🔁 Đổi GPU giữa chừng (vd L4 OOM → A100) — không xoá file, không wav trên Drive
#
# | Tình huống | Chạy | Cell 4 làm gì |
# |---|---|---|
# | **OOM ở Cell 6** (codes đã xong) | `1 → 2 → 3 → 4 → 5 (skip) → 6` | tái tạo **chỉ `ref_audio`** (1 file, nhanh) |
# | **OOM ở Cell 5** (codes chưa xong) | `1 → 2 → 3 → 4 → 5 → 6` | tái tạo **toàn bộ train trims** (~10') |
#
# ## 📂 Output nằm ở đâu?
#
# Gốc Drive: `WORK_DIR = /content/drive/MyDrive/qwen3tts_karlsson_de/`
#
# | Loại output | Đường dẫn | Sống sót reset? |
# |---|---|---|
# | Checkpoint train (mỗi epoch) | `WORK_DIR/checkpoints/checkpoint-epoch-N/` | ✅ Drive |
# | Model export (Cell 8, bản tốt nhất) | `WORK_DIR/export/` (`EXPORT_DIR`) | ✅ Drive |
# | Audio eval (Cell 7) | local `/content/eval_out/` **→ copy** `WORK_DIR/eval_out/<tag>_NN.wav` | ✅ bản Drive |
# | Audio "nghe nhanh" (snippet `EP=…`) | `/content/try_epN_*.wav` | ❌ chỉ local |
# | Manifest + ref_meta | `WORK_DIR/data/*.jsonl`, `ref_meta.json` | ✅ Drive |
#
# > `generate_custom_voice(...)` / `generate_voice_clone(...)` **không tự ghi file** — chúng trả
# > `(wavs, sr)` vào RAM. File chỉ sinh ra ở nơi BẠN gọi `sf.write(...)`. **Luôn ghi ra `/content`
# > rồi copy sang Drive** (FUSE không hỗ trợ seek-on-write → lỗi *"Format not recognised"* nếu ghi thẳng).
#
# ## ⚠️ Điều dễ vấp (đã xử lý sẵn trong code)
#
# - `sft_12hz.py` **chỉ nhận 7 cờ**: `init_model_path, output_model_path, train_jsonl, batch_size,
#   lr, num_epochs, speaker_name`. `grad_accum=4`, `flash_attention_2`, `bf16` bị **hardcode** →
#   batch hiệu dụng = `batch_size × 4`. Cell 6 tự bỏ các cờ thừa.
# - Cell 6 tự: `snapshot_download` model về **local path** (script `copytree` cần path thật, không nhận
#   repo-id); cài `flash-attn`; vá `dataset.py` (resample ref-audio về **24kHz**) + `sft_12hz.py`
#   (tensorboard cần `project_dir`). Các patch nằm ở `/content` → **mất khi reset**, Cell 6 tự áp lại.
# - **Không `sf.write` thẳng lên Drive** (FUSE không hỗ trợ seek-on-write của libsndfile → lỗi
#   *"Format not recognised"*). Luôn ghi WAV ra `/content` rồi copy sang Drive.
# - Qwen3-TTS **không có tham số `batch_size`** khi sinh audio — batch bằng cách truyền `text=[list]`;
#   cả 2 hàm trả `(List[np.ndarray], int)`.
# - Eval tốn giờ chủ yếu do **load model từ Drive**, không phải sinh audio. Cell 7 copy checkpoint
#   ra local trước khi load + cho `EVAL_EPOCHS` để giới hạn số model.

# %% [1] Kiểm tra GPU & mount Drive --------------------------------------------
# @title [1] Kiểm tra GPU & mount Drive
import os
import subprocess
import textwrap
from pathlib import Path

# --- GPU ---
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout
      or "⚠️ Không thấy GPU!")

try:
    import torch
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM ≈ {vram_gb:.1f} GB")
        if vram_gb < 38:
            print(textwrap.dedent("""\
                ⚠️  VRAM < ~40GB. Full SFT 1.7B có thể OOM.
                    → Cell 6 sẽ giảm batch_size + tăng grad-accum tương ứng.
                    → Nếu vẫn OOM: cân nhắc model 0.6B-Base hoặc Colab Pro+ (A100 40GB)."""))
    else:
        print("⚠️ torch không thấy CUDA — chọn Runtime > Change runtime type > GPU.")
except ImportError:
    print("torch chưa cài — sẽ cài ở Cell 2 (Colab thường đã có sẵn torch).")

# --- Mount Drive (chỉ chạy trên Colab) ---
from google.colab import drive  # noqa: E402
drive.mount("/content/drive")


os.environ['HF_TOKEN'] = userdata.get('hf_token')
# --- Đường dẫn: TÁCH backup (Drive, bền) vs working (local Colab, nhanh) ---
# Lý do: prepare_data/train đọc hàng nghìn wav nhỏ. Đọc qua Drive FUSE rất chậm,
# dễ treo → xử lý trên LOCAL /content (khớp workflow trong CLAUDE.md). Drive chỉ
# giữ những thứ cần sống sót reset: zip backup, checkpoint, HF cache, export.
WORK_DIR   = Path("/content/drive/MyDrive/qwen3tts_karlsson_de")  # gốc dự án FT (Drive)
DATA_DIR   = WORK_DIR / "data"         # backup zip + manifest (Drive, bền)
CKPT_DIR   = WORK_DIR / "checkpoints"  # output sft_12hz.py (Drive, bền)
HF_CACHE   = WORK_DIR / "hf_cache"     # cache model/tokenizer → không tải lại (Drive)
EXPORT_DIR = WORK_DIR / "export"       # checkpoint tốt nhất để mang về dự án (Drive)

# Working copy trên LOCAL Colab (nhanh; mất khi reset session → giải nén lại từ backup)
LOCAL_DATA = Path("/content/karlsson_local")  # nơi giải nén wav để xử lý
REPO_DIR   = Path("/content/Qwen3-TTS")        # repo clone (local Colab)
FT         = REPO_DIR / "finetuning"

for d in (WORK_DIR, DATA_DIR, CKPT_DIR, HF_CACHE, EXPORT_DIR, LOCAL_DATA):
    d.mkdir(parents=True, exist_ok=True)

# Trỏ HF cache vào Drive (đặt TRƯỚC khi import transformers/qwen_tts)
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

BASE_MODEL      = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TOKENIZER_MODEL = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
SPEAKER_NAME    = "karlsson_de"

print("\n✅ Drive mounted. WORK_DIR =", WORK_DIR)


# %% [2] Cài đặt & clone repo finetuning ---------------------------------------
# @title [2] Cài đặt & clone repo finetuning
# Chạy khối shell này trên Colab (hoặc bọc trong %%bash cell):
#   !pip install -q qwen-tts soundfile librosa num2words hf_transfer jiwer openai-whisper resemblyzer
#   !pip install -q https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl || echo "flash-attn fail -> eager"
#   !apt-get -y -q install sox libsox-fmt-all
#   !test -d /content/Qwen3-TTS/.git || git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS.git /content/Qwen3-TTS
#   !ls -1 /content/Qwen3-TTS/finetuning/
import subprocess

def _sh(cmd):
    print("$", cmd)
    print(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout[-2000:])

_sh("pip install -q qwen-tts soundfile librosa num2words hf_transfer jiwer "
    "openai-whisper resemblyzer")
# flash-attn: dùng prebuilt wheel (cu128 + torch2.10 + cp312) cho Colab A100, tránh build lâu.
_sh("pip install -q https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl "
    "|| echo 'flash-attn prebuilt fail -> eager (Cell 6 tự xử lý)'")
_sh("apt-get -y -q install sox libsox-fmt-all")
if not (REPO_DIR / ".git").exists():
    _sh(f"git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS.git {REPO_DIR}")
_sh(f"ls -1 {FT}")

# In README + --help để XÁC MINH tên cờ thật (ẩn số #1 trong kế hoạch)
readme = FT / "README.md"
if readme.exists():
    print("=" * 70, "\nfinetuning/README.md\n", "=" * 70)
    print(readme.read_text(encoding="utf-8")[:6000])
for script in ("prepare_data.py", "sft_12hz.py"):
    print("\n" + "=" * 70, f"\n{script} --help\n", "=" * 70)
    r = subprocess.run(["python", str(FT / script), "--help"],
                       capture_output=True, text=True, cwd=str(FT))
    print(r.stdout or r.stderr)


# %% [3.0] EXPLORE — Tải + backup zip (Drive) + giải nén working (local) + KHẢO SÁT
# @title [3.0] EXPLORE — Tải + backup zip (Drive) + giải nén working (local) + KHẢO SÁT
# Cell ĐỘC LẬP, chạy TRƯỚC TIÊN để kiểm tra cấu trúc HUI Karlsson clean trước khi
# build manifest. KHÔNG train gì cả. An toàn chạy đầu tiên.
#   - Zip backup: GIỮ trên Drive (DATA_DIR) — tải 1 lần, sống qua reset session.
#   - Giải nén: vào LOCAL /content (LOCAL_DATA) — nhanh, prepare_data/train đọc ở đây.
import zipfile
import collections
import soundfile as sf

HUI_CLEAN_URL = ("https://opendata.iisys.de/systemintegration/Datasets/"
                 "HUI-Audio-Corpus-German/dataset_clean/Karlsson_Clean.zip")
zip_path = DATA_DIR / "Karlsson_Clean.zip"   # <-- backup zip gốc, GIỮ trên Drive
raw_dir  = LOCAL_DATA / "Karlsson"           # <-- working copy, giải nén ở LOCAL

# --- 1. Tải zip về Drive (resume nếu đứt) làm backup ---
if not zip_path.exists():
    print("⏳ Đang tải về Drive (~7GB, có thể vài chục phút)...")
    rc = subprocess.run(["wget", "-c", "-q", "--show-progress",
                         HUI_CLEAN_URL, "-O", str(zip_path)]).returncode
    assert rc == 0, "Tải thất bại — kiểm tra HUI_CLEAN_URL trên trang dataset."
else:
    print(f"✅ Zip backup đã có trên Drive: {zip_path} "
          f"({zip_path.stat().st_size/1e9:.2f} GB) — skip tải.")

# --- 2. Soi nội dung zip TRƯỚC khi giải nén (xem cây thư mục gốc) ---
with zipfile.ZipFile(zip_path) as z:
    names = z.namelist()
print(f"\nSố entry trong zip: {len(names)}")
top = sorted({n.split('/')[0] for n in names if n.strip()})
print("Thư mục/file cấp 1 trong zip:", top[:10])
print("10 entry mẫu:")
for n in names[:10]:
    print("   ", n)

# --- 3. Giải nén zip backup (Drive) -> LOCAL /content để xử lý nhanh ---
if raw_dir.exists() and any(raw_dir.rglob("*.wav")):
    print(f"\n✅ Đã có working copy tại {raw_dir} (local) — skip giải nén.")
else:
    print(f"\n⏳ Đang giải nén từ Drive vào local {LOCAL_DATA} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(LOCAL_DATA)
    print("✅ Giải nén xong (local).")

# Dò thư mục gốc nếu tên khác 'Karlsson'
if not raw_dir.exists():
    cands = [p for p in LOCAL_DATA.iterdir() if p.is_dir() and "karlsson" in p.name.lower()]
    if cands:
        raw_dir = cands[0]
        print("→ Dùng thư mục:", raw_dir)

# --- 4. KHẢO SÁT cấu trúc ---
print("\n" + "=" * 60, "\nKHẢO SÁT CẤU TRÚC DATASET\n", "=" * 60)

# 4a. Cây thư mục (2 cấp đầu)
print("\n[Cây thư mục — 2 cấp đầu]")
for sub in sorted([p for p in raw_dir.iterdir() if p.is_dir()])[:15]:
    n_wav_sub = sum(1 for _ in sub.rglob("*.wav"))
    inner = sorted({c.name for c in sub.iterdir()})[:6]
    print(f"  {sub.name}/  ({n_wav_sub} wav)  -> {inner}")

# 4b. Tổng quan file
wavs = list(raw_dir.rglob("*.wav"))
csvs = list(raw_dir.rglob("*.csv"))
print(f"\n[Tổng quan]  wav={len(wavs)}  |  csv={len(csvs)}")
print("File .csv (metadata) mẫu:")
for c in csvs[:5]:
    print("   ", c.relative_to(raw_dir))

# 4c. Định dạng metadata (3 dòng đầu của csv đầu tiên)
if csvs:
    print(f"\n[Định dạng metadata: {csvs[0].name}]")
    for ln in csvs[0].read_text(encoding="utf-8").splitlines()[:3]:
        print("   ", repr(ln))   # repr để thấy rõ dấu phân cách '|'

# 4d. Thuộc tính audio (sr/channel) + phân bố độ dài trên mẫu 300 wav
import random as _rnd
_rnd.seed(0)
sample = _rnd.sample(wavs, min(300, len(wavs)))
srs, chans, durs = collections.Counter(), collections.Counter(), []
for w in sample:
    try:
        info = sf.info(str(w))
        srs[info.samplerate] += 1
        chans[info.channels] += 1
        durs.append(info.frames / info.samplerate)
    except Exception:
        pass
durs.sort()
print(f"\n[Audio — mẫu {len(sample)} file]")
print("   Sample rate:", dict(srs))
print("   Channels   :", dict(chans))
if durs:
    import statistics
    print(f"   Độ dài (s) : min={durs[0]:.1f}  p50={statistics.median(durs):.1f}  "
          f"max={durs[-1]:.1f}  mean={statistics.mean(durs):.1f}")
    tot_h = (statistics.mean(durs) * len(wavs)) / 3600
    print(f"   Ước tổng thời lượng ≈ {tot_h:.1f} giờ ({len(wavs)} wav)")
print("\n✅ Khảo sát xong. Xem output trên rồi mới chạy Cell 4 (build manifest).")


# %% [3] Đảm bảo working copy (local) sẵn sàng --------------------------------
# @title [3] Đảm bảo working copy (local) sẵn sàng
# CHẠY MỖI SESSION sau khi đã có backup. Sau reset session, /content (local) bị xóa
# nhưng zip backup trên Drive còn → cell này DÒ zip backup trên Drive rồi GIẢI NÉN
# vào LOCAL /content để prepare_data/train đọc nhanh. Nếu chưa có backup, chạy [3.0] trước.
import zipfile

zip_path = DATA_DIR / "Karlsson_Clean.zip"   # backup trên Drive
raw_dir  = LOCAL_DATA / "Karlsson"           # working copy local

if raw_dir.exists() and any(raw_dir.rglob("*.wav")):
    print(f"✅ Working copy local đã sẵn: {raw_dir} — skip.")
else:
    assert zip_path.exists(), (
        f"Chưa thấy zip backup {zip_path} trên Drive. "
        f"Chạy cell [3.0] EXPLORE để tải về trước.")
    print(f"⏳ Giải nén backup Drive → local {LOCAL_DATA} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(LOCAL_DATA)
    print("✅ Working copy local sẵn sàng.")

# Dò thư mục gốc nếu tên khác 'Karlsson'
if not raw_dir.exists():
    cands = [p for p in LOCAL_DATA.iterdir() if p.is_dir() and "karlsson" in p.name.lower()]
    if cands:
        raw_dir = cands[0]
        print("→ Dùng thư mục:", raw_dir)

print(f"Số file .wav (local): {sum(1 for _ in raw_dir.rglob('*.wav'))}")
metas = list(raw_dir.rglob("metadata*.csv")) or list(raw_dir.rglob("*.csv"))
print("File metadata mẫu:", metas[:3])
if metas:
    print("\n--- 3 dòng đầu metadata ---")
    print("\n".join(metas[0].read_text(encoding="utf-8").splitlines()[:3]))


# %% [4a] Dựng manifest từ transcript HUI + làm sạch + TRIM SILENCE -------------
# @title [4a] Dựng manifest từ transcript HUI + làm sạch + trim silence
# PHẦN QUYẾT ĐỊNH CHẤT LƯỢNG. Nhãn = transcript HUI (KHÔNG ASR).
# HUI = LJSpeech-style: metadata.csv phân cách '|', cột = id | text | normalized_text.
# Ưu tiên normalized_text (cột 3). Nếu Cell 3 cho thấy định dạng khác -> chỉnh parse_metadata().
#
# TRIM SILENCE (quan trọng — fix gốc cho audio sinh ra bị silence cuối loạn + rè/bíp/bộp):
# bản thu HUI có silence/room-tone đầu-cuối DÀI NGẮN THẤT THƯỜNG. Train thẳng wav thô ->
# model học token "dừng" (EOS) không nhất quán -> lúc inference đuôi clip lúc dài lúc cụt,
# kèm token codec lạ cuối câu render thành click/pop. Ở đây mỗi wav được trim theo năng
# lượng rồi pad ĐỒNG NHẤT 2 đầu, ghi sang TRIM_DIR (local, tạm); manifest trỏ vào bản đã trim.
# (TRIM_DIR là dữ liệu dẫn xuất, ephemeral — artifact bền là train_codes.jsonl ở Cell 5.
#  Nếu reset session khi MỚI dựng manifest mà CHƯA chạy Cell 5: xóa *.jsonl để dựng lại.)
import re
import json
import random
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from num2words import num2words

random.seed(42)
MIN_SEC, MAX_SEC = 3.0, 30.0   # HUI Karlsson: min≈5s, max=27.1s (sample 300 file) — 30s để buffer an toàn

TRIM_TOP_DB = 30               # ngưỡng trim theo năng lượng (dB dưới đỉnh); thấp hơn = cắt mạnh tay hơn
PAD_SEC     = 0.1             # pad 50ms đồng nhất 2 đầu sau khi trim -> mục tiêu EOS nhất quán
TRIM_DIR    = LOCAL_DATA / "Karlsson_trimmed"   # bản đã trim (local, tạm)
TRIM_DIR.mkdir(parents=True, exist_ok=True)


def trim_wav(src):
    """Trim silence đầu-cuối + pad đồng nhất -> ghi bản mới. Trả (path, dur) hoặc None nếu hỏng.
    Idempotent per-file: đã có bản trim thì dùng lại (rerun/đứt giữa chừng không tốn công)."""
    dst = TRIM_DIR / (Path(src).stem + ".wav")
    if dst.exists():
        info = sf.info(str(dst))
        return str(dst), info.frames / info.samplerate
    try:
        y, sr = librosa.load(str(src), sr=None, mono=True)
    except Exception:
        return None
    y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    if len(y) == 0:                       # toàn silence/hỏng -> bỏ
        return None
    p = int(sr * PAD_SEC)
    y = np.pad(y, (p, p))
    sf.write(str(dst), y, sr)
    return str(dst), len(y) / sr


def normalize_de(text: str) -> str:
    """Chuẩn hóa nhẹ text Đức: giữ äöüß, đọc số thành chữ, bỏ ký tự lạ."""
    text = text.strip()

    def _num(m):
        try:
            return num2words(int(m.group()), lang="de")
        except Exception:
            return m.group()

    text = re.sub(r"\d+", _num, text)
    text = re.sub(r"[^0-9A-Za-zÄÖÜäöüß .,!?;:'\"-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_metadata(meta_path):
    """HUI/LJSpeech: id|text|normalized_text -> list (wav_path, text)."""
    base = meta_path.parent
    rows = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 2:
                continue
            uid = parts[0].strip()
            text = (parts[2] if len(parts) >= 3 and parts[2].strip() else parts[1]).strip()
            cand = base / "wavs" / f"{uid}.wav"
            if not cand.exists():
                hits = list(base.rglob(f"{uid}.wav"))
                cand = hits[0] if hits else None
            if cand:
                rows.append((cand, text))
    return rows


# Đường dẫn manifest — định nghĩa SỚM để dùng ở cả nhánh build lẫn nhánh skip.
train_raw = DATA_DIR / "train_raw.jsonl"
val_jsonl = DATA_DIR / "val.jsonl"
ref_meta  = DATA_DIR / "ref_meta.json"

# Idempotent: đã có đủ manifest trên Drive -> skip build (không cần wav local).
MANIFESTS_READY = (train_raw.exists() and train_raw.stat().st_size > 0
                   and val_jsonl.exists() and val_jsonl.stat().st_size > 0
                   and ref_meta.exists())

if MANIFESTS_READY:
    # Nạp lại biến cần cho Cell 5/6/7 (KHÔNG đọc lại duration mọi wav).
    _rm = json.load(open(ref_meta, encoding="utf-8"))
    REF_AUDIO, REF_TEXT = _rm["ref_audio"], _rm["ref_text"]
    print("✅ Manifest đã có — skip dựng lại manifest.")
    print("   train_raw =", train_raw, "| val =", val_jsonl, "| ref =", REF_AUDIO)

    # --- SELF-HEAL trims: tái tạo từ RAW, KHÔNG lưu wav lên Drive --------------
    # Bản trim ở /content là ephemeral; sau reset chúng mất nhưng manifest trỏ vào.
    # trim_wav() deterministic -> tái tạo từ raw (Cell 3 giải nén lại) == bản cũ
    # (cùng stem -> ghi đúng vào TRIM_DIR/<stem>.wav). Chỉ tái tạo cái ĐANG thiếu:
    #   codes đã xong -> chỉ cần ref_audio (Cell 6/7);  chưa xong -> cả train trims (Cell 5).
    _codes = DATA_DIR / "train_codes.jsonl"
    _codes_ready = _codes.exists() and _codes.stat().st_size > 0
    _need = ([] if _codes_ready
             else [json.loads(l)["audio"] for l in open(train_raw, encoding="utf-8")])
    _need.append(REF_AUDIO)
    _missing = [p for p in dict.fromkeys(_need) if not Path(p).exists()]
    if _missing:
        assert raw_dir.exists() and any(raw_dir.rglob("*.wav")), (
            "Thiếu wav raw để tái tạo trims — chạy Cell 3 (giải nén zip) trước.")
        _raw_idx = {p.stem: p for p in raw_dir.rglob("*.wav")}
        print(f"♻️  Mất /content → tái tạo {len(_missing)} bản trim từ raw…")
        _nf = 0
        for _k, _p in enumerate(_missing):
            _src = _raw_idx.get(Path(_p).stem)
            if _src is None:
                _nf += 1; continue
            trim_wav(_src)                       # ghi vào TRIM_DIR/<stem>.wav == _p
            if (_k + 1) % 500 == 0:
                print(f"   …{_k + 1}/{len(_missing)}")
        print("✅ Trims sẵn sàng." + (f" ({_nf} không thấy raw)" if _nf else ""))
    else:
        print("   (trims local còn đủ — không cần tái tạo.)")
else:
    meta_files = sorted(set(raw_dir.rglob("metadata*.csv"))) or sorted(set(raw_dir.rglob("*.csv")))
    assert meta_files, "Không tìm thấy metadata — xem lại output Cell 3."

    pairs = []
    for mf in meta_files:
        pairs.extend(parse_metadata(mf))
    print(f"Tổng utterance thô: {len(pairs)}")

    clean = []
    for k, (wav, text) in enumerate(pairs):
        try:
            info = sf.info(str(wav))
            dur = info.frames / info.samplerate
        except Exception:
            continue
        if not (MIN_SEC <= dur <= MAX_SEC):       # lọc theo độ dài THÔ (chưa trim) cho nhanh
            continue
        t = normalize_de(text)
        if len(t) < 3:
            continue
        tr = trim_wav(wav)                        # trim silence -> bản đã trim trong TRIM_DIR
        if tr is None:
            continue
        twav, tdur = tr
        clean.append({"audio": twav, "text": t, "dur": round(tdur, 2)})   # dur = sau trim
        if (k + 1) % 500 == 0:
            print(f"  …đã quét {k + 1}/{len(pairs)} (trim)")

    print(f"Sau lọc + trim: {len(clean)} utterance | ≈ {sum(c['dur'] for c in clean)/3600:.1f} giờ")


# %% [4b] Chọn ref_audio + tách train/val + xuất manifest ----------------------
# @title [4b] Chọn ref_audio + tách train/val + xuất manifest
if not MANIFESTS_READY:
    import IPython.display as ipd

    # 1 ref_audio duy nhất cho MỌI mẫu (khuyến nghị chính thức): clip sạch ~7s.
    ref_cands = [c for c in clean if 5.0 <= c["dur"] <= 10.0]
    ref_cands.sort(key=lambda c: abs(c["dur"] - 7.0))
    assert ref_cands, "Không có clip 5-10s làm ref — nới điều kiện."
    REF_AUDIO = ref_cands[0]["audio"]
    REF_TEXT = ref_cands[0]["text"]
    print("ref_audio =", REF_AUDIO, "|", ref_cands[0]["dur"], "s")
    print("ref_text  =", REF_TEXT)
    ipd.display(ipd.Audio(REF_AUDIO))

    # Tách validation (giữ riêng, KHÔNG đưa vào train). random.seed(42) ở 4a -> deterministic.
    random.shuffle(clean)
    N_VAL = 40
    clean = [c for c in clean if c["audio"] != REF_AUDIO]
    val_set, train_set = clean[:N_VAL], clean[N_VAL:]
    print(f"train = {len(train_set)} | val = {len(val_set)}")

    # train_raw.jsonl — fields chính thức: audio, text, ref_audio
    with open(train_raw, "w", encoding="utf-8") as f:
        for c in train_set:
            f.write(json.dumps({"audio": c["audio"], "text": c["text"],
                                "ref_audio": REF_AUDIO}, ensure_ascii=False) + "\n")

    # val.jsonl — phục vụ eval Cell 7
    with open(val_jsonl, "w", encoding="utf-8") as f:
        for c in val_set:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # ref_meta.json — transcript của REF_AUDIO cho base (ICL voice-clone) ở Cell 7
    with open(ref_meta, "w", encoding="utf-8") as f:
        json.dump({"ref_audio": REF_AUDIO, "ref_text": REF_TEXT}, f, ensure_ascii=False)

    print("✅ Đã ghi", train_raw, "và", val_jsonl)


# %% [5] Trích xuất audio codes (prepare_data.py) ------------------------------
# @title [5] Trích xuất audio codes (prepare_data.py)
# Sinh train_codes.jsonl (thêm audio_codes). Cache ra Drive -> skip nếu đã có.
# ĐỐI CHIẾU tên cờ với --help ở Cell 2.
train_codes = DATA_DIR / "train_codes.jsonl"

if train_codes.exists() and train_codes.stat().st_size > 0:
    print("✅ Đã có", train_codes, "— skip. (Xóa file nếu muốn làm lại.)")
else:
    cmd = [
        "python", str(FT / "prepare_data.py"),
        "--device", "cuda:0",
        "--tokenizer_model_path", TOKENIZER_MODEL,
        "--input_jsonl",  str(train_raw),
        "--output_jsonl", str(train_codes),
    ]
    print("$", " ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(FT)).returncode
    assert rc == 0, "prepare_data.py lỗi — đối chiếu tên cờ với --help ở Cell 2."
    print("✅ Xong:", train_codes)


# %% [6] Train full SFT (checkpoint ra Drive) ---------------------------------
# @title [6] Train full SFT (checkpoint ra Drive)
# QUAN TRỌNG — sft_12hz.py CHỈ nhận 7 cờ: init_model_path, output_model_path,
# train_jsonl, batch_size, lr, num_epochs, speaker_name. device/attn/grad_accum
# bị HARDCODE trong script (grad_accum=4, flash_attention_2, bf16). Batch hiệu
# dụng = batch_size * 4. Checkpoint lưu mỗi epoch -> resume thủ công bằng cách
# trỏ --init_model_path vào checkpoint-epoch-N gần nhất (resume mức epoch).
import torch
from huggingface_hub import snapshot_download

# (1) Tải snapshot model về LOCAL path. Bắt buộc: script copytree(init_model_path)
#     khi lưu checkpoint -> phải là thư mục thật trên disk, KHÔNG phải repo-id.
LOCAL_MODEL = snapshot_download(BASE_MODEL)
print("Local model:", LOCAL_MODEL)

# (2) Cài flash-attn nếu thiếu (script hardcode flash_attention_2). Nếu cài/biên
#     dịch không được, patch sang sdpa để vẫn chạy được trên A100/L4.
HAS_FA2 = False
try:
    import flash_attn  # noqa: F401
    HAS_FA2 = True
except Exception:
    print("flash-attn chưa có — thử cài (build ~5-10').")
    subprocess.run(["pip", "install", "-q", "flash-attn", "--no-build-isolation"])
    try:
        import flash_attn  # noqa: F401
        HAS_FA2 = True
    except Exception:
        print("⚠️ flash-attn vẫn lỗi → patch script sang sdpa.")

# (3) Patch script (idempotent): tensorboard cần project_dir; sdpa nếu không có FA2.
def _patch(path, old, new, tag):
    p = Path(path); s = p.read_text()
    if new in s:
        print(f"ℹ️ {tag}: đã áp."); return
    assert old in s, f"{tag}: không thấy chuỗi gốc — script có thể đã đổi."
    p.write_text(s.replace(old, new)); print(f"✅ {tag}")

_patch(FT / "sft_12hz.py",
       'Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")',
       'Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", '
       'log_with="tensorboard", project_dir=args.output_model_path)',
       "Patch tensorboard project_dir")

_patch(FT / "dataset.py",
       '        assert sr == 24000, "Only support 24kHz audio"',
       '        if sr != 24000:\n'
       '            import librosa, numpy as np\n'
       '            audio = librosa.resample(np.asarray(audio, dtype="float32"), '
       'orig_sr=sr, target_sr=24000)\n'
       '            sr = 24000',
       "Patch resample 24kHz")

if not HAS_FA2:
    _patch(FT / "sft_12hz.py", 'attn_implementation="flash_attention_2"',
           'attn_implementation="sdpa"', "Patch FA2 -> sdpa (default transformers)")

# (4) batch_size theo VRAM (grad_accum=4 cố định -> batch hiệu dụng = *4).
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
batch_size = 18 if vram_gb >= 70 else 8 if vram_gb >= 38 else 4 if vram_gb >= 22 else 2
print(f"VRAM ≈ {vram_gb:.0f}GB → batch_size={batch_size} (hiệu dụng {batch_size*4})")

cmd = [
    "python", str(FT / "sft_12hz.py"),
    "--init_model_path",   LOCAL_MODEL,        # local path (không phải repo-id)
    "--output_model_path", str(CKPT_DIR),
    "--train_jsonl",       str(train_codes),
    "--speaker_name",      SPEAKER_NAME,
    "--batch_size",        str(batch_size),
    "--lr",                "2e-6",
    "--num_epochs",        "6",
]
print("$", " ".join(cmd), "\n")
rc = subprocess.run(cmd, cwd=str(FT)).returncode
assert rc == 0, "sft_12hz.py lỗi — xem STDERR phía trên (ô chẩn đoán capture_output ở Cell 2)."
print("\n✅ Train xong. Checkpoint tại:", CKPT_DIR)
print(sorted(p.name for p in CKPT_DIR.glob("checkpoint-epoch-*")))


# %% [7a] Setup eval: ASR (WER) + speaker embedding ----------------------------
# @title [7a] Setup eval: ASR (WER) + speaker embedding
import gc
import shutil
import numpy as np

# attn_implementation: bắt buộc flash_attention_2 nếu có, fallback eager (KHÔNG sdpa vì
# sdpa có thể tạo ra output khác với những gì model được train với FA2).
try:
    import flash_attn  # noqa: F401
    ATTN_IMPL = "flash_attention_2"
except ImportError:
    ATTN_IMPL = None   # không truyền → transformers tự chọn default
print("attn_implementation:", ATTN_IMPL)
import torch
import soundfile as sf
import IPython.display as ipd
from jiwer import wer
from qwen_tts import Qwen3TTSModel

val = [json.loads(l) for l in open(val_jsonl, encoding="utf-8")]
val_eval = val[:15]                       # đủ đo, không quá chậm

# REF_TEXT (transcript của REF_AUDIO) — bắt buộc cho base ICL voice-clone
with open(DATA_DIR / "ref_meta.json", encoding="utf-8") as f:
    _rm = json.load(f)
REF_AUDIO, REF_TEXT = _rm["ref_audio"], _rm["ref_text"]

# Ghi WAV ra LOCAL trước (Drive FUSE không hỗ trợ seek-on-write của libsndfile),
# rồi copy sang Drive ở cuối.
out_dir = Path("/content/eval_out")
out_dir.mkdir(parents=True, exist_ok=True)
drive_out = WORK_DIR / "eval_out"
drive_out.mkdir(exist_ok=True)

import whisper                            # openai-whisper
asr = whisper.load_model("small")


def asr_de(wav_path):
    return asr.transcribe(str(wav_path), language="de").get("text", "").strip().lower()


try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    spk_enc = VoiceEncoder()

    def spk_emb(wav_path):
        return spk_enc.embed_utterance(preprocess_wav(Path(wav_path)))

    ref_emb = spk_emb(REF_AUDIO)
    HAS_SPK = True
except Exception as e:
    print("⚠️ resemblyzer lỗi — bỏ qua speaker-sim.", e)
    HAS_SPK = False


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# %% [7b] Eval từng checkpoint + base (zero-shot) ------------------------------
# @title [7b] Eval từng checkpoint + base (zero-shot)
# Nút thắt thời gian = LOAD model từ Drive (mỗi model ~3.5GB). Tối ưu:
#   1) EVAL_EPOCHS: chỉ eval epoch ứng viên thay vì tất cả.
#   2) Copy checkpoint Drive->local /content rồi mới from_pretrained (đọc nhanh hơn).
#   3) Batch generate cả bộ câu trong 1 lần gọi (fallback loop nếu API không nhận list).
EVAL_EPOCHS = None        # None = mọi epoch tìm thấy; hoặc [3, 4, 5] để chỉ eval ứng viên


def _to_np(w):
    w = w.detach().cpu().numpy() if isinstance(w, torch.Tensor) else np.asarray(w)
    return np.asarray(w, dtype="float32").reshape(-1)


def load_local(ckpt_path, tag):
    """base -> repo-id (HF tự cache). checkpoint -> copy Drive→/content 1 lần rồi
    load từ local. Lý do: from_pretrained mmap file ~3.5GB; mmap qua Drive FUSE
    chậm + dễ lỗi I/O ngắt quãng. Copy tuần tự 1 lần chịu đựng tốt hơn nhiều."""
    if tag == "base":
        return str(BASE_MODEL)
    ckpt_path = Path(ckpt_path)
    if str(ckpt_path).startswith("/content/"):        # đã ở local -> khỏi copy
        return str(ckpt_path)
    dst = Path("/content/ckpt_eval") / ckpt_path.name
    if not dst.exists():
        print(f"  copy {ckpt_path.name} Drive→/content…")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ckpt_path, dst)
    return str(dst)


def gen_all(model, texts, tag):
    """Sinh audio cho CẢ bộ texts trong 1 lần gọi (batch native, list input).
    Cả 2 hàm trả (List[np.ndarray], int) — wavs LUÔN là list, kể cả 1 câu.
    Không có tham số batch_size: batch điều khiển qua độ dài list input."""
    n = len(texts)
    if tag == "base":
        wavs, sr = model.generate_voice_clone(
            text=texts, language=["German"] * n,
            ref_audio=REF_AUDIO, ref_text=[REF_TEXT] * n)
    else:
        wavs, sr = model.generate_custom_voice(
            text=texts, speaker=SPEAKER_NAME, language=["German"] * n)
    return [_to_np(w) for w in wavs], int(sr)


def eval_checkpoint(ckpt_path, tag):
    """Sinh audio val (batch) + đo WER & speaker-sim. Trả về dict metric."""
    kw = {"attn_implementation": ATTN_IMPL} if ATTN_IMPL else {}
    model = Qwen3TTSModel.from_pretrained(load_local(ckpt_path, tag),
                                          device_map="cuda:0", dtype=torch.bfloat16, **kw)
    texts = [it["text"] for it in val_eval]
    try:
        wavs, sr = gen_all(model, texts, tag)
    except Exception as e:
        print(f"  [{tag}] sinh audio lỗi: {e}")
        wavs, sr = [], 24000
    wers, sims, first_audio = [], [], None
    for i, w in enumerate(wavs):
        outp = out_dir / f"{tag}_{i:02d}.wav"               # ghi local
        sf.write(str(outp), w, sr, format="WAV", subtype="PCM_16")
        shutil.copy2(outp, drive_out / outp.name)           # backup sang Drive
        if first_audio is None:
            first_audio = outp
        wers.append(wer(val_eval[i]["text"].lower(), asr_de(outp)))
        if HAS_SPK:
            sims.append(cos(ref_emb, spk_emb(outp)))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"tag": tag,
            "WER": round(float(np.mean(wers)), 3) if wers else None,
            "spk_sim": round(float(np.mean(sims)), 3) if sims else None,
            "sample": first_audio}


ckpts = sorted(CKPT_DIR.glob("checkpoint-epoch-*"))
if EVAL_EPOCHS is not None:
    ckpts = [c for c in ckpts if int(c.name.split("-")[-1]) in EVAL_EPOCHS]

results = [eval_checkpoint(BASE_MODEL, "base")]   # mốc so sánh zero-shot
for ckpt in ckpts:
    results.append(eval_checkpoint(ckpt, ckpt.name))

print(f"\n{'tag':<24}{'WER↓':>8}{'spk_sim↑':>10}")
print("-" * 42)
for r in results:
    print(f"{r['tag']:<24}{str(r['WER']):>8}{str(r['spk_sim']):>10}")
    if r["sample"]:
        ipd.display(ipd.HTML(f"<b>{r['tag']}</b>"))
        ipd.display(ipd.Audio(str(r["sample"])))


# %% [8] Export checkpoint tốt nhất --------------------------------------------
# @title [8] Export checkpoint tốt nhất
# Chọn epoch cân bằng WER thấp + spk_sim cao + nghe ổn (dựa bảng Cell 7b).
# Export ra Drive (bền) -> copy ra /content -> load từ LOCAL để sanity-check.
# (Load thẳng từ Drive FUSE chậm + dễ lỗi I/O với file ~3.5GB -> luôn copy local trước.)
import shutil
from pathlib import Path

BEST = "checkpoint-epoch-1"   # <-- SỬA theo kết quả eval

src = CKPT_DIR / BEST
dst = EXPORT_DIR / "qwen3tts-karlsson-de"          # bản bền trên Drive
assert src.exists(), f"Không thấy {src} — kiểm tra lại tên BEST."
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print("✅ Đã export (Drive):", dst)
print("   Dung lượng:", round(sum(f.stat().st_size for f in dst.rglob('*')) / 1e9, 2), "GB")

# --- Copy export Drive→/content rồi load từ LOCAL để sanity-check ---
import torch
import soundfile as sf
import IPython.display as ipd
from qwen_tts import Qwen3TTSModel

local_export = Path("/content/export_local/qwen3tts-karlsson-de")
if not local_export.exists():
    print("⏳ Copy export Drive→/content (load nhanh & ổn định)…")
    local_export.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dst, local_export)

try:
    import flash_attn  # noqa: F401
    _attn = "flash_attention_2"
except ImportError:
    _attn = "eager"
model = Qwen3TTSModel.from_pretrained(str(local_export), device_map="cuda:0",
                                      dtype=torch.bfloat16, attn_implementation=_attn)
_w, _sr = model.generate_custom_voice(
    text=["Guten Tag! Dies ist ein Test der exportierten Stimme."],
    speaker=SPEAKER_NAME, language=["German"])
_sanity = Path("/content/export_sanity.wav")
sf.write(str(_sanity), _w[0].reshape(-1), int(_sr), format="WAV", subtype="PCM_16")
print("✅ Load + sinh audio từ bản export (local) OK.")
ipd.display(ipd.Audio(str(_sanity)))
del model
torch.cuda.empty_cache()

print(textwrap.dedent(f"""\
    ── Cách dùng lại (khi mang về dự án) ──
      # Drive = lưu bền; nhưng LOAD thì copy ra /content trước
      # (FUSE chậm + dễ lỗi I/O khi from_pretrained mmap file ~3.5GB).
      import shutil
      from pathlib import Path
      from qwen_tts import Qwen3TTSModel
      src   = "{dst}"                       # bản export trên Drive
      local = "/content/qwen3tts-karlsson-de"
      if not Path(local).exists():
          shutil.copytree(src, local)       # copy 1 lần -> sau đó load nhanh & ổn định
      model = Qwen3TTSModel.from_pretrained(local, device_map="cuda:0", dtype=torch.bfloat16,
                                            attn_implementation="flash_attention_2")  # hoặc "eager"
      wav, sr = model.generate_custom_voice(text="...", speaker="{SPEAKER_NAME}")

    LƯU Ý tích hợp ngược (ngoài phạm vi file này):
      Engine dự án hiện gọi generate_voice_clone(). Checkpoint SFT dùng
      generate_custom_voice(speaker=...) → cần chỉnh lời gọi + trỏ model_path vào folder export."""))



# %% [9] Muốn nghe nhanh 1 checkpoint với câu bạn tự gõ --------------------------------------------
# @title Muốn nghe nhanh 1 checkpoint với câu bạn tự gõ

import soundfile as sf, torch, shutil
from pathlib import Path
from qwen_tts import Qwen3TTSModel
import IPython.display as ipd

EP = 4                                   # đổi epoch: 3/4/5
ckpt = CKPT_DIR / f"checkpoint-epoch-{EP}"
local = Path("/content/ckpt_eval") / ckpt.name
if not local.exists(): shutil.copytree(ckpt, local)   # load từ local cho nhanh

texts = [
    "Stell dir vor, du wachst eines Morgens auf, und die ganze Welt sieht plötzlich anders aus. Die Straßen sind leer, die Vögel singen, und über allem liegt eine seltsame, fast magische Stille. Was würdest du tun? Würdest du einfach weiterschlafen, oder würdest du hinausgehen und das Geheimnis lüften? Manche Menschen warten ihr ganzes Leben auf diesen einen Moment. Doch nur die Mutigen, die wirklich Mutigen, finden am Ende die Antwort!",
    "An jenem Abend war alles still, ruhig, beinahe friedlich. Doch dann hörte ich es: ein leises Klopfen an der Tür. Wer konnte das nur sein, so spät in der Nacht? Ich öffnete, und da stand er, lächelnd, mit Tränen in den Augen. „Du bist zurückgekommen! Ich kann es nicht glauben!\" Wir hatten so viel verloren, so viele Jahre. Aber jetzt, endlich, war er wieder hier. Und vielleicht, ganz vielleicht, würde alles wieder gut werden.",
]
model = Qwen3TTSModel.from_pretrained(str(local), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2")
wavs, sr = model.generate_custom_voice(text=texts, speaker=SPEAKER_NAME,   # batch 1 lần
                                       language=["German"]*len(texts))
for i, (t, w) in enumerate(zip(texts, wavs)):
    p = Path(f"/content/try_ep{EP}_{i}.wav")
    sf.write(str(p), w.reshape(-1), int(sr), format="WAV", subtype="PCM_16")
    print(t); ipd.display(ipd.Audio(str(p)))
del model; torch.cuda.empty_cache()
