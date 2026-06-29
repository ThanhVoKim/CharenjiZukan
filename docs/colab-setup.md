# Colab — Thiết lập môi trường & khoá version (lock)

Tài liệu này giải quyết lỗi kiểu **"hôm qua chạy, hôm nay không"** trên Colab (điển hình:
`ERROR:qwen_tts: libcudart.so.13: cannot open shared object file`).

## Flow tổng quan

```
                    ┌─────────────────────────────────────────────┐
                    │  Lần đầu (chưa có lock) → chạy mục A       │
                    │                                             │
                    │  install → verify → freeze → lưu lock      │
                    └──────────────────────┬──────────────────────┘
                                           │ lock đã có trên Drive
                                           ▼
                    ┌─────────────────────────────────────────────┐
                    │  Mỗi ngày sau → chỉ chạy mục B             │
                    │                                             │
                    │  restore từ lock (không freeze lại)         │
                    └─────────────────────────────────────────────┘

  Chỉ quay lại mục A khi:
    - Colab đổi base image (torch 2.10 → 2.11)
    - Cố tình nâng cấp một package
    - Lock cũ bị hỏng / restore không được
```

Lock file **không tự sinh ra mới mỗi lần chạy**. Một lần freeze → một file lock → dùng mãi cho
đến khi có lý do cụ thể phải freeze lại.

---

## Vì sao lỗi xảy ra

Flow cài đặt mặc định **không khoá version**. Mỗi lần chạy, `uv`/`pip` resolve lại **bản mới
nhất** trên PyPI, nên chỉ cần một dependency publish bản mới qua đêm là cả stack lệch. Hai khác
biệt then chốt:

- **`pip` thường chạy được** vì nó giữ nguyên `torch` mà Colab đã tiền cài (khớp CUDA driver).
- **`uv` hay hỏng** vì nó dễ nâng cấp `torch`/`onnxruntime-gpu` lên bản build cho **CUDA 13**
  (`libcudart.so.13`), trong khi driver Colab là **CUDA 12.8** (chỉ có `libcudart.so.12`).

Thủ phạm đã gặp (xem `logs/JOURNAL.md`, 2026-06-19): **`onnxruntime-gpu 1.27.0` build cho CUDA 13**.
`qwen_tts` gọi `import onnxruntime` (tokenizer VQ) → nạp `.so` link `libcudart.so.13` → fail. Bản
chạy được là `onnxruntime-gpu==1.26.0` (CUDA 12).

## Nguyên tắc: **1 môi trường = 1 file lock**

Mỗi CLI có bộ phụ thuộc khác nhau (thậm chí xung đột nhau) → môi trường riêng → lock riêng. Không
nhồi cả ba vào một env.

| CLI          | Cần gì                                                                              | Môi trường       | Lock            |
| ------------ | ----------------------------------------------------------------------------------- | ---------------- | --------------- |
| `sync-video` | qwen-tts + audio-separator + qwen-asr (forced alignment) + `onnxruntime-gpu==1.26.0`. Gộp **1 lệnh** + `--override transformers==4.57.6` + `-c cuda-base.txt` (ghim cả họ torch cu128). Restore lock **bắt buộc `--no-deps`**. | `.venv-sync`     | `sync_lock.txt` |
| `qwen3_asr`  | qwen-asr + `transformers==4.57.6` (torch do qwen-asr[vllm] tự kéo)                  | `.venv-qwen3asr` | `asr_lock.txt`  |
| `video-ocr`  | extra `ocr` (cv2 + Pillow + torch + **torchvision** + transformers + qwen-vl-utils) + `-c cuda-base.txt`. KHÔNG cần qwen-tts/onnxruntime-gpu | `.venv-ocr`      | `ocr_lock.txt`  |

> **Cả 3 đều là venv riêng, CÔ LẬP — KHÔNG `--system`, KHÔNG `--system-site-packages`.**
> - Bỏ `--system`: `uv pip freeze` trên venv chỉ liệt kê gói trong venv, không nuốt cả bộ
>   RAPIDS/CUDA Colab tiền cài (`cudf-cu12`, `cuda-toolkit`, `nvidia-cuda-nvcc-cu12`...) — vốn pin
>   mâu thuẫn nội bộ làm restore "unsatisfiable".
> - Bỏ `--system-site-packages`: dù để cờ này, `uv` vẫn cài lại torch riêng vào venv (uv#2500), nên
>   nó **không giúp tiết kiệm gì** mà còn để lộ **TensorFlow/keras của Colab** vào venv → `transformers`
>   dò nhầm backend TF (lệch version) → `ImportError: cannot import name 'AutoProcessor'`.
>
> Đổi lại: mỗi venv tự tải torch (~2GB). Đây là cái giá cho lock sạch, restore được. Xem `logs/JOURNAL.md`.

> **Hai quy luật rút ra từ mọi lỗi đã gặp (áp cho CẢ 3 venv):**
> - **Quy luật A — gói "Colab cài ngầm":** code (hoặc *thư viện khác*) import gói mà Colab có sẵn
>   nhưng `pyproject.toml` không khai báo → venv cô lập thiếu. Gồm cả gói *cần ngầm*: `qwen_tts` →
>   `torchaudio`, `transformers VL` → `torchvision`, `edgetts` → `aiohttp`. Khắc phục: khai báo đủ
>   trong extra (`qwen-tts`, `ocr`...) rồi verify import trước khi freeze.
> - **Quy luật B — đồng bộ CUDA:** mọi gói họ-torch (`torch`/`torchvision`/`torchaudio`) **và**
>   `onnxruntime-gpu` phải cùng `cu128`. **Mọi** lệnh `uv pip install` đụng tới chúng đều phải mang
>   `-c /content/cuda-base.txt` + `--extra-index-url ...cu128` + `--index-strategy unsafe-best-match`,
>   nếu không `torch` trôi sang bản CUDA-13 của PyPI → `libcudart.so.13` / "PyTorch and TorchAudio
>   compiled with different CUDA versions". Đây là lý do A.1 gộp **1 lệnh** thay vì cài tách.
>
> **Xung đột transformers ở `.venv-sync`:** `qwen-tts==0.1.1` pin cứng `==4.57.3`, `qwen-asr` pin cứng
> `==4.57.6` → loại trừ nhau. Xử lý bằng `--override transformers==4.57.6` (xem A.1). `qwen-asr` cài
> **trơn** (không `[vllm]`) nên không kéo vllm/torch nặng. `.venv-qwen3asr` riêng cho ASR đầy đủ thì
> dùng `transformers==4.57.6` tự nhiên (chỉ có qwen-asr).

---

## A. THIẾT LẬP LẦN ĐẦU (chưa có lock) — rồi freeze ra lock

Chạy một lần để dựng môi trường sạch, verify, rồi `uv pip freeze` để có lock dùng mãi về sau.

### A.0 — Chụp lớp CUDA gốc của Colab (constraints, làm trước mọi thứ)

`constraints.txt` chỉ **giới hạn version nếu gói được cài**, KHÁC `requirements.txt` (luôn cài).
Ta dùng nó để ép `torch`/`nvidia-*` về đúng bản Colab tiền cài, không cho trôi.

```python
import subprocess
freeze = subprocess.run(["pip","freeze"], capture_output=True, text=True).stdout
keep = [l for l in freeze.splitlines()
        if l.lower().startswith(("torch","nvidia-","triton"))]
open("/content/cuda-base.txt","w").write("\n".join(keep) + "\n")
print("\n".join(keep))
import torch; print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
```

### A.1 — sync-video (TTS) → `sync_lock.txt`

**Gộp TẤT CẢ vào MỘT lệnh resolve** + `--override` + `-c cuda-base.txt`. Một resolve duy nhất giữ cả
họ torch (`torch`/`torchvision`/`torchaudio`) đồng bộ `cu128`; cài tách từng lệnh thì lệnh sau (không
mang `-c cuda-base.txt`) sẽ kéo `torch` lên bản CUDA-13 của PyPI → lệch CUDA với phần còn lại.

```bash
%cd /content/CharenjiZukan
!uv venv .venv-sync       # venv CÔ LẬP (không --system-site-packages: tránh rò TF/keras Colab)

# override: qwen-tts pin CỨNG ==4.57.3, qwen-asr pin CỨNG ==4.57.6 → ép về 4.57.6 (qwen-asr cần,
#           qwen-tts vẫn chạy tốt trên đó). + onnxruntime-gpu==1.26.0 chống libcudart.so.13.
!printf "onnxruntime-gpu==1.26.0\ntransformers==4.57.6\n" > /content/overrides.txt

# MỘT lệnh resolve duy nhất:
#  -c cuda-base.txt           : ghim torch/torchvision/torchaudio về 2.10.0+cu128 (đồng bộ CUDA)
#  --extra-index-url cu128 + unsafe-best-match : lấy wheel +cu128 (chỉ có ở index pytorch)
#  --override                 : bỏ qua pin cứng transformers của qwen-tts, ép cả graph về 4.57.6
#  qwen-asr TRƠN (không [vllm]): chỉ cần Qwen3ForcedAligner cho forced-alignment subtitle
!uv pip install -p .venv-sync/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  -c /content/cuda-base.txt \
  --override /content/overrides.txt \
  -e ".[qwen-tts,openai-provider]" "audio-separator[gpu]" "qwen-asr"

# pin onnxruntime cu12 sau cùng (lặp --override để bước này không resolve lại theo pin gốc)
!uv pip install -p .venv-sync/bin/python --override /content/overrides.txt \
  --reinstall-package onnxruntime-gpu "onnxruntime-gpu==1.26.0"

# VERIFY trước khi chạy cả pipeline. torch/torchaudio PHẢI cùng CUDA 12.8 (nếu lệch → torch đã trôi cu13).
!.venv-sync/bin/python -c "import torch,torchaudio; print('torch',torch.__version__,torch.version.cuda,'| ta',torchaudio.__version__)"
!.venv-sync/bin/python -c "import aiohttp;  print('✅ aiohttp', aiohttp.__version__)"
!.venv-sync/bin/python -c "import qwen_tts; print('✅ qwen_tts OK')"
!.venv-sync/bin/python -c "from qwen_asr import Qwen3ForcedAligner; print('✅ aligner OK')"

# Khi OK → freeze ra lock.
# Lọc bỏ dòng `file:///` (google-colab nội bộ Colab + editable videocolab) — chúng không restore
# được trên runtime khác. videocolab sẽ được gắn lại bằng `-e . --no-deps` ở mục B.
!mkdir -p config/colab
!uv pip freeze -p .venv-sync/bin/python | grep -v "file:///" > config/colab/sync_lock.txt
```

> **Vì sao cần `--override transformers==4.57.6`?** `qwen-tts==0.1.1` pin **cứng** `==4.57.3`, còn
> `qwen-asr` pin **cứng** `==4.57.6` → resolve thường loại trừ nhau ("unsatisfiable"). `--override`
> bắt uv bỏ qua mọi pin transformers và ép cả graph về `4.57.6` (bản qwen-asr cần; qwen-tts vẫn chạy
> tốt vì chênh `.3`→`.6` chỉ là patch). Chọn `4.57.6` chứ KHÔNG phải `4.57.3` — vì qwen-asr cũng pin
> cứng, ép `4.57.3` sẽ làm qwen-asr unsatisfiable.

> **Hệ quả cho restore:** lock chứa `transformers==4.57.6` + `qwen-tts==0.1.1` (metadata qwen-tts vẫn
> đòi `4.57.3`) → env hợp lệ runtime nhưng "phạm luật" graph. Vì vậy lock này **chỉ restore được bằng
> `--no-deps`** (mục B.1), `-r lock` thường sẽ "unsatisfiable".

> **`qwen-asr` ở đây chỉ phục vụ forced-alignment subtitle** (`Qwen3ForcedAligner`). Nếu render config
> để `forced_alignment_subtitle.enabled: false` thì **bỏ `qwen-asr`** khỏi lệnh + bỏ dòng
> `transformers==4.57.6` trong overrides cho env gọn — khi đó transformers giữ `4.57.3` (pin qwen-tts)
> và lock restore được bằng `-r lock` bình thường, không cần `--no-deps`. Phần ASR đầy đủ (full + vllm)
> vẫn ở `.venv-qwen3asr` riêng.

### A.2 — qwen3_asr (ASR) → `asr_lock.txt`

> **ASR khác hẳn TTS — đọc kỹ 3 điểm này:**
>
> 1. **KHÔNG dùng `-c cuda-base.txt`.** Constraint đó ghim `torch==2.10.0+cu128`, trong khi
>    `qwen-asr[vllm]` tự quyết torch của nó → áp vào sẽ "No solution found".
> 2. **KHÔNG dùng `--extra-index-url ...cu128`.** `qwen-asr[vllm]` cần `vllm==0.14.0` — bản này
>    chỉ có trên **PyPI**, không có trên index cu128. Nếu thêm cu128 làm index, uv sẽ chỉ tìm
>    vllm ở đó (chống dependency-confusion) và báo "No solution found". Để mặc định PyPI.
> 3. **Thứ tự bắt buộc:** cài `qwen-asr[vllm]` **TRƯỚC** (nó kéo đúng torch), flash-attn **SAU**
>    (dùng lại torch đã có). Nếu cài flash-attn vào venv rỗng, uv tự kéo `torch` mới nhất
>    (vd 2.12.1 + CUDA 13) → hỏng. Flash-attn cho ASR là bản `torch2.9` (khác `torch2.10` ở A.1).

```bash
%cd /content/CharenjiZukan
!rm -rf .venv-qwen3asr          # tạo lại sạch nếu venv cũ đã nhiễm torch sai
!uv venv .venv-qwen3asr

# qwen-asr[vllm] TRƯỚC, flash-attn (torch 2.9) SAU
!uv pip install -p .venv-qwen3asr/bin/python "qwen-asr[vllm]"
!uv pip install -p .venv-qwen3asr/bin/python \
  https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl

!.venv-qwen3asr/bin/python -c "import qwen_asr; print('✅ qwen_asr OK')"

# tạo thư mục lock (repo clone chưa có) rồi freeze (lọc dòng file:/// như A.1)
!mkdir -p config/colab
!uv pip freeze -p .venv-qwen3asr/bin/python | grep -v "file:///" > config/colab/asr_lock.txt
```

### A.3 — video-ocr (OCR) → `ocr_lock.txt`

```bash
!uv venv .venv-ocr
# -e .[ocr] = opencv-python-headless + Pillow + torch + torchvision + transformers + accelerate
#             + qwen-vl-utils (xem extra `ocr` trong pyproject.toml).
# Bộ index cu128 + -c cuda-base.txt (Quy luật B): OCR cũng dính torch-drift cu13; torchvision được
# ghim 0.25.0+cu128 theo cuda-base. torchvision là gói transformers Qwen3VLVideoProcessor cần NGẦM.
!uv pip install -p .venv-ocr/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  -c /content/cuda-base.txt \
  -e ".[ocr]"
# (cài Qwen3-VL OCR — xem colab-guide §2.9/§2.10)

# VERIFY import top-level + đồng bộ CUDA trước khi freeze
!.venv-ocr/bin/python -c "import cv2, PIL, torchvision; print('✅ cv2', cv2.__version__, '| tv', torchvision.__version__)"
!.venv-ocr/bin/python -c "import torch; print('torch', torch.__version__, torch.version.cuda)"   # kỳ vọng cu 12.8
!mkdir -p config/colab
!uv pip freeze -p .venv-ocr/bin/python | grep -v "file:///" > config/colab/ocr_lock.txt
```

---

## B. DÙNG HẰNG NGÀY (đã có lock) — chỉ restore, KHÔNG cài lại extras

> **Khi nào dùng mục B?** Mỗi lần Colab runtime khởi động lại (mọi thứ trong `/content/` biến
> mất). Lock file nằm trong repo tại `config/colab/`, lấy về bằng `git pull` → chỉ cần restore,
> không cần cài lại từ đầu, không cần freeze lại.

> **QUAN TRỌNG:** khi restore từ lock, **không** chạy lại `uv pip install -e .[qwen-tts]`,
> `audio-separator[gpu]`, `qwen-asr`... Lock đã chứa sẵn toàn bộ. Cài lại = resolve lại = trôi
> version. Lock thay thế tất cả các lệnh cài đó.

Ba lưu ý chung khi restore (đã áp vào cả 3 lệnh dưới):

- `torch==...+cu128` là wheel local-version, không có trên PyPI → **bắt buộc**
  `--extra-index-url https://download.pytorch.org/whl/cu128`.
- Vì lock trộn 2 nguồn (gói `+cu128` ở index pytorch, phần lớn còn lại như `certifi` ở PyPI) →
  **bắt buộc** `--index-strategy unsafe-best-match`, nếu không uv chỉ tra mỗi gói ở index đầu
  tiên và báo "No solution found" cho các gói chỉ có trên PyPI.
- Dòng `videocolab` (editable) trong lock nên bỏ qua; gắn lại entry point bằng `-e . --no-deps`.

### B.1 — Cần sync-video

> **Bắt buộc `--no-deps`.** Lock có `transformers==4.57.6` + `qwen-tts==0.1.1` (metadata qwen-tts
> đòi `4.57.3`). Nếu để uv resolve dep (bỏ `--no-deps`) → "unsatisfiable". `--no-deps` = cài đúng
> các bản đã ghim trong lock, không suy diễn lại dep. Lock là closure đầy đủ (freeze) nên an toàn.
> (Trường hợp env không có qwen-asr — xem ghi chú cuối A.1 — thì lock không mâu thuẫn, bỏ `--no-deps` được.)

```bash
%cd /content/CharenjiZukan
!git pull                                    # lấy lock mới nhất từ repo
!uv venv .venv-sync
!uv pip install -p .venv-sync/bin/python \
  --no-deps \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  -r config/colab/sync_lock.txt
!uv pip install -p .venv-sync/bin/python -e . --no-deps
# sox: cho qwen_tts (resample audio nội bộ engine).
# rubberband-cli: cho time-stretch giữ pitch của pipeline; thiếu nó sẽ tụt xuống FFmpeg atempo.
!apt-get -y install sox libsox-fmt-all rubberband-cli
!.venv-sync/bin/sync-video --task-file /content/video_sync_tasks.json --tts-provider qwen ...
```

### B.2 — Cần ASR

```bash
%cd /content/CharenjiZukan
!git pull
!uv venv .venv-qwen3asr
!uv pip install -p .venv-qwen3asr/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  -r config/colab/asr_lock.txt
!.venv-qwen3asr/bin/python cli/qwen3_asr.py ...
```

### B.3 — Cần OCR

```bash
%cd /content/CharenjiZukan
!git pull
!uv venv .venv-ocr
!uv pip install -p .venv-ocr/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  -r config/colab/ocr_lock.txt
!.venv-ocr/bin/python -m cli.video_ocr ...
```

---

## Câu hỏi thường gặp

**`constraints.txt` vs `requirements.txt`?** `requirements` = "cài các gói này" (`-r`).
`constraints` = "nếu gói này được cài thì phải đúng version này, không tự cài" (`-c`). Khi đã có
**lock đầy đủ** (freeze) thì không cần constraints riêng nữa — lock đã ghim cả `torch`/`nvidia`/
`onnxruntime`. `constraints.txt` chỉ dùng ở bước thiết lập lần đầu (mục A) để chống trôi torch.

**Nếu Colab đổi base torch (vd lên 2.11)?** Đừng hard-code torch trong `pyproject.toml`. Mục A.0
chụp lại torch THẬT của Colab mỗi runtime; nếu base đổi thì freeze lại lock mới. Khi đó kiểm tra
wheel `flash-attn` (đang là bản `torch2.10`) còn khớp không — nếu không, đổi sang wheel cùng repo
`mjun0812` khớp torch mới.

**`sync-video` báo `ModuleNotFoundError: No module named 'qwen_asr'` / "Forced alignment thất bại"?**
Tính năng `forced_alignment_subtitle` cần `Qwen3ForcedAligner` (gói `qwen-asr`) ngay trong
`.venv-sync`. Pipeline **không chết** (mặc định fallback remap SRT) nhưng mất căn chỉnh timestamp
word-level. Hai cách:
- **Cần forced alignment:** cài `qwen-asr` (đã có sẵn trong lệnh A.1, kèm `--override
  transformers==4.57.6`), rồi freeze lại `sync_lock.txt`.
- **Không cần:** đặt `forced_alignment_subtitle.enabled: false` trong render config; lúc đó bỏ
  `qwen-asr` khỏi lệnh A.1 + bỏ dòng `transformers==4.57.6` trong overrides cho env gọn.

**`PyTorch and TorchAudio were compiled with different CUDA versions` (hoặc `torch.version.cuda`
ra `13.0`)?** Vi phạm **Quy luật B**: một lệnh `uv pip install` đụng `torch` (trực tiếp hay
transitive qua `audio-separator`/`transformers`/`qwen-asr`) đã **thiếu** `-c /content/cuda-base.txt`
+ index cu128 → uv nâng `torch` lên bản CUDA-13 của PyPI, lệch với `torchaudio`/`torchvision` cu128.
Khắc phục: dựng lại venv bằng **MỘT lệnh gộp** mang đủ `-c cuda-base.txt` + `--extra-index-url
...cu128` + `--index-strategy unsafe-best-match` (A.1 / A.3). Luôn verify
`torch.version.cuda == 12.8` trước khi freeze.

**`ModuleNotFoundError: No module named 'cv2'` / `Qwen3VLVideoProcessor requires the Torchvision
library` (OCR)?** Vi phạm **Quy luật A**: `.venv-ocr` thiếu gói Colab cài ngầm. `cv2` (opencv) là
import top-level của `video_subtitle_extractor`; `torchvision` là gói `transformers` cần **ngầm** cho
Qwen3-VL video processor (code mình không import trực tiếp nên dễ sót). Cả hai đã nằm trong extra
`ocr` — cài `.venv-ocr` theo mục A.3 (`-e ".[ocr]"`) rồi freeze lại `ocr_lock.txt`.
