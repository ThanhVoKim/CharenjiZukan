# Project Journal

## 2026-06-23: Xóa rubberband khỏi TTS speed-up — dùng FFmpeg atempo thuần túy

### Bối cảnh

Provider `qwen_custom` + `speed_scale` đã được tích hợp (session 2026-06-22). Khi kiểm tra
lại, `speed_rate.py` dùng rubberband (pitch-preserving) làm backend ưu tiên cho `speedup_to_factor()`
và `SpeedRate._speedup_all()`, fallback sang FFmpeg atempo nếu binary không có. Người dùng
quyết định **loại bỏ hoàn toàn rubberband khỏi TTS speed-up** và dùng atempo làm backend duy nhất.

### Phân tích phạm vi

Rubberband được dùng ở hai nơi **độc lập**:
1. **`speed_rate.py`** — TTS path (`speedup_to_factor`, `_speedup_audio`, `SpeedRate._speedup_all`).
2. **`utils/media_utils.py` + `cli/media_speed.py`** — general media audio stretching CLI.

→ Chỉ xóa ở nơi 1; giữ nguyên nơi 2 (tính năng độc lập). `pyrubberband` trong `pyproject.toml`
vẫn giữ vì `utils/media_utils.py` còn dùng.

### Thay đổi

**`speed_rate.py`**:
- Xóa toàn bộ detection block lúc import: `_RUBBERBAND_BIN`, `_has_rubberband_binary()`,
  `_RUBBERBAND_AVAILABLE`, `_PYRUBBERBAND_INSTALLED` và các print/log liên quan.
- Xóa `_speedup_rubberband()` (pyrubberband wrapper).
- Xóa `_speedup_ffmpeg()` (alias backward compat đã không dùng).
- `_speedup_audio()` rút gọn thành thin wrapper gọi thẳng `_speedup_with_atempo()`.
- Cập nhật docstring module-level và comment block: thay `rubberband/atempo` → `atempo`.

Không thay đổi bất kỳ logic nào khác (atempo chain, `speedup_to_factor`, `SpeedRate`).

### Kết quả test

**706 passed, 28 skipped** — toàn bộ test suite xanh (tăng so với 695 trước do fix pre-existing
test `test_engine_run_with_strip_silence` trong session 2026-06-22 đã được tính vào lần chạy này).

---

## 2026-06-23: Fix `forced_alignment_subtitle` — dấu phân cách số bị mất (500.000 → 500000.000)

### Bối cảnh

Subtitle gốc `"...Preisgeld von 500.000 Dollar."` (dấu chấm = phân cách hàng nghìn
kiểu Đức) bị output thành `"...500000.000 Dollar."`. Lỗi xảy ra ở bước forced
alignment subtitle của `sync-video`.

### Nguyên nhân (root cause)

`utils/asr_subtitle_utils.py::merge_punctuation`. Qwen3-ForcedAligner normalize số
`"500.000"` → token phẳng `"500000"`. Trong vòng partial-match, code chỉ khớp được
`"500"` với `"500.000"` của full_text (dừng tại `.`) nhưng lại:
1. Output **toàn bộ** token normalized `"500000"`,
2. Nuốt `.` của full_text vào trailing punctuation → `"500000."`,
3. Để lại `"000"` trong full_text → bị token kế tiếp (`Dollar`) lấy làm prefix.
→ Ghép lại thành `"500000.000 Dollar."`.

Code đã có cơ chế xử lý y hệt cho dash-compound (`47-round` → `47round`) nhưng chưa
có cho dấu phân cách số.

### Thay đổi

- **`utils/asr_subtitle_utils.py`**:
  - Thêm hằng `NUMERIC_SEPARATOR_CHARS = set(".,，．")`.
  - Thêm `_is_numeric_separator_at()` (dấu `./,` nằm giữa 2 chữ số) và
    `_match_numeric_separator_remainder()` (khôi phục text gốc, song song với
    `_match_dash_compound_remainder`). Hỗ trợ nhiều dấu phân cách trong 1 số
    (`1.234.567`, `1,234.5`).
  - Gọi nhánh numeric trong `merge_punctuation` khi nhánh dash không khớp.
- **`tests/utils/test_asr_subtitle_utils.py`**: thêm regression test 500.000 +
  parametrize cho `3.5`, `1,234`, `1.234.567`, `1,234.5`, và 2 ca âm tính (dấu chấm
  cuối câu / dấu phẩy theo sau khoảng trắng vẫn là dấu câu thường).

### Trạng thái

Hoàn thành. `tests/utils/test_asr_subtitle_utils.py` (38) +
`tests/sync_engine/test_forced_alignment_subtitle.py` + `tests/cli/test_qwen3_asr.py`
(tổng 64) đều pass. Không cần cập nhật `test_matrix.yaml` (tái dùng entry sẵn có cho
2 file test này).

## 2026-06-22: Provider `qwen_custom` + `speed_scale` toàn cục + bỏ `--slow-cap`

### Bối cảnh

Checkpoint fine-tune Qwen3-TTS giọng Karlsson DE (từ session trước) dùng
`generate_custom_voice(speaker="karlsson_de")` — khác với engine `qwen` hiện tại vốn gọi
`generate_voice_clone(ref_audio=, ref_text=)`. Cần tích hợp ngược vào pipeline dubbing.
Đồng thời đổi chiến lược stretch video: thay vì dùng `--slow-cap` giới hạn mức kéo chậm
(kéo theo nén audio khi TTS quá dài), chuyển sang **audio `speed_scale` làm đòn bẩy duy nhất**
và video stretch tự do không giới hạn.

### Thay đổi kiến trúc

**`tts/base.py`** — thêm `apply_speed_scale()` vào `BaseTTSEngine`: tăng tốc các clip đã
sinh theo `self.speed_scale` (rubberband giữ pitch, fallback atempo). Voicevox không gọi
(đã có `speedScale` native). Engine edge/qwen/qwen_custom gọi cuối `run()`.

**`speed_rate.py`** — thêm public `speedup_to_factor(wav_path, speed_scale)`: quy đổi
factor → `target_ms` rồi tái dùng `_speedup_audio` nội bộ. `speed_scale <= 1.0` → no-op.

**`tts/qwen.py`** — refactor:
- Tách `DEFAULT_QWEN_GEN_KWARGS` thành hằng module-level (dùng chung với `qwen_custom`).
- Thêm `@staticmethod _postprocess(wav, sr, clean_tail, pre, post, fade_ms, top_db)`: gói
  logic clean_tail-vs-pad để tái dùng.
- Thêm `speed_scale: float = 1.0`, gọi `apply_speed_scale()` cuối `run()`.

**`tts/qwen_custom.py`** (FILE MỚI) — `QwenCustomTTSEngine`:
- Gọi `model.generate_custom_voice(text=, speaker=, language=)`.
- `_resolve_model_path()`: ưu tiên `local_ckpt`; nếu chưa có thì copy `drive_ckpt→local_ckpt`
  (Drive FUSE chậm + dễ lỗi I/O với file ~3.5GB khi `from_pretrained` mmap).
- Tái dùng `QwenTTSEngine._postprocess` + `DEFAULT_QWEN_GEN_KWARGS`.
- Gọi `apply_speed_scale()` cuối `run()`.

**`tts/edgetts.py`** — thêm `speed_scale: float = 1.0`, gọi `apply_speed_scale()` cuối `run()`.

**`cli/sync_video.py`** — flow stretch mới:
- **Xóa `--slow-cap`** khỏi argparse.
- `compute_speeds(..., no_cap=True)` cho **mọi provider** → video stretch tự do, audio không
  bao giờ bị nén. `speed_scale` trong YAML là đòn bẩy duy nhất điều chỉnh tốc độ audio.
- Thêm nhánh `qwen_custom`, import `QwenCustomTTSEngine`.
- Nhánh `edge`: thêm `speed_scale=edge_cfg.get("speed_scale", 1.0)`.

**`cli/tts.py`** — thêm nhánh `qwen_custom` + cập nhật `choices`.

**`config/tts_config.yaml`** — thêm `speed_scale: 1.0` vào `edge` và `qwen`; thêm section
`qwen_custom` với đủ tham số (drive_ckpt, local_ckpt, speaker, language, clean_tail, speed_scale…).

### Fix đi kèm

- `tests/tts/test_tts_edgetts.py` — sửa assertion sai từ trước: `test_engine_run_with_strip_silence`
  kỳ vọng 1500ms nhưng `strip_audio_silence` cắt CẢ HAI ĐẦU nên kết quả đúng là 1000ms
  (`start_ms = max(0, 500-0) = 500`, không phải 0 như comment cũ nói).
- `tests/sync_engine/test_note_overlay_layout.py` — thêm mock `tts.qwen_custom` vào block
  `sys.modules` (pipeline mới import module này nên test cần stub).
- `tests/sync_engine/test_sync_video_pipeline.py` — xóa `slow_cap=0.5` thừa khỏi `Namespace`.
- `docs/colab-guide.md` — xóa `--slow-cap` khỏi ví dụ lệnh + bảng tham số; thêm `qwen_custom`
  vào danh sách provider.

### Kết quả test

**699 passed, 28 skipped** — toàn bộ test suite xanh sau thay đổi.

### Pending

- Chạy trên Colab với checkpoint Karlsson thật để xác nhận luồng copy Drive→local hoạt động.
- Cân nhắc `speed_scale` phù hợp cho giọng Karlsson DE (bắt đầu từ 1.0, tăng nếu video kéo quá chậm).

---

## 2026-06-20: Thêm script fine-tune Qwen3-TTS giọng Đức Karlsson (HUI Audio Corpus)

### Bối cảnh

Dự án cần giọng TTS tiếng Đức chất lượng cao (khán giả Đức). Quyết định fine-tune
`Qwen/Qwen3-TTS-12Hz-1.7B-Base` với dataset HUI Audio Corpus German, giọng Karlsson (clean, ~29h).

### Quyết định kiến trúc

- **Full SFT, single-speaker** via script chính thức `QwenLM/Qwen3-TTS/finetuning/` (không LoRA, không wrapper).
- **Nhãn = transcript ground-truth HUI** — tuyệt đối không ASR phiên âm lại (làm bẩn nhãn).
- **Drive vs local**: zip backup + manifest + checkpoint trên Drive (bền); wav giải nén trên local `/content` (nhanh, tạm). Không lưu thư mục wav lên Drive.
- Chọn checkpoint bằng WER + speaker-similarity + nghe thử, không chỉ theo loss.

### File tạo mới

`qwen3tts_finetune_de_karlsson.py` — đặt ở gốc repo, nội dung độc lập với codebase. Cấu trúc:
- Cell 1: env + mount Drive + định nghĩa đường dẫn (Drive + local)
- Cell 2: cài đặt + clone repo + in `--help` để xác minh tên cờ CLI
- Cell 3.0 EXPLORE: tải zip về Drive, giải nén vào local, khảo sát cấu trúc dataset
- Cell 3: đảm bảo working copy local sẵn (dò zip backup Drive → giải nén vào local mỗi session)
- Cell 4: dựng manifest từ transcript HUI + lọc + chọn ref_audio + tách val
- Cell 5: trích audio codes (`prepare_data.py`)
- Cell 6: train full SFT, checkpoint ra Drive, resumable
- Cell 7: eval WER + speaker-sim + nghe thử để chọn checkpoint
- Cell 8: export checkpoint tốt nhất ra Drive

### Tích hợp ngược (chưa làm)

Checkpoint SFT dùng `generate_custom_voice(speaker="karlsson_de")`, khác với engine hiện tại
dùng `generate_voice_clone(...)`. Khi mang checkpoint về sẽ cần chỉnh lời gọi trong `tts/qwen.py`.

### Pending

- Chạy Cell 3.0 EXPLORE trên Colab để xác nhận URL tải, định dạng metadata, sample rate thực tế.
- Đối chiếu tên cờ CLI từ `--help` trước khi chạy Cell 5/6.

---

## 2026-06-20: Colab — 2 quy luật venv cô lập + sửa override 4.57.6 + fix `.venv-ocr` thiếu torchvision

### Bối cảnh

Sau chuỗi lỗi "hôm qua chạy hôm nay không" trên mọi venv cô lập, gom lại rút **2 quy luật** dùng
chung cho cả `.venv-sync` / `.venv-qwen3asr` / `.venv-ocr`:

- **Quy luật A — gói "Colab cài ngầm":** code (hoặc *thư viện khác*) import gói Colab có sẵn nhưng
  `pyproject.toml` không khai báo → venv cô lập thiếu. Gồm cả gói *cần ngầm*: `qwen_tts`→`torchaudio`,
  `transformers VL`→`torchvision`, `edgetts`→`aiohttp`, `video_subtitle_extractor`→`cv2`/`PIL`.
- **Quy luật B — đồng bộ CUDA:** mọi gói họ-torch (`torch`/`torchvision`/`torchaudio`) **và**
  `onnxruntime-gpu` phải cùng `cu128`. **Mọi** lệnh `uv pip install` đụng chúng phải mang
  `-c /content/cuda-base.txt` + `--extra-index-url ...cu128` + `--index-strategy unsafe-best-match`.

### Sửa `.venv-sync` (đính chính entry 2026-06-19 ngay dưới)

- **Override transformers đúng là `4.57.6`, KHÔNG phải `4.57.3`.** Entry dưới ghi `4.57.3` là sai:
  `qwen-asr` cũng pin **cứng** `==4.57.6`, nên ép `4.57.3` làm qwen-asr unsatisfiable. Ép `4.57.6`
  (qwen-asr cần; qwen-tts vẫn chạy tốt). Override 2 dòng: `onnxruntime-gpu==1.26.0` + `transformers==4.57.6`.
- **Lỗi torch-drift CUDA-13:** cài tách (`audio-separator`, `qwen-asr` không kèm `-c cuda-base.txt`)
  làm `torch` trôi lên bản CUDA-13 PyPI → `RuntimeError: PyTorch and TorchAudio compiled with
  different CUDA versions` (torch cu13 vs torchaudio cu128). → Chốt dùng **1 lệnh gộp** mang đủ
  `-c cuda-base.txt` + index cu128 (giữ cả họ torch đồng bộ trong một resolve).
- Lock vẫn "phạm luật" graph (qwen-tts metadata đòi 4.57.3) → restore **bắt buộc `--no-deps`**.

### Fix `.venv-ocr` (áp Quy luật A + B)

- Lỗi `ModuleNotFoundError: cv2`, rồi `Qwen3VLVideoProcessor requires the Torchvision library`. `cv2`
  là import top-level của `video_subtitle_extractor`; `torchvision` là gói `transformers` cần **ngầm**
  cho Qwen3-VL (code không import trực tiếp nên dễ sót — y hệt `torchaudio` của `qwen_tts`).
- **Thêm extra `ocr` vào `pyproject.toml`:** opencv-python-headless + Pillow + torch + **torchvision**
  + transformers + accelerate + qwen-vl-utils + matplotlib. `torch` để unversioned (ghim bởi
  cuda-base lúc cài). Không thêm torchcodec (OCR đưa frame ảnh, không decode video).
- **A.3 thêm bộ index cu128 + `-c cuda-base.txt`** và dùng `-e ".[ocr]"`; verify `import cv2,
  torchvision` + `torch.version.cuda==12.8` trước khi freeze `ocr_lock.txt`.

Chi tiết: **`docs/colab-setup.md`** mục A.1 (Option A), A.3, và 2 quy luật ở phần "Nguyên tắc".

## 2026-06-19: Colab — Đưa lại `qwen-asr` vào `.venv-sync` cho forced alignment (override transformers)

### Triệu chứng

`.venv-sync/bin/sync-video` chạy được nhưng log:
`WARNING - Forced alignment thất bại ... ModuleNotFoundError: No module named 'qwen_asr'`.
Pipeline không chết (fail_policy mặc định = warn → fallback remap SRT), nhưng mất tính năng
forced-alignment subtitle (timestamp word-level). Nguyên do: entry trước đã **bỏ `qwen-asr`** khỏi
`.venv-sync`, trong khi `Qwen3ForcedAligner` lại được gọi trong tiến trình sync-video.

### Quyết định / thay đổi workflow (đảo lại một phần entry "bỏ qwen-asr" bên dưới)

- **Đưa lại `qwen-asr` (TRƠN, không `[vllm]`) vào `.venv-sync`.** Chỉ cần `Qwen3ForcedAligner`,
  không cần vllm/torch nặng của bản ASR đầy đủ.
- **Xung đột `transformers` quay lại** (`qwen-tts==4.57.3` vs `qwen-asr==4.57.6`, cả hai `==` →
  loại trừ nhau bằng resolve thường → "your requirements are unsatisfiable"). Xử lý bằng
  **`--override transformers==4.57.3`** (ép về bản qwen-tts; chênh `.3`→`.6` chỉ là patch nên
  `Qwen3ForcedAligner` vẫn chạy). Override giờ gồm 2 dòng: `onnxruntime-gpu==1.26.0` +
  `transformers==4.57.3`.
- **Gộp cả 3 vào 1 lệnh resolve** (`-e ".[qwen-tts,...]" "audio-separator[gpu]" "qwen-asr"`); cài lẻ
  từng lệnh sẽ resolve lại từ đầu mỗi lần → đảo/trôi version. `--override` phải lặp lại ở **mọi**
  lệnh `uv pip install` đụng các gói này (kể cả bước `--reinstall-package onnxruntime-gpu`).
- **Tùy chọn gọn:** nếu render config để `forced_alignment_subtitle.enabled: false` thì bỏ
  `qwen-asr` + bỏ override transformers → quay về trạng thái entry bên dưới (qwen-tts tự dùng 4.57.3).
- Phân biệt: `.venv-qwen3asr` vẫn là `qwen-asr[vllm]` đầy đủ cho CLI `qwen3_asr`; `.venv-sync` chỉ
  mượn phần aligner.

Chi tiết quy trình: **`docs/colab-setup.md`** mục A.1.

## 2026-06-19: Colab — `sync-video` chuyển từ `--system` sang venv riêng `.venv-sync`

### Triệu chứng

Restore lock `sync_lock.txt` (freeze từ `--system`) chết khi resolve:
`cudf-cu12==26.2.1 → cuda-toolkit[nvcc]==12.* → nvidia-cuda-nvcc-cu12==12.8.93`, nhưng lock ghim
`nvidia-cuda-nvcc-cu12==12.5.82` → "your requirements are unsatisfiable".

### Nguyên nhân gốc

`uv pip freeze --system` **chụp luôn toàn bộ môi trường Colab**, gồm cả bộ RAPIDS/CUDA tiền cài
(`cudf-cu12`, `cuda-toolkit`, `nvidia-cuda-nvcc-cu12`...). Lúc freeze chỉ là "đang cài" nên không
ai kiểm tra; lúc restore, `uv pip install -r` resolve lại và phát hiện các pin này **mâu thuẫn nội
bộ** với nhau. ASR/OCR không dính vì chúng là **venv** — `uv pip freeze` trên venv chỉ liệt kê gói
cài trong venv, không nhìn xuyên xuống `--system-site-packages` (xác nhận: astral-sh/uv#2500).

### Quyết định / thay đổi workflow

- `sync-video` **bỏ `--system`**, dùng venv riêng **CÔ LẬP** `.venv-sync` (KHÔNG
  `--system-site-packages`) — đồng nhất với `.venv-qwen3asr` và `.venv-ocr`. Lock sinh ra sạch
  (không còn cudf/cuda-toolkit). → **Thay thế** ghi chú "`uv pip install --system`" ở entry
  2026-06-19 (onnxruntime) bên dưới.
- **Vì sao KHÔNG `--system-site-packages`:** dù để cờ này, `uv` vẫn cài lại torch riêng vào venv
  (uv#2500) → không tiết kiệm gì, mà còn rò **TensorFlow/keras Colab** vào venv → `transformers`
  4.57.6 dò nhầm backend TF (lệch version) → `ImportError: cannot import name 'AutoProcessor' from
  'transformers'` (dấu hiệu: log nạp TF cuFFT/cuDNN xuất hiện khi `import qwen_tts`). Venv cô lập:
  `is_tf_available()` = False → đường torch-only → import sạch. Cái giá: mỗi venv tự tải torch ~2GB.
- Lệnh gọi đổi thành `.venv-sync/bin/sync-video`. Đã xác minh an toàn: sync-video spawn tiến trình
  con bằng `multiprocessing.get_context('spawn')` ([cli/sync_video.py:756]), không giả định Python
  hệ thống, không hardcode interpreter.
- **Bỏ `qwen-asr` khỏi env sync-video.** Trước đây nó bị nhét chung → kéo `transformers==4.57.6`
  đụng `qwen-tts==4.57.3` → phải `--override`. Venv cô lập siết chặt nên báo "unsatisfiable". ASR
  đã có `.venv-qwen3asr` riêng → sync-video chỉ cần qwen-tts (dùng đúng `transformers==4.57.3`,
  bỏ luôn override transformers; chỉ còn override `onnxruntime-gpu==1.26.0`).
- Restore lock 2-nguồn (gói `+cu128` ở index pytorch, phần còn lại ở PyPI) bắt buộc
  `--index-strategy unsafe-best-match`, nếu không uv chỉ tra index đầu tiên → fail gói chỉ có ở PyPI.
- `pyproject.toml`: `pyrubberband` đã ở base deps → bỏ khỏi dòng lệnh cài (thừa). `all-providers`
  hiện trùng `openai-provider` (cả hai chỉ `openai`) vì `vertexai-provider` rỗng — alias, giữ được.

Chi tiết quy trình: **`docs/colab-setup.md`**.

## 2026-06-19: Colab — Lỗi `libcudart.so.13` do `onnxruntime-gpu` trôi version

### Triệu chứng

`sync-video --tts-provider qwen` chết với:
`ERROR:qwen_tts: ❌ Thiếu thư viện cho QwenTTS: libcudart.so.13: cannot open shared object file`.
Kiểu lỗi "hôm qua chạy, hôm nay không" — script không đổi.

### Chẩn đoán (loại trừ từng lớp)

- `torch 2.10.0+cu128` import OK; `flash_attn` import OK; trên máy **chỉ có `libcudart.so.12`**
  (CUDA 12.8), không có `.so.13`; **không** gói `*-cu13` nào trong `pip list`. → Không phải
  torch/CUDA.
- `grep -rl 'libcudart.so.13'` site-packages + traceback đầy đủ của `import qwen_tts` chỉ thẳng:
  `qwen_tts → core/tokenizer_25hz/vq/speech_vq.py → import onnxruntime →
  onnxruntime/capi/onnxruntime_pybind11_state.so → libcudart.so.13`.

### Nguyên nhân gốc

**`onnxruntime-gpu` tự nhảy `1.26.0` → `1.27.0` qua đêm** (flow không khoá version, `uv`/`pip`
nhặt bản mới nhất). Bản `1.27.0` build cho **CUDA 13** (`libcudart.so.13`), trong khi Colab là
CUDA 12.8. Không phải Colab đổi, không phải torch, không phải Qwen3-TTS đổi version (qwen-tts vẫn
0.1.1). Bản chạy được: **`onnxruntime-gpu==1.26.0`** (CUDA 12).

### Bài học / phòng ngừa

- Stack TTS trên Colab phải **khoá version**. Hai pin tối thiểu hiện tại: `transformers==4.57.6`
  (giải xung đột `qwen-tts==4.57.3` vs `qwen-asr==4.57.6` bằng `--override`) và
  `onnxruntime-gpu==1.26.0` (CUDA 12).
- `uv pip install --system` để cài vào python Colab (giữ torch tiền cài), **không** dùng `uv run`
  (nó sync lại lockfile → kéo torch/onnxruntime mới). `--reinstall-package onnxruntime-gpu` cuối
  cùng để bản CUDA-12 thắng trên đĩa (vì cả `onnxruntime` lẫn `onnxruntime-gpu` ghi đè cùng thư mục).
- Giải pháp bền vững: **1 môi trường = 1 file lock** (`uv pip freeze`). Quy trình đầy đủ cho cả 3
  CLI (`sync-video` / `qwen3_asr` / `video-ocr`) đã viết ở **`docs/colab-setup.md`**.
- KHÔNG hard-code `torch` trong `pyproject.toml` (Colab có thể đổi base torch 2.10↔2.11) — để
  `cuda-base.txt` chụp torch thật mỗi runtime quyết định.

## 2026-06-17: srt_batch — Refactor I/O sang định dạng Numbered-Line

### Vấn đề

Payload gửi LLM mỗi batch là SRT đầy đủ (`N\nTIMESTAMP\nTEXT` × block, nối `\n\n`). Timestamp
LLM KHÔNG dùng (map kết quả theo vị trí, timestamp cuối lấy từ bản gốc) → token thừa + "rác" làm
loãng mạch văn.

### Quyết định

Đổi I/O batch sang **numbered-line**: mỗi block một dòng `"N. text"`, N = `item['line']` (chỉ số
SRT toàn cục, duy nhất). Bỏ timestamp khỏi cả input lẫn output. N giữ vai trò **neo chống lệch
hàng**: map theo SỐ (không theo vị trí mù).

- `runner.py` (`_process_one_batch`): builder mới `f"{item['line']}. {item['text'].strip()}"`, nối `\n`.
- `batching.py`: thêm `parse_numbered_lines()` (regex `^\s*(\d+)\.\s?(.*)$`, orphan-line nối vào
  entry đang mở, số lặp → raise). `merge_translated_batch()` map theo line id; integrity: tập số
  parse được PHẢI khớp ĐÚNG tập `item['line']` của batch (thiếu/thừa → BatchIntegrityError). Giữ
  chữ ký hàm `(translated_str, original_batch) -> list[dict]` nên không đụng caller/validator.
  KHÔNG còn dùng `parse_srt` để parse output LLM.
- Lỗi batch sau retry: GIỮ NGUYÊN cả batch (all-or-nothing, như cũ — quyết định của user).

### Prompt (đồng bộ hợp đồng LLM)

Sửa cả 3: `prompts/translation/srt_translate_ja.txt`, `srt_translate.txt`,
`prompts/llm_tasks/punctuation_restoration.txt`. Bỏ "Copy Index/Timestamp", mô tả input `N. text`
(N là id ổn định), yêu cầu output `N. <kết quả>` giữ nguyên tập id, không gộp/tách dòng, không
chèn timestamp, không xuống dòng giữa entry. Thay ví dụ SRT 3-dòng bằng ví dụ numbered-line. Bản
Nhật giữ Fragment-Mapping (SOV); punctuation giữ rule #6 (neo "dấu cuối mỗi dòng").

### Test (docs/testing-guide.md — Domain-Based, TestLayer*)

- MỚI `tests/llm_ai/test_srt_batch_batching.py` (Layer 1): parse_numbered_lines (orphan line,
  digit giữa câu, số lặp) + merge map-by-id (đảo thứ tự vẫn đúng, thiếu/thừa raise, id không bắt
  đầu từ 1, không mutate batch gốc). Thêm entry `test_matrix.yaml` (tags: unit).
- Sửa `test_srt_batch_concurrency.py` (FakeProvider echo theo dòng `N. text`),
  `test_punctuate_srt.py` (mock response numbered-line; mismatch = thiếu id),
  `test_translation_providers.py` (FakeProvider Layer 2 trả numbered-line).
- Input SRT thật (file `.srt`) vẫn parse bằng `parse_srt` như cũ — chỉ payload/response LLM đổi.

### Kết quả

`tests/translation/ tests/llm_ai/` (trừ api/probe): 81 passed, 3 skipped. Bộ liên quan
(batching + concurrency + punctuate + translation L2): 49 passed.

### Không đụng

`<FULL_SOURCE_CONTEXT>` (text-trần, đã cache provider), `parse_srt`/`segments_to_srt`, timestamp,
`align-srt`/`resegment_srt_by_sentence`.

---

## 2026-06-17: punctuate-srt — Rule #6 clause-chaining cho TTS tự nhiên

### Vấn đề

Block OCR sau OCR rất ngắn (mỗi caption 1 mảnh). Bước gom câu `resegment_srt_by_sentence`
(`cli/align_srt.py`) chỉ cắt block khi gặp ký tự kết thúc câu `DEFAULT_GRAMMAR_SPLIT_CHARS =
".!?:。！？：；"` — **dấu phẩy KHÔNG cắt**. Nhưng LLM ở `punctuate-srt` lại có xu hướng đóng mỗi
mệnh đề ngắn bằng `。`, khiến resegment cắt vụn → TTS đọc giật từng câu, mất tự nhiên.

### Quyết định

Đòn bẩy đúng là **prompt punctuation**, không phải sửa logic gom câu. Thêm Rule #6 vào
`prompts/llm_tasks/punctuation_restoration.txt`: hướng dẫn LLM đặt `。！？` đúng nơi ý trọn vẹn,
nối các mệnh đề cùng một ý bằng dấu phẩy (，/、). Nhờ vậy resegment gộp các mệnh đề nối-phẩy thành
1 block dài → TTS đọc liền mạch. Rule vẫn tuân Rule #1 (chỉ chèn dấu, không đổi chữ) nên validator
`_validate_content_preserved` không bị vi phạm.

**Sửa lại Rule #6 (cùng ngày):** Bản đầu có guardrail "~3 mệnh đề/câu" và biện minh "tránh tường
chữ trên màn hình" — cả hai SAI. File này KHÔNG dùng làm subtitle hiển thị (chỉ feed TTS) nên không
có ràng buộc độ-dài-đọc; và cap theo số mệnh đề là tùy tiện, ép đóng câu khi ý chưa trọn → flow
gượng ép, đổi "giật vì ngắn" thành "đứt vì cắt máy móc". Ranh giới câu phải do NGỮ NGHĨA quyết định:
đặt dấu kết thúc đúng nơi tác giả/người nói tự nhiên dừng, không cap bằng số đếm, đồng thời không
fuse các ý khác nhau thành run-on.

**Làm rõ Rule #6 (cùng ngày):** LLM bị Rule #3 + validator `merge_translated_batch`
(`llm_ai/srt_batch/batching.py`) BẮT giữ nguyên số block (1 vào → 1 ra), KHÔNG được tự gộp text.
Việc gộp vật lý xảy ra ở bước sau (`align-srt` → `resegment_srt_by_sentence`), và hàm đó CHỈ nhìn
KÝ TỰ CUỐI mỗi block: `。！？` → cắt; `，`/`、`/không dấu → gộp tiếp. Vậy đòn bẩy thật sự là dấu
LLM đặt ở RANH GIỚI block, không phải dấu bên trong. Đã sửa Rule #6 nhấn mạnh điều này + nhắc rõ
vẫn tuân Rule #1 (chỉ chèn dấu) và Rule #3 (giữ cấu trúc 1-block, không tự merge).

**Bonus — giải toả hiểu lầm:** input text-trần (không index/timestamp) chỉ là
`<FULL_SOURCE_CONTEXT>` do `build_global_context` join `it["text"]`. Phần LLM thật sự xử lý là
`<INPUT>` = `batch_srt_str` (`runner.py:116`) CÓ đủ index + timestamp + text. Nên các hướng dẫn
nói về "blocks/Index/Timestamp" trong prompt vẫn chính xác.

### Knob liên quan

`align-srt --split-on-comma` → `EXTENDED_GRAMMAR_SPLIT_CHARS` (gồm dấu phẩy) là hướng NGƯỢC lại,
cắt nhỏ hơn khi cần.

## 2026-06-17: video-ocr — `--strip-punctuation` thay cho `keep_punctuation`

### Tóm tắt

OCR (Qwen3-VL/DeepSeek) hay ảo giác chèn dấu câu (`· . - ， / :`) vào subtitle dù video không
có. Input bẩn này phá bước `punctuate-srt` (vốn cần text sạch để LLM tự khôi phục dấu). Thêm flag
`--strip-punctuation` cho `video-ocr` để bỏ MỌI dấu câu khỏi text OCR.

### Quyết định kiến trúc

- **Nhận diện dấu câu bằng `unicodedata.category(ch).startswith("P")`** (helper mới
  `utils/srt_parser.py::strip_punctuation`), trung lập ngôn ngữ — phủ trọn 7 nhóm P* (ASCII
  `. , - / : ; ! ?` lẫn fullwidth CJK `， 。 、 ！ ？ … （ ） 【 】 《 》`). Cùng cách nhận diện với
  `cli/punctuate_srt.py::_content_signature`. KHÔNG đụng nhóm ký hiệu S* (`~ + = < >`).
- **`--strip-punctuation` là NGUỒN DUY NHẤT xử lý dấu câu.** Gỡ hẳn cơ chế `keep_punctuation` cũ
  (param `ChineseFilter`/`extractor`, flag `--no-punctuation`, key YAML) để tránh hai đường bỏ-dấu
  song song. `ChineseFilter` giờ LUÔN giữ dấu câu CJK; việc bỏ dấu tách riêng, độc lập
  `--enable-chinese-filter` (chạy cả khi filter tắt — vá đúng chỗ mặc định text chỉ `.strip()`).
- Áp dụng strip **sau** nhánh filter/`.strip()` ở cả 2 vòng OCR (batch giữa chừng + tail) trong
  `extractor.py`, nên xếp chồng đúng khi bật kèm chinese-filter.

### Files

- `utils/srt_parser.py` — thêm `strip_punctuation()` (+ `import unicodedata`).
- `video_subtitle_extractor/chinese_filter.py` — gỡ param `keep_punctuation`, luôn giữ dấu CJK.
- `video_subtitle_extractor/extractor.py` — gỡ `keep_punctuation`, thêm param `strip_punctuation`.
- `cli/video_ocr.py` — gỡ `--no-punctuation`, thêm `--strip-punctuation` (YAML: `output.strip_punctuation`).
- `config/extractor_config.yaml` — bỏ `chinese_filter.keep_punctuation`, thêm `output.strip_punctuation`.
- Tests: `tests/utils/test_srt_parser.py` (Layer1 cho strip_punctuation), `tests/cli/test_extractor_config.py`.
- Docs: `docs/video-subtitle-extractor.md`, `docs/colab-guide.md` (bảng tham số).

### Pending / Next

- Nếu OCR ảo giác cả ký hiệu S* (`~ + =`), cân nhắc mở rộng `strip_punctuation` (hiện chỉ P*).
- Test `tests/cli/test_extractor_config.py` cần `cv2` → chỉ pass trên Colab/GPU env (local skip).

## 2026-06-16: blend-overlay-parallel — phủ blend video song song, frame-accurate

### Tóm tắt

CLI mới `cli/blend_overlay_parallel.py`: phủ một video **blend** (scratch/dust/noise loop) lên
video gốc bằng FFmpeg blend mode, nhưng **render SONG SONG** nhiều tiến trình thay cho lệnh
1-pass. Nút thắt của lệnh gốc KHÔNG ở encode (đã NVENC) mà ở **filter_complex chạy đơn luồng**
phải xử lý tuần tự cả tiếng video → mô hình fan-out nhiều tiến trình ffmpeg mới là nguồn
concurrency thật (bản thân ffmpeg không có batch/queue/async ở tầng CLI).

### Quyết định kiến trúc

- **Chia đoạn theo bội số NGUYÊN của độ dài blend (L):** mọi điểm nối nội bộ rơi đúng `k·L` nên
  khi mỗi đoạn loop blend lại từ 0, pha texture tự khớp như 1-pass → **liền mạch qua mối nối**
  mà không cần seek vào giữa blend. Phần dư lẻ (vd 0.5·L) dồn vào **đoạn cuối** (sau nó không
  còn mối nối nên vô hại). Chia đều `integer_loops` cho `workers` (vd 10.5L/4w → [3,3,2,2.5]).
- **An toàn timeline:** quy mọi mốc cắt về **frame nguyên** + ép CFR (`fps=`) + đồng nhất
  `-video_track_timescale 90000` → concat `-c:v copy` frame-exact, không drift (cùng tư duy
  `sync_engine/video_processor.py`).
- **Chống render vô hạn:** mỗi đoạn chốt cứng `-frames:v N` thay vì tin vào `shortest` của blend
  loop (`-stream_loop -1`). Đây là nguyên nhân gốc của hiện tượng "render mãi" ở lệnh 1-pass.
- **Audio zero-drift:** mỗi đoạn render `-an`; audio gốc ghép cuối trong `concat_and_mux` bằng
  `-c:a copy` (không encode lại) → output dài đúng bằng video gốc.
- **Giữ nguyên độ phân giải gốc:** main KHÔNG scale (full W×H probe được); chỉ blend được
  scale-crop cho khớp W×H gốc. (Bỏ tham số `--width/--height` cứng 1920×1080 của bản nháp.)

### CLI

`--video/--output` cho 1 video, hoặc `--task-file` JSON `[{input, output}]` cho batch nhiều
video (dùng chung 1 `--blend`). Tham số: `--workers` (mặc định 4), `--mode` (subtract),
`--opacity` (0.9), `--keep-tmp`, `-v`. File tạm vào `tmp/blend_<ts>_<uid>/`, tự xoá sau khi xong.
Đăng ký script `blend-overlay-parallel` trong `pyproject.toml`.

### Lưu ý vận hành

`--workers N` = N phiên `hevc_nvenc` đồng thời. GPU GeForce consumer giới hạn ~3–8 phiên NVENC;
`OpenEncodeSessionEx failed` → giảm `--workers`. Colab T4/L4 thường không bị giới hạn.

### File thay đổi

- `cli/blend_overlay_parallel.py` — TẠO MỚI: probe (fps/W×H/duration), `plan_segments`,
  `build_segment_cmd`, `concat_and_mux`, `resolve_tasks`, orchestration song song.
- `tests/cli/test_blend_overlay_parallel.py` — TẠO MỚI: 30 tests (Layer 1 + Layer 2), 30/30 pass.
- `tests/test_matrix.yaml` — thêm 2 entry `unit` (Layer 1, Layer 2).
- `pyproject.toml` — thêm script `blend-overlay-parallel`.
- `docs/colab-guide.md` — thêm Mục 2.14: flow + an-toàn-timeline + bảng tham số + task-file.

---

## 2026-06-16: align-srt v2 — bỏ Forced Aligner, thay bằng tách câu thuần CPU

### Tóm tắt

`cli/align_srt.py` viết lại: bỏ hoàn toàn nhánh `Qwen3ForcedAligner` + tách vocal + GPU.
Lý do: (1) OOM VRAM với video > 1 giờ; (2) overengineer — mục đích thật chỉ là **gom block SRT
thành câu hoàn chỉnh theo dấu ngắt câu**, trong khi `_punct.srt` đã có timestamp OCR gốc.

### Thuật toán mới (v1)

Duyệt block SRT theo thứ tự, gom vào buffer; khi block kết thúc bằng dấu ngắt câu
(`.!?:。！？：；`, không gồm phẩy, bỏ qua ngoặc/nháy đóng ở đuôi) → flush thành 1 block câu:
- `start = buffer[0].start_time` (timestamp OCR thật)
- `end   = buffer[-1].end_time`  (timestamp OCR thật)
- Không nội suy, không model, không GPU.

Block dư cuối file (không có dấu ngắt) → flush thành câu cuối.

### NOTE hoãn lại

Dấu ngắt câu nằm GIỮA block (vd `去学校。然后`) → v1 mặc kệ. Khi cần: nội suy timestamp theo
tỉ lệ ký tự trong block đó (dùng `start_time`/`end_time` thật của chính block).

### File thay đổi

- `utils/srt_sentence_segmenter.py` — TẠO MỚI: lõi thuần (testable, no I/O)
- `cli/align_srt.py` — VIẾT LẠI: bỏ forced-align/vocal/GPU; args mới đơn giản
- `tests/utils/test_srt_sentence_segmenter.py` — TẠO MỚI: 30 unit tests (Layer 1), 30/30 pass
- `tests/test_matrix.yaml` — thêm entry `unit` cho segmenter
- `docs/colab-guide.md` — cập nhật Mục 2.0c: flow + bảng tham số mới + NOTE hoãn

---

## 2026-06-15: Forced Alignment per-clip TTS — sửa OOM dây chuyền + không sót dòng mute

### Tóm tắt

Pipeline `sync-video` cho video dài (76 phút) chết tại Phase 5 (tuber NVENC) do **CUDA OOM dây
chuyền**: Phase 3.5 đưa cả `mixed_audio` vào `Qwen3ForcedAligner.align()` (giới hạn ~5 phút/call);
align() ném OOM; thiếu `try/finally` nên model kẹt ~16 GiB trong GPU; FFmpeg `hevc_nvenc` cùng process
không xin được CUDA context → chết.

**Giải pháp:** Đổi hướng sang **align từng clip TTS `dubb-{i}.wav`** (đã sinh sẵn ở Phase 0, 1:1 mỗi
dòng phụ đề). Clip chỉ vài giây → không bao giờ OOM, chạy video dài tùy ý. Word timing của mỗi clip
được offset về timeline cuối qua `seg.new_start + word_ms / audio_speed`. Voicevox family dùng `no_cap`
nên `audio_speed=1.0`, map thẳng không cần scale. Dòng phụ đề trong **vùng mute** (không có clip TTS)
tự động được **remap timeline rồi gộp** vào SRT cuối (không sót dòng nào).

### Thay đổi

- **`utils/forced_aligner.py`**:
  - Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` đầu module (giảm phân mảnh VRAM).
  - Bọc `align()` trong **try/finally** → `del aligner + clear_vram()` luôn chạy, kể cả khi OOM. Sửa
    bug domino sang NVENC Phase 5.
  - Thêm `execute_forced_alignment_clips(clips, align_cfg)`: load aligner 1 lần, loop batch clip, offset
    word timing, gọi `merge_punctuation → segment_words_to_subtitles` per-clip, trả `(aligned_segs, failed_lines)`.

- **`sync_engine/forced_alignment_subtitle.py`**:
  - Re-export `execute_forced_alignment_clips`.
  - `_resolve_aligner_config` thêm `batch_size` (default 16).
  - Thêm `_build_clips_from_timeline(timeline, subtitle_segments, mute_segments)`: zip tts_only ↔
    TTS TimelineSegments, nhận diện dòng mute và clip thiếu → remap_lines.
  - Thêm `_run_forced_alignment_clips(...)`: align per-clip + remap dòng còn lại (mute/fail) + gộp +
    sort theo time + ghi SRT hoàn chỉnh.
  - `run_forced_alignment_subtitle` nhận thêm `timeline, subtitle_segments, mute_segments, fps_float,
    remap_max_chars`. Khi có `timeline` → per-clip mode; khi không → nhánh cũ (backward compat cho
    test cũ và `align_srt.py`).

- **`sync_engine/timestamp_remapper.py`**: tách `recalculate_segments(...)` (trả list dict, không ghi
  file) từ `recalculate_srt` để dùng lại trong per-clip flow.

- **`cli/sync_video.py`** Phase 3.5: truyền `timeline, subtitle_segments, mute_segments, fps_float,
  remap_max_chars` vào `run_forced_alignment_subtitle`.

- **Config JSON** (`assets/*render_config.json` × 3): thêm `"batch_size": 16`.

- **Tests** (`tests/sync_engine/test_forced_alignment_subtitle.py`): thêm
  `TestLayer1_PerClipBuildClips` (5 test: mute→remap, missing clip, offset_ms, batch_size config) và
  `TestLayer2_PerClipAlignMerge` (5 test: offset đúng, mute present trong SRT, VRAM freed on OOM,
  empty clip skip, fallback to mixed-audio khi không có timeline).

- **`tests/test_matrix.yaml`**: thêm 2 entry mới (`Layer1_PerClip`, `Layer2_PerClip`).

### Pending

- Xác minh end-to-end trên Colab GPU với video 76 phút: Phase 3.5 align per-clip, không OOM, Phase 5
  NVENC chạy được, `*_synced.srt` có đủ dòng (cả vùng mute), timing word-level khớp giọng.
- edge/qwen với `audio_speed > 1.0`: xác minh chiều scale `word_offset / audio_speed` đúng (atempo
  tăng tốc → word cùng offset trong clip raw → vị trí nhỏ hơn trên timeline cuối).

## 2026-06-14: Fan-out song song cho SRT batch (translate/punctuate) + an-toàn-thread cho cache/anchor

### Tóm tắt

`run_srt_batch_task` trước nay chạy **tuần tự** (1 batch/lần). Với video dài (hàng nghìn block) wall-clock
rất lâu dù mỗi batch độc lập. Thêm chế độ **fan-out song song** (`max_workers > 1`) theo mô hình
**warm-up rồi fan-out**: batch ĐẦU chạy tuần tự để provider kịp tạo/ấm context cache (Vertex
`CachedContent`) hoặc anchor R0 (OpenAI Responses) TRƯỚC khi bung các batch còn lại qua
`ThreadPoolExecutor`. Cache/anchor là read-only dùng chung nên **không ảnh hưởng tính nhất quán bản
dịch** giữa các batch (fork-from-anchor, không chain tuần tự N→N-1). Mặc định vẫn `max_workers=1`
(giữ nguyên hành vi cũ).

Theo tài liệu chính thức (Vertex + OpenAI): caching **không** nới rate limit (cached tokens vẫn tính vào
TPM/RPM), nên concurrency phải **cap** + **exponential backoff** chứ không fan-out vô hạn — đặc biệt với
model free-tier/preview RPM thấp.

### Thay đổi

- **`llm_ai/retry.py`**: thêm `calculate_exponential_retry_wait_seconds` + `build_exponential_retry_wait`
  (gấp đôi mỗi attempt, truncated tại max, + jitter để tránh "retry storm"). Vertex & OpenAI provider
  chuyển sang dùng exponential thay cho linear.
- **`llm_ai/providers/vertexai.py` & `openai.py`**: telemetry chuyển sang **thread-local**
  (`threading.local`) — chạy song song mỗi thread đọc đúng số liệu call của mình, không bị clobber.
- **`llm_ai/providers/openai.py`**: thêm `_recreate_anchor_locked` (**double-checked locking** quanh
  `_anchor_lock`) — khi R0 hết hạn giữa chừng, nhiều thread chỉ tạo lại **đúng 1** anchor mới, không trùng.
- **`llm_ai/provider_chain.py`**: `FallbackLLMProvider.call()` **thread-safe** (RLock chốt snapshot
  active_index + áp context dưới khoá, network call ngoài khoá; fallover-guard chống double-advance).
  Thêm `last_telemetry_record` (delegate per-thread xuống provider active).
- **`llm_ai/srt_batch/batching.py`**: `CacheTelemetryAccumulator.record_dict()` — cộng dồn từ dict
  telemetry (worker đọc thread-local rồi trả về **main thread** cộng đơn luồng, không cần khoá).
- **`llm_ai/srt_batch/runner.py`**: tách `_process_one_batch` (thuần, không ghi state chia sẻ) trả
  `_BatchOutcome`; thêm `_RateLimiter` (trần nhịp phát request theo `wait_sec` khi song song); warm-up
  batch 0 → fan-out `ThreadPoolExecutor`; áp kết quả + telemetry ở main thread. `max_workers` param mới.
- **CLI/wrapper**: `translate_srt_file`, `restore_punctuation_srt` + `cli/translate_srt.py`,
  `cli/punctuate_srt.py` thêm `--workers/-w` (ưu tiên CLI > task YAML `max_workers` > 1).
- **Config**: `srt_translation.yaml` + `punctuation_restoration.yaml` thêm `max_workers: 1` (kèm chú
  thích cảnh báo RPM). `docs/colab-guide.md`: bảng tham số + ghi chú song song/warm-up/rate-limit.
- **Tests**: mới `tests/llm_ai/test_srt_batch_concurrency.py` (15 test: backoff math, RateLimiter,
  record_dict, song song==tuần tự cùng kết quả, overlap thật, warm-up đơn độc, context set 1 lần,
  fallback chain thread-safe). Thêm `test_concurrent_anchor_recreate_creates_single_r0` vào
  `test_translation_providers.py`. `test_matrix.yaml`: 2 entry mới. Đã chạy `tests/{translation,llm_ai}`
  xanh (67 passed + mới, 6 skip Layer4/no-cred).

> Cơ chế **theo capability profile**, không theo primary/fallback: provider nào hỗ trợ explicit cache/
> Responses anchor thì an toàn song song; cached tokens vẫn tính TPM nên giữ workers nhỏ (3–5).

---

## 2026-06-14: Gộp `punctuation/srt_punctuator.py` vào `cli/punctuate_srt.py` — xóa package `punctuation/`

### Tóm tắt

Sau khi tách `punctuate-srt` thành CLI riêng (entry cùng ngày, bên dưới), package `punctuation/` chỉ
còn 1 file mỏng phục vụ đúng 1 CLI → không đáng có thư mục riêng. Inline toàn bộ vào CLI.

### Thay đổi

- **`cli/punctuate_srt.py`**: gộp 4 hàm core (`_content_signature`, `_validate_content_preserved`,
  `restore_punctuation_srt`, `flatten_srt_to_text`) trực tiếp vào file. Import đổi từ
  `punctuation.srt_punctuator` → dùng `llm_ai.srt_batch` + `utils.srt_parser` tại chỗ.
- **Xóa** thư mục `punctuation/` (cả `__init__.py` + `srt_punctuator.py`). Bỏ `punctuation*` khỏi
  `pyproject.toml` `packages.find`.
- **Tests**: gộp `tests/punctuation/test_srt_punctuator.py` vào `tests/cli/test_punctuate_srt.py`
  (thêm `TestLayer1_ContentSignature`, `TestLayer1_Flatten`, `TestLayer2_RestorePunctuation` — tham
  chiếu hàm qua `punctuate_srt.*`). Xóa `tests/punctuation/`. `test_matrix.yaml`: gộp entry, gỡ 2 mục
  trỏ file đã xóa. Đã chạy: `tests/{cli,translation,llm_ai}` xanh (133 passed, 7 skip Layer4).

> Hạ tầng batch vẫn ở `llm_ai/srt_batch/` (dùng chung `translate-srt`); chỉ phần wrapper đặc thù
> punctuation chuyển từ package riêng vào thẳng CLI.

---

## 2026-06-14: Tách `punctuate-srt` thành CLI riêng + gom hạ tầng batch-SRT về `llm_ai/srt_batch/`

### Tóm tắt

Phục hồi dấu câu trước nay là **một phase nhúng trong `video-ocr`** (`--punctuate`), khó retry/debug
độc lập (LLM tốn kém, hay phải gọi lại; log thực tế từng thấy `45/47 batch, 100 block giữ gốc`). Tách
hẳn thành CLI standalone **`punctuate-srt`**. Đồng thời sửa một "mùi" kiến trúc: `punctuation/` import
hạ tầng batch từ `translation/` (mũi tên phụ thuộc sai — thứ bị mượn là batch-loop/context-cache/
integrity-retry, không dính dịch thuật). Gom hạ tầng đó về **`llm_ai/srt_batch/`** (nhà ngữ nghĩa
đúng); `translation/` và `punctuation/` chỉ còn wrapper mỏng.

### Thay đổi

- **Mới `llm_ai/srt_batch/`**: `batching.py` + `prompting.py` (chuyển từ `translation/`), `runner.py`
  với `run_srt_batch_task(...)` — vòng lặp batch generic nhận `response_tag` + `validator` (optional).
  `load_prompt` hợp nhất thay cả `{lang}` lẫn `{language}` (no-op với prompt chỉ có 1 placeholder).
- **`translation/srt_translator.py`**: wrapper mỏng gọi `run_srt_batch_task(response_tag="TRANSLATE_TEXT",
  validator=None)` + sidecar `.txt`. Giữ chữ ký `translate_srt_file` + alias `GeminiCaller`/
  `parse_gemini_response`. **Xóa** `translation/batching.py`, `translation/prompting.py`.
- **`punctuation/srt_punctuator.py`**: wrapper mỏng gọi `run_srt_batch_task(response_tag="PUNCT_TEXT",
  validator=_validate_content_preserved)`. Giữ nguyên tên public (`_content_signature`,
  `restore_punctuation_srt`, `flatten_srt_to_text`) → test cũ không đổi.
- **Mới `cli/punctuate_srt.py`** → script `punctuate-srt` (pyproject). Args: `--input/--task-file/
  --output`, `--task-config` (SSOT, mặc định `config/llm_tasks/punctuation_restoration.yaml`),
  `--lang/--batch/--no-context/--no-flatten`, provider overrides. Dùng `resolve_cli_tasks`
  (default_ext `_punct.srt`) + `create_task_provider` (như `llm-task`).
- **`cli/video_ocr.py`**: **gỡ hẳn** `run_punctuation_phase()` + 2 args `--punctuate`/
  `--punctuation-task-config` + 2 call-site. `video-ocr` giờ chỉ làm OCR.
- **Tests**: `tests/translation/...` đổi import `translation.batching` → `llm_ai.srt_batch.batching`.
  Mới `tests/cli/test_punctuate_srt.py` (Layer1 arg-parser + Layer3 pipeline mock provider). Thêm 2
  entry vào `test_matrix.yaml`. Đã chạy: `tests/{llm_ai,cli,punctuation,translation}` xanh
  (107+35 passed, skip Layer4 real-API).
- **Docs**: `colab-guide.md` mục 2.0c → luồng **3 bước rời** `video-ocr → punctuate-srt → align-srt`,
  thêm bảng tham số `punctuate-srt`, gỡ bảng `--punctuate` của video-ocr.

### Flow mới (OCR-centric)

```
video-ocr <video> --config flow.yaml      → <stem>.srt                  (chỉ OCR)
punctuate-srt <stem>.srt                   → <stem>_punct.srt + <stem>_punct_flat.txt
align-srt <stem>_punct_flat.txt --video …  → <stem>_punct_flat_aligned.srt
```

Mỗi bước retry/debug độc lập. Hạ tầng batch (cache/context/retry) dùng chung `translate-srt` qua
`llm_ai/srt_batch/`.

### Pending / Next

- Câu hỏi mở của user (chưa làm): **phương án non-LLM** cho phục hồi dấu câu (deepmultilingualpunctuation,
  ct-punctuator...) — có thể thêm như một provider/validator thay thế trong `punctuate-srt` sau này.

---

## 2026-06-13: Tối ưu token LLM — context cache đo được + thinking_level per-task + Responses anchor-fork

### Tóm tắt

Giải bài toán "đốt token" khi `use_full_context: true` cho video dài (hàng nghìn block). Kết luận
nghiên cứu: **Agent Engine Sessions KHÔNG giúp** (gửi lại history mỗi lượt). C��ng cụ đúng là **context
caching** — vốn đã có cho Vertex nhưng (1) translation dùng `provider_chain` nên bị vô hiệu, (2) không
có telemetry để biết cache có hit. Đồng thời thêm điều khiển `thinking_level` (Gemini 3) per-task.

### Thay đổi

- **`llm_ai/providers/vertexai.py`**: `_build_generate_config` pop `thinking_level` (low/medium/high) →
  `ThinkingConfig`, loại `thinking_budget` để tránh lỗi 400 (Gemini 3). Thêm `_capture_telemetry` đọc
  `usage_metadata` (`cached_content_token_count`...) → `last_telemetry_record`.
- **`llm_ai/provider_chain.py`**: `FallbackLLMProvider` giờ **tự quản global context**.
  `set_global_context` trả `True` (caller bỏ inline); `_ensure_context_on_active()` set context cho
  provider khi active (cache nếu được, prepend inline nếu không) — re-apply khi fallover.
  `apply_provider_chain_entry_overrides` deep-merge `generation_config` (thay vì replace).
- **`llm_ai/providers/openai.py`**: thêm `set_global_context` cho Responses-API profile → tạo **anchor
  R0** (`store: true`) gửi full context 1 lần; mọi batch **fork từ R0 cố định** (biến mới
  `_anchor_response_id`, KHÔNG sequential-chaining để tránh phình token). Error-recovery: R0 invalid →
  tạo lại anchor 1 lần. Nhánh chat_completions giữ nguyên.
- **`llm_ai/task_runner.py`**: `apply_provider_overrides` deep-merge `generation_config` từ task config.
- **`translation/batching.py`**: thêm `CacheTelemetryAccumulator` cộng dồn token qua batch.
- **`translation/srt_translator.py`** + **`punctuation/srt_punctuator.py`**: ghi nhận telemetry mỗi
  batch, log 1 dòng tổng kết `🧠 Cache: X/Y prompt tokens hit (Z%)`.
- **Configs**: `srt_translation.yaml` đảo chain → **Vertex primary** (thinking_level: low) +
  **freemodel_dev fallback** (Responses API). `punctuation_restoration.yaml` thêm `thinking_level: low`.
- **Tests**: `tests/translation/test_translation_providers.py` — sửa import stale
  (`translation.vertexai_provider` → `llm_ai.providers.vertexai`); thêm Layer1 (telemetry accumulator,
  thinking_level, chain-aware context) + Layer3 (Responses anchor fork-from-R0). Matrix entries hiện có
  đã bao qua keyword Layer1/Layer3.

### Lý do

OpenAI prompt cache là _giảm giá best-effort_ (token vẫn bị tính, evict 5-10 phút idle) — KHÁC Vertex
`CachedContent` (bỏ token khỏi prompt, ~90% off, TTL deterministic). Anchor-fork: cache discount hết
hạn KHÔNG cần gửi lại full_context — fork cùng R0 tự re-warm; tạo R0 mới là strictly worse. Vì vậy
chọn Vertex làm primary cho translation (tiết kiệm chắc chắn), freemodel_dev Responses làm fallback.
`cached_tokens` telemetry chỉ để **quan sát**, không phải trigger gửi lại context.

### Pending / Next

- E2E trên 1 video dài thật: xác nhận `cached_content_token_count`/`cached_tokens` > 0 và so token
  Console Billing trước/sau.
- Tuỳ chọn follow-up: dùng `prompt_cache_key` ổn định cho OpenAI chat_completions profile khi backend
  hỗ trợ (`supports_prompt_cache_key`).
- `google-api-core` + `openai` đã được cài vào `.venv` local để chạy test (CPU-only, an toàn).

---

## 2026-06-13: Dọn config + môi trường flow OCR (align-srt CLI thuần, dedup punctuation, .venv chính)

### Tóm tắt

Refactor theo review người dùng — 3 điểm: (1) `align-srt` đứng riêng nên **bỏ đọc YAML config**, dùng
**CLI args thuần + `--task-file`** (đúng idiom repo); (2) **dedup** config punctuation — giữ
`config/llm_tasks/punctuation_restoration.yaml` làm **SSOT**, bỏ key trùng khỏi config video-ocr;
(3) `align-srt` chạy ở **`.venv` chính** như `sync-video` (cần `qwen-asr` + `audio-separator` cài ở
`.venv` chính — cùng bộ deps forced-align của sync-video; KHÔNG cần `flash-attn`).

### Thay đổi

- **`cli/align_srt.py`**: xoá `_load_yaml` + arg `--config`; knob forced-alignment thành **default của
  argparse** (`_ALIGN_DEFAULTS`); `resolve_align_cfg` build từ args; vocal-sep preset qua
  `--separator-preset` (default `vocal_extraction`), bỏ section `vocal_separation:`. Toggle tách vẫn
  `--no-separate`/`--vocals`.
- **`cli/video_ocr.py`**: thêm 2 CLI args `--punctuate` (mặc định TẮT) + `--punctuation-task-config`;
  `run_punctuation_phase(output_paths, args, config, format)` đọc enabled/task_config từ CLI
  (CLI>YAML>default), mọi tham số LLM (language/batch_size/use_full_context/provider) lấy **chỉ từ
  task_cfg**. Bỏ các key trùng trong section `punctuation:`.
- **`docs/colab-guide.md`**: Mục 2.0c — `align-srt` chạy `!uv run align-srt` ở `.venv` chính (gỡ
  `.venv-qwen3asr` + cài audio-separator vào venv đó); gỡ `forced_alignment:`/`vocal_separation:` khỏi
  `flow.yaml`; cập nhật bảng tham số (bỏ `--config`, thêm `--separator-preset`, `--no-split-on-comma`,
  `--punctuate`/`--punctuation-task-config`).

### Lý do

align-srt là CLI độc lập, repo không có master-config-per-CLI → CLI args đủ và nhất quán. Punctuation
config trùng lặp → SSOT = llm_tasks yaml (như `srt_translation.yaml`). align-srt dùng chung lõi
`utils/forced_aligner.py` (`from qwen_asr import Qwen3ForcedAligner`, `attn_implementation=None`) nên
**cần `qwen-asr`** (không cần `flash-attn` — aligner không gọi flash_attention_2) — y hệt forced-align
của sync-video; chọn chạy ở `.venv` chính (cài qwen-asr + audio-separator ở đó) cho nhất quán với
sync-video, thay vì `.venv-qwen3asr`.

---

## 2026-06-13: OCR-centric source SRT — Punctuation (LLM) + Forced-alignment timing

### Tóm tắt

Tách trách nhiệm tạo SRT nguồn: **text từ OCR** (ground-truth), **dấu câu từ LLM** (vertexAI),
**timing + ngắt block từ Qwen3-ForcedAligner**. Thay cho qwen3-asr (text+timing đều do ASR → text dễ sai).
Các bước CLI rời, dùng chung 1 file YAML config; aligner tự tách vocal từ video.

### Flow

```
video-ocr <video> --config flow.yaml
  → <stem>_<box>.srt  + (nếu punctuation.enabled) _punct.srt + _flat.txt
align-srt <stem>_<box>_flat.txt --video <video> --config flow.yaml
  → tự tách vocal (reuse audio-separator) → _aligned.srt (text OCR + timing align)
```

### Thay đổi

- **`utils/forced_aligner.py`** (MỚI): tách lõi `load_forced_aligner` + `execute_forced_alignment`
  (trung lập, nhận `align_cfg` dict) ra khỏi `sync_engine/`. Dùng chung sync-video + align-srt.
- **`sync_engine/forced_alignment_subtitle.py`**: chỉ còn glue render_config
  (`_resolve_aligner_config`, `run_forced_alignment_subtitle`), re-export lõi. **sync_video không đổi.**
- **`punctuation/srt_punctuator.py`** (MỚI): `restore_punctuation_srt` chạy BATCH (mô phỏng
  srt_translator: parse_srt + translation/batching + render_batch_prompt). Validator
  `_content_signature` (strip Unicode P\* + whitespace) chống ảo giác — chỉ thêm dấu, không đổi chữ;
  lệch → BatchIntegrityError → retry → giữ nguyên block gốc. `flatten_srt_to_text` nối 1 dòng
  (CJK không space / Latin có space) cho aligner.
- **`cli/align_srt.py`** (MỚI) + script `align-srt`: nhận `--video`, tự trích audio (ffmpeg) +
  tách vocal (reuse `cli/audio_separator.separate_audio`, preset vocal_extraction), align → SRT.
  Đọc section `forced_alignment` trong --config (CLI > YAML > default). Hỗ trợ `--vocals`, `--no-separate`, `--task-file`.
- **`cli/video_ocr.py`**: thêm `run_punctuation_phase` (tuỳ chọn, bật bằng `punctuation.enabled`
  trong --config YAML); provider dựng qua `llm_ai.task_runner.create_task_provider`.
- **`config/llm_tasks/punctuation_restoration.yaml`** + **`prompts/llm_tasks/punctuation_restoration.txt`**
  (MỚI): vertexai; prompt ràng buộc đa ngôn ngữ + block-là-mảnh-câu + chỉ thêm dấu + giữ số dòng (`<PUNCT_TEXT>`).
- **`pyproject.toml`**: đăng ký `align-srt`, thêm `punctuation*` vào packages.find.
- **Tests**: `tests/punctuation/test_srt_punctuator.py` (9 test L1/L2, mock provider — PASS local);
  cập nhật patch target trong `tests/sync_engine/test_forced_alignment_subtitle.py` sang
  `utils.forced_aligner.*` (25 test PASS). Thêm 2 entry `test_matrix.yaml`.

### Quyết định thiết kế

- **Không dùng `llm-task`/`generic_text_task`** cho punctuation (gửi nguyên file 1 call, không batch,
  không kiểm tra toàn vẹn) → dùng SRT-batch như translate (chống drift theo số block + char-preservation).
- **Validator trung lập ngôn ngữ** (Unicode category P\*) thay vì hardcode bộ dấu CJK.
- **align-srt là CLI riêng** (không gọi từ video-ocr) — người dùng chạy 2 bước rời, dễ debug.
- Aligner cần text **phẳng 1 dòng** (`merge_punctuation` nuốt `\n`) → có bước flatten riêng.

### Trạng thái

- ✅ Lõi + 2 CLI/phase + config/prompt + test L1/L2 (local PASS) + matrix + journal.
- ⏳ Pending: Layer 3/4 (align-srt + tách vocal) cần GPU + `qwen-asr` + `audio-separator` → chạy Colab;
  real-API test punctuation (vertexAI) trên dữ liệu OCR thật để tinh chỉnh prompt + `max_chars` tiếng Trung.

---

## 2026-06-12: Text Isolation — Lọc watermark/overlay mờ khỏi OCR phụ đề

### Tóm tắt

Thêm tính năng tách phụ đề lời thoại khỏi watermark/text-overlay (opacity < 70%) **trước khi OCR**,
dựa trên opacity/color masking thuần OpenCV. Giải quyết ca creator có overlay động đè lên dải phụ đề.
Mặc định TẮT — chỉ bật cho video có watermark.

### Nguyên lý

Watermark mờ (α<0.7) bị blend với nền → giảm tương phản/gradient, mất viền tối, lệch màu. `text_isolator`
quyết định GIỮ/XÓA ở mức từng connected-component: color gate (màu chỉ định, Lab) + contrast gate
(morph gradient ~ opacity) + stroke gate (viền tối) + min_component_area (chỉ diệt nhiễu). Deterministic,
không model → không hallucinate.

### Thay đổi

- **`video_subtitle_extractor/text_isolator.py`** (MỚI): `TextIsolationConfig` + `isolate_subtitle_text`
  - `parse_color_spec` (tên/hex/RGB).
- **`tools/calibrate_text_isolation.py`** (MỚI): script hiệu chỉnh giám sát 2 thư mục mẫu
  (`--subtitle-samples`, `--watermark-samples`), xuất JSON config + ảnh preview before/after + báo cáo
  độ tách bạch. Tham số suy ra: min_contrast/color_tolerance (mạnh), stroke_max_luminance/require_stroke
  (khá), min_component_area (yếu, chỉ diệt nhiễu — P5×0.5).
- **`video_subtitle_extractor/extractor.py`**: thêm param `text_isolation`; mask ảnh TRƯỚC CV prefilter
  (frame chỉ-watermark → ROI trống → skip OCR, tiết kiệm GPU); scene-detect vẫn trên ROI gốc.
- **`cli/video_ocr.py`**: flags `--isolate-text`, `--isolate-config`, `--subtitle-colors`,
  `--color-tolerance`, `--subtitle-min-contrast`, `--stroke-max-luminance`, `--min-component-area`,
  `--no-require-stroke`. Ưu tiên CLI > YAML > JSON-calibrate > default.
- **Tests**: `tests/video_ocr/test_text_isolator.py` (Layer1 unit parse_color_spec/config — verified
  PASS; Layer2 component masking — chạy trên Colab vì cần cv2). Thêm 2 entry vào `test_matrix.yaml`.
- **Docs**: `docs/text-isolation-guide.md` — quy tắc cắt mẫu + cách đọc/dùng 5 tham số + giới hạn.

### Quyết định thiết kế

- **Bỏ lọc thời gian (`min_persistence_frames`)**: watermark trôi chậm sống LÂU (không bị loại), phụ đề
  câu ngắn sống ngắn (bị loại nhầm) → phản tác dụng. Loại hẳn.
- Màu phụ đề **do người dùng chỉ định** (không tự dò), hỗ trợ cả chữ trắng lẫn chữ màu.
- VLM tự phân loại subtitle: loại (dễ ảo giác).

### Trạng thái

- ✅ Module + script + tích hợp + CLI + test + doc hoàn tất. Layer1 test PASS local.
- ⏳ Pending: chạy Layer2 trên Colab (cần cv2); người dùng cung cấp bảng màu thực tế + crop mẫu để
  hiệu chỉnh ngưỡng; xác nhận chất lượng mask qua ảnh preview.
- 🔮 Tương lai (nếu watermark đặc đè khít dòng phụ đề): text detection + lọc hình học/tracking vị trí.

## 2026-06-12: Phase 1 — Gỡ hoàn toàn path Remotion, prerender là path duy nhất

### Tóm tắt

Xóa subproject `remotion_tuber/` (Node/Remotion) và toàn bộ code chết liên quan. Path "prerender" (Python/PIL/FFmpeg) trở thành path render duy nhất (TODO dòng 351 DONE ✓).

### Thay đổi

- **`sync_engine/tuber_overlay.py`**: Xóa `_run_render_driver`, `_which_npm`, `prepare_assets`, `composite_group` (V1), `build_group_base`. Bỏ param `use_prerender`/`project_dir`/`do_prepare_assets` khỏi `render_and_composite_groups`, `render_groups_to_video`, `prepare_groups_and_base`. Đơn giản hóa `run_tuber_flow_all_in` — loại bỏ nhánh `overlay_mode=="remotion"`.
- **`sync_engine/tuber_config.py`**: Xóa `_REMOTION_REQUIRED_KEYS`, `remotion_project_dir()`. `overlay_mode` giờ luôn trả về `"prerender"`.
- **`sync_engine/tuber_manifest.py`**: Bỏ param `remotion` + block `"remotion"` khỏi `build_run_manifest`.
- **`cli/tuber_repair.py`**: Raise `TuberOverlayError` nếu thiếu prerenderManifest (không còn fallback Remotion). Bỏ đọc `run_manifest["remotion"]`.
- **Config**: Xóa block `"remotion": {...}` khỏi `assets/charenjizukan_tuber_overlay_config.json` và `assets/tuber_overlay_config.json`.
- **Tests**: Xóa `tests/sync_engine/test_tuber_remotion_validation.py`. Viết lại `TestLayer3_RetryAndCleanup` mock `_pipe_prerender_frames`. Đổi env `REMOTION_TUBER_E2E` → `PRERENDER_E2E` trong `test_tuber_prerender.py`. Xóa 3 entry "Tuber Remotion" khỏi `test_matrix.yaml`.
- **Docs**: Dọn `docs/tuber-overlay-guide.md` + `docs/colab-guide.md` — bỏ mọi hướng dẫn Remotion/npm.
- **`remotion_tuber/`**: `git rm -r` (xóa 17 file).

### Trạng thái

- ✅ Prerender là path duy nhất, không còn dead code Remotion.
- ✅ `overlay_mode` config cũ `"remotion"`/`"auto"` được coi như `"prerender"` (backward-compat cảnh báo).
- ⏳ Phase 2: gom `sync_engine/tuber_*.py` → package `sync_engine/tuber/` (commit 2).

---

## 2026-06-11: Local testing — venv `uv` cô lập (CPU-only), sửa runner dùng `sys.executable`

### Bối cảnh

Cần chạy test ở máy local (Windows, Python global 3.13) mà không đụng Python global và không kéo package GPU (chúng được test trên Colab).

### Đã làm

- Tạo `.venv` bằng `uv venv --python 3.12 .venv` (khớp Colab `requires-python >= 3.12`).
- Cài CPU-only vào `.venv`: `uv pip install -e .` + `pytest pyyaml pydub numpy`. KHÔNG dùng `--system`, KHÔNG cài group `[qwen-tts]` (torch/flash-attn). Đã verify `Location` trỏ `.venv` và không có package GPU; global Python sạch.
- **Sửa `run_colab_tests.py::_build_pytest_cmd`**: `["python", ...]` → `[sys.executable, ...]` để subprocess pytest chạy đúng interpreter `.venv` thay vì rơi vào python PATH khác (uv cpython).
- **Sửa `run_colab_tests.py`**: thêm `_safe_report_name()` (sanitize ký tự Windows-invalid `< > : " / \ | ? *`) → hết crash `OSError [Errno 22]` khi tên report chứa `>`. Thêm `PYTHONIOENCODING=utf-8`/`PYTHONUTF8=1` vào env subprocess → test in emoji/CJK (vd `📄`) không crash cp1252 trên Windows.
- **Sửa `pyproject.toml`**: thêm `tts*` vào `packages.find.include` (trước đây thiếu → `import tts.edgetts` fail khi không có cwd trên path). Reinstall editable.
- Sửa các test stale: `test_extractor_config.py` (thêm `importorskip("cv2")`), `test_ass_utils.py` (2 test sai expectation), `test_tts_edgetts.py` (import `cli.tts_srt`→`cli.tts`, skip Layer 3 dùng API `run_tts` cũ, cập nhật `test_strip_silence_only_tail` theo logic "strip cả hai đầu" hiện tại).
- **Xóa Demucs** (đã bị thay bằng audio-separator): bỏ 2 entry trong `test_matrix.yaml` trỏ file đã xoá `tests/cli/test_demucs_audio.py`; cập nhật wording trong `audio_assembler.py`, `colab-guide.md`, `testing-guide.md`. Giữ nguyên 2 file `plans/` (bản ghi lịch sử, theo yêu cầu user).
- Cập nhật `docs/testing-guide.md`: thêm **Mục 0** ở đầu file (hướng dẫn `uv` + `.venv`, cấm `--system`, cấm package GPU local) để Agent scan nắm ngay.

### Kết quả

- `python run_colab_tests.py --tags unit` → **35 passed, 0 failed**, 1 all-skipped (Translation L2 cần API), 3 no-collection (cần `cv2`/opencv — cố ý KHÔNG cài local).

### Next steps đề xuất

- Khi cần chạy L4/gpu hoặc các test cần `cv2`: chạy trên Colab (venv Colab kế thừa torch/cv2).
- Cân nhắc viết test thay thế cho audio-separator (chỗ Demucs cũ để trống).

## 2026-06-11: qwen3_asr — expandable_segments + `--max-new-tokens` (mặc định 1024) giảm OOM VRAM audio dài

### Bối cảnh

Trên L4 Colab (22GB), transcribe video ~1h thấy VRAM tăng tịnh tiến gần chạm trần, suýt OOM ở `--batch-size 8` và OOM hẳn ở `16`. Câu hỏi: đây là hành vi mặc định của Qwen3-ASR hay code thiếu clear VRAM.

### Nguyên nhân gốc

Vì CLI gọi `transcribe(..., return_time_stamps=True)` nên `qwen_asr` ép chunk audio theo `MAX_FORCE_ALIGN_INPUT_SECONDS = 180s` (3 phút) → video 1h ≈ 20 chunk, gom batch theo `max_inference_batch_size` (= `--batch-size`). Thư viện KHÔNG `empty_cache()`/`del` tensor/giải phóng KV-cache giữa các batch; PyTorch giữ lại reserved memory → nhìn `nvidia-smi` thấy leo dần tới đỉnh rồi plateau. **Không phải leak** (text trả về string, timestamp merge bằng list — đều ở CPU); đỉnh VRAM ∝ `batch_size × max_new_tokens`. Batch 16 = 16 chunk×3phút đồng thời → KV-cache gấp đôi → vượt 22GB. `clear_vram()` đặt ở `finally` chỉ chạy SAU cả file, vô dụng trong-file và không chèn được vào vòng lặp chunk nội bộ thư viện.

### Thay đổi

1. `cli/qwen3_asr.py`:
   - `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")` ngay đầu module (TRƯỚC khi torch khởi tạo CUDA allocator) → giảm phân mảnh, trị đúng kiểu "leo tới trần rồi OOM". Dùng `setdefault` để tôn trọng giá trị user set sẵn.
   - Bỏ hardcode `max_new_tokens=4096` → tham số `--max-new-tokens` (mặc định **1024**), xuyên suốt: chữ ký `run_batch_transcribe` → `from_pretrained` → arg → lời gọi trong `main()`. (Official transformers example dùng 256; chunk 3 phút không cần tới 4096, cap cao chỉ làm trần KV-cache khi 1 chunk lặp/hallucinate.)

### Lưu ý còn lại

- Chưa verify thật trên Colab L4 — cần chạy lại video 1h xác nhận batch 16 không còn OOM.
- `batch_size` vẫn là đòn bẩy chính cho đỉnh VRAM; `--max-new-tokens` là cap an toàn — nếu chunk dài bị cắt cụt text thì tăng lên.

### File thay đổi

- `cli/qwen3_asr.py`

---

## 2026-06-11: Watermark width + layer_order config + nung black_strip ở stretch (dưới tuber)

### Bối cảnh

Yêu cầu: (1) thêm `width` optional cho watermark_img (height auto theo aspect, mặc định width gốc); (2) sắp lại layer order `Base → Black strip → Tuber → Image → Note → Watermark → Subtitle`; (3) tuber phải nằm TRÊN black_strip. Vấn đề kiến trúc: tuber baked vào base TRƯỚC final render nên vốn là layer dưới cùng (dưới cả strip). Promote tuber thành alpha layer độc lập thì mất NVENC (NVENC không hỗ trợ alpha) + file ProRes/VP9 alpha cực lớn cho video 2-3h.

### Quyết định

Nung `black_strip` ngay ở **Phase 2 (stretch)** thay vì Phase 5 → gấp chung vào encode batch sẵn có, **0 encode thêm**, và strip tự nằm DƯỚI tuber (tuber composite seek vào `video_stretched.mp4` sau đó). Tuber giữ nguyên cơ chế baked. Các layer Phase 5 còn lại điều khiển thứ tự bằng `render_config.layer_order`.

### Thay đổi

1. `sync_engine/video_processor.py`: `build_ffmpeg_batch_cmd(..., strip, video_width)` + helper `_build_strip_overlay` (full-width clamp) + `_probe_video_width`. `process_video_chunks_parallel(..., strip)` probe width source rồi overlay strip lên concat trước khi encode. Strip phủ full width video gốc; `scale_width` config > width video → kẹp về full width.
2. `sync_engine/renderer.py`: refactor 6 layer (black_strip/image_overlay/note_overlay/watermark_img/watermark_text/subtitles) thành builder + dispatch theo `render_config["layer_order"]` (thiếu → default order). Thêm param `skip_layers`. Watermark_img: `width` > 0 → `scale={width}:-1` (height auto giữ aspect); null/absent/<=0 → giữ ảnh gốc.
3. `cli/sync_video.py`: helper `_resolve_black_strip`; Phase 2 truyền `strip=` vào stretch; Phase 5 `skip_layers={"black_strip"}` khi strip bật.
4. `cli/tuber_repair.py`: final render `skip_layers={"black_strip"}` (base promote đã chứa strip nung).
5. `assets/default_render_config.json`: thêm `"width": null` vào watermark_img + mảng `"layer_order"`.
6. Docs: `sync-video-guide.md` (sơ đồ layer, mục layer_order, watermark_img width, black_strip nung ở Phase 2), `tuber-overlay-guide.md` (z-order tuber trên strip).

### Layer order kết quả

`Base → Black strip (nung Phase 2) → Tuber (baked) → Image overlay → Note overlay → Watermark img → Watermark text → Subtitle`

### Lưu ý còn lại

- Resume hash tuber anchor theo video gốc + group_manifest, KHÔNG theo stretched → đổi black_strip config KHÔNG tự re-render tuber group (strip nằm dưới, không đổi hình tuber). Muốn ép thì `resume.skipDone=false`.
- Chưa verify FFmpeg thật trên Colab GPU — cần chạy pipeline thật để xác nhận strip hiển thị đúng vị trí dưới tuber. Test unit (80 passed) chỉ verify command/filter string + pipeline mock.
- Muốn tuber xen kẽ z-order tùy ý (vd trên image overlay) → phải tách alpha layer (mất NVENC + disk lớn), chưa làm.

### File thay đổi

- `sync_engine/video_processor.py`, `sync_engine/renderer.py`, `cli/sync_video.py`, `cli/tuber_repair.py`, `assets/default_render_config.json`, `docs/sync-video-guide.md`, `docs/tuber-overlay-guide.md`

---

## 2026-06-09: Tuber group cuối fail "Duration lệch" — clamp group theo frame THỰC của video_stretched

### Vấn đề

Video dài (1h30–1h40) fail ở Phase 5.0 tuber overlay, **chỉ group cuối** (hoặc vài group đuôi) fail với `Duration lệch: 286.067s vs expected 286.200s (tol 0.1s)` — hụt đúng ~4 frame (0.133s). Group cuối là group duy nhất chạm **EOF thật** của `base_video_stretched.mp4`.

### Nguyên nhân gốc

Manifest tính `groupStartFrame`/`renderDurationFrames` bằng tổng dồn `segment_output_frames()` (frame **lý thuyết**). Nhưng `build_ffmpeg_batch_cmd` dùng `trim=end_frame=N` — chỉ **chặn-trên**, không đảm bảo đủ N: segment chạm EOF nguồn ra thiếu vài frame. File `video_stretched.mp4` thật vì thế luôn ≤ lý thuyết, phần hụt dồn vào group cuối. Composite seek theo thời gian tuyệt đối → group cuối trim không đủ N → lệch tolerance 0.1s → fail → fallback render KHÔNG tuber.

### Thay đổi

**A. Fix gốc ở Phase 2 — pad tail (frame lý thuyết thành nguồn sự thật duy nhất thật sự):**

- `sync_engine/video_processor.py`: thêm hằng `_TAIL_PAD_SECONDS=2.0` + chèn `tpad=stop_mode=clone:stop_duration=2.0` NGAY TRƯỚC `trim=end_frame=N` trong cả `build_ffmpeg_batch_cmd` (Phase 2 dùng) và `build_ffmpeg_chunk_cmd` (legacy). Segment chạm EOF được clone frame cuối cho đủ N rồi trim chốt đúng N → `video_stretched.mp4` thật == frame lý thuyết == audio == sub. Segment thường (đủ/dư N) bị trim cắt bỏ tpad ngay, không encode, output không đổi → không phá test filter-string hiện có (tpad nằm giữa `fps=` và `trim=end_frame=N[vi]`, mọi substring assert vẫn còn).

**B. Lưới an toàn ở lớp tuber — clamp theo frame thật:**

1. `sync_engine/tuber_overlay.py`: thêm `probe_frame_count(video, fps)` (ưu tiên `nb_frames`, fallback `duration*fps`, KHÔNG `-count_frames` để tránh decode toàn bộ — đọc metadata header, tức thì, không GPU). `run_tuber_flow_all_in` đo frame thật của base stretched rồi truyền xuống; `prepare_groups_and_base` nhận `real_total_frames`.
2. `sync_engine/tuber_manifest.py`: `build_render_groups(..., real_total_frames=None)` + helper `_clamp_groups_to_real()` — cắt `group_end_frame` của group chứa EOF, bỏ group nằm hẳn sau EOF. Deficit > 2s → WARNING (nghi Phase 2 truncate thật, vd ca 1h30 hụt ~25 phút), ≤ 2s → INFO.
3. `tests/sync_engine/test_tuber_overlay_pipeline.py`: 4 unit test mới trong `TestLayer1_GroupBuilding`.

Sau (A), file thật khớp lý thuyết nên (B) hầu như không kích hoạt với rounding đuôi — chỉ còn là lưới bắt truncate bất thường.

### Lưu ý còn lại

- Ca 1h30 hụt ~25 phút là **truncate thật ở Phase 2** (một batch ra ngắn). tpad (A) chỉ bù ≤2s nên KHÔNG che được ca này → clamp (B) bắt + WARNING. Gốc vẫn nên thêm validate frame-count sau `_concat_chunks` (chỉ đang validate size) — việc riêng.
- Chưa verify FFmpeg thật trên Colab — cần chạy lại pipeline video 1h40 để xác nhận group cuối pass và `video_stretched` đủ frame.

### File thay đổi

- `sync_engine/video_processor.py`, `sync_engine/tuber_overlay.py`, `sync_engine/tuber_manifest.py`, `tests/sync_engine/test_tuber_overlay_pipeline.py`

---

## 2026-06-08: Fix chromakey — bỏ qua key khi bodySource đã có alpha sẵn

### Vấn đề

Body tuber overlay hiện ra bán trong suốt ("như ẩn như hiện", chỉ rõ mouth) dù user KHÔNG khai báo `chromakey` trong config. Hai nguyên nhân:

1. **Code luôn chromakey**: `extract_body_transparent()` không có đường tắt — khi `chroma_color=None` nó _auto-detect_ màu 4 góc rồi vẫn `chromakey=...`. Với nguồn đã trong suốt (hoặc H264 nền bị nướng đen), chromakey key luôn cả vùng tối của body → bán trong suốt.
2. **Hiểu nhầm H264**: `loop_mouthless_h264.mp4` không thể trong suốt thật — H264/.mp4 không mang kênh alpha. Muốn nền trong suốt phải dùng ProRes4444 `.mov` / VP9 `.webm` / PNG sequence.

### Thay đổi

1. `sync_engine/tuber_prerender.py`:
   - Thêm `_source_has_alpha()` (ffprobe pix_fmt → nhận diện yuva\*/rgba/...).
   - `extract_body_transparent()` thêm tham số `chromakey_enabled: Optional[bool]`. Tri-state: None=auto (bỏ qua chromakey nếu nguồn có alpha), False=luôn giữ alpha gốc (`vf=format=rgba`), True=luôn chromakey. Cảnh báo khi tắt chromakey nhưng nguồn không alpha (sẽ ra frame đặc).
2. `sync_engine/tuber_config.py`: property `chromakey_enabled` đọc `asset.chromakey.enabled` (tri-state).
3. `sync_engine/tuber_overlay.py`: `_auto_run_prerender()` truyền `config.chromakey_enabled` xuống.
4. `tests/sync_engine/test_tuber_overlay_pipeline.py`: 3 unit test cho `chromakey_enabled` (None/False/True).

### Hành động cho user

- Re-export body sang `.mov` (ProRes4444) / `.webm` (VP9 alpha) / PNG seq → code TỰ nhận alpha và bỏ qua chromakey. Hoặc đặt `asset.chromakey.enabled=false` để tắt tường minh.
- Còn dùng H264 green-screen thì khai `asset.chromakey.color` (vd `0x00FF00`) thay vì để auto-detect màu đen.

### Lưu ý còn lại

- Đường Remotion (`remotion_tuber/scripts/prepare-assets.ts`, mode `remotion`/`direct`) vẫn luôn chromakey — chưa sửa vì config hiện dùng `overlay.mode=prerender`. Cần port logic skip-alpha tương tự nếu sau này dùng Remotion với nguồn alpha.
- Chưa verify FFmpeg thật trên nguồn .mov/.webm (máy dev không có asset); cần chạy lại pipeline trên Colab để xác nhận frame body đặc/trong suốt đúng.

### File thay đổi

- `sync_engine/tuber_prerender.py`, `sync_engine/tuber_config.py`, `sync_engine/tuber_overlay.py`, `tests/sync_engine/test_tuber_overlay_pipeline.py`
- `docs/tuber-overlay-guide.md` (doc: key `asset.chromakey.enabled`, cảnh báo H264 không alpha, Phase 0 diagram)

---

## 2026-06-08: Prerender song song thật — ThreadPool → ProcessPoolExecutor (vượt GIL)

### Vấn đề

`prerender_character()` (bake body×mouth) đọc đúng `maxWorkers=4`, truyền đúng vào hàm, và _đã_ vào nhánh `ThreadPoolExecutor` — nhưng 705 frame vẫn mất ~48s như đơn luồng. **Nguyên nhân:** warp là CPU-bound Python thuần + PIL (`compute_affine`, `_invert_affine`, `apply_mouth_calibration`, dựng mask `ImageDraw.polygon`, `Image.transform/alpha_composite/paste`) → giữ **GIL** → thread không song song. (Thread chỉ hiệu quả ở composite groups / video chunks vì chúng gọi FFmpeg subprocess, nhả GIL.)

### Thay đổi (`sync_engine/tuber_prerender.py`)

1. Thêm cấp module: `_PRERENDER_CTX` + `_prerender_pool_init()` (nạp sprite + track 1 lần/worker) + `_prerender_body_worker(body_idx)` (render mọi mouth_state cho 1 body frame). Gom task theo **body_idx** → mở ảnh body 705→141 lần, mỗi worker chỉ nhận 1 int qua pickle.
2. Nhánh `max_workers>1`: đổi sang `ProcessPoolExecutor` với `mp_context=get_context("spawn")` (an toàn khi tiến trình cha `worker_task` đã spawn + init CUDA; child chỉ làm PIL). Tiến độ cộng dồn qua `as_completed` + mốc 100.
3. Nhánh `max_workers<=1`: giữ tuần tự (eager cache) — hành vi & output không đổi → test integration mặc định (`max_workers=1`) vẫn xanh.

### Lý do thiết kế

- `initargs` đều picklable (track dict, path str, list, int); worker/init/`warp_sprite_to_quad`/`apply_mouth_calibration` đều module-level → spawn re-import được.
- `_PRERENDER_CTX` reset trong mỗi child qua initializer → không rò trạng thái.

### Kiểm chứng

- `pytest tests/sync_engine/test_tuber_prerender.py` → 25 passed, 5 skipped (integration cần body-transparent/ chưa extract local; PIL không có trên máy dev nên nhánh process chỉ chạy thật trên Colab).
- **Còn phải verify trên Colab:** thời gian prerender giảm rõ + output PNG y hệt nhánh tuần tự (cùng thuật toán warp). Chưa chạy được nhánh process tại local do thiếu PIL.

### File thay đổi

- `sync_engine/tuber_prerender.py`

---

## 2026-06-08: V5 — Mouth vowel selection (spectral centroid, port ③ライブ実行)

### Tóm tắt

PNGTuber chọn khẩu hình chỉ theo biên độ (closed/half/open) nên 2 khẩu hình nguyên âm `e.png` (え) / `u.png` (う) không bao giờ được chọn. Port **Tầng 2** từ nút `③ライブ実行` của repo gốc [MotionPNGTuber](https://github.com/rotejin/MotionPNGTuber): chọn `e`/`u` bằng **spectral centroid** (FFT brightness) tại đỉnh sóng khi đang `open`. Đây là proxy độ sáng phổ (KHÔNG phải nhận diện phoneme thật): centroid thấp → `u`, cao → `e`.

### Thiết kế (offline, 2 tầng)

- **Tầng 1** (như cũ): RMS amplitude → per-frame level closed/half/open.
- **Tầng 2** (mới): tính centroid mỗi frame (`numpy.fft.rfft`, chuẩn hoá [0,1] theo Nyquist), ngưỡng `U_TH`/`E_TH` = percentile 20/80 của centroid các frame `open` trên toàn clip (offline → ổn định hơn live adaptive). Tại đỉnh sóng + cooldown → cập nhật khẩu hình sticky.
- **Backward-compatible tuyệt đối:** chỉ kích hoạt khi `mouth.mouthStates` có `e`/`u`. Cấu hình 3-state cũ, thiếu numpy, hoặc quá ít frame `open` → tự bỏ qua Tầng 2 (fail mềm), hành vi y hệt trước.

### Thay đổi chính

1. **`tuber_mouth_events.py`**: tách `_read_wav_samples`/`_samples_to_rms` (giữ `_read_wav_rms`); thêm `_frame_centroids` (numpy lazy+guarded), `_percentile`, `_select_vowel_shapes`, `_apply_vowel_selection`. `analyze_tts_amplitude` tái cấu trúc 3 bước (levels → vowel rewrite → collapse events) + nhận `mouth_states`, `peak_margin`, `min_vowel_interval_ms`, `vowel_low/high_percentile`.
2. **`tuber_config.py`**: accessor `mouth_peak_margin`, `mouth_min_vowel_interval_ms`, `mouth_vowel_low_pct`, `mouth_vowel_high_pct`.
3. **`tuber_overlay.py`**: `mouth_opts` truyền thêm `mouth_states` + vowel knobs; thêm `_prerender_is_stale()` → tự bake bổ sung frame `e`/`u` khi prerendered/ cũ thiếu state (không cần `resume.skipDone=false`).
4. **`pyproject.toml`**: khai báo tường minh `numpy>=1.24` (trước chỉ có gián tiếp qua imagehash/torch).
5. **Docs sync**: `docs/tuber-overlay-guide.md` (bảng `mouth` V5 + config mẫu 5 state + ghi rõ prerender-only) và `remotion_tuber/README.md` (Remotion path không hỗ trợ vowel).

### Resume / Hash

`compute_group_input_hash` đã băm `mouthEvents` → state e/u đổi → group tự re-render. Prerender staleness fix lo phần bake frame nguyên âm. Không sửa hash.

### Lưu ý

- Centroid là proxy thô → e/u đôi lúc lệch nguyên âm thật (đúng kỳ vọng, giống bản gốc).
- **Chỉ hỗ trợ ở `overlay.mode="prerender"`**; path Remotion (`mouthState.ts`) chưa có Tầng 2.
- Config mặc định `assets/tuber_overlay_config.json` GIỮ 3-state (asset `nike_loop_fix` chỉ có 3 sprite); ví dụ 5-state nằm trong docs.

### Tests

- `tests/sync_engine/test_tuber_mouth_events.py`: `TestLayer1_VowelSelection` (`_select_vowel_shapes`, `_percentile`, backward-compat 3-state), `TestLayer2_VowelFromWav` (WAV sin tần thấp→`u`/cao→`e`).
- `tests/test_matrix.yaml`: thêm entry Layer1/Layer2 vowel.

### File thay đổi

- `sync_engine/tuber_mouth_events.py`, `sync_engine/tuber_config.py`, `sync_engine/tuber_overlay.py`
- `pyproject.toml`, `docs/tuber-overlay-guide.md`, `remotion_tuber/README.md`
- `tests/sync_engine/test_tuber_mouth_events.py`, `tests/test_matrix.yaml`

---

## 2026-06-05: Fix resume.skipDone luôn miss — inputHash anchor sai vào video tái tạo

### Triệu chứng

Chạy `sync-video` lần 2 (cùng video/SRT/config), group đáng lẽ `skipped` nhưng status ra `done` → re-render lại từ đầu, resume vô tác dụng.

### Nguyên nhân gốc

`compute_group_input_hash()` (`tuber_status.py`) trộn `st_mtime_ns` của `video_stretched.mp4` (base video đã promote) vào hash. NHƯNG `sync-video` tái tạo `video_stretched.mp4` từ đầu mỗi lần chạy (stretch lại → mtime mới), `promote_media` copy lại file mới → hash lưu ở lần 1 KHÔNG BAO GIỜ khớp lần 2 → skip-check fail → re-render → status `done`.

(Ngay cả content-hash video stretched cũng không cứu được: NVENC không byte-deterministic giữa các lần encode.)

### Cách sửa

Đổi anchor của hash từ **intermediate tái tạo** sang **input ổn định**: video GỐC (`--video`). Mọi yếu tố stretch/mouth đã nằm trong `group_manifest` rồi, nên video gốc + manifest là đủ để định danh output 1 group, và video gốc không đổi giữa các lần chạy.

### Thay đổi chính

1. **`compute_group_input_hash(group_manifest, prerender_manifest, source_video)`** (`tuber_status.py`): param `stretched_video` → `source_video`; guard `None` (graceful, không raise).
2. **Đường truyền `source_video`**: `run_tuber_flow_all_in` truyền `Path(video_path)` → `render_groups_to_video` → `render_and_composite_groups` → 2 chỗ gọi hash trong `_process_one_group`.
3. **`run_manifest["sourceVideo"]`** (`tuber_manifest.py::build_run_manifest`): ghi path video gốc để repair tái lập đúng hash.
4. **`tuber_repair`**: đọc `run_manifest["sourceVideo"]`, fallback `baseVideo` nếu manifest cũ chưa có field (hash khác → re-render, an toàn).

### Tests

- `TestLayer1_GroupHash::test_hash_stable_when_intermediate_regenerated` — regression: cùng video gốc → hash khớp dù intermediate đổi mtime; đổi video gốc → hash đổi.
- `TestLayer1_GroupHash::test_hash_tolerates_missing_source_video` — `source_video=None` không raise.
- Sửa 2 test stale có sẵn (do default `direct` từ V4): `test_defaults` (assert `direct`), `test_pipe_cmd_no_overlay_frames_dir` (kiểm tra intent qua AST, bỏ docstring thay vì grep chuỗi).
- `tests/sync_engine/test_tuber_overlay_pipeline.py` + `test_tuber_repair.py`: 77 passed, 7 skipped.

### File thay đổi

- `sync_engine/tuber_status.py`, `sync_engine/tuber_overlay.py`, `sync_engine/tuber_manifest.py`, `cli/tuber_repair.py`
- `tests/sync_engine/test_tuber_overlay_pipeline.py`

---

## 2026-06-05: V4 — Direct RGB Pipe (bỏ I/O PNG sequence trung gian)

### Tóm tắt

V4 thêm `overlay.format: "direct"` (default) — pipe raw RGBA từ `prerendered/` cache thẳng vào FFmpeg stdin, bỏ hoàn toàn `overlay_frames/*.png` trung gian. Giữ `"png_sequence"` làm debug mode. Resume/skipDone không thay đổi.

### Thay đổi chính

1. **`overlay.format: "direct"` (default mới):** Thêm giá trị `"direct"` vào `overlay.format` accessor (`tuber_config.py`). Default đổi từ `"png_sequence"` → `"direct"`. Giá trị lạ → fallback `"direct"` + warning. Đã bỏ `pipeMode` riêng — gộp chung vào `format` để tránh 2 field cùng mô tả 1 thứ.

2. **`_pipe_prerender_frames()`** (`tuber_overlay.py`): Hàm mới thay `_build_prerender_frame_list` + `composite_group_from_stretched`. Đọc kích thước THẬT từ frame probe đầu tiên (PIL `.size`), dùng hybrid seek y hệt V3, bơm raw RGBA tuần tự vào `process.stdin`. Stderr ghi ra file log (không PIPE) để tránh deadlock buffer.

3. **`_make_mouth_lookup()`** (`tuber_overlay.py`): Tách hàm binary-search lookup mouth state ra module-level để dùng chung cho cả direct pipe và png_sequence path (DRY).

4. **Rẽ nhánh trong `_process_one_group()`**: Khi `use_prerender=True`, kiểm tra `overlay_format` → `"direct"` gọi `_pipe_prerender_frames`, `"png_sequence"` gọi `_build_prerender_frame_list` + `composite_group_from_stretched`. Fallback tự động: direct fail hết retry → thử png_sequence 1 lần cuối.

5. **Đường truyền `overlay_format`**: `run_tuber_flow_all_in` lấy `config.overlay_format` → `render_groups_to_video` → `render_and_composite_groups`. `tuber_repair` đọc `run_manifest["overlayFormat"]` (đã được ghi lúc all-in) với fallback `cfg.overlay_format`.

6. **Tests Layer 1 mới**: `TestLayer1_OverlayFormatConfig` (config accessor), `TestLayer1_DirectPipeCmd` (source inspection cmd + `_make_mouth_lookup`), `TestLayer1_RepairResolveFormat` (`test_tuber_repair.py`).

7. **Docs**: `tuber-overlay-guide.md` — thêm V4 summary, bảng `overlay.format` với 2 giá trị + debug workflow, note `overlay_frames/` chỉ tồn tại ở `png_sequence`. Sample config đổi sang `"direct"`.

### Đã bác bỏ

- **Split Alpha (2 HEVC video)**: 3 generation HEVC loss làm hỏng viền alpha. Không phù hợp production pipeline. Direct pipe đạt cùng lợi ích I/O mà lossless và 1 encode.

### File thay đổi

- `sync_engine/tuber_config.py`, `sync_engine/tuber_overlay.py`, `cli/tuber_repair.py`
- `assets/tuber_overlay_config.json`
- `tests/sync_engine/test_tuber_overlay_pipeline.py`, `tests/sync_engine/test_tuber_repair.py`, `tests/test_matrix.yaml`
- `docs/tuber-overlay-guide.md`

---

## 2026-06-05: V3 — Tối ưu hiệu năng + Resume cho pre-render pipeline

### Tóm tắt

V3 không thêm tính năng người dùng mới, tập trung vào hiệu năng, resume skip-done, sửa bug repair cho V2 prerender.

### Thay đổi chính

1. **Composite-seek (bỏ build_group_base):** Thay `build_group_base(video_gốc, trim+stretch lại)` bằng `composite_group_from_stretched(seek vào video_stretched.mp4, trim-by-frame, overlay 1 lần)`. Giảm từ 8 encode còn 5 encode (N=3 groups). Không thay đổi `render_final_video`.

2. **Fix tuber_repair.py cho V2 prerender:** Đọc `prerenderManifest` từ `run_manifest`, truyền `use_prerender`/`prerender_dir`/`prerender_manifest` xuống `render_groups_to_video`, bỏ hardcode `asset_id`/`nike_loop_fix`. Repair V2 không còn gọi npm/Remotion.

3. **Parallel prerender_character (`max_workers`):** `prerender_character()` nhận `max_workers: int`. Khi >1 dùng `ThreadPoolExecutor` cho loop body×mouth (170×3=510 frame độc lập). Body cache + sprite cache load trước ở main thread (read-only).

4. **Parallel groups (`max_workers`):** Vòng `for g in groups` trong `render_and_composite_groups` được bọc `ThreadPoolExecutor`. Retry per-group vẫn hoạt động. Sort kết quả theo group index. Mặc định 2 worker.

5. **Hash + skipDone (`resume.skipDone`):** `compute_group_input_hash()` = SHA-256(segments, renderStartFrame, renderDurationFrames, compOffsets, prerender outputSize, stretched_video mtime+size). Lưu vào `status.json["inputHash"]`. Khi `skipDone=true` + hash khớp + output validate OK → skip group (status=Skipped). Khi `skipDone=false` → xóa sạch groups/ + prerendered/ và dựng lại.

6. **Mode hybrid (mouth):** `analyze_tts_amplitude` nhận `mode` param. Hybrid: RMS gate (nói/im), debounce non-closed transitions bằng `cadenceMs` (half↔open phải chờ đủ cadence_frames). Silence luôn override → closed ngay. Thêm `"mode" key` vào `mouth_opts` chain.

7. **debugFrameOutput:** `debug.frameOutput.{enabled, marginFrames}`. Khi enabled, dump overlay + composited frames quanh boundary (start + end) vào `logs/debug_frames/{group_id}/` + `boundary.json`.

8. **Config mới:** 3 blocks trong `tuber_overlay_config.json`: `performance.maxWorkers`, `resume.skipDone`, `debug.frameOutput.{enabled, marginFrames}`.

### File thay đổi

- `sync_engine/tuber_overlay.py` — composite-seek (mới), parallel (sửa), skip-done, debug dump
- `sync_engine/tuber_prerender.py` — parallel prerender_character với max_workers
- `sync_engine/tuber_status.py` — compute_group_input_hash, inputHash field
- `sync_engine/tuber_config.py` — accessors: max_workers, resume_skip_done, debug_frame_output_enabled, debug_frame_margin
- `sync_engine/tuber_mouth_events.py` — mode hybrid debounce
- `cli/tuber_repair.py` — fix V2 prerender path (prerenderManifest, use_prerender, stretched_video)
- `assets/tuber_overlay_config.json` — blocks performance, resume, debug
- `tests/sync_engine/test_tuber_mouth_events.py` — Hybrid Layer 1
- `tests/sync_engine/test_tuber_overlay_pipeline.py` — CompositeSeek, Performance, Hash Layer 1
- `tests/sync_engine/test_tuber_repair.py` — MỚI: Repair Layer 1
- `tests/test_matrix.yaml` — 3 entry mới
- `docs/tuber-overlay-guide.md` — cập nhật

### Verification

- 60 Layer 1 tests pass (13 existing + 47 new/expanded)

---

## 2026-06-04: Bugfix — miệng mở trễ ~1s (bug gộp event \_merge_short_silence)

### Tóm tắt

- **Triệu chứng**: PNGTuber mở miệng trễ ~1s so với audio, xảy ra cả EdgeTTS lẫn Voicevox.
- **Chẩn đoán**: Sinh script `diag_mouth_onset.py` để in onset tại nhiều ngưỡng dB và events từ pipeline thật. Output trên `dubb-2.wav` (EdgeTTS) cho thấy onset thật = frame 5-6, nhưng `analyze_tts_amplitude()` trả `open` ở frame 39.
- **Root cause**: Bug trong `_merge_short_silence()` (tuber_mouth_events.py). Bước gộp consecutive same-state: `merged[-1]["frame"] = ev["frame"]` ghi đè frame onset sớm bằng frame muộn. Ví dụ: 2 đoạn `open` (7-34 và 39-44) quanh 1 `closed` ngắn 4 frame bị drop → khi gộp, onset `open` bị dời từ 7 → 39 (~1s). Đây là gốc rễ dùng chung cho CẢ hai engine.
- **Giả thuyết đã loại**: Voicevox lead-in (strip-silence). WAV onset ~0.17s, không phải 1s.
- **Fix**: Bỏ dòng ghi đè frame — giữ nguyên frame chuyển trạng thái sớm nhất khi gộp. Thêm `exclude_mute → whole_video` fallback khi không có mute segment (tường minh hoá policy).

### File thay đổi

- `sync_engine/tuber_mouth_events.py` — fix gộp event trong `_merge_short_silence()`
- `sync_engine/audio_assembler.py` — fallback exclude_mute → whole_video khi `has_mute_segment = False`
- `tests/sync_engine/test_tuber_mouth_events.py` — thêm regression test `test_merge_short_silence_keeps_early_onset_frame`
- `diag_mouth_onset.py` — script chẩn đoán onset miệng (giữ lại làm công cụ)

### Verification

Chạy diagnostic sau fix: `uv run python diag_mouth_onset.py --glob ".../dubb-2.wav"` → `open` ở frame ~6-7 (không còn 39). Tất cả 13 Layer1 unit test pass.

---

## 2026-06-04: V2 — Pre-render character + lip-sync amplitude

### Tóm tắt

- **Mục tiêu**: Giải quyết 2 vấn đề blocking V1 — miệng không khép khi im lặng, hiệu năng video dài.
- **Kết quả chính**: Port thuật toán affine warp từ TypeScript → Python/PIL, module `tuber_prerender.py` pre-render 170×N body×mouth, module `tuber_mouth_events.py` phân tích RMS amplitude TTS → mouthEvents chính xác đến frame.
- **Kiến trúc**: Pipeline thuần Python + FFmpeg khi có pre-render. Remotion code giữ nguyên làm reference, không gọi runtime. Config thêm `mouth.mode=amplitude`, `asset.prerender`, composite hỗ trợ offset cho character box.

### File tạo mới

| File                                                                        | Mô tả                                                                  |
| --------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| [`sync_engine/tuber_mouth_events.py`](../sync_engine/tuber_mouth_events.py) | Phân tích RMS amplitude TTS → mouthEvents [{frame, state}] per segment |
| [`sync_engine/tuber_prerender.py`](../sync_engine/tuber_prerender.py)       | Port mouthWarp.ts (affine 2-triangle), pre-render body×mouth → PNG     |

### File sửa

| File                                                                      | Thay đổi                                                                                                                   |
| ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| [`sync_engine/tuber_config.py`](../sync_engine/tuber_config.py)           | +mouth (mode, silenceDb, minSilenceMs, cadenceMs, mouthStates) +prerender config                                           |
| [`sync_engine/tuber_manifest.py`](../sync_engine/tuber_manifest.py)       | +compute_character_box, +mouthEvents build, +compWidth/compHeight/compOffset, +prerenderManifest trong run_manifest        |
| [`sync_engine/tuber_overlay.py`](../sync_engine/tuber_overlay.py)         | +composite offset_x/offset_y, +prerender path (\_build_prerender_frame_list, use_prerender param), +mouthEvents map lookup |
| [`assets/tuber_overlay_config.json`](../assets/tuber_overlay_config.json) | Config mới với mode=amplitude + prerender section                                                                          |
| [`docs/tuber-overlay-guide.md`](../docs/tuber-overlay-guide.md)           | Update architecture, mouth config, prerender docs                                                                          |

### Quyết định kiến trúc

1. **Pre-render là default**: Nếu `prerender_manifest.json` tồn tại → tự động dùng. Không cần config flag renderMode.
2. **Không CLI mới**: Pre-render qua `python -m sync_engine.tuber_prerender`. MouthEvents tích hợp sẵn vào `build_group_manifest()`.
3. **Backward compatible**: Manifest cũ không có mouthEvents → V1 legacy vẫn hoạt động.
4. **TTS amplitude → mouthEvents**: Đọc WAV trực tiếp từ TTS clip path có sẵn trong TimelineSegment. Không cần build speech_control_audio.wav riêng.
5. **Character box crop**: Pre-render output được crop từ 1920×1080 → kích thước ô character (vd 512×288) → composite với offset, giảm pixel xử lý ~14×.

### Trạng thái

- ✓ P1 (MouthEvents) + P2 (Pre-render) core code
- ✓ Config + docs
- ⬜ Test files: `test_tuber_mouth_events.py`, `test_tuber_prerender.py` (cần viết)
- ⬜ Smoke test Colab
- ⬜ Cleanup: xóa `remotion_tuber/` runtime calls khi V2 ổn định

---

## 2026-06-03: MotionPNGTuber — implement Python orchestration + test suite

### Tóm tắt

- **Mục tiêu**: Implement toàn bộ Python orchestration cho tuber overlay theo plan `motionpngtuber-remotion-prototype-plan.md`.
- **Kết quả chính**: 5 module Python mới (`tuber_config`, `tuber_manifest`, `tuber_artifacts`, `tuber_status`, `tuber_overlay`), CLI `tuber_repair`, tích hợp `--tuber-config` vào `sync_video`, sample config, test Layer 1-3 (46 passed, 7 skipped), docs mới.
- **Subproject `remotion_tuber/`**: Đã hoàn thiện trước đó (render driver, component, mouth warp, prepare-assets auto-detect màu nền `0x08A702`, width ưu tiên cho character).

### File tạo

- [`sync_engine/tuber_config.py`](../sync_engine/tuber_config.py) — Load/validate config, resolve layout (jobName sentinel, tuberRoot, artifactPolicy).
- [`sync_engine/tuber_manifest.py`](../sync_engine/tuber_manifest.py) — Build groups từ timeline (Phase F), export run/group manifest absolute paths (Phase H).
- [`sync_engine/tuber_artifacts.py`](../sync_engine/tuber_artifacts.py) — Promote media/final_render_inputs (Phase E), cleanup overlay_frames/failedGroups theo policy (Phase T), serialize/deserialize ImageOverlayEvent.
- [`sync_engine/tuber_status.py`](../sync_engine/tuber_status.py) — status.json per group (pending/running/done/failed/skipped).
- [`sync_engine/tuber_overlay.py`](../sync_engine/tuber_overlay.py) — Orchestration: prepare-assets, build group base (tái dùng `build_ffmpeg_batch_cmd` — B5), render driver, composite, validate, concat, retry, coordinator `run_tuber_flow_all_in`.
- [`cli/tuber_repair.py`](../cli/tuber_repair.py) — Late repair CLI: đọc run_manifest + final_render_inputs, render tuber muộn, gọi `render_final_video`.
- [`assets/tuber_overlay_config.json`](../assets/tuber_overlay_config.json) — Sample config với `chromakey: {color:"0x08A702", similarity:0.12, blend:0.1}`.
- [`docs/tuber-overlay-guide.md`](../docs/tuber-overlay-guide.md) — Hướng dẫn đầy đủ tham số JSON config + CLI repair + retry logic.
- [`tests/sync_engine/test_tuber_overlay_pipeline.py`](../tests/sync_engine/test_tuber_overlay_pipeline.py) — Test Layer 1 (frame math, group, config, status, serialize), Layer 2 (manifest export, artifact promote), Layer 3 (composite, validate, retry/cleanup mock). 46 passed, 7 skipped (thiếu GPU).

### File sửa

- [`cli/sync_video.py`](../cli/sync_video.py) — Thêm `--tuber-config` arg, load config + swap `stretched_video` → `video_stretched_with_tuber` + fallback ở Phase 5.
- [`pyproject.toml`](../pyproject.toml) — Thêm `tuber-repair = "cli.tuber_repair:main"`.
- [`tests/test_matrix.yaml`](../tests/test_matrix.yaml) — Thêm 3 entry Tuber Overlay Pipeline Layer 1/2/3.
- [`docs/colab-guide.md`](../docs/colab-guide.md) — Thêm `--tuber-config` vào bảng tham số sync-video + section 2.12 Tuber Overlay + CLI `tuber-repair`.

### Quyết định kiến trúc

1. **Width ưu tiên trong `resolveCharacterBox`**: Có `width` → suy `height = width / aspect(mouth_track)`, bỏ qua `height`. Không cần cờ `maintainAspect`. Giữ `objectFit:'fill'` để body + canvas đồng scale, miệng không lệch.
2. **Auto-detect màu nền chromakey**: `prepare-assets` lấy median 4 góc frame đầu làm key color, không hardcode `0x00FF00`. Asset hiện tại nền `~0x08A702`.
3. **Concatenation ngoài retry loop**: Concat `-c copy` chỉ chạy 1 lần sau khi tất cả group pass validate. Chỉ render + composite + validate mới có retry.
4. **`attempt` counter chỉ tăng khi composite/validate fail**: Batch render fail → re-render riêng group đó trong cùng attempt (soft fallback), không tính là retry.
5. **Dùng chung `detect_hevc_nvenc` (SSOT)**: Test skip GPU dùng `_GPU_OK = detect_hevc_nvenc()` khớp fixture `use_gpu` trong `tests/conftest.py`.

### Trạng thái hiện tại

- ✓ 5 module Python orchestration đã implement (tuber_config, tuber_manifest, tuber_artifacts, tuber_status, tuber_overlay).
- ✓ [`cli/tuber_repair.py`](../cli/tuber_repair.py) hoàn chỉnh với late repair flow.
- ✓ Tích hợp vào [`sync_video`](../cli/sync_video.py) (load config → resolve layout → build groups → base.mp4 → render → composite → validate → concat → fallback).
- ✓ Sample config [`assets/tuber_overlay_config.json`](../assets/tuber_overlay_config.json) có chromakey params.
- ✓ Test Layer 1-3 (46 passed, 7 skipped do thiếu GPU NVENC).
- ✓ [`docs/tuber-overlay-guide.md`](../docs/tuber-overlay-guide.md) hoàn chỉnh.
- ✓ [`docs/colab-guide.md`](../docs/colab-guide.md) đã cập nhật.
- ✓ Không hồi quy: [`test_tuber_remotion_validation.py`](../tests/sync_engine/test_tuber_remotion_validation.py) vẫn 6 passed.

### Pending / Next steps

- Smoke test end-to-end trên Colab với video thật 30-60s: `sync-video --tuber-config` → kiểm tra output có tuber overlay + alpha sạch.
- Khi có GPU: bật `enabled: true` trong config, chạy `REMOTION_TUBER_E2E=1` verify Layer 4 Remotion thật.

## 2026-06-01: Sync-video audio mix volume schema

### Tóm tắt

- **Mục tiêu**: Thêm schema volume rõ ràng cho Phase 3 audio assembly của [`sync-video`](../cli/sync_video.py): [`non_voicevox_tts_volume`](../sync_engine/audio_assembler.py:611) cho TTS non-Voicevox và [`mute_audio_volume`](../sync_engine/audio_assembler.py:612) cho audio trong mute chunks.
- **Kết quả chính**: [`compress_tts_clip()`](../sync_engine/audio_assembler.py:159) không còn hardcode boost `volume=1.75`; volume non-Voicevox lấy từ render config, còn Voicevox/Voicevox Nemo tiếp tục dùng `volume_scale` trong [`config/tts_config.yaml`](../config/tts_config.yaml).
- **Tác động workflow**: Backward compatibility được giữ bằng default runtime/config `non_voicevox_tts_volume=1.75`; `mute_audio_volume=1.0` mặc định không đổi loudness mute audio hiện tại.

### File sửa

- [`sync_engine/audio_assembler.py`](../sync_engine/audio_assembler.py) — Đọc volume mới từ `audio_mix_config`, áp gain cho non-Voicevox TTS và mute chunks (`original`, `vocals`, `instrumental`), bỏ qua render volume cho provider bắt đầu bằng `voicevox`.
- [`assets/default_render_config.json`](../assets/default_render_config.json) — Thêm `non_voicevox_tts_volume` và `mute_audio_volume` vào block `audio_mix` mặc định.
- [`docs/sync-video-guide.md`](../docs/sync-video-guide.md) — Cập nhật schema `audio_mix`, mô tả rõ Voicevox family dùng `volume_scale` riêng trong TTS config.
- [`tests/sync_engine/test_audio_assembler.py`](../tests/sync_engine/test_audio_assembler.py) — Thêm Layer 1 regression tests cho filter volume non-Voicevox, Voicevox ignore render volume, và truyền `mute_audio_volume` vào mute chunk.

### Verification

- `python -m compileall -q sync_engine/audio_assembler.py tests/sync_engine/test_audio_assembler.py && python -m pytest tests/sync_engine/test_audio_assembler.py -v` → `15 passed`.

### Trạng thái hiện tại

- ✓ Schema [`audio_mix`](../docs/sync-video-guide.md:324) đã có `non_voicevox_tts_volume`, `mute_audio_volume`, `ambient_volume`, `bgm_volume`.
- ✓ Voicevox/Voicevox Nemo không bị ảnh hưởng bởi render volume mới.
- ✓ Mute chunk volume có default `1.0` và chỉ đổi khi render config override.
- ✓ Compile và test mục tiêu cho [`tests/sync_engine/test_audio_assembler.py`](../tests/sync_engine/test_audio_assembler.py) đã pass.

### Pending / Next steps

- Không còn pending trong phạm vi thay đổi schema volume audio hiện tại.

## 2026-06-01: Fix pre-cut hybrid-copy keep topology collapse

### Tóm tắt

- **Mục tiêu**: Sửa lỗi [`hybrid-copy`](../cli/pre_cut_video.py:53) chỉ tạo 2 file `keep_*` trong khi cùng remove SRT với [`reencode-smooth`](../cli/pre_cut_video.py:54) tạo đúng 4 file.
- **Nguyên nhân**: Nhánh [`run_pre_cut()`](../utils/video_cutter.py:434) đang dùng [`expand_to_keyframes()`](../utils/video_cutter.py:209) để mở rộng **remove ranges** tới keyframe trước/sau. Khi GOP/keyframe thưa, các vùng xóa sau mở rộng bị overlap rồi bị [`normalize_and_merge()`](../utils/video_cutter.py:178) gộp lại, làm mất topology các đoạn keep ở giữa.
- **Kết quả chính**: [`hybrid-copy`](../utils/video_cutter.py:488) vẫn query keyframes để fail-fast nhưng không còn dùng keyframe expansion để thay đổi remove ranges; topology SRT sau safe margin/normalize là SSOT cho số lượng `keep_*`.

### File sửa

- [`utils/video_cutter.py`](../utils/video_cutter.py) — Đổi nhánh [`method == "hybrid-copy"`](../utils/video_cutter.py:488) để đặt `final_remove = normalized`, tránh merge các remove range chỉ vì keyframe expansion.
- [`tests/utils/test_video_cutter.py`](../tests/utils/test_video_cutter.py) — Thêm [`TestLayer2_HybridCopyKeepTopology`](../tests/utils/test_video_cutter.py:742) mô phỏng case 3 remove ranges thực tế và assert tạo đúng 4 `keep_*` parts; chỉnh temp cleanup tests để tạo fake part theo từng FFmpeg command.

### Verification

- `python -m compileall -q utils/video_cutter.py tests/utils/test_video_cutter.py && python -m pytest tests/utils/test_video_cutter.py -v` → `58 passed`.

### Trạng thái hiện tại

- ✓ [`hybrid-copy`](../utils/video_cutter.py:488) không còn collapse các keep ranges hợp lệ do mở rộng vùng xóa theo keyframe.
- ✓ Case 3 remove SRT ranges với safe margin `100ms` tạo 4 keep ranges/4 temp files trong regression test.
- ✓ [`reencode-smooth`](../utils/video_cutter.py:495) vẫn giữ logic snap-to-frame-grid riêng.

### Pending / Next steps

- Chạy lại [`pre-cut-video`](../cli/pre_cut_video.py:119) với video Colab thực tế để xác nhận số file `keep_*` khớp kỳ vọng end-to-end.

## 2026-06-01: HEVC NVENC runtime probe shared utility

### Tóm tắt

- **Mục tiêu**: Sửa false-positive khi phát hiện `hevc_nvenc`: FFmpeg có advertise encoder nhưng encode thật fail do thiếu CUDA/NVIDIA runtime (`libcuda.so.1`).
- **Kết quả chính**: [`utils.ffmpeg_probe.detect_hevc_nvenc()`](../utils/ffmpeg_probe.py:45) là SSOT dùng chung cho runtime probe 2 bước: kiểm tra `ffmpeg -encoders`, sau đó dummy encode 1 frame bằng đúng tham số production `hevc_nvenc -preset p4 -tune hq -cq 28`.
- **Tác động workflow**: [`sync_engine.video_processor.process_video_chunks_parallel()`](../sync_engine/video_processor.py:231), [`sync_engine.renderer.render_final_video()`](../sync_engine/renderer.py:199), [`utils.video_cutter.run_pre_cut()`](../utils/video_cutter.py:443) và fixture test dùng cùng detector, tránh drift giữa sync-video, renderer, pre-cut và tests.

### File sửa

- [`utils/ffmpeg_probe.py`](../utils/ffmpeg_probe.py) — Thêm shared runtime probe có cache, lưu lý do fail gần nhất qua [`utils.ffmpeg_probe.get_hevc_nvenc_unavailable_reason()`](../utils/ffmpeg_probe.py:33), reset cache cho tests và hằng số encoder chung.
- [`sync_engine/video_processor.py`](../sync_engine/video_processor.py) — Dùng shared detector/encoder args, fail-fast phase video chunks kèm chi tiết probe.
- [`sync_engine/renderer.py`](../sync_engine/renderer.py) — Import detector và encoder args trực tiếp từ shared utils để final render không phụ thuộc vào module video processor.
- [`utils/video_cutter.py`](../utils/video_cutter.py) — Bỏ detector local chỉ đọc encoder list; flow `reencode-smooth` dùng shared runtime probe và đưa reason vào RuntimeError.
- [`tests/conftest.py`](../tests/conftest.py) — Fixture `use_gpu` gọi shared detector thay vì duplicate dummy encode.
- [`tests/utils/test_ffmpeg_probe.py`](../tests/utils/test_ffmpeg_probe.py) — Thêm regression tests cho false-positive `ffmpeg -encoders`, cache success và thiếu FFmpeg.
- [`tests/sync_engine/test_video_processor.py`](../tests/sync_engine/test_video_processor.py), [`tests/sync_engine/test_concat_demuxer.py`](../tests/sync_engine/test_concat_demuxer.py), [`tests/sync_engine/test_sync_video_pipeline.py`](../tests/sync_engine/test_sync_video_pipeline.py) — Chuyển test/users sang detector shared utils.
- [`tests/test_matrix.yaml`](../tests/test_matrix.yaml) — Thêm entry Layer 1 cho FFmpeg probe shared utility.

### Verification

- `python -m compileall -q utils sync_engine tests && python -m pytest tests/utils/test_ffmpeg_probe.py tests/sync_engine/test_video_processor.py tests/utils/test_video_cutter.py tests/sync_engine/test_image_overlay.py -v` → `71 passed, 1 skipped`.
- `python -m pytest tests/sync_engine/test_sync_video_pipeline.py -v` → `0 collected / 1 skipped`, exit code `5` vì module pipeline bị skip toàn bộ trong môi trường hiện tại trước collection đầy đủ.

### Quyết định kiến trúc

1. **Runtime usability là SSOT trong utils**: [`ffmpeg -encoders`](../utils/ffmpeg_probe.py:63) chỉ là bước lọc nhanh; dummy encode mới quyết định `hevc_nvenc` usable hay không.
2. **Dùng đúng tham số production trong probe**: Probe dùng chung [`utils.ffmpeg_probe.HEVC_NVENC_VIDEO_ARGS`](../utils/ffmpeg_probe.py:17) để tránh pass giả với cấu hình khác.
3. **Cache theo process**: Kết quả probe được cache để tránh mỗi phase/test gọi FFmpeg dummy encode nhiều lần; tests có [`utils.ffmpeg_probe.reset_hevc_nvenc_probe_cache()`](../utils/ffmpeg_probe.py:38).
4. **Không đặt detector trong sync_engine**: Detector nằm ở [`utils/ffmpeg_probe.py`](../utils/ffmpeg_probe.py) để sync-video, final renderer, pre-cut CLI path và tests dùng chung.

### Trạng thái hiện tại

- ✓ False-positive NVENC do thiếu `libcuda.so.1` đã được chặn ở shared runtime probe.
- ✓ [`sync_engine/renderer.py`](../sync_engine/renderer.py) và [`utils/video_cutter.py`](../utils/video_cutter.py) không còn giữ logic detect riêng gây drift.
- ✓ Test Layer 1 cho video processor không phụ thuộc OpenCV/Numpy nhờ lazy import trong fixture Layer 2.
- ✓ Target compile/tests đã chạy pass; pipeline integration vẫn skip trong môi trường hiện tại do dependency/hardware.

### Pending / Next steps

- Chạy lại sync-video end-to-end trên máy có NVIDIA driver/CUDA runtime usable để xác nhận nhánh render thật không bị skip.
