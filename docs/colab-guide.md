# Hướng dẫn sử dụng CharenjiZukan trên Google Colab

Tài liệu này hướng dẫn cách sử dụng project CharenjiZukan trên Google Colab với `uv` - công cụ quản lý package Python nhanh.

---

## 1. Cài đặt môi trường

### 1.1. Cấu hình Google Colab Secrets

Trước khi bắt đầu, cần cấu hình Secrets để bảo mật token và API keys:

1. Trong Google Colab, click vào biểu tượng 🔑 **Secrets** ở sidebar bên trái
2. Thêm các secret sau:

| Tên Secret     | Giá trị                                 | Mô tả                        |
| -------------- | --------------------------------------- | ---------------------------- |
| `github_token` | `ghp_xxxx...` hoặc `github_pat_xxxx...` | GitHub Personal Access Token |
| `gemini_token` | `AIza...`                               | Gemini API Key               |
| `hf_token`     | `hf_...`                                | Hugging Face Access Token    |

**Cách tạo GitHub Personal Access Token:**

1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Tick `repo` (để truy cập private repo)
4. Copy token và thêm vào Secrets

**Cách tạo Gemini API Key:**

1. Truy cập [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy key và thêm vào Secrets

**Cách tạo Hugging Face Token:**

1. Truy cập [Hugging Face Settings - Tokens](https://huggingface.co/settings/tokens)
2. Click "Create new token"
3. Chọn quyền Read
4. Copy token và thêm vào Secrets

### 1.2. Cài đặt uv và clone project

```colab
# Cài đặt uv
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ['PATH'] += ':/root/.local/bin'

# Clone project (Private Repository với Secrets)
from google.colab import userdata
token = userdata.get('github_token')
!git clone https://{token}@github.com/ThanhVoKim/CharenjiZukan.git /content/CharenjiZukan
%cd /content/CharenjiZukan

!pip install -q pyyaml pytest

# Cài đặt project ở chế độ editable (để sử dụng CLI commands)
!uv pip install -e .

# Cài đặt rubberband-cli (cần cho time-stretch)
!apt-get install -y rubberband-cli
```

> **Lưu ý:** Sử dụng `userdata.get()` để lấy token từ Secrets, không hardcode token vào code để tránh lộ thông tin nhạy cảm.

### 1.3. Cài đặt môi trường cho DeepSeek-OCR-2 (tùy chọn - cho trích xuất phụ đề cứng)

DeepSeek-OCR-2 là mô hình AI trên Hugging Face, không phải là một package Python thông thường. Mã nguồn và weights của mô hình sẽ được tự động tải về thông qua thư viện `transformers` khi chạy script.

Các thư viện nền (như `transformers`, `torch`, `einops`, `PyMuPDF`) đã được cấu hình trong `pyproject.toml` và sẽ tự động cài đặt khi chạy `!uv pip install -e .`. Tuy nhiên, bạn cần cài thêm `flash-attn` để tăng tốc xử lý:

```colab
# Cài đặt Flash Attention (yêu cầu cho DeepSeek-OCR-2)
!uv pip install -p .venv/bin/python flash-attn==2.7.3 --no-build-isolation
```

### 1.4. Cài đặt Qwen3-VL OCR (tùy chọn)

Nếu muốn dùng Qwen3-VL thay cho DeepSeek-OCR-2 (có tốc độ chậm hơn nhưng đọc chính xác hơn, đặc biệt khi dùng bản Thinking), cần cài đặt thủ công do yêu cầu phiên bản `transformers` khác với DeepSeek:

> **⚠️ Lưu ý phiên bản transformers:**
>
> - DeepSeek-OCR-2: yêu cầu `transformers==4.45.2`
> - Qwen3-VL: yêu cầu `transformers>=4.57.0`
>   Hai model **không thể dùng chung** một phiên bản `transformers` cùng lúc.

```colab
# Nâng transformers cho Qwen3-VL
!uv pip install --upgrade "transformers>=4.57.0"

# Cài qwen-vl-utils (phiên bản khuyến nghị)
!uv pip install qwen-vl-utils==0.0.14
```

**Chạy với Qwen3-VL (Bước 2):**

```colab
# Đọc nhanh (Instruct)
!uv run video-ocr /content/video.mp4 --ocr-model Qwen/Qwen3-VL-8B-Instruct --device cuda

# Đọc chính xác với suy luận (Thinking)
!uv run video-ocr /content/video.mp4 --ocr-model Qwen/Qwen3-VL-8B-Thinking --device cuda
```

---

## 2. Các script chính

### 2.0 Speech-to-Text với WhisperX (cho video có giọng đọc rõ ràng)

#### Cài đặt môi trường

Nếu cần chuyển video thành subtitle, cài đặt thêm WhisperX bằng Optional Dependency `whisper` đã cấu hình sẵn trong project:

```colab
# !uv pip install -e .[whisper]
# Tạo môi trường ảo riêng biệt cho Whisper
!uv venv .venv-whisper
# Cài đặt whisperx và các thư viện cần thiết vào môi trường này
!uv pip install -p .venv-whisper/bin/python whisperx pydub
# Cài đặt thư viện hệ thống
!apt install libcudnn8 libcudnn8-dev -y
# Đặt biến môi trường
%env MPLBACKEND=agg
%env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
%env LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/
```

Chuyển video/audio thành file subtitle `.srt` dùng WhisperX. Công cụ đã được tối ưu hóa cho **Batch Processing** (chạy nhiều file cùng lúc) giúp tiết kiệm VRAM và giảm thời gian tải mô hình.

#### Chạy 1 file đơn lẻ

```colab
# Lưu ý: Gọi file python bên trong .venv-whisper
!.venv-whisper/bin/python cli/whisper_srt.py \
  --input /content/7620801394840177960_hd.mp4 \
  --model large-v2 \
  --lang zh \
  --pause-thresh 100 \
  --batch-size 32 \
  --verbose
```

_Output mặc định sẽ lưu cùng thư mục với file input: `/content/video.srt`_

#### Chạy hàng loạt nhiều file (Tối ưu VRAM) bằng JSON Batch

Để tối ưu, hãy truyền vào một file JSON chứa danh sách các task. WhisperX sẽ tải model đúng **1 lần** cho toàn bộ danh sách, giúp tăng tốc cực nhanh.

Ví dụ file `tasks.json`:

```json
[
  {
    "input": "/content/Video/bai1.mp4",
    "output": "/content/drive/MyDrive/PhuDe/bai1.srt"
  },
  {
    "input": "/content/Video/bai2.mp4",
    "output": "/content/drive/MyDrive/PhuDe/bai2.srt"
  }
]
```

Chạy CLI với file JSON:

```colab
!.venv-whisper/bin/python cli/whisper_srt.py \
  --task-file tasks.json \
  --model large-v3 \
  --batch-size 32
```

#### Bảng tham số chính

| Tham số             | Mô tả                                                                     | Mặc định                            |
| ------------------- | ------------------------------------------------------------------------- | ----------------------------------- |
| `--input`, `-i`     | File video hoặc audio đầu vào                                             | (bắt buộc nếu không dùng task-file) |
| `--task-file`, `-t` | File JSON cấu hình chạy hàng loạt (`{"input": "...", "output": "..."}`)   | (không dùng)                        |
| `--output`, `-o`    | File .srt hoặc folder đầu ra (chỉ dùng với `--input`)                     | `<input_dir>/<name>.srt`            |
| `--model`, `-m`     | Model Whisper (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`) | `large-v3`                          |
| `--lang`, `-l`      | Ép buộc mã ngôn ngữ (`vi`, `en`, `ja`, `zh`...)                           | (auto-detect)                       |
| `--batch-size`      | Batch size quá trình nhận dạng                                            | `16` (GPU L4 dùng `32`)             |
| `--max-speech-ms`   | Cắt các câu thoại dài hơn ngưỡng này (milliseconds)                       | `6000`                              |
| `--pause-thresh`    | Khoảng lặng tối thiểu để cắt câu (nếu < 300ms sẽ tắt cắt thông minh)      | `800`                               |
| `--min-seg-ms`      | Gộp các câu thoại ngắn hơn ngưỡng này (tránh đọc cụt lủn)                 | `1000`                              |
| `--maxlen`          | Ký tự tối đa mỗi dòng (ngắt dòng nếu dài hơn)                             | `0` (KHÔNG ngắt dòng)               |
| `--vad-chunk`       | Giới hạn cứng (giây) cho mỗi đoạn audio mà VAD cắt ra                     | `0` (mặc định 30s của WhisperX)     |
| `--max-chars`       | Tách câu theo độ dài ký tự tối đa                                         | `0` (auto: CJK 35, Latin 80)        |
| `--no-align`        | Bỏ qua bước Forced Alignment (nhanh hơn nhưng kém chính xác thời gian)    | (tắt)                               |
| `--verbose`         | Bật log chi tiết                                                          | (tắt)                               |

### 2.0 b. Speech-to-Text với Qwen3-ASR (qwen3-asr-srt)

Chuyển video/audio thành file subtitle `.srt` dùng mô hình **Qwen3-ASR** (backend Transformers, không dùng vLLM). Đây là lựa chọn thay thế WhisperX khi cần nhận dạng giọng nói tiếng Trung (CJK) với độ chính xác cao và timestamp chi tiết từng từ.

#### Cài đặt môi trường

Qwen3-ASR yêu cầu `transformers` và `flash-attn`. Do `flash-attn` là package biên dịch nặng, project đã cấu hình sẵn prebuilt wheel trong optional dependency `[qwen-asr]`:

> **⚠️ Lưu ý phiên bản môi trường:**
> Prebuilt wheel hiện tại yêu cầu: **Python 3.12**, **CUDA 12.8**, **PyTorch 2.9**, **Linux x86_64**.

```colab
!uv venv .venv-qwen3asr
# Cài đặt qwen-asr và flash-attn thẳng vào môi trường .venv-qwen3asr
!uv pip install -p .venv-qwen3asr/bin/python qwen-asr[vllm]
!uv pip install -p .venv-qwen3asr/bin/python https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl
```

#### Chạy 1 file đơn lẻ

```colab
!.venv-qwen3asr/bin/python cli/qwen3_asr.py \
  --input /content/video.mp4 \
  --output /content/subs/ \
  --language Chinese \
  --max-chars 15 \
  --batch-size 32 \
  --max-new-tokens 1024 \
  --offset-seconds 0.24
```

_Output mặc định sẽ tạo 3 file trong thư mục output:_

- `/content/subs/video.srt` — File phụ đề
- `/content/subs/video.txt` — Toàn bộ văn bản transcript
- `/content/subs/video.json` — Dữ liệu timestamp gốc (đã merge dấu câu)

#### Chạy hàng loạt nhiều file (Batch JSON)

Tương tự, truyền vào file JSON chứa danh sách task để xử lý batch:

Ví dụ file `tasks.json`:

```json
[
  {
    "input": "/content/Video/bai1.mp4",
    "output": "/content/drive/MyDrive/PhuDe/bai1.srt"
  },
  {
    "input": "/content/Video/bai2.mp4",
    "output": "/content/drive/MyDrive/PhuDe/bai2.srt"
  }
]
```

Chạy CLI với file JSON:

```colab
!uv run qwen3-asr-srt \
  --task-file tasks.json \
  --language Chinese \
  --batch-size 32
```

#### Bảng tham số chính

| Tham số             | Mô tả                                                                                     | Mặc định                            |
| ------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------- |
| `--input`, `-i`     | File video hoặc audio đầu vào                                                             | (bắt buộc nếu không dùng task-file) |
| `--task-file`, `-t` | File JSON cấu hình chạy hàng loạt (`{"input": "...", "output": "..."}`)                   | (không dùng)                        |
| `--output`, `-o`    | File .srt hoặc folder đầu ra (chỉ dùng với `--input`)                                     | `<input_dir>/<name>.srt`            |
| `--language`, `-l`  | Ngôn ngữ audio (`Chinese`, `English`, `Japanese`...)                                      | `Chinese`                           |
| `--max-chars`       | Số ký tự tối đa trên mỗi dòng phụ đề (CJK thường 15, Latin 40), đặt 0 để tắt              | `15`                                |
| `--min-chars`       | Số ký tự tối thiểu trên mỗi dòng phụ đề, đặt 0 để tắt                                     | `8`                                 |
| `--batch-size`      | Batch size cho inference (tăng lên 32~64 nếu GPU L4 22GB)                                 | `32`                                |
| `--max-new-tokens`  | Số token tối đa sinh ra mỗi chunk (giảm để tiết kiệm VRAM, tăng nếu chunk dài bị cắt cụt) | `1024`                              |
| `--offset-seconds`  | Độ lệch bù trừ thời gian (giây, ví dụ: 0.24 = 6 frames @ 25fps)                           | `0.24`                              |
| `--model-path`      | Đường dẫn model ASR trên HuggingFace hoặc local                                           | `Qwen/Qwen3-ASR-1.7B`               |
| `--aligner-path`    | Đường dẫn model Forced Aligner                                                            | `Qwen/Qwen3-ForcedAligner-0.6B`     |
| `--device`, `-d`    | Thiết bị chạy (`cuda:0`, `cuda:1`, `cpu`)                                                 | `cuda:0`                            |
| `--verbose`         | Bật log chi tiết                                                                          | (tắt)                               |

#### Lưu ý quan trọng

- **Flash Attention**: Bắt buộc phải có `flash-attn` để tối ưu VRAM. Nếu không cài được prebuilt wheel, có thể cài từ source (rất lâu trên Colab):
  ```colab
  !uv pip install flash-attn --no-build-isolation
  ```
- **VRAM với audio dài (~1h)**: Vì CLI luôn bật timestamp, `qwen_asr` cắt audio thành chunk ~3 phút rồi gom theo `--batch-size`. VRAM sẽ leo dần tới đỉnh rồi plateau (do PyTorch giữ reserved memory — không phải leak); đỉnh tỉ lệ với `batch_size × max_new_tokens`. Trên L4 22GB, `--batch-size 16` dễ OOM. Để khắc phục:
  - CLI đã tự đặt `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (giảm phân mảnh, dùng `setdefault` nên có thể override bằng env var của bạn).
  - Hạ `--max-new-tokens` (mặc định 1024) hoặc `--batch-size` nếu vẫn sát trần; tăng `--max-new-tokens` nếu thấy phụ đề cuối chunk bị cắt cụt.

### 2.0 c. SRT nguồn từ OCR — Phục hồi dấu câu (LLM) + Tách câu theo dấu câu (align-srt)

Hướng tiếp cận thay thế cho `qwen3-asr-srt` khi muốn **text chính xác tuyệt đối**: lấy text từ
**OCR phụ đề cứng** (ground-truth), LLM **chỉ thêm dấu câu** (không đổi chữ), rồi `align-srt`
gom các block subtitle liên tiếp thành câu hoàn chỉnh theo dấu ngắt câu. Gồm **3 bước CLI rời,
độc lập** — mỗi bước retry/debug riêng được, không CLI nào đọc `flow.yaml` của CLI khác:

```
video-ocr <video> --config flow.yaml  → <stem>_<box>.srt           (chỉ OCR)
punctuate-srt <stem>_<box>.srt        → <stem>_<box>_punct.srt
align-srt <stem>_<box>_punct.srt      → <stem>_<box>_punct_seg.srt
```

> **`align-srt` v2 — thuần CPU, không model, không GPU:** đọc `_punct.srt`, gom các block liên
> tiếp cho tới khi gặp block kết thúc bằng dấu ngắt câu (`.!?:。！？：；`), tạo 1 block câu mới.
> Timestamp lấy thẳng từ biên block OCR gốc — không nội suy, không tách vocal, không VRAM.
> Thêm `--split-on-comma` để cắt cả tại dấu phẩy (，、,;) — mặc định tắt.
>
> **NOTE (hoãn lại):** Dấu ngắt câu nằm GIỮA một block subtitle (vd `去学校。然后`) → v1 mặc kệ,
> không cắt tại đó. Khi cần: nội suy timestamp theo tỉ lệ ký tự trong block.

#### Bước phục hồi dấu câu — `punctuate-srt` (CLI riêng)

Tách hẳn khỏi `video-ocr` để dễ **retry/debug** (LLM tốn kém, hay phải gọi lại). Mọi tham số LLM
(provider, language, batch_size, use_full_context, prompt, response_parser) lấy **DUY NHẤT** từ
`config/llm_tasks/punctuation_restoration.yaml` (SSOT, đúng pattern `srt_translation.yaml`). Đổi task
config khác bằng `--task-config <path>`; override nhanh bằng `--lang`, `--batch`, `--no-context`...

> Dùng provider **vertexAI** (xem `config/llm/vertexai.yaml`, cấu hình ADC như `translate-srt`). LLM
> chỉ được chèn dấu câu; mỗi dòng được kiểm tra "xoá dấu phải khớp text gốc", lệch thì giữ nguyên text
> OCR (chống ảo giác). Batch nào fail sau hết retry → cả batch giữ text gốc (stats `… block giữ gốc`).
> Block phụ đề có thể là **mảnh câu** — prompt không ép mỗi dòng thành câu hoàn chỉnh.
>
> Hạ tầng batch (vòng lặp lô + context cache + integrity retry) dùng chung với `translate-srt` qua
> `llm_ai/srt_batch/`; `punctuate-srt` chỉ là wrapper mỏng thêm validator chống ảo giác.
>
> Cờ `--flatten` (mặc định bật) sinh thêm `_flat.txt` — không còn cần cho `align-srt` v2 nhưng
> vẫn hữu ích nếu dùng forced-aligner của `sync-video`.

#### Chạy

```colab
# Bước 1: OCR thuần (cần môi trường OCR ở Mục 1.3/1.4)
!uv run video-ocr /content/video.mp4 --config /content/flow.yaml

# Bước 2: Phục hồi dấu câu (cần vertexAI ADC). Retry chỉ cần chạy lại đúng lệnh này.
!uv run punctuate-srt /content/video_subtitle.srt --lang Chinese

# Bước 3: Tách câu theo dấu câu (thuần CPU, không cần GPU/model)
!uv run align-srt /content/video_subtitle_punct.srt
```

#### Bảng tham số `punctuate-srt`

| Tham số             | Mô tả                                                                                                 | Mặc định                                        |
| ------------------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `--input`, `-i`     | File SRT nguồn (text OCR chưa dấu)                                                                    | (bắt buộc nếu không `--task-file`)              |
| `--task-file`, `-t` | JSON `[{"input": "...", "output": "..."}]`                                                            | (không dùng)                                    |
| `--output`, `-o`    | File `.srt` hoặc folder đầu ra                                                                        | `<input>_punct.srt`                             |
| `--task-config`     | Task YAML — SSOT mọi tham số LLM                                                                      | `config/llm_tasks/punctuation_restoration.yaml` |
| `--lang`, `-l`      | Ngôn ngữ nguồn (override task config)                                                                 | task config → `Chinese`                         |
| `--batch`, `-b`     | Số block SRT mỗi batch (override task config)                                                         | task config → `30`                              |
| `--no-context`      | Tắt full-context (mặc định bật)                                                                       | (context bật)                                   |
| `--no-flatten`      | Không sinh `_flat.txt` (mặc định sinh; không còn cần cho `align-srt` v2)                              | (flatten bật)                                   |
| `--provider`, `-p`  | Override provider (gemini/openai/vertexai)                                                            | task config / provider_chain                    |
| `--model`, `-m`     | Override model name                                                                                   | task config                                     |
| `--wait`            | Tuần tự: giây chờ giữa batch. Song song: trần nhịp request, tối đa 1 request mỗi `min_interval` giây. | task config → `0`                               |
| `--workers`, `-w`   | Số batch chạy song song (batch đầu warm-up tuần tự)                                                   | task config (`max_workers`) → `1`               |
| `--verbose`, `-v`   | Bật log chi tiết                                                                                      | (tắt)                                           |

#### Bảng tham số `align-srt`

> **Không cần `qwen-asr`, `audio-separator`, hay GPU.** Chạy thẳng trong `.venv` chính.

| Tham số               | Mô tả                                                                                      | Mặc định              |
| --------------------- | ------------------------------------------------------------------------------------------ | --------------------- |
| `input_srt` / `-i`   | File `_punct.srt` (đã có dấu câu từ `punctuate-srt`)                                       | (bắt buộc nếu không `--task-file`) |
| `--output`, `-o`      | File `.srt` hoặc folder đầu ra                                                             | `<input>_seg.srt`     |
| `--task-file`, `-t`   | JSON `[{"input": "..._punct.srt", "output": "..."}]`                                       | (không dùng)          |
| `--offset-seconds`    | Offset cộng vào timestamp (giây)                                                           | `0.0`                 |
| `--split-on-comma`    | Cắt câu tại dấu phẩy (，、,;) — mặc định **tắt** (chỉ cắt tại `.!?:。！？：；`)             | (tắt)                 |
| `--verbose`, `-v`     | Bật log chi tiết                                                                           | (tắt)                 |

> **Auto-detect ngôn ngữ:** `align-srt` tự nhận biết CJK hay Latin từ nội dung SRT và in ra
> trong output (vd `[CJK]` hay `[Latin]`). CJK được nối không khoảng trắng; Latin nối bằng space.

### 2.1. Mute Audio (mute-srt)

Dùng khi audio có 2 ngôn ngữ (ví dụ: bình luận + video gốc trích dẫn). Thay thế các đoạn được đánh dấu bằng silence, giữ nguyên độ dài audio.

#### Mute audio nhanh

```colab
!uv run mute-srt --input /content/video.mp4 --mute /content/mute.srt
```

#### Đầy đủ tham số

```colab
!uv run mute-srt \
    --input       /content/video.mp4 \
    --mute        /content/mute.srt \
    --output      /content/audio_muted.wav \
    --sample-rate 16000 \
    --verbose
```

#### Tham số

| Tham số           | Mô tả                                | Mặc định               |
| ----------------- | ------------------------------------ | ---------------------- |
| `--input`, `-i`   | File audio/video đầu vào             | (bắt buộc)             |
| `--mute`, `-m`    | File mute.srt chứa các đoạn cần mute | (bắt buộc)             |
| `--output`, `-o`  | File audio đầu ra                    | `<input>_muted.wav`    |
| `--sample-rate`   | Sample rate output                   | `16000` (cho WhisperX) |
| `--verbose`, `-v` | Hiển thị log chi tiết                | (tắt)                  |

#### File mute.srt format

Tạo file `mute.srt` đánh dấu các đoạn cần mute:

```srt
1
00:01:24,233 --> 00:01:27,566
[MUTE] Đoạn video gốc được trích dẫn

2
00:05:30,000 --> 00:05:45,500
[MUTE] Đoạn ngôn ngữ thứ hai
```

> **Lưu ý:** Text trong file mute.srt không quan trọng, chỉ cần timestamp đúng format SRT.

---

### 2.2. Extract Audio (extract-srt)

Ngược với mute-srt: Giữ lại CHỈ các đoạn được đánh dấu trong mute.srt, các đoạn khác thành silence.

#### Extract audio nhanh

```colab
!uv run extract-srt --input /content/video.mp4 --mute /content/mute.srt
```

#### Đầy đủ tham số

```colab
!uv run extract-srt \
    --input       /content/video.mp4 \
    --mute        /content/mute.srt \
    --output      /content/audio_extracted.wav \
    --sample-rate 16000 \
    --verbose
```

#### Tham số

| Tham số           | Mô tả                                   | Mặc định                |
| ----------------- | --------------------------------------- | ----------------------- |
| `--input`, `-i`   | File audio/video đầu vào                | (bắt buộc)              |
| `--mute`, `-m`    | File mute.srt chứa các đoạn cần extract | (bắt buộc)              |
| `--output`, `-o`  | File audio đầu ra                       | `<input>_extracted.wav` |
| `--sample-rate`   | Sample rate output                      | `16000` (cho WhisperX)  |
| `--verbose`, `-v` | Hiển thị log chi tiết                   | (tắt)                   |

---

### 2.3. Merge SRT (merge-srt)

Merge 2 file SRT thành 1 file hoàn chỉnh, sắp xếp theo timestamp.

#### Merge nhanh

```colab
!uv run merge-srt \
    --commentary /content/video_subtitle.srt \
    --quoted     /content/subtitle_quoted.srt
```

#### Đầy đủ tham số

```colab
!uv run merge-srt \
    --commentary       /content/video_subtitle.srt \
    --quoted           /content/subtitle_quoted.srt \
    --output           /content/subtitle_merged.srt \
    --no-check-overlap \
    --verbose
```

#### Tham số

| Tham số              | Mô tả                                       | Mặc định              |
| -------------------- | ------------------------------------------- | --------------------- |
| `--commentary`, `-c` | File SRT chứa subtitle phần bình luận       | (bắt buộc)            |
| `--quoted`, `-q`     | File SRT chứa subtitle phần video trích dẫn | (bắt buộc)            |
| `--output`, `-o`     | File SRT output                             | `subtitle_merged.srt` |
| `--no-check-overlap` | Bỏ qua kiểm tra overlapping segments        | (mặc định check)      |
| `--verbose`, `-v`    | Hiển thị log chi tiết                       | (tắt)                 |

---

### 2.4. Dịch SRT (translate-srt)

#### Dịch nhanh (Gemini - Mặc định)

```colab
from google.colab import userdata
gemini_key = userdata.get('gemini_key')

!uv run translate-srt \
    --input /content/video.srt \
    --keys  "{gemini_key}"
```

#### Dịch với OpenAI-Compatible (DeepSeek)

```colab
!uv run translate-srt \
    --input /content/video.srt \
    --provider openai \
    --provider-config config/llm/openai_compat.yaml \
    --keys "sk-deepseek-xxx" \
    --lang "Japanese"
```

#### Dịch với Vertex AI (Application Default Credentials)

```colab
!uv run translate-srt \
    --input /content/video.srt \
    --provider vertexai \
    --provider-config config/llm/vertexai.yaml \
    --lang "Japanese"
```

#### Chạy hàng loạt nhiều file (Batch JSON)

Chạy:

```colab
from google.colab import userdata
gemini_key = userdata.get('gemini_key')

!uv run translate-srt \
    --task-file /content/tasks.json \
    --provider gemini \
    --lang "Japanese" \
    --keys "{gemini_key}" \
    --batch 30
```

#### Đầy đủ tham số

```colab
from google.colab import userdata
gemini_key = userdata.get('gemini_key')

!uv run translate-srt \
    --input     /content/video.srt \
    --output    /content/video_ja.srt \
    --provider  gemini \
    --provider-config config/llm/gemini.yaml \
    --task-config config/llm_tasks/srt_translation.yaml \
    --lang      "Japanese" \
    --keys      "{gemini_key}" \
    --model     "gemini-3-flash-preview" \
    --prompt    /content/prompts/translation/srt_translate.txt \
    --batch     30 \
    --budget    24576 \
    --wait      0.5 \
    --workers   3 \
    --no-context \
    --verbose
```

> **Chạy song song (`--workers`):** Mặc định `1` (tuần tự). Đặt `--workers 3` để dịch
> nhiều batch cùng lúc — batch ĐẦU luôn chạy tuần tự ("warm-up") để provider kịp tạo/ấm
> context cache (Vertex `CachedContent`) hoặc anchor R0 (OpenAI Responses) TRƯỚC khi
> fan-out, tránh cả wave đầu cùng cold-start cache. Cache/anchor là read-only dùng chung
> nên **không ảnh hưởng tính nhất quán bản dịch** giữa các batch. Lưu ý rate limit: cached
> tokens vẫn tính vào TPM/RPM; với model free-tier/preview (RPM thấp) nên giữ workers nhỏ
> (3–5). 429 được xử lý tự động bằng exponential backoff + jitter. Có thể đặt sẵn
> `max_workers` trong task YAML thay cho cờ CLI.

#### Tham số

| Tham số             | Mô tả                                                                                                                     | Mặc định                                     |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| `--input`, `-i`     | File .srt đầu vào                                                                                                         | (bắt buộc nếu không dùng task-file)          |
| `--task-file`, `-t` | File JSON chứa danh sách task (`{"input": "...", "output": "..."}`)                                                       | (không dùng)                                 |
| `--output`, `-o`    | File .srt đầu ra (chỉ dùng với `--input`)                                                                                 | `<input>_<lang>.srt`                         |
| `--provider`, `-p`  | Provider (gemini/openai/vertexai)                                                                                         | `gemini`                                     |
| `--provider-config` | Đường dẫn provider YAML                                                                                                   | `config/llm/<provider>.yaml`                 |
| `--task-config`     | Đường dẫn task YAML của SRT translation                                                                                   | `config/llm_tasks/srt_translation.yaml`      |
| `--base-url`        | Override base URL cho OpenAI provider                                                                                     | `None`                                       |
| `--keys`, `-k`      | API key(s), phân cách bằng dấu phẩy; có thể dùng `GEMINI_API_KEY` hoặc `OPENAI_API_KEY`                                   | (bắt buộc gemini/openai nếu chưa có env var) |
| `--lang`, `-l`      | Ngôn ngữ đích (tên tiếng Anh đầy đủ)                                                                                      | `Vietnamese`                                 |
| `--model`, `-m`     | Model provider                                                                                                            | theo provider config                         |
| `--prompt`          | Đường dẫn prompt dịch SRT                                                                                                 | `prompts/translation/srt_translate.txt`      |
| `--batch`, `-b`     | Số dòng dịch mỗi lần                                                                                                      | `30`                                         |
| `--budget`          | Thinking budget tokens (Gemini only)                                                                                      | `24576`                                      |
| `--wait`            | Tuần tự: giây chờ giữa batch. Song song: trần nhịp phát request (RPM cap)                                                 | `0`                                          |
| `--workers`, `-w`   | Số batch dịch chạy song song (batch đầu warm-up tuần tự để tạo/ấm cache trước)                                            | task config (`max_workers`) → `1`            |
| `--no-context`      | Tắt global context, gộp toàn bộ text của file SRT gốc thành một đoạn "Read-Only Reference" và gửi kèm trong prompt cho AI | (mặc định bật)                               |
| `--verbose`, `-v`   | Hiển thị log chi tiết                                                                                                     | (tắt)                                        |

#### Generic LLM task: tạo SEO metadata

Flow LLM generic dùng `llm-task` để chạy các tác vụ dạng text-in/text-out bằng provider trong `llm_ai`.
Metadata SEO mặc định dùng task config `config/llm_tasks/seo_metadata.yaml`, prompt `prompts/llm_tasks/seo_metadata.txt`, input raw text `.txt` và output markdown `.md`.

```colab
!uv run llm-task \
    --task-config config/llm_tasks/seo_metadata.yaml \
    --input /content/video_content.txt \
    --output /content/video_metadata.md \
    --provider openai \
    --provider-config config/llm/openai_compat.yaml \
    --keys "sk-xxx"
```

Chạy hàng loạt bằng JSON task-file:

```json
[
  {
    "input": "/content/video_001.txt",
    "output": "/content/video_001_metadata.md"
  },
  {
    "input": "/content/video_002.txt",
    "output": "/content/video_002_metadata.md"
  }
]
```

```colab
!uv run llm-task \
    --task-config config/llm_tasks/seo_metadata.yaml \
    --task-file /content/metadata_tasks.json \
    --provider openai \
    --provider-config config/llm/openai_compat.yaml \
    --keys "sk-xxx"
```

---

### 2.5. SRT to ASS (srt-to-ass)

Chuyển file SRT thành ASS trung gian cho note overlay. Với `note_overlay.mode=dynamic_ass_box`, mỗi block SRT nhiều dòng có thể đặt dòng đầu tiên làm layout key (`top_left`, `bottom_right`, `center_panel`, ...). Dòng layout key sẽ được ghi vào field `Name` của ASS và không render ra video.

#### Chuyển đổi nhanh

```colab
!uv run srt-to-ass --input /content/note_translated.srt --layout-key top_left
```

#### Ví dụ SRT nhiều layout

```srt
1
00:00:03,000 --> 00:00:10,000
top_left
Quick field note:
The north gate opens only after the second bell.

2
00:00:14,000 --> 00:00:22,000
bottom_right
Gear checklist:
1. Small knife
2. Water pouch
```

#### Đầy đủ tham số

```colab
!uv run srt-to-ass \
    --input     /content/note_translated.srt \
    --output    /content/note_overlay.ass \
    --template  /content/CharenjiZukan/assets/sample.ass \
    --max-chars 0 \
    --style     NoteStyle \
    --layout-key top_left \
    --srt-layout-key-mode warn \
    --verbose
```

#### Tham số

| Tham số                 | Mô tả                                                                 | Mặc định            |
| ----------------------- | --------------------------------------------------------------------- | ------------------- |
| `--input`, `-i`         | File SRT đầu vào                                                      | (bắt buộc)          |
| `--output`, `-o`        | File ASS đầu ra                                                       | `<input>.ass`       |
| `--template`, `-t`      | File ASS template                                                     | `assets/sample.ass` |
| `--max-chars`           | Wrap ký tự legacy cho ASS trung gian; nên đặt `0` cho dynamic overlay | `14`                |
| `--style`               | Tên style placeholder trong ASS                                       | `NoteStyle`         |
| `--layout-key`          | Layout fallback khi block SRT không khai báo dòng layout đầu tiên     | `""`                |
| `--srt-layout-key-mode` | `warn`, `strict`, `off` cho parser dòng layout đầu tiên               | `warn`              |
| `--verbose`, `-v`       | Hiển thị log chi tiết                                                 | (tắt)               |

---

### 2.6. Text-to-Speech (tts)

Hỗ trợ 4 engine: **EdgeTTS** (mặc định, cloud), **Voicevox Nemo** (local server), **Voicevox** (local server), và **Qwen3-TTS** (HuggingFace, voice-clone).

Cấu hình engine được đặt trong file YAML (`config/tts_config.yaml`). CLI chỉ cần trỏ `--config` và `--provider`.

#### Xem danh sách giọng (EdgeTTS)

```colab
# Giọng tiếng Việt
!uv run tts --list-voices vi

# Giọng tiếng Nhật
!uv run tts --list-voices ja
```

#### TTS nhanh (EdgeTTS)

```colab
!uv run tts \
    --input /content/video_ja.srt \
    --config /content/CharenjiZukan/config/tts_config.yaml
```

#### TTS với autorate (tự động nén giọng)

```colab
!uv run tts \
    --input    /content/video_ja.srt \
    --output   /content/video_ja.wav \
    --config   /content/CharenjiZukan/config/tts_config.yaml \
    --autorate
```

#### Sử dụng Voicevox Nemo

**Bước 1: Cài đặt và khởi động Server Voicevox Nemo ngầm**

```colab
!python setup_voicevox_nemo.py
```

**Bước 2: Chạy TTS với Voicevox Nemo**

```colab
!uv run tts \
    --input /content/video_ja.srt \
    --provider voicevox_nemo \
    --config /content/CharenjiZukan/config/tts_config.yaml
```

#### Sử dụng Voicevox (Chính thức)

**Bước 1: Cài đặt và khởi động Server Voicevox ngầm**

```colab
!python setup_voicevox.py
```

**Bước 2: Chạy TTS với Voicevox**

```colab
!uv run tts \
    --input /content/video_ja.srt \
    --provider voicevox \
    --config /content/CharenjiZukan/config/tts_config.yaml
```

#### Sử dụng Qwen3-TTS (Voice Clone)

#### Cài đặt môi trường

```colab
!uv pip install -e .[qwen-tts]
!apt-get -y install sox libsox-fmt-all
```

```colab
!uv run tts \
    --input /content/script.txt \
    --provider qwen \
    --config /content/CharenjiZukan/config/tts_config.yaml
```

> **Lưu ý:** Qwen3-TTS yêu cầu cài đặt `qwen-tts`, `transformers`, `accelerate`, `soundfile` và `flash-attn`. Cấu hình `ref_audio` và `ref_text` trong `config/tts_config.yaml` để voice-clone.

#### Chạy hàng loạt (Batch JSON)

Tạo file `tasks.json`:

```json
[
  {
    "input": "/content/video1_ja.srt",
    "output": "/content/audio1.wav"
  },
  {
    "input": "/content/video2_ja.srt",
    "output": "/content/audio2.wav"
  }
]
```

Chạy:

```colab
!uv run tts \
    --task-file /content/tasks.json \
    --config /content/CharenjiZukan/config/tts_config.yaml
```

#### Đầy đủ tham số

```colab
!uv run tts \
    --input      /content/video_ja.srt \
    --output     /content/video_ja.wav \
    --config     /content/CharenjiZukan/config/tts_config.yaml \
    --provider   edge \
    --autorate \
    --max-speed  100.0 \
    --silence-ms 0 \
    --cache      /content/cache_tts \
    --keep-cache \
    --verbose
```

#### Tham số

| Tham số             | Mô tả                                                | Mặc định                            |
| ------------------- | ---------------------------------------------------- | ----------------------------------- |
| `--input`, `-i`     | File .srt hoặc .txt đầu vào                          | (bắt buộc nếu không dùng task-file) |
| `--output`, `-o`    | File audio đầu ra (.wav/.mp3)                        | `output/<input_stem>.wav`           |
| `--task-file`, `-t` | File JSON chứa danh sách task                        | (không dùng)                        |
| `--config`, `-c`    | File cấu hình YAML                                   | `config/tts_config.yaml`            |
| `--provider`, `-p`  | TTS engine (edge/voicevox_nemo/voicevox/qwen)        | `edge`                              |
| `--autorate`        | Tự động nén audio khớp slot SRT (chỉ .srt)           | (tắt)                               |
| `--max-speed`       | Giới hạn tốc độ nén tối đa                           | `100.0`                             |
| `--silence-ms`      | Độ dài silence giữa các dòng khi không dùng autorate | `0`                                 |
| `--cache`           | Thư mục cache audio tạm                              | `tmp/<stem>_<ts>/`                  |
| `--keep-cache`      | Giữ lại thư mục cache tạm sau khi xử lý xong         | (tắt)                               |
| `--list-voices`     | Liệt kê giọng EdgeTTS                                | (không dùng)                        |
| `--verbose`         | Bật logging debug                                    | (tắt)                               |

#### File cấu hình `config/tts_config.yaml`

---

### 2.7. Audio Separation (audio-separator)

Tách voice/background từ audio sử dụng thư viện audio-separator với các mô hình ROFORMER.

#### Tách background music (mặc định)

```colab
!uv run audio-separator --input /content/audio_muted.wav
```

#### Tách vocals (giữ lại giọng nói)

```colab
!uv run audio-separator --input /content/audio.wav --preset vocal_extraction
```

#### Đầy đủ tham số

```colab
!uv run audio-separator \
    --input   /content/audio_muted.wav \
    --output-dir  /content \
    --preset  bgm_extraction
```

#### Tham số

| Tham số              | Mô tả                                                 | Mặc định         |
| -------------------- | ----------------------------------------------------- | ---------------- |
| `--input`, `-i`      | File audio đầu vào                                    | (bắt buộc)       |
| `--output-dir`, `-o` | Thư mục output đầu ra                                 | `.`              |
| `--preset`, `-p`     | Preset cấu hình: `bgm_extraction`, `vocal_extraction` | `bgm_extraction` |

#### Lưu ý quan trọng

- `audio-separator` nhận diện tự động và sẽ sử dụng GPU L4 để chạy ROFORMER nhanh nhất thông qua `use_autocast=true` được thiết lập trong yaml.

---

### 2.8. Media Speed (media-speed)

Thay đổi tốc độ media (video, audio, SRT, ASS). Hỗ trợ cả slow down và speed up.

#### Slow down video 0.65x

```colab
!uv run media-speed --input /content/video.mp4 --speed 0.65
```

#### Slow down audio

```colab
!uv run media-speed --input /content/audio.wav --speed 0.65
```

#### Scale SRT timestamps

```colab
!uv run media-speed --input /content/subtitle.srt --speed 0.65
```

#### Scale ASS timestamps

```colab
!uv run media-speed --input /content/note_overlay.ass --speed 0.65
```

#### Đầy đủ tham số

```colab
!uv run media-speed \
    --input          /content/video.mp4 \
    --output         /content/video_slow.mp4 \
    --speed          0.65 \
    --type           auto \
    --no-keep-audio \
    --verbose
```

#### Tham số

| Tham số           | Mô tả                                   | Mặc định                               |
| ----------------- | --------------------------------------- | -------------------------------------- |
| `--input`, `-i`   | File input (video, audio, SRT, ASS)     | (bắt buộc)                             |
| `--output`, `-o`  | File output                             | `<input>_slow.*` hoặc `<input>_fast.*` |
| `--speed`, `-s`   | Hệ số tốc độ (< 1.0: slow, > 1.0: fast) | `0.65`                                 |
| `--type`, `-t`    | Loại file: auto, video, audio, srt, ass | `auto` (auto-detect)                   |
| `--no-keep-audio` | Không giữ audio trong video output      | (mặc định giữ audio)                   |
| `--verbose`, `-v` | Hiển thị log chi tiết                   | (tắt)                                  |

---

### 2.9. Trích xuất phụ đề cứng Multi-Box (video-ocr)

Trích xuất phụ đề (hardsub) trực tiếp từ khung hình video sử dụng mô hình DeepSeek-OCR-2, hỗ trợ nhiều vùng box độc lập.

#### Trích xuất nhanh (với Secrets)

```colab
from google.colab import userdata
hf_token = userdata.get('hf_token')

!uv run video-ocr /content/video.mp4 \
    --boxes-file /content/CharenjiZukan/assets/boxesOCR.txt \
    --hf-token "{hf_token}"
```

#### Đầy đủ tham số

```colab
from google.colab import userdata
hf_token = userdata.get('hf_token')

!uv run video-ocr /content/video.mp4 \
    --boxes-file /content/CharenjiZukan/assets/boxesOCR.txt \
    --output-dir /content \
    --frame-interval 3 \
    --scene-threshold 1.5 \
    --min-scene-frames 3 \
    --phash-threshold 4 \
    --noise-threshold 25 \
    --cv-prefilter \
    --cv-min-edge-density 0.03 \
    --cv-edge-low 50 \
    --cv-edge-high 150 \
    --min-chars 2 \
    --device cuda \
    --warn-english \
    --save-minify-txt \
    --hf-token "{hf_token}" \
    --format srt \
    --enable-chinese-filter
```

#### File `boxesOCR.txt`

```text
subtitle 370 930 1180 140
```

Mỗi dòng gồm `box_name x y w h`.

#### Tham số

| Tham số                   | Mô tả                                                                            | Mặc định                                                    |
| ------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `input_video`             | File video đầu vào (hoặc directory nếu dùng --input-dir)                         | (bắt buộc)                                                  |
| `--boxes-file`            | File cấu hình các vùng OCR theo format `name x y w h`                            | `assets/boxesOCR.txt` (nếu có config yaml thì config > cli) |
| `--output-dir`            | Thư mục output cho các file theo box                                             | cùng thư mục video                                          |
| `--frame-interval`        | Số frame bỏ qua giữa mỗi lần xử lý                                               | `30`                                                        |
| `--scene-threshold`       | Ngưỡng phần trăm thay đổi pixel trên tổng diện tích box để phát hiện chuyển cảnh | `1.5`                                                       |
| `--min-scene-frames`      | Số frame tối thiểu giữa 2 lần chuyển cảnh để tránh nhiễu                         | `3`                                                         |
| `--phash-threshold`       | Ngưỡng Hamming distance cho perceptual hash                                      | `4`                                                         |
| `--noise-threshold`       | Ngưỡng loại bỏ nhiễu nén video khi so sánh pixel                                 | `25`                                                        |
| `--cv-prefilter`          | Bật tiền lọc OpenCV để bỏ qua ROI không có dấu hiệu chữ                          | (tắt)                                                       |
| `--cv-min-edge-density`   | Ngưỡng mật độ cạnh tối thiểu cho CV prefilter                                    | `0.03`                                                      |
| `--cv-edge-low`           | Ngưỡng thấp Canny edge detector                                                  | `50`                                                        |
| `--cv-edge-high`          | Ngưỡng cao Canny edge detector                                                   | `150`                                                       |
| `--min-chars`             | Số ký tự tối thiểu để ghi nhận                                                   | `2`                                                         |
| `--no-scene-detection`    | Tắt bỏ tính năng Scene detection (tương đương threshold=0)                       | (tắt)                                                       |
| `--enable-chinese-filter` | Bật bộ lọc chỉ giữ lại tiếng Trung                                               | (tắt)                                                       |
| `--no-punctuation`        | Không giữ dấu câu tiếng Trung (khi bật filter)                                   | (tắt)                                                       |
| `--ocr-model`             | Tên model Hugging Face (DeepSeek-OCR-2 hoặc Qwen3-VL)                            | `deepseek-ai/DeepSeek-OCR-2`                                |
| `--qwen-max-new-tokens`   | [Chỉ Qwen3-VL] Số token tối đa sinh ra                                           | `256`                                                       |
| `--qwen-min-pixels`       | [Chỉ Qwen3-VL] Pixel blocks tối thiểu (ảnh hưởng VRAM)                           | `256`                                                       |
| `--qwen-max-pixels`       | [Chỉ Qwen3-VL] Pixel blocks tối đa (ảnh hưởng VRAM)                              | `1280`                                                      |
| `--device`                | Thiết bị xử lý (cuda/cpu)                                                        | `cuda`                                                      |
| `--hf-token`              | Hugging Face Token                                                               | (không dùng)                                                |
| `--batch-size`            | Batch size cho OCR batching                                                      | `8`                                                         |
| `--format`                | Định dạng output theo box (srt/txt)                                              | `srt`                                                       |
| `--default-duration`      | Thời lượng mặc định mỗi subtitle                                                 | `3.0s`                                                      |
| `--min-duration`          | Thời lượng tối thiểu sau deduplicate                                             | `1.0s`                                                      |
| `--max-duration`          | Thời lượng tối đa sau deduplicate                                                | `7.0s`                                                      |
| `--no-deduplicate`        | Tắt gộp subtitle trùng lặp                                                       | (tắt)                                                       |
| `--warn-english`          | Tạo file cảnh báo riêng nếu subtitle chứa tiếng Anh/số                           | (tắt)                                                       |
| `--save-minify-txt`       | Lưu file `<video>_script.txt` thuần văn bản, mỗi câu 1 dòng                      | (tắt)                                                       |
| `--no-timestamp`          | Tắt timestamp (chỉ với format=txt)                                               | (tắt)                                                       |
| `--isolate-text`          | Bật lọc watermark/overlay mờ (opacity masking) trước khi OCR                     | (tắt)                                                       |
| `--isolate-config`        | File JSON ngưỡng từ `tools/calibrate_text_isolation.py`                          | (không dùng)                                                |
| `--subtitle-colors`       | Màu phụ đề chỉ định: `"white,#FFD700"` hoặc `"255,255,255"`                      | (rỗng → dựa độ sáng)                                        |
| `--color-tolerance`       | Sai số khoảng cách màu Lab                                                       | `40`                                                        |
| `--subtitle-min-contrast` | Ngưỡng tương phản (proxy opacity); component mờ hơn bị xóa                       | `40`                                                        |
| `--stroke-max-luminance`  | Ngưỡng độ sáng coi là viền tối                                                   | `80`                                                        |
| `--min-component-area`    | Diện tích tối thiểu giữ component — chỉ diệt nhiễu, đặt nhỏ                      | `8`                                                         |
| `--no-require-stroke`     | Tắt kiểm tra viền tối (phụ đề không có viền)                                     | (tắt)                                                       |
| `--config`                | Đường dẫn file cấu hình `.yaml`                                                  | (không dùng)                                                |

> **Mức ưu tiên Cấu hình**: CLI parameters có mức ưu tiên cao nhất, sau đó là tham số khai báo trong `--config`, và cuối cùng là Default values.

#### Lọc watermark/overlay mờ (Text Isolation)

Khi video có **watermark / text overlay mờ** (opacity < 70%) lọt vào vùng OCR — kể cả overlay di chuyển đè lên dải phụ đề — bật `--isolate-text` để xóa chúng **trước khi OCR**. Cơ chế dựa trên opacity/color masking thuần OpenCV (giữ glyph phụ đề đặc, đúng màu, có viền; xóa glyph mờ). **Mặc định TẮT** — video sạch không cần bật.

**Bước 1 — Hiệu chỉnh ngưỡng** từ vài crop mẫu (cắt ở độ phân giải gốc, lưu PNG, chừa vài pixel nền):

```colab
# samples/subtitle/  : crop phụ đề sạch + crop phụ đề bị watermark ĐÈ chồng
# samples/watermark/ : crop CHỈ watermark/overlay
!uv run python /content/CharenjiZukan/tools/calibrate_text_isolation.py \
    --subtitle-samples /content/samples/subtitle/ \
    --watermark-samples /content/samples/watermark/ \
    --subtitle-colors "white,#FFD700" \
    --out /content/samples/text_isolation_config.json
# → kiểm tra ảnh trong samples/preview/ : phụ đề còn nguyên, watermark thành đen
```

**Bước 2 — Chạy OCR với config đã hiệu chỉnh:**

```colab
!uv run video-ocr /content/video.mp4 \
    --boxes-file /content/CharenjiZukan/assets/boxesOCR.txt \
    --ocr-model Qwen/Qwen3-VL-8B-Instruct \
    --isolate-text \
    --isolate-config /content/samples/text_isolation_config.json \
    --subtitle-colors "white,#FFD700" \
    --hf-token "{hf_token}"
```

Hoặc truyền tay ngưỡng (không cần file JSON):

```colab
!uv run video-ocr /content/video.mp4 \
    --isolate-text \
    --subtitle-colors "white,#FFD700" \
    --color-tolerance 40 \
    --subtitle-min-contrast 45 \
    --stroke-max-luminance 80 \
    --min-component-area 8
# Phụ đề không viền: thêm --no-require-stroke
```

> Chi tiết quy tắc cắt mẫu và cách đọc/dùng từng tham số: xem `docs/text-isolation-guide.md`.
> Lưu ý `--min-component-area` chỉ để diệt nhiễu lốm đốm (đặt nhỏ 5–15), KHÔNG dùng để loại watermark.

#### Output theo từng box

Ví dụ input là `/content/video.mp4` với 2 box `subtitle` và `note`, output sẽ là:

- `/content/video_subtitle.srt`
- `/content/video_note.srt`

#### Hướng dẫn chỉnh ngưỡng

| Tình huống                       | scene_threshold | phash_threshold |
| :------------------------------- | :-------------- | :-------------- |
| Box nhỏ, subtitle thay đổi nhanh | 1.0             | 3               |
| Box lớn, subtitle thay đổi chậm  | 0.5             | 5               |
| Video nhiều nhiễu/hiệu ứng       | 2.0             | 6               |
| Mặc định cân bằng                | 1.5             | 4               |

> **Phục hồi dấu câu** đã tách khỏi `video-ocr` thành CLI riêng `punctuate-srt` (xem Mục 2.0c) —
> chạy sau khi có SRT từ OCR. `video-ocr` giờ chỉ làm OCR.

---

### 2.10. Native Video Subtitle Extractor (video-ocr-native)

CLI này dùng Qwen3-VL Native Video mode để xử lý subtitle theo từng batch video (mặc định 60 giây), giữ context theo chiến lược multi-turn giữa các batch.

#### Chạy nhanh (khuyến nghị cho Colab)

```colab
from google.colab import userdata
hf_token = userdata.get('hf_token')

!uv run video-ocr-native /content/video.mp4 \
    --hf-token "{hf_token}" \
    --device cuda
```

#### Chạy đầy đủ tham số chính

```colab
from google.colab import userdata
hf_token = userdata.get('hf_token')

!uv run video-ocr-native /content/video.mp4 \
    --config /content/CharenjiZukan/config/native_video_ocr_config.yaml \
    --boxes-file /content/CharenjiZukan/assets/boxesOCR.txt \
    --output-dir /content \
    --prompt-file /content/CharenjiZukan/prompts/native_video_ocr_prompt.txt \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --device cuda \
    --hf-token "{hf_token}" \
    --frame-interval 6 \
    --batch-duration 60 \
    --sample-fps 5.0 \
    --max-new-tokens 2048 \
    --total-pixels 20971520 \
    --min-pixels 65536 \
    --max-frames 2048 \
    --warn-english \
    --save-minify-txt \
    --verbose
```

#### Tham số chính

| Tham số             | Mô tả                                                           | Mặc định                              |
| ------------------- | --------------------------------------------------------------- | ------------------------------------- |
| `video`             | Đường dẫn video input                                           | (bắt buộc)                            |
| `--config`          | YAML config cho native pipeline                                 | `config/native_video_ocr_config.yaml` |
| `--boxes-file`      | File ROI dạng `name x y w h`                                    | theo config (`assets/boxesOCR.txt`)   |
| `--output-dir`      | Thư mục output                                                  | cùng thư mục video                    |
| `--prompt-file`     | Prompt template cho native extraction                           | `prompts/native_video_ocr_prompt.txt` |
| `--model`           | Model Qwen3-VL                                                  | `Qwen/Qwen3-VL-8B-Instruct`           |
| `--device`          | Thiết bị chạy model                                             | `cuda`                                |
| `--hf-token`        | Hugging Face token                                              | (không dùng)                          |
| `--frame-interval`  | Lấy mẫu 1 frame mỗi N frame                                     | `6`                                   |
| `--batch-duration`  | Số giây mỗi batch video                                         | `60.0`                                |
| `--sample-fps`      | FPS khai báo cho frame-list native video                        | `5.0`                                 |
| `--max-new-tokens`  | Số token output tối đa mỗi batch                                | `2048`                                |
| `--total-pixels`    | Giới hạn tổng pixel video input                                 | `20971520`                            |
| `--min-pixels`      | Giới hạn pixel tối thiểu                                        | `65536`                               |
| `--max-frames`      | Giới hạn số frame mỗi batch                                     | `2048`                                |
| `--warn-english`    | Lưu file cảnh báo English/number                                | (tắt)                                 |
| `--save-minify-txt` | Lưu file script thuần văn bản (mỗi câu 1 dòng, không timestamp) | (tắt)                                 |
| `-v`, `--verbose`   | Tăng chi tiết log                                               | (tắt)                                 |
| `--quiet`           | Chỉ in lỗi                                                      | (tắt)                                 |

#### Output files

Với input `/content/video.mp4`:

- `/content/video_native.srt` (luôn tạo)
- `/content/video_native_script.txt` (khi bật `--save-minify-txt`)
- `/content/video_subtitle_english_warnings.txt` (khi bật `--warn-english`)

---

### 2.11. TTS-Video Sync Pipeline (sync-video)

CLI `sync-video` dùng pipeline `sync_engine` để đồng bộ video + TTS theo timeline subtitle (gồm 7 phase: phân tích timeline, xử lý video chunks, ghép audio, forced alignment subtitle (optional), remap timestamps, render final, LLM metadata (optional)).

#### Forced Alignment Subtitle (Phase 3.5)

Khi bật `forced_alignment_subtitle.enabled = true` trong `render_config.json`, pipeline sẽ align **từng clip TTS `dubb-N.wav`** (Phase 0 đã sinh) bằng `Qwen3ForcedAligner`, tạo SRT với timestamp chính xác cho từng từ. Mỗi clip chỉ vài giây nên không OOM dù video dài nhiều tiếng. Dòng phụ đề vùng mute tự động remap timeline rồi gộp vào SRT cuối — không sót dòng nào.

**Cấu hình forced alignment trong `render_config.json`:**

**Output:**

- `<name>_synced.srt` — Forced alignment SRT (nếu bật) hoặc remap SRT (mặc định)
- `<name>_tts_synced_debug.srt` — Remap SRT debug (chỉ khi `keep_tts_synced_debug=true`)

#### Chạy nhanh

```colab
!uv run sync-video \
    --video /content/video.mp4 \
    --subtitle /content/subtitle_translated.srt \
    --tts-provider edge \
    --tts-voice ja-JP-KeitaNeural \
    --output-dir /content/output_sync
```

#### Chạy nhanh với Voicevox Nemo

Yêu cầu đã bật server Voicevox Nemo ngầm (xem phần 2.6).

```colab
!uv run sync-video \
    --video /content/video.mp4 \
    --subtitle /content/subtitle_translated.srt \
    --tts-provider voicevox_nemo \
    --tts-voice 10008 \
    --output-dir /content/output_sync
```

#### Chạy nhanh với Voicevox (Chính thức)

Yêu cầu đã bật server Voicevox ngầm (xem phần 2.6).

```colab
!uv run sync-video \
    --video /content/video.mp4 \
    --subtitle /content/subtitle_translated.srt \
    --tts-provider voicevox \
    --tts-voice 100 \
    --output-dir /content/output_sync
```

#### Chạy nhanh với Qwen3-TTS

Yêu cầu đã cài đặt `qwen-tts` và `transformers` (xem phần 2.6).

```colab
!uv run sync-video \
    --video /content/video.mp4 \
    --subtitle /content/subtitle_translated.srt \
    --tts-provider qwen \
    --tts-config /content/CharenjiZukan/config/tts_config.yaml \
    --output-dir /content/output_sync
```

#### Chạy với Image Overlay static image theo SRT

Image overlay dùng một SRT riêng để điều khiển thời gian hiển thị static image full-screen. Text mỗi block SRT là basename ảnh, không có đuôi tệp. Khi `file_ext="auto"`, pipeline chấp nhận nhiều định dạng ảnh tĩnh; PNG vẫn là định dạng khuyến nghị nếu cần alpha channel để video bên dưới còn hiển thị.

Ví dụ `/content/image_overlay.srt`:

```srt
1
00:00:01,000 --> 00:00:03,500
frame_intro

2
00:00:05,000 --> 00:00:08,000
callout_01
```

Cấu trúc thư mục ảnh tĩnh:

```text
/content/overlay_images/frame_intro.png
/content/overlay_images/callout_01.webp
/content/overlay_images/diagram.jpg
```

Lưu ý vận hành trên Colab:

- Nên export ảnh đúng kích thước output trong `render_config.json` (ví dụ 1920x1080 hoặc 1080x1920) để tránh ảnh bị kéo méo do `fit=stretch_to_output`.
- Nếu cần alpha channel để giữ video nền hiển thị, ưu tiên PNG hoặc WebP hỗ trợ trong suốt.
- `file_ext="auto"` cho phép resolver nhận diện nhiều định dạng ảnh tĩnh; text SRT vẫn chỉ dùng basename không có extension.
- Timestamp overlay được remap theo video stretch timeline, không phụ thuộc forced alignment subtitle.
- Nếu SRT có nhiều event, `render_strategy=auto` sẽ tự chuyển từ `-filter_complex` sang `-filter_complex_script` khi vượt ngưỡng `direct_overlay_max_events` hoặc `command_line_max_chars`.
- `intermediate` overlay video hiện chỉ là stub cho phase tối ưu sau, chưa được thực thi.

#### Chạy đầy đủ tham số

```colab
!uv run sync-video \
    --video /content/video.mp4 \
    --subtitle /content/subtitle_translated.srt \
    --tts-provider edge \
    --tts-voice ja-JP-KeitaNeural \
    --mute /content/mute.srt \
    --note-overlay-ass /content/note_overlay.ass \
    --image-overlay-srt /content/image_overlay.srt \
    --image-overlay-dir /content/overlay_images \
    --render-config /content/CharenjiZukan/assets/default_render_config.json \
    --ambient /content/ambient.mp3 \
    --slow-cap 0.5 \
    --output-dir /content/output_sync \
    --output-name video_synced \
    --no-hardsub \
    --workers 4 \
    --batch-size 100 \
    --no-gpu \
    --subtitle-max-chars 0
```

#### Chạy hàng loạt nhiều video (Batch JSON)

Yêu cầu: Truyền danh sách tasks qua file JSON thông qua `--task-file`. Mỗi task chạy trong **Process riêng biệt** (dùng `multiprocessing` context `spawn`) để đảm bảo giải phóng VRAM hoàn toàn sau khi xử lý QwenTTS/audio-separator.

**Cấu trúc JSON:**

```json
[
  {
    "input": "/content/video1.mp4",
    "subtitle": "/content/video1_translated.srt",
    "mute": "/content/video1_mute.srt",
    "note_overlay_ass": "/content/video1_note.ass",
    "image_overlay_srt": "/content/video1_image_overlay.srt",
    "image_overlay_dir": "/content/video1_overlay_images",
    "output": "/content/output/video1_synced.mp4"
  }
]
```

**Lưu ý:**

- Task JSON bắt buộc phải có `input` (video) và `subtitle`.
- `mute`, `note_overlay_ass`, `image_overlay_srt`, `image_overlay_dir` là tùy chọn.
- `image_overlay_srt` và `image_overlay_dir` chỉ có hiệu lực khi `image_overlay.enabled=true` trong render config.
- `output` phải là đường dẫn file `.mp4` đầy đủ (hệ thống tự tách `output_dir` và `output_name` từ đường dẫn này).
- Mỗi video chạy trong Process riêng — khi process kết thúc, OS tự động giải phóng VRAM cho model (QwenTTS, audio-separator) để task tiếp theo không bị OOM.

#### Tham số

| Tham số                | Mô tả                                                                                                | Mặc định                                |
| ---------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------- |
| `--task-file`          | File JSON chứa danh sách tasks cho xử lý hàng loạt                                                   | (không dùng)                            |
| `--video`              | File video gốc (`.mp4`, `.mkv`)                                                                      | (bắt buộc khi không dùng `--task-file`) |
| `--subtitle`           | File subtitle `.srt` đầy đủ (bao gồm cả vùng mute nếu có)                                            | (bắt buộc khi không dùng `--task-file`) |
| `--tts-provider`       | Provider TTS (`edge`, `voicevox_nemo`, `voicevox`, `qwen`)                                           | `edge`                                  |
| `--tts-voice`          | Giọng đọc EdgeTTS hoặc ID nhân vật Voicevox/Voicevox Nemo (ghi đè YAML)                              | (lấy từ `tts_config.yaml`)              |
| `--tts-config`         | File YAML cấu hình TTS (dùng cho `edge`, `voicevox_nemo`, `voicevox`, `qwen`)                        | `config/tts_config.yaml`                |
| `--mute`               | File mute `.srt` cho vùng quoted (không TTS)                                                         | (không dùng)                            |
| `--note-overlay-ass`   | File ASS text cho note overlay                                                                       | (không dùng)                            |
| `--image-overlay-srt`  | File SRT điều khiển static image overlay; text block là basename không có extension                  | (không dùng)                            |
| `--image-overlay-dir`  | Thư mục chứa static image overlay                                                                    | (không dùng)                            |
| `--render-config`      | File JSON cấu hình render (style, resolution, dải đen, watermark...)                                 | `assets/default_render_config.json`     |
| `--ambient`            | Nhạc nền ambient cho toàn bộ video                                                                   | `assets/ambient.mp3`                    |
| `--slow-cap`           | Giới hạn tốc độ video thấp nhất (cap cho stretch)                                                    | `0.5`                                   |
| `--output-dir`         | Thư mục output                                                                                       | `./sync_output/`                        |
| `--output-name`        | Tên base cho tất cả file output                                                                      | `video_synced`                          |
| `--no-hardsub`         | Bỏ render MP4 hardsub, chỉ xuất các file đã remap                                                    | (tắt)                                   |
| `--workers`            | Số worker FFmpeg chạy song song khi xử lý chunk video                                                | `4`                                     |
| `--batch-size`         | Số segments mỗi batch Filter Complex (giảm = ít RAM hơn)                                             | `100`                                   |
| `--no-gpu`             | Tùy chọn tương thích cũ; video render vẫn bắt buộc dùng `hevc_nvenc -preset p4 -tune hq -cq 28`      | (tắt)                                   |
| `--keep-tmp`           | Giữ lại thư mục tạm chứa các chunks video để debug                                                   | (tắt)                                   |
| `--subtitle-max-chars` | Số ký tự tối đa mỗi dòng khi wrap text subtitle                                                      | `0`                                     |
| `--tuber-config`       | File JSON cấu hình tuber overlay (bỏ trống = tắt). Xem [Tuber Overlay Guide](tuber-overlay-guide.md) | (không dùng)                            |

> **Lưu ý về Tuber Overlay:** Khi bật `--tuber-config`, pipeline sẽ render overlay nhân vật PNGTuber (Python/PIL + FFmpeg) lên video trước khi final render. Chỉ cần Pillow + FFmpeg (không cần Node.js).

> **Lưu ý về âm lượng (Volume):** Khi cấu hình tách BGM trong `render_config`, bạn có thể điều chỉnh âm lượng của BGM đã tách và ambient thông qua block `audio_mix`:
>
> ```json
> "audio_mix": {
>   "ambient_volume": 0.03,
>   "bgm_volume": 1.0
> }
> ```

#### Quy ước input/output quan trọng

- Chương trình tự động sinh audio theo `--tts-voice`.
- Khi chạy đủ pipeline (không bật `--no-hardsub`), output chính bao gồm:
  - `<output-name>.mp4`
  - `<output-name>_tts_synced.srt`
  - `<output-name>_synced.srt`
- Output tùy chọn nếu có input tương ứng:
  - `<output-name>_mute_synced.srt` (khi có `--mute`)
  - `<output-name>_image_overlay_synced.srt` chỉ được giữ khi bật `--keep-tmp` hoặc `image_overlay.keep_intermediate_srt=true`
  - `<output-name>_note_overlay.ass` (ASS cuối có `NoteBox` + `NoteText`, khi có `--note-overlay-ass`)
  - `<output-name>_note_synced.ass` chỉ được giữ khi bật `--keep-tmp` hoặc `note_overlay.keep_intermediate_ass=true`

#### Image Overlay static image theo SRT

- Layer render: Base video → Image overlay static image → Note overlay → Black strip → Watermark → Subtitle.
- SRT overlay dùng timestamp video gốc; pipeline remap timestamp sau khi video stretch.
- Text block là tên ảnh không có extension, ví dụ `frame_intro` → một file trong `/content/overlay_images/` có basename khớp như `frame_intro.png`, `frame_intro.webp` hoặc `frame_intro.jpg`.
- Renderer deduplicate static image inputs: nhiều event dùng cùng file thì FFmpeg chỉ load một input và dùng `split=N`.
- `render_strategy=auto` là khuyến nghị; `direct` ép `-filter_complex`, `script` ép `-filter_complex_script`.

#### Note Overlay dynamic ASS box

- Không cần upload `assets/note_overlay.png`; asset này đã deprecated.
- Cần giữ font CJK, ví dụ `assets/NotoSansCJKsc-VF.ttf`, để Pillow đo pixel khi wrap text.
- ASS nguồn có thể chọn layout per dialogue bằng field `Name`; SRT nguồn có thể chọn layout bằng dòng text đầu tiên của mỗi block nhiều dòng rồi convert bằng `srt-to-ass`.
- Trên Colab, libass và Pillow có thể đo glyph lệch nhẹ; nên giữ `padding_bottom` và `height_safety_margin` đủ lớn trong `render_config.json`.

---

### 2.12. Tuber Overlay (sync-video --tuber-config + tuber-repair)

Thêm overlay nhân vật ảo PNGTuber vào video output. Pipeline thuần Python/PIL + FFmpeg (không cần Node.js). Xem chi tiết tại [Tuber Overlay Guide](tuber-overlay-guide.md).

#### Chạy sync-video với tuber overlay

```colab
!uv run sync-video \
    --video /content/video.mp4 \
    --subtitle /content/subtitle_ja.srt \
    --tuber-config /content/CharenjiZukan/assets/tuber_overlay_config.json
```

> **Lưu ý:** `--tuber-config` dùng file JSON mẫu tại `assets/tuber_overlay_config.json`. Sửa `enabled: true` để bật.

#### Late repair (sau khi fallback non-tuber)

```colab
!uv run tuber-repair --tuber-root tuber-output/<job>/tuber
```

#### File cấu hình `assets/tuber_overlay_config.json`

Các key chính: `enabled` (bật/tắt), `asset` (PNGTuber + chromakey), `character` (vị trí/kích thước — width ưu tiên giữ tỉ lệ), `mouth.mode` (cue/amplitude/hybrid), `retry.retryAttempts`, `artifactPolicy.mode=repairable`. Xem [Tuber Overlay Guide](tuber-overlay-guide.md) để biết đầy đủ tham số.

#### Test full flow tuber overlay trên Colab

---

### 2.13. Pre-cut Video (pre-cut-video)

Loại bỏ các đoạn thừa từ video gốc **trước khi** chạy transcript/translate/sync. CLI này tạo video clean và manifest JSON để trace timeline.

#### Re-encode smooth (HEVC NVENC)

```colab
!uv run pre-cut-video \
    --input /content/source.mp4 \
    --output /content/clean.mp4 \
    --remove-srt /content/remove.srt \
    --method reencode-smooth
```

#### Đầy đủ tham số

```colab
!uv run pre-cut-video \
    --input /content/source.mp4 \
    --output /content/clean.mp4 \
    --remove-srt /content/remove.srt \
    --manifest /content/clean_manifest.json \
    --method hybrid-copy \
    --audio-bitrate 256k \
    --audio-fade-ms 10 \
    --safe-margin-ms 100 \
    --keep-tmp \
    --verbose
```

#### File remove.srt format

Mỗi block SRT là một đoạn cần **xóa** khỏi video gốc. Text trong block dùng làm ghi chú:

```srt
1
00:00:12,500 --> 00:00:18,000
CUT intro mistake

2
00:03:10,000 --> 00:03:25,200
CUT sponsor
```

#### Tham số

| Tham số                | Mô tả                                                | Mặc định                     |
| ---------------------- | ---------------------------------------------------- | ---------------------------- |
| `--input`, `-i`        | File video gốc cần cắt                               | (bắt buộc)                   |
| `--output`, `-o`       | File video clean sau khi cắt                         | (bắt buộc)                   |
| `--remove-srt`, `-r`   | File SRT chứa các đoạn cần xóa (timestamp video gốc) | (bắt buộc)                   |
| `--manifest`           | Path manifest JSON                                   | `<output>_cut_manifest.json` |
| `--method`             | Phương pháp: `hybrid-copy` hoặc `reencode-smooth`    | `hybrid-copy`                |
| `--hevc-cq`            | CQ cho reencode-smooth                               | `28`                         |
| `--maxrate-ratio`      | Nhân input bitrate để tính maxrate (reencode-smooth) | `1.15`                       |
| `--hevc-preset`        | Preset cho hevc_nvenc                                | `p4`                         |
| `--audio-bitrate`      | Bitrate AAC output                                   | `256k`                       |
| `--audio-fade-ms`      | Fade audio ở rìa mỗi keep part (ms)                  | `10`                         |
| `--safe-margin-ms`     | Mở rộng remove ranges trên source timeline (ms)      | `100`                        |
| `--disable-audio-fade` | Tắt audio fade                                       | (tắt)                        |
| `--keep-tmp`           | Giữ part files tạm sau concat để debug               | (tắt)                        |
| `--verbose`, `-v`      | Bật log chi tiết (DEBUG level)                       | (tắt)                        |

> **Lưu ý:** Sau pre-cut, tất cả timestamp đều thuộc timeline của video clean. Không dùng lại timestamp của video gốc cho các bước sau.

---

## 3. Chạy test trên Google Colab với `run_colab_tests.py`

File nằm tại: `run_colab_tests.py` (project root)

### 3.1 Cú pháp đầy đủ

```colab
!python run_colab_tests.py [OPTIONS]

OPTIONS:
  --matrix PATH       File test_matrix.yaml (mặc định: tests/test_matrix.yaml)
  --tags TAG [TAG...] Lọc entry theo tags (OR logic: khớp bất kỳ tag nào)
  --name SUBSTR       Lọc entry theo tên (substring, case-insensitive)
  --reports-dir DIR   Thư mục lưu báo cáo fail (mặc định: test_reports/)
  --list              Hiển thị danh sách test sẽ chạy, không chạy thật
```

### 3.2 Các trường hợp sử dụng thường gặp

**Xem danh sách không chạy**:

```colab
!python run_colab_tests.py --list
!python run_colab_tests.py --tags gpu --list
```

**Chạy toàn bộ tests nhanh (không GPU)**:

```colab
!python run_colab_tests.py --tags unit
!python run_colab_tests.py --tags unit integration
```

**Tìm và chạy tất cả các hàm/class có chứa chữ "vertexai" trong tên**:

```colab
!uv run pytest tests/test_translation_providers.py -k "vertexai" -v
```

**Chạy test 1 hàm**:

```colab
!python -m pytest tests/test_translation_providers.py::TestLayer4_RealAPIs::test_vertexai_real_api
```

**Chạy tất cả tests liên quan 1 feature**:

```colab
!python run_colab_tests.py --name "Native Video"
!python run_colab_tests.py --name "SRT Parser"
```

**Chạy toàn bộ** (enabled tests):

```colab
!python run_colab_tests.py
```

**Dùng file matrix khác** (khi có nhiều môi trường):

```colab
!python run_colab_tests.py --matrix tests/test_matrix_ci.yaml
```

### 3.3 Chạy toàn bộ test trong một file

Khi một entry trong [`tests/test_matrix.yaml`](tests/test_matrix.yaml) **bỏ trống `keyword`**, script [`run_colab_tests.py`](tests/run_colab_tests.py) sẽ **không thêm cờ `-k`** và `pytest` sẽ collect toàn bộ test trong file đó.

#### Ví dụ entry trong `test_matrix.yaml`

```yaml
- name: "Native Video OCR — All Layers"
  file: "tests/test_native_video_ocr_pipeline.py"
  # keyword: bỏ trống hoàn toàn
  timeout_sec: 900 # Tổng timeout của cả 4 layers cộng lại
  pytest_args: ["-v", "-s"]
  tags: ["unit", "integration", "gpu", "native_ocr"]
  enabled: true
```

> `timeout_sec` ở đây là timeout tổng cho toàn bộ các layer trong cùng một file.

#### Chạy trực tiếp bằng pytest (không qua `run_colab_tests.py`)

```bash
python -m pytest tests/test_native_video_ocr_pipeline.py -v
```

### 3.4 Quy trình làm việc trên Google Colab

#### 3.4.1 Workflow chuẩn khi develop một feature mới

```text
Bước 1: Viết code feature
Bước 2: Viết file test (4 layers) + thêm vào test_matrix.yaml
Bước 3: Chạy Layer 1 → fix cho đến khi pass
Bước 4: Chạy Layer 2 → fix cho đến khi pass
Bước 5: Chạy Layer 3 → fix cho đến khi pass
Bước 6: (Khi có GPU) Chạy Layer 4 → confirm chất lượng AI
```

```colab
# Cell: Layer 1 và 2 (không cần GPU)
!python run_colab_tests.py --tags unit integration --name "TÊN FEATURE"

# Cell: Layer 3 (không cần GPU)
!python run_colab_tests.py --name "TÊN FEATURE" --name "Layer 3"

# Cell: Nếu có lỗi, xem report
!ls test_reports/
!cat "test_reports/failed_*.md"
```

#### 3.4.2 Workflow debug khi có fail

```colab
# Bước 1: Xem report tóm tắt
!ls -la test_reports/*.md

# Bước 2: Đọc report chi tiết (hoặc download file .md để gửi AI)
import subprocess
result = subprocess.run(["cat", "test_reports/failed_xxx.md"], capture_output=True, text=True)
print(result.stdout[:5000])  # Print 5000 chars đầu

# Bước 3: Chạy lại command từ mục "1. Lệnh đã chạy" trong report
!python -m pytest tests/test_native_video_ocr_pipeline.py -k "Layer3" -v --tb=long -s

# Bước 4: Chạy 1 test duy nhất để isolate lỗi
!python -m pytest tests/test_native_video_ocr_pipeline.py::TestLayer3_FullPipeline::test_entries_count -v --tb=long
```

---

## 4. Cách dùng truyền thống (không có uv)

Nếu không muốn dùng uv, bạn có thể cài đặt thủ công:

```colab
!pip install google-genai tenacity edge-tts pydub pyrubberband soundfile aiohttp -q
!apt-get install -y rubberband-cli

# Sử dụng Secrets cho API key
from google.colab import userdata
gemini_key = userdata.get('gemini_key')

# Chạy trực tiếp với python (đường dẫn từ thư mục project)
!python cli/translate_srt.py --input video.srt --keys "{gemini_key}"
!python cli/tts.py --input video_vi.srt --config config/tts_config.yaml
```

---

## 5. Xử lý sự cố

### WhisperX lỗi CUDA

```colab
# Kiểm tra GPU
!nvidia-smi

# Cài lại CUDA dependencies
!apt install libcudnn8 libcudnn8-dev -y
```

### EdgeTTS lỗi kết nối

```colab
# Thử với proxy (nếu cần)
!uv run tts --input video.srt --config config/tts_config.yaml
```

### Output không có extension

Script sẽ tự động thêm `.wav` nếu output không có extension.

### Lỗi "Failed to spawn: mute-srt"

Chạy lệnh sau để cài đặt package:

```colab
%cd /content/CharenjiZukan
!uv pip install -e .
```

Hoặc chạy trực tiếp file Python:

```colab
!uv run python cli/mute_srt.py --input video.mp4 --mute mute.srt
```

---

## 7. Lưu ý quan trọng

1. **Cài đặt project**: Sau khi clone, cần chạy `!uv pip install -e .` để cài đặt project ở chế độ editable, cho phép sử dụng các CLI commands (`mute-srt`, `translate-srt`, `tts`, `video-ocr`).

2. **rubberband-cli**: Cần cài đặt bằng `apt-get` vì đây là binary hệ thống, không phải Python package. Dùng cho time-stretch audio chất lượng cao.

3. **API Keys**: Sử dụng Google Colab Secrets để bảo mật API keys. Không hardcode token vào code.

4. **Multi-Box OCR**: File `boxesOCR.txt` phải đúng format `name x y w h`, mỗi box một dòng.

5. **Output Multi-Box**: `video-ocr` xuất nhiều file theo mẫu `<video_stem>_<box_name>.srt` hoặc `.txt`, không còn dùng `--output` file đơn.

6. **Autorate**: Khi bật `--autorate`, audio sẽ được nén/giãn để khớp với thời lượng slot trong file SRT.

7. **Gemini API Key**: Sử dụng cú pháp `--keys "{gemini_key}"` với biến từ `userdata.get('gemini_key')`.

8. Trình tự chạy lại để xác thực fix:

```colab
!uv cache clean
!uv sync --reinstall
!uv pip install -e .
```
