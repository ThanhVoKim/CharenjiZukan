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

| Tham số             | Mô tả                                                                        | Mặc định                            |
| ------------------- | ---------------------------------------------------------------------------- | ----------------------------------- |
| `--input`, `-i`     | File video hoặc audio đầu vào                                                | (bắt buộc nếu không dùng task-file) |
| `--task-file`, `-t` | File JSON cấu hình chạy hàng loạt (`{"input": "...", "output": "..."}`)      | (không dùng)                        |
| `--output`, `-o`    | File .srt hoặc folder đầu ra (chỉ dùng với `--input`)                        | `<input_dir>/<name>.srt`            |
| `--language`, `-l`  | Ngôn ngữ audio (`Chinese`, `English`, `Japanese`...)                         | `Chinese`                           |
| `--max-chars`       | Số ký tự tối đa trên mỗi dòng phụ đề (CJK thường 15, Latin 40), đặt 0 để tắt | `15`                                |
| `--min-chars`       | Số ký tự tối thiểu trên mỗi dòng phụ đề, đặt 0 để tắt                        | `8`                                 |
| `--batch-size`      | Batch size cho inference (tăng lên 32~64 nếu GPU L4 22GB)                    | `32`                                |
| `--offset-seconds`  | Độ lệch bù trừ thời gian (giây, ví dụ: 0.24 = 6 frames @ 25fps)              | `0.24`                              |
| `--model-path`      | Đường dẫn model ASR trên HuggingFace hoặc local                              | `Qwen/Qwen3-ASR-1.7B`               |
| `--aligner-path`    | Đường dẫn model Forced Aligner                                               | `Qwen/Qwen3-ForcedAligner-0.6B`     |
| `--device`, `-d`    | Thiết bị chạy (`cuda:0`, `cuda:1`, `cpu`)                                    | `cuda:0`                            |
| `--verbose`         | Bật log chi tiết                                                             | (tắt)                               |

#### Lưu ý quan trọng

- **Flash Attention**: Bắt buộc phải có `flash-attn` để tối ưu VRAM. Nếu không cài được prebuilt wheel, có thể cài từ source (rất lâu trên Colab):
  ```colab
  !uv pip install flash-attn --no-build-isolation
  ```

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
    --provider-config config/openai_compat_translate.yaml \
    --keys "sk-deepseek-xxx" \
    --lang "Japanese"
```

#### Dịch với Vertex AI (Application Default Credentials)

```colab
!uv run translate-srt \
    --input /content/video.srt \
    --provider vertexai \
    --provider-config config/vertexai_translate.yaml \
    --lang "Japanese"
```

#### Đầy đủ tham số

```colab
from google.colab import userdata
gemini_key = userdata.get('gemini_key')

!uv run translate-srt \
    --input     /content/video.srt \
    --output    /content/video_ja.srt \
    --provider  gemini \
    --lang      "Japanese" \
    --keys      "{gemini_key}" \
    --model     "gemini-3-flash-preview" \
    --prompt    /content/prompts/gemini.txt \
    --batch     30 \
    --budget    24576 \
    --wait      0.5 \
    --no-context \
    --verbose
```

#### Tham số

| Tham số             | Mô tả                                   | Mặc định                 |
| ------------------- | --------------------------------------- | ------------------------ |
| `--input`, `-i`     | File .srt đầu vào                       | (bắt buộc)               |
| `--provider`, `-p`  | Provider (gemini/openai/vertexai)       | `gemini`                 |
| `--provider-config` | Đường dẫn file config YAML              | (tuỳ provider)           |
| `--base-url`        | Override base URL cho OpenAI provider   | `None`                   |
| `--keys`, `-k`      | API key(s), phân cách bằng dấu phẩy     | (bắt buộc gemini/openai) |
| `--output`, `-o`    | File .srt đầu ra                        | `<input>_<lang>.srt`     |
| `--lang`, `-l`      | Ngôn ngữ đích (tên tiếng Anh đầy đủ)    | `Vietnamese`             |
| `--model`, `-m`     | Model Gemini (nếu dùng gemini provider) | `gemini-3-flash-preview` |
| `--prompt`          | Đường dẫn tới file prompt gemini.txt    | `prompts/gemini.txt`     |
| `--batch`, `-b`     | Số dòng dịch mỗi lần                    | `30`                     |
| `--budget`          | Thinking budget tokens (Gemini only)    | `24576`                  |
| `--wait`            | Giây chờ giữa mỗi batch                 | `0`                      |
| `--no-context`      | Tắt global context                      | (mặc định bật)           |
| `--verbose`, `-v`   | Hiển thị log chi tiết                   | (tắt)                    |

---

### 2.5. SRT to ASS (srt-to-ass)

Chuyển file SRT thành file ASS để overlay lên video.

#### Chuyển đổi nhanh

```colab
!uv run srt-to-ass --input /content/note_translated.srt
```

#### Đầy đủ tham số

```colab
!uv run srt-to-ass \
    --input     /content/note_translated.srt \
    --output    /content/note_overlay.ass \
    --template  /content/CharenjiZukan/assets/sample.ass \
    --max-chars 14 \
    --style     NoteStyle \
    --verbose
```

#### Tham số

| Tham số            | Mô tả                                   | Mặc định            |
| ------------------ | --------------------------------------- | ------------------- |
| `--input`, `-i`    | File SRT đầu vào                        | (bắt buộc)          |
| `--output`, `-o`   | File ASS đầu ra                         | `<input>.ass`       |
| `--template`, `-t` | File ASS template                       | `assets/sample.ass` |
| `--max-chars`      | Số ký tự tối đa mỗi dòng (tự động ngắt) | `14`                |
| `--style`          | Tên style trong ASS                     | `NoteStyle`         |
| `--verbose`, `-v`  | Hiển thị log chi tiết                   | (tắt)               |

---

### 2.6. Text-to-Speech (tts)

Hỗ trợ 3 engine: **EdgeTTS** (mặc định, cloud), **Voicevox** (local server), và **Qwen3-TTS** (HuggingFace, voice-clone).

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

#### Sử dụng Voicevox

**Bước 1: Khởi động Server Voicevox ngầm**

```colab
!python setup_voicevox.py
```

Chạy trong một cell riêng biệt trước khi gọi lệnh TTS:

```colab
import subprocess
import time

print("🚀 Đang khởi động Voicevox Server ngầm...")
process = subprocess.Popen(
    ["uv", "run", "run.py", "--use_gpu", "--host", "127.0.0.1", "--port", "50121"],
    cwd="/content/voicevox_nemo_engine",
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

while True:
    line = process.stdout.readline()
    if "Application startup complete" in line or "Uvicorn running on" in line:
        print("✅ Voicevox Server đã sẵn sàng tại 127.0.0.1:50121")
        break
```

**Bước 2: Chạy TTS với Voicevox**

```colab
!uv run tts \
    --input /content/video_ja.srt \
    --provider voicevox \
    --config /content/CharenjiZukan/config/tts_config.yaml
```

#### Sử dụng Qwen3-TTS (Voice Clone)

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
| `--provider`, `-p`  | TTS engine (edge/voicevox/qwen)                      | `edge`                              |
| `--autorate`        | Tự động nén audio khớp slot SRT (chỉ .srt)           | (tắt)                               |
| `--max-speed`       | Giới hạn tốc độ nén tối đa                           | `100.0`                             |
| `--silence-ms`      | Độ dài silence giữa các dòng khi không dùng autorate | `0`                                 |
| `--cache`           | Thư mục cache audio tạm                              | `tmp/<stem>_<ts>/`                  |
| `--keep-cache`      | Giữ lại thư mục cache tạm sau khi xử lý xong         | (tắt)                               |
| `--list-voices`     | Liệt kê giọng EdgeTTS                                | (không dùng)                        |
| `--verbose`         | Bật logging debug                                    | (tắt)                               |

#### File cấu hình `config/tts_config.yaml`

```yaml
provider: "edge"

edge:
  voice: "vi-VN-HoaiMyNeural"
  rate: "+0%"
  volume: "+0%"
  pitch: "+0Hz"
  concurrent: 10
  strip_silence: true
  silence_thresh_dbfs: -50

voicevox:
  voice_id: 10008
  host: "127.0.0.1"
  port: 50121
  speed_scale: 1.12
  pitch_scale: -0.05
  concurrent_requests: 100

qwen:
  model_path: "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
  ref_audio: ""
  ref_text: ""
  batch_size: 32
  device: "cuda:0"
```

---

### 2.7. Demucs Audio Separation (demucs-audio)

Tách voice/background từ audio sử dụng AI model Demucs.

#### Tách background music (mặc định)

```colab
!uv run demucs-audio --input /content/audio_muted.wav
```

#### Tách vocals (giữ lại giọng nói)

```colab
!uv run demucs-audio --input /content/audio.wav --keep vocals
```

#### Đầy đủ tham số

```colab
!uv run demucs-audio \
    --input   /content/audio_muted.wav \
    --output  /content/audio_bgm.wav \
    --model   htdemucs \
    --keep    bgm \
    --bitrate 128k \
    --device  cuda \
    --segment 7 \
    --verbose
```

#### Ví dụ nâng cao với `--keep`

```colab
# Chỉ lấy source "other" (index 2)
!uv run demucs-audio --input audio.wav --keep 2 --output other.mp3

# Lấy drums + bass (index 0,1)
!uv run demucs-audio --input audio.wav --keep 0,1 --output drums_bass.m4a

# Lấy drums + bass + other (index 0-2)
!uv run demucs-audio --input audio.wav --keep 0-2 --output bgm.mp3

# Lấy tất cả trừ vocals
!uv run demucs-audio --input audio.wav --keep 0-2 --output no_vocals.mp3
```

#### Tham số

| Tham số           | Mô tả                                               | Mặc định          |
| ----------------- | --------------------------------------------------- | ----------------- |
| `--input`, `-i`   | File audio đầu vào                                  | (bắt buộc)        |
| `--output`, `-o`  | File audio đầu ra (.wav, .mp3, .m4a, .aac)          | `<input>_bgm.wav` |
| `--model`, `-m`   | Model Demucs: htdemucs, htdemucs_ft, mdx, mdx_extra | `htdemucs`        |
| `--keep`, `-k`    | Sources giữ lại (xem bảng dưới)                     | `bgm`             |
| `--bitrate`, `-b` | Bitrate cho MP3/M4A output                          | `192k`            |
| `--device`, `-d`  | Device: cuda, cuda:0, cpu                           | auto-detect       |
| `--segment`       | Độ dài chunk (giây) để xử lý                        | `7`               |
| `--verbose`, `-v` | Hiển thị log chi tiết                               | (tắt)             |

#### Tham số `--keep` (chọn sources)

Demucs tách audio thành 4 sources với index:

| Index | Source | Mô tả     |
| ----- | ------ | --------- |
| 0     | drums  | Trống     |
| 1     | bass   | Bass      |
| 2     | other  | Nhạc khác |
| 3     | vocals | Giọng hát |

**Cách sử dụng `--keep`:**

```colab
# Presets
--keep bgm      # drums + bass + other (mặc định)
--keep vocals   # chỉ giọng hát
--keep drums    # chỉ trống
--keep bass     # chỉ bass
--keep other    # chỉ nhạc khác

# Index đơn lẻ
--keep 2        # chỉ other
--keep 3        # chỉ vocals

# Nhiều index (phẩy)
--keep 0,2      # drums + other
--keep 1,2      # bass + other

# Range index (gạch nối)
--keep 0-2      # drums + bass + other (tương đương bgm)
--keep 1-3      # bass + other + vocals
```

#### Output formats

```colab
# WAV (mặc định - chất lượng cao, file lớn)
--output bgm.wav

# MP3 (nén, file nhỏ hơn)
--output bgm.mp3 --bitrate 128k

# M4A/AAC (tốt cho video)
--output bgm.m4a --bitrate 192k
```

#### Lưu ý quan trọng

- Demucs yêu cầu audio **stereo (2 channels)** và **44.1kHz+** để có chất lượng tốt
- Nếu input là mono, script sẽ tự động convert sang stereo
- Nên dùng audio chất lượng cao thay vì audio 16kHz mono từ WhisperX
- WAV có chất lượng cao nhất nhưng file lớn; MP3/M4A nhỏ hơn nhưng mất chất lượng

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
| `--config`                | Đường dẫn file cấu hình `.yaml`                                                  | (không dùng)                                                |

> **Mức ưu tiên Cấu hình**: CLI parameters có mức ưu tiên cao nhất, sau đó là tham số khai báo trong `--config`, và cuối cùng là Default values.

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

CLI `sync-video` dùng pipeline `sync_engine` để đồng bộ video + TTS theo timeline subtitle (gồm 5 phase: phân tích timeline, xử lý video chunks, ghép audio, remap timestamps, render final).

#### Chạy nhanh

```colab
!uv run sync-video \
    --video /content/video.mp4 \
    --subtitle /content/subtitle_translated.srt \
    --tts-provider edge \
    --tts-voice ja-JP-KeitaNeural \
    --output-dir /content/output_sync
```

#### Chạy nhanh với Voicevox

Yêu cầu đã bật server Voicevox ngầm (xem phần 2.6).

```colab
!uv run sync-video \
    --video /content/video.mp4 \
    --subtitle /content/subtitle_translated.srt \
    --tts-provider voicevox \
    --tts-voice 10008 \
    --output-dir /content/output_sync
```

#### Chạy đầy đủ tham số

```colab
!uv run sync-video \
    --video /content/video.mp4 \
    --subtitle /content/subtitle_translated.srt \
    --tts-provider edge \
    --tts-voice ja-JP-KeitaNeural \
    --mute /content/mute.srt \
    --note-overlay-png /content/note-overlay.png \
    --note-overlay-ass /content/note_overlay.ass \
    --black-bg /content/black-background.png \
    --ambient /content/ambient.mp3 \
    --slow-cap 0.5 \
    --use-demucs \
    --output-dir /content/output_sync \
    --output-name video_synced \
    --no-hardsub \
    --workers 4 \
    --no-gpu \
    --subtitle-fontname "Noto Sans CJK JP" \
    --subtitle-fontsize 22 \
    --subtitle-color "&H00EEF5FF" \
    --subtitle-margin-v 6 \
    --note-max-chars 15
    --subtitle-max-chars 0
```

#### Tham số

| Tham số                | Mô tả                                                     | Mặc định             |
| ---------------------- | --------------------------------------------------------- | -------------------- |
| `--video`              | File video gốc (`.mp4`, `.mkv`)                           | (bắt buộc)           |
| `--subtitle`           | File subtitle `.srt` đầy đủ (bao gồm cả vùng mute nếu có) | (bắt buộc)           |
| `--tts-provider`       | Provider TTS (`edge` hoặc `voicevox`)                     | `edge`               |
| `--tts-voice`          | Giọng đọc EdgeTTS hoặc ID nhân vật Voicevox               | `vi-VN-HoaiMyNeural` |
| `--mute`               | File mute `.srt` cho vùng quoted (không TTS)              | (không dùng)         |
| `--note-overlay-png`   | Ảnh PNG tĩnh nền note                                     | (không dùng)         |
| `--note-overlay-ass`   | File ASS text cho note overlay                            | (không dùng)         |
| `--black-bg`           | Ảnh dải đen nền note (tự tạo nếu không truyền)            | (không dùng)         |
| `--ambient`            | Nhạc nền ambient cho toàn bộ video                        | `assets/ambient.mp3` |
| `--slow-cap`           | Giới hạn tốc độ video thấp nhất (cap cho stretch)         | `0.5`                |
| `--use-demucs`         | Dùng Demucs tách lời (vocals) cho các đoạn quoted audio   | (tắt)                |
| `--output-dir`         | Thư mục output                                            | `./sync_output/`     |
| `--output-name`        | Tên base cho tất cả file output                           | `video_synced`       |
| `--no-hardsub`         | Bỏ render MP4 hardsub, chỉ xuất các file đã remap         | (tắt)                |
| `--workers`            | Số worker FFmpeg chạy song song khi xử lý chunk video     | `4`                  |
| `--batch-size`         | Số segments mỗi batch Filter Complex (giảm = ít RAM hơn)  | `100`                |
| `--no-gpu`             | Dùng `libx264` thay `h264_nvenc` (CPU mode)               | (tắt)                |
| `--keep-tmp`           | Giữ lại thư mục tạm chứa các chunks video để debug        | (tắt)                |
| `--subtitle-fontname`  | Font subtitle dùng khi burn hardsub                       | `Noto Sans CJK JP`   |
| `--subtitle-fontsize`  | Cỡ chữ subtitle                                           | `22`                 |
| `--subtitle-color`     | Màu chữ subtitle (ASS hex format)                         | `&H00EEF5FF`         |
| `--subtitle-margin-v`  | Margin dọc subtitle (px)                                  | `6`                  |
| `--note-max-chars`     | Số ký tự tối đa mỗi dòng khi wrap text ASS note           | `15`                 |
| `--subtitle-max-chars` | Số ký tự tối đa mỗi dòng khi wrap text subtitle           | `0`                  |

#### Quy ước input/output quan trọng

- Chương trình tự động sinh audio theo `--tts-voice`.
- Khi chạy đủ pipeline (không bật `--no-hardsub`), output chính bao gồm:
  - `<output-name>.mp4`
  - `<output-name>_tts_synced.srt`
  - `<output-name>_synced.srt`
- Output tùy chọn nếu có input tương ứng:
  - `<output-name>_mute_synced.srt` (khi có `--mute`)
  - `<output-name>_note_synced.ass` (khi có `--note-overlay-ass`)

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
