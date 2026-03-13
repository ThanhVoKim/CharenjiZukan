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

**Cách tạo GitHub Personal Access Token:**

1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Tick `repo` (để truy cập private repo)
4. Copy token và thêm vào Secrets

**Cách tạo Gemini API Key:**

1. Truy cập [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy key và thêm vào Secrets

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

# Cài đặt project ở chế độ editable (để sử dụng CLI commands)
!uv pip install -e .

# Cài đặt rubberband-cli (cần cho time-stretch)
!apt-get install -y rubberband-cli
```

> **Lưu ý:** Sử dụng `userdata.get()` để lấy token từ Secrets, không hardcode token vào code để tránh lộ thông tin nhạy cảm.

### 1.3. Cài đặt WhisperX (tùy chọn - cho Speech-to-Text)

Nếu cần chuyển video thành subtitle, cài đặt thêm WhisperX:

```colab
# Tạo virtual environment và cài whisperx
!uv venv
!uv pip install -p .venv/bin/python whisperx
!uv pip install -p .venv/bin/python pyrubberband

# Cài CUDA dependencies
!apt install libcudnn8 libcudnn8-dev -y
!pip install modelscope addict

# Set environment variables
%env MPLBACKEND=agg
%env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
%env LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/
```

---

## 2. Các script chính

### 2.1. Mute Audio (mute-srt)

Dùng khi audio có 2 ngôn ngữ (ví dụ: bình luận + video gốc trích dẫn). Thay thế các đoạn được đánh dấu bằng silence, giữ nguyên độ dài audio.

#### Mute audio nhanh

```colab
!uv run mute-srt \
    --input /content/video.mp4 \
    --mute /content/mute.srt
```

#### Đầy đủ tham số

```colab
!uv run mute-srt \
    --input       /content/video.mp4 \
    --mute        /content/mute.srt \
    --output      /content/video_muted.wav \
    --sample-rate 16000 \
    --verbose
```

#### Tham số

| Tham số         | Mô tả                                | Mặc định               |
| --------------- | ------------------------------------ | ---------------------- |
| `--input`       | File audio/video đầu vào             | (bắt buộc)             |
| `--mute`        | File mute.srt chứa các đoạn cần mute | (bắt buộc)             |
| `--output`      | File audio đầu ra                    | `<input>_muted.wav`    |
| `--sample-rate` | Sample rate output                   | `16000` (cho WhisperX) |
| `--verbose`     | Hiển thị log chi tiết                | (tắt)                  |

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

### 2.2. Dịch SRT (translate-srt)

#### Dịch nhanh (với Secrets)

```colab
from google.colab import userdata
gemini_key = userdata.get('gemini_token')
!uv run translate-srt \
    --input /content/video.srt \
    --output /content/video_vi.srt \
    --keys "{gemini_key}"
```

#### Đầy đủ tham số

```colab
from google.colab import userdata
gemini_key = userdata.get('gemini_token')
!uv run translate-srt \
    --input   /content/video.srt         \
    --output  /content/video_vi.srt      \
    --lang    "Vietnamese"               \
    --keys    "{gemini_key}"             \
    --model   "gemini-2.5-flash"         \
    --batch   30                         \
    --budget  8192                       \
    --no-context
```

#### Tham số

| Tham số        | Mô tả                                      | Mặc định                      |
| -------------- | ------------------------------------------ | ----------------------------- |
| `--input`      | File .srt đầu vào                          | (bắt buộc)                    |
| `--output`     | File .srt đầu ra                           | `<input_stem>_translated.srt` |
| `--lang`       | Ngôn ngữ đích                              | `Vietnamese`                  |
| `--keys`       | Gemini API key(s), phân cách bằng dấu phẩy | (bắt buộc)                    |
| `--model`      | Model Gemini                               | `gemini-2.5-flash`            |
| `--batch`      | Số dòng dịch mỗi lần                       | `30`                          |
| `--budget`     | Token budget                               | `8192`                        |
| `--no-context` | Tắt global context                         | (mặc định bật)                |

---

### 2.3. Text-to-Speech (tts-srt)

#### Xem danh sách giọng

```colab
# Giọng tiếng Việt
!uv run tts-srt --list-voices vi

# Giọng tiếng Nhật
!uv run tts-srt --list-voices ja
```

#### TTS nhanh

```colab
!uv run tts-srt \
    --input /content/video_vi.srt \
    --voice vi-VN-HoaiMyNeural
```

#### TTS với autorate (tự động nén giọng)

```colab
!uv run tts-srt \
    --input   /content/video_vi.srt \
    --output  /content/video_vi.mp3 \
    --voice   vi-VN-HoaiMyNeural    \
    --rate    +5%                    \
    --autorate
```

#### Đầy đủ tham số

```colab
!uv run tts-srt \
    --input   /content/video_vi.srt \
    --output  /content/video_vi.wav \
    --voice   vi-VN-HoaiMyNeural    \
    --rate    +10%                   \
    --volume  +0%                    \
    --pitch   +0Hz                   \
    --autorate                       \
    --cache   /content/cache_tts
```

#### Tham số

| Tham số        | Mô tả                           | Mặc định                  |
| -------------- | ------------------------------- | ------------------------- |
| `--input`      | File .srt đầu vào               | (bắt buộc)                |
| `--output`     | File audio đầu ra (.wav/.mp3)   | `output/<input_stem>.wav` |
| `--voice`      | Tên giọng EdgeTTS               | (bắt buộc)                |
| `--rate`       | Tốc độ giọng (vd: +10%, -5%)    | `+0%`                     |
| `--volume`     | Âm lượng (vd: +20%)             | `+0%`                     |
| `--pitch`      | Cao độ (vd: +50Hz)              | `+0Hz`                    |
| `--autorate`   | Tự động nén audio khớp slot SRT | (tắt)                     |
| `--max-speed`  | Giới hạn tốc độ nén tối đa      | `100.0`                   |
| `--concurrent` | Số request EdgeTTS song song    | `10`                      |
| `--cache`      | Thư mục cache audio tạm         | `tmp/<stem>_<ts>/`        |
| `--verbose`    | Bật logging debug               | (tắt)                     |

---

## 3. Quy trình hoàn chỉnh

### 3.1. Tải video từ Google Drive

Đồng bộ video từ Google Drive sang local storage của Colab

### 3.2. Speech-to-Text với WhisperX

Chuyển audio thành file subtitle .srt:

```colab
!uv run whisperx "/content/video.mp4" \
  --model large-v2 \
  --language zh \
  --align_model jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn \
  --device cuda \
  --compute_type float16 \
  --batch_size 16 \
  --output_dir "/content/output"
```

#### Tham số WhisperX

| Tham số          | Mô tả           | Giá trị khuyến nghị                              |
| ---------------- | --------------- | ------------------------------------------------ |
| `--model`        | Model Whisper   | `large-v2` (chính xác cao nhất)                  |
| `--language`     | Ngôn ngữ audio  | `zh` (Trung), `ja` (Nhật), `en` (Anh)            |
| `--align_model`  | Model alignment | Xem [HuggingFace](https://huggingface.co/models) |
| `--device`       | Thiết bị xử lý  | `cuda` (GPU)                                     |
| `--compute_type` | Precision       | `float16` (GPU), `int8` (CPU)                    |
| `--batch_size`   | Batch size      | `16` (GPU L4)                                    |

#### Output

- File `.srt` tại thư mục output
- File `.json` với thông tin chi tiết

---

### 3.3. Dịch SRT

Dịch file subtitle sang ngôn ngữ đích:

```colab
from google.colab import userdata
gemini_key = userdata.get('gemini_token')
!uv run translate-srt \
    --input /content/output/video.srt \
    --output /content/output/video_ja.srt \
    --lang "Japanese" \
    --keys "{gemini_key}"
```

---

### 3.4. Text-to-Speech

Chuyển subtitle đã dịch thành file audio:

```colab
!uv run tts-srt \
    --input   /content/output/video_ja.srt \
    --output  /content/output/video_ja.wav \
    --voice   ja-JP-KeitaNeural \
    --rate    +5% \
    --autorate
```

---

### 3.5. Ghép Audio vào Video

---

## 4. Cách dùng truyền thống (không có uv)

Nếu không muốn dùng uv, bạn có thể cài đặt thủ công:

```colab
!pip install google-genai tenacity edge-tts pydub pyrubberband soundfile aiohttp -q
!apt-get install -y rubberband-cli

# Sử dụng Secrets cho API key
from google.colab import userdata
gemini_key = userdata.get('gemini_token')

# Chạy trực tiếp với python (đường dẫn từ thư mục project)
!python cli/translate_srt.py --input video.srt --keys "{gemini_key}"
!python cli/tts_srt.py --input video_vi.srt --voice vi-VN-HoaiMyNeural
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
!uv run tts-srt --input video.srt --voice ja-JP-KeitaNeural \
    --proxy http://127.0.0.1:7890
```

### Output không có extension

Script sẽ tự động thêm `.wav` nếu output không có extension.

---

## 6. Lưu ý quan trọng

1. **Cài đặt project**: Sau khi clone, cần chạy `!uv pip install -e .` để cài đặt project ở chế độ editable, cho phép sử dụng các CLI commands (`mute-srt`, `translate-srt`, `tts-srt`).

2. **rubberband-cli**: Cần cài đặt bằng `apt-get` vì đây là binary hệ thống, không phải Python package. Dùng cho time-stretch audio chất lượng cao.

3. **API Keys**: Sử dụng Google Colab Secrets để bảo mật API keys. Không hardcode token vào code.

4. **Output không có extension**: Nếu `--output` không có extension, script sẽ tự động thêm `.wav`.

5. **Autorate**: Khi bật `--autorate`, audio sẽ được nén/giãn để khớp với thời lượng slot trong file SRT.
