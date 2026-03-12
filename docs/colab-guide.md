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

### 2.1. Dịch SRT (translate_srt.py)

#### Dịch nhanh (với Secrets)

```colab
from google.colab import userdata
gemini_key = userdata.get('gemini_token')
!uv run translate_srt.py \
    --input /content/video.srt \
    --output /content/video_vi.srt \
    --keys "{gemini_key}"
```

#### Đầy đủ tham số

```colab
from google.colab import userdata
gemini_key = userdata.get('gemini_token')
!uv run translate_srt.py \
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

### 2.2. Text-to-Speech (tts_srt.py)

#### Xem danh sách giọng

```colab
# Giọng tiếng Việt
!uv run tts_srt.py --list-voices vi

# Giọng tiếng Nhật
!uv run tts_srt.py --list-voices ja
```

#### TTS nhanh

```colab
!uv run tts_srt.py \
    --input /content/video_vi.srt \
    --voice vi-VN-HoaiMyNeural
```

#### TTS với autorate (tự động nén giọng)

```colab
!uv run tts_srt.py \
    --input   /content/video_vi.srt \
    --output  /content/video_vi.mp3 \
    --voice   vi-VN-HoaiMyNeural    \
    --rate    +5%                    \
    --autorate
```

#### Đầy đủ tham số

```colab
!uv run tts_srt.py \
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

Đồng bộ video từ Google Drive sang local storage của Colab:

```colab
from google.colab import drive
import os
import subprocess
import glob

# --- CẤU HÌNH ---
FOLDER_NAME = "Survival"
DRIVE_ROOT_PATH = "/content/drive/MyDrive/CharenjiZukan/2603"

SOURCE_PARENT_PATH = os.path.join(DRIVE_ROOT_PATH)
LOCAL_PARENT_PATH = os.path.join("/content", FOLDER_NAME)

# Mount Drive
drive.mount('/content/drive', force_remount=True)

# Làm sạch thư mục local cũ
subprocess.run(["rm", "-rf", LOCAL_PARENT_PATH], check=False)
os.makedirs(LOCAL_PARENT_PATH, exist_ok=True)

# Duyệt và copy video
if os.path.exists(SOURCE_PARENT_PATH):
    print(f"🚀 Bắt đầu đồng bộ từ: {SOURCE_PARENT_PATH}")
    subfolders = sorted([f for f in os.listdir(SOURCE_PARENT_PATH)
                         if os.path.isdir(os.path.join(SOURCE_PARENT_PATH, f))])

    for item_name in subfolders:
        source_item_path = os.path.join(SOURCE_PARENT_PATH, item_name)
        local_item_path = os.path.join(LOCAL_PARENT_PATH, item_name)
        drive_assets_folder = os.path.join(source_item_path, "assets")

        # Bỏ qua nếu đã có folder assets (đã xử lý)
        if os.path.exists(drive_assets_folder) and os.path.isdir(drive_assets_folder):
            print(f"⏩ [BỎ QUA] Đã có folder 'assets': {item_name}")
            continue

        print(f"📥 [COPY] Đang tải dữ liệu nguồn: {item_name}")
        subprocess.run(["mkdir", "-p", local_item_path], check=True)

        # Copy file mp4
        mp4_files = glob.glob(os.path.join(source_item_path, "*.mp4"))
        if mp4_files:
            source_mp4_path = mp4_files[0]
            subprocess.run(["cp", source_mp4_path, local_item_path], check=True)
            print(f"  ✅ Đã copy: {os.path.basename(source_mp4_path)}")
        else:
            print(f"  ⚠️ Không tìm thấy file video (.mp4)")

    print("\n✅ Hoàn tất quá trình copy!")
else:
    print(f"❌ Không tìm thấy thư mục gốc: {SOURCE_PARENT_PATH}")
```

---

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
!uv run translate_srt.py \
    --input /content/output/video.srt \
    --output /content/output/video_ja.srt \
    --lang "Japanese" \
    --keys "{gemini_key}"
```

---

### 3.4. Text-to-Speech

Chuyển subtitle đã dịch thành file audio:

```colab
!uv run tts_srt.py \
    --input   /content/output/video_ja.srt \
    --output  /content/output/video_ja.wav \
    --voice   ja-JP-KeitaNeural \
    --rate    +5% \
    --autorate
```

---

### 3.5. Ghép Audio vào Video

#### Ghép audio mới vào video (thay thế audio gốc)

```colab
!ffmpeg -i /content/video.mp4 -i /content/output/video_ja.wav \
    -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 \
    /content/output/video_final.mp4
```

#### Giữ cả audio gốc (2 track)

```colab
!ffmpeg -i /content/video.mp4 -i /content/output/video_ja.wav \
    -c:v copy -c:a aac \
    -map 0:v:0 -map 1:a:0 -map 0:a:0 \
    /content/output/video_dual.mp4
```

---

## 4. Ví dụ Workflow hoàn chỉnh

### Ví dụ: Dịch video Trung → Nhật

```colab
# 1. Cài đặt
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os; os.environ['PATH'] += ':/root/.local/bin'
!apt-get install -y rubberband-cli

# 2. Clone project
from google.colab import userdata
token = userdata.get('github_token')
!git clone https://{token}@github.com/ThanhVoKim/CharenjiZukan.git /content/CharenjiZukan
%cd /content/CharenjiZukan

# 3. Upload file .srt gốc
from google.colab import files
uploaded = files.upload()  # Upload video.srt

# 4. Dịch sang tiếng Nhật (sử dụng Secrets)
gemini_key = userdata.get('gemini_token')
!uv run translate_srt.py \
    --input /content/video.srt \
    --output /content/video_ja.srt \
    --lang "Japanese" \
    --keys "{gemini_key}"

# 5. Tạo audio với giọng Nhật
!uv run tts_srt.py \
    --input /content/video_ja.srt \
    --output /content/video_ja.wav \
    --voice ja-JP-KeitaNeural \
    --autorate

# 6. Download kết quả
from google.colab import files
files.download('/content/video_ja.wav')
```

---

## 5. Cách dùng truyền thống (không có uv)

Nếu không muốn dùng uv, bạn có thể cài đặt thủ công:

```colab
!pip install google-genai tenacity edge-tts pydub pyrubberband soundfile aiohttp -q
!apt-get install -y rubberband-cli

# Sử dụng Secrets cho API key
from google.colab import userdata
gemini_key = userdata.get('gemini_token')

# Chạy trực tiếp với python
!python translate_srt.py --input video.srt --keys "{gemini_key}"
!python tts_srt.py --input video_vi.srt --voice vi-VN-HoaiMyNeural
```

---

## 6. Xử lý sự cố

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
!uv run tts_srt.py --input video.srt --voice ja-JP-KeitaNeural \
    --proxy http://127.0.0.1:7890
```

### Output không có extension

Script sẽ tự động thêm `.wav` nếu output không có extension.

---

## 7. Lưu ý quan trọng

1. **rubberband-cli**: Cần cài đặt bằng `apt-get` vì đây là binary hệ thống, không phải Python package. Dùng cho time-stretch audio chất lượng cao.

2. **API Keys**: Sử dụng Google Colab Secrets để bảo mật API keys. Không hardcode token vào code.

3. **Output không có extension**: Nếu `--output` không có extension, script sẽ tự động thêm `.wav`.

4. **Autorate**: Khi bật `--autorate`, audio sẽ được nén/giãn để khớp với thời lượng slot trong file SRT.
