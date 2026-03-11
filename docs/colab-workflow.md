# Hướng dẫn Quy trình chi tiết trên Google Colab

Tài liệu này hướng dẫn từng bước thực hiện quy trình lồng tiếng video trên Google Colab.

## Bước 0: Cài đặt môi trường

### 0.1. Cài đặt uv và clone project

```colab
# Cài đặt uv
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ['PATH'] += ':/root/.local/bin'

# Clone project
!git clone https://github.com/your-repo/videocolab.git /content/videocolab
%cd /content/videocolab

# Cài đặt rubberband-cli (cần cho time-stretch)
!apt-get install -y rubberband-cli
```

### 0.2. Cài đặt WhisperX và dependencies

```colab
# Tạo virtual environment và cài whisperx
!uv venv
!uv pip install -p .venv/bin/python whisperx

# Cài CUDA dependencies
!apt install libcudnn8 libcudnn8-dev -y
!pip install modelscope addict

# Set environment variables
%env MPLBACKEND=agg
%env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
%env LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/
```

---

## Bước 1: Tải video từ Google Drive

Đồng bộ video từ Google Drive sang local storage của Colab.

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

## Bước 2: Speech-to-Text với WhisperX

Chuyển audio thành file subtitle .srt.

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

### Tham số WhisperX

| Tham số          | Mô tả           | Giá trị khuyến nghị                              |
| ---------------- | --------------- | ------------------------------------------------ |
| `--model`        | Model Whisper   | `large-v2` (chính xác cao nhất)                  |
| `--language`     | Ngôn ngữ audio  | `zh` (Trung), `ja` (Nhật), `en` (Anh)            |
| `--align_model`  | Model alignment | Xem [HuggingFace](https://huggingface.co/models) |
| `--device`       | Thiết bị xử lý  | `cuda` (GPU)                                     |
| `--compute_type` | Precision       | `float16` (GPU), `int8` (CPU)                    |
| `--batch_size`   | Batch size      | `16` (GPU L4)                                    |

### Output

- File `.srt` tại thư mục output
- File `.json` với thông tin chi tiết

---

## Bước 3: Dịch SRT với Gemini API

Dịch file subtitle sang ngôn ngữ đích.

### 3.1. Dịch nhanh

```colab
!uv run translate_srt.py \
    --input /content/output/video.srt \
    --output /content/output/video_ja.srt \
    --lang "Japanese" \
    --keys "AIzaSyB-..."
```

### 3.2. Đầy đủ tham số

```colab
!uv run translate_srt.py \
    --input   /content/output/video.srt \
    --output  /content/output/video_ja.srt \
    --lang    "Japanese" \
    --keys    "AIzaSyB-key1,AIzaSyB-key2" \
    --model   "gemini-2.5-flash" \
    --batch   30 \
    --budget  8192 \
    --no-context
```

### Tham số translate_srt.py

| Tham số    | Mô tả             | Mặc định                 |
| ---------- | ----------------- | ------------------------ |
| `--input`  | File .srt đầu vào | (bắt buộc)               |
| `--output` | File .srt đầu ra  | `<input>_translated.srt` |
| `--lang`   | Ngôn ngữ đích     | `Vietnamese`             |
| `--keys`   | Gemini API key(s) | (bắt buộc)               |
| `--model`  | Model Gemini      | `gemini-2.5-flash`       |
| `--batch`  | Số dòng/batch     | `30`                     |
| `--budget` | Token budget      | `8192`                   |

---

## Bước 4: Text-to-Speech với EdgeTTS

Chuyển subtitle đã dịch thành file audio.

### 4.1. Xem danh sách giọng

```colab
# Giọng tiếng Nhật
!uv run tts_srt.py --list-voices ja

# Giọng tiếng Việt
!uv run tts_srt.py --list-voices vi
```

### 4.2. TTS với autorate

```colab
!uv run tts_srt.py \
    --input   /content/output/video_ja.srt \
    --output  /content/output/video_ja.wav \
    --voice   ja-JP-KeitaNeural \
    --rate    +5% \
    --autorate
```

### 4.3. Đầy đủ tham số

```colab
!uv run tts_srt.py \
    --input   /content/output/video_ja.srt \
    --output  /content/output/video_ja.mp3 \
    --voice   ja-JP-KeitaNeural \
    --rate    +10% \
    --volume  +0% \
    --pitch   +0Hz \
    --autorate \
    --max-speed 2.0 \
    --concurrent 10
```

### Tham số tts_srt.py

| Tham số       | Mô tả               | Mặc định             |
| ------------- | ------------------- | -------------------- |
| `--input`     | File .srt đầu vào   | (bắt buộc)           |
| `--output`    | File audio đầu ra   | `output/<input>.wav` |
| `--voice`     | Giọng EdgeTTS       | (bắt buộc)           |
| `--rate`      | Tốc độ giọng        | `+0%`                |
| `--volume`    | Âm lượng            | `+0%`                |
| `--pitch`     | Cao độ              | `+0Hz`               |
| `--autorate`  | Tự động nén audio   | (tắt)                |
| `--max-speed` | Giới hạn nén tối đa | `100.0`              |

---

## Bước 5: Ghép Audio vào Video

> **Lưu ý:** Bước này đang được hoàn thiện.

### 5.1. Ghép audio với ffmpeg

```colab
# Ghép audio mới vào video (thay thế audio gốc)
!ffmpeg -i /content/video.mp4 -i /content/output/video_ja.wav \
    -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 \
    /content/output/video_final.mp4
```

### 5.2. Giữ cả audio gốc (2 track)

```colab
# Ghép audio mới làm track 1, audio gốc làm track 2
!ffmpeg -i /content/video.mp4 -i /content/output/video_ja.wav \
    -c:v copy -c:a aac \
    -map 0:v:0 -map 1:a:0 -map 0:a:0 \
    /content/output/video_dual.mp4
```

---

## Xử lý sự cố

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
