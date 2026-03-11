# Hướng dẫn sử dụng trên Google Colab

Tài liệu này hướng dẫn cách sử dụng project videocolab trên Google Colab với `uv` - công cụ quản lý package Python nhanh.

## 1. Cài đặt ban đầu

### Bước 1: Cài đặt uv

```colab
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ['PATH'] += ':/root/.local/bin'
```

### Bước 2: Clone project

```colab
!git clone https://github.com/your-repo/videocolab.git /content/videocolab
%cd /content/videocolab
```

### Bước 3: Cài đặt rubberband-cli (cần cho time-stretch)

```colab
!apt-get install -y rubberband-cli
```

## 2. Dịch SRT (translate_srt.py)

### Dịch nhanh

```colab
!uv run translate_srt.py \
    --input /content/video.srt \
    --output /content/video_vi.srt \
    --keys "AIza..."
```

### Đầy đủ tham số

```colab
!uv run translate_srt.py \
    --input   /content/video.srt         \
    --output  /content/video_vi.srt      \
    --lang    "Vietnamese"               \
    --keys    "AIza...key1,AIza...key2"  \
    --model   "gemini-2.5-flash"         \
    --batch   30                         \
    --budget  8192                       \
    --no-context
```

### Tham số

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

## 3. Text-to-Speech (tts_srt.py)

### Xem danh sách giọng

```colab
# Giọng tiếng Việt
!uv run tts_srt.py --list-voices vi

# Giọng tiếng Nhật
!uv run tts_srt.py --list-voices ja
```

### TTS nhanh

```colab
!uv run tts_srt.py \
    --input /content/video_vi.srt \
    --voice vi-VN-HoaiMyNeural
```

### TTS với autorate (tự động nén giọng)

```colab
!uv run tts_srt.py \
    --input   /content/video_vi.srt \
    --output  /content/video_vi.mp3 \
    --voice   vi-VN-HoaiMyNeural    \
    --rate    +5%                    \
    --autorate
```

### Đầy đủ tham số

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

### Tham số

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

## 4. Workflow hoàn chỉnh

### Ví dụ: Dịch video Trung → Nhật

```colab
# 1. Cài đặt
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os; os.environ['PATH'] += ':/root/.local/bin'
!apt-get install -y rubberband-cli

# 2. Clone project
!git clone https://github.com/your-repo/videocolab.git /content/videocolab
%cd /content/videocolab

# 3. Upload file .srt gốc
from google.colab import files
uploaded = files.upload()  # Upload video.srt

# 4. Dịch sang tiếng Nhật
!uv run translate_srt.py \
    --input /content/video.srt \
    --output /content/video_ja.srt \
    --lang "Japanese" \
    --keys "AIza..."

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

## 5. Cách dùng truyền thống (không có uv)

Nếu không muốn dùng uv, bạn có thể cài đặt thủ công:

```colab
!pip install google-genai tenacity edge-tts pydub pyrubberband soundfile aiohttp -q
!apt-get install -y rubberband-cli

# Chạy trực tiếp với python
!python translate_srt.py --input video.srt --keys "AIza..."
!python tts_srt.py --input video_vi.srt --voice vi-VN-HoaiMyNeural
```

## 6. Lưu ý quan trọng

1. **rubberband-cli**: Cần cài đặt bằng `apt-get` vì đây là binary hệ thống, không phải Python package. Dùng cho time-stretch audio chất lượng cao.

2. **API Keys**: Gemini API keys có thể lấy từ [Google AI Studio](https://aistudio.google.com/app/apikey). Có thể dùng nhiều key phân cách bằng dấu phẩy để rotate.

3. **Output không có extension**: Nếu `--output` không có extension, script sẽ tự động thêm `.wav`.

4. **Autorate**: Khi bật `--autorate`, audio sẽ được nén/giãn để khớp với thời lượng slot trong file SRT.
