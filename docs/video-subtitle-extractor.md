# Video Subtitle Extractor

Trích xuất subtitle tiếng Trung từ video sử dụng Qwen3-VL.

## Tính năng

- **Frame Sampling**: Lấy mỗi N frame để tối ưu hiệu suất
- **ROI Cropping**: Chỉ OCR vùng subtitle (thường ở dưới video)
- **Scene Detection**: Chỉ xử lý khi có chuyển cảnh
- **Chinese Filter**: Lọc chỉ giữ text tiếng Trung
- **Deduplication**: Loại bỏ text trùng lặp liên tiếp
- **Multiple Output Formats**: SRT hoặc TXT
- **UV-friendly CLI**: Có thể chạy trực tiếp bằng `uv run extract-subtitles`

## Cài đặt

### 1. Cách khuyến nghị: uv + virtual environment

#### Local (Windows/Linux/macOS)

```bash
# Cài uv (nếu chưa có)
pip install uv

# Tạo virtual environment trong project
uv venv .venv

# Cài dependencies từ pyproject.toml
uv sync

# Cài project để dùng CLI entrypoint (extract-subtitles)
uv pip install -e .
```

#### Google Colab (!uv)

```colab
# Cài uv
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ["PATH"] += ":/root/.local/bin"

# Clone project và vào thư mục
!git clone <repo-url> /content/CharenjiZukan
%cd /content/CharenjiZukan

# Tạo venv + cài dependencies
!uv venv .venv
!uv sync
!uv pip install -e .
```

### 2. Cài đặt trên Google Colab (khuyến nghị dùng `.venv-ocr`)

Trên Colab, OCR cần venv riêng để tránh xung đột torch/CUDA. Xem **[docs/colab-setup.md](colab-setup.md) mục A.3** để dựng `.venv-ocr` và tạo lock file lần đầu, sau đó restore bằng mục B.3 mỗi ngày.

Tóm tắt nhanh lần đầu thiết lập:

```bash
# Chụp CUDA gốc Colab (làm trước)
# → python snippet ở colab-setup.md mục A.0

# Dựng venv-ocr
!uv venv .venv-ocr
!uv pip install -p .venv-ocr/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  -c /content/cuda-base.txt \
  -e ".[ocr]"

# Freeze lock
!uv pip freeze -p .venv-ocr/bin/python | grep -v "file:///" > config/colab/ocr_lock.txt
```

Restore hằng ngày (đã có lock):

```bash
!uv venv .venv-ocr
!uv pip install -p .venv-ocr/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  -r config/colab/ocr_lock.txt
```

> **Xem chi tiết tại [docs/colab-setup.md](colab-setup.md)** — gồm lý do dùng venv riêng, cách xử lý lỗi CUDA, và quy trình freeze/restore đầy đủ.

### 3. Cài đặt thủ công (không dùng uv)

```bash
pip install opencv-python pyyaml numpy
```

## Sử dụng

### Cơ bản (khuyến nghị dùng uv)

```bash
# Trích xuất subtitle từ video bằng script entrypoint
uv run extract-subtitles video.mp4

# Output mặc định: video_chinese.srt
```

### Với các tùy chọn

```bash
# Chỉ định file output
uv run extract-subtitles video.mp4 -o subtitles.srt

# Frame sampling mỗi 60 frames (nhanh hơn)
uv run extract-subtitles video.mp4 --frame-interval 60

# Điều chỉnh vùng ROI (subtitle ở dưới hơn)
uv run extract-subtitles video.mp4 --roi-start 0.9

# Sử dụng CPU thay vì GPU
uv run extract-subtitles video.mp4 --device cpu

# Output format TXT
uv run extract-subtitles video.mp4 --format txt
```

### Batch mode

```bash
# Xử lý tất cả video trong thư mục
uv run extract-subtitles --input-dir ./videos --output-dir ./subtitles
```

### Sử dụng config file

```bash
uv run extract-subtitles video.mp4 --config config/extractor_config.yaml
```

### Chạy trực tiếp Python (fallback)

```bash
uv run python cli/video_ocr.py video.mp4
```

## Tham số

| Tham số                   | Mô tả                                                      | Mặc định                                                    |
| ------------------------- | ---------------------------------------------------------- | ----------------------------------------------------------- |
| `input_video`             | File video đầu vào (hoặc directory nếu dùng --input-dir)   | (bắt buộc)                                                  |
| `--boxes-file`            | File cấu hình các vùng OCR theo format `name x y w h`      | `assets/boxesOCR.txt` (nếu có config yaml thì config > cli) |
| `--output-dir`            | Thư mục output cho các file theo box                       | cùng thư mục video                                          |
| `--frame-interval`        | Số frame bỏ qua giữa mỗi lần xử lý                         | `30`                                                        |
| `--scene-threshold`       | Ngưỡng phát hiện chuyển cảnh cho từng box                  | `30.0`                                                      |
| `--min-scene-frames`      | Số frame tối thiểu giữa 2 lần chuyển cảnh để tránh nhiễu   | `10`                                                        |
| `--cv-prefilter`          | Bật tiền lọc OpenCV để bỏ qua ROI không có dấu hiệu chữ    | (tắt)                                                       |
| `--cv-min-edge-density`   | Ngưỡng mật độ cạnh tối thiểu cho CV prefilter              | `0.03`                                                      |
| `--cv-edge-low`           | Ngưỡng thấp Canny edge detector                            | `50`                                                        |
| `--cv-edge-high`          | Ngưỡng cao Canny edge detector                             | `150`                                                       |
| `--min-chars`             | Số ký tự tối thiểu để ghi nhận                             | `2`                                                         |
| `--no-scene-detection`    | Tắt bỏ tính năng Scene detection (tương đương threshold=0) | (tắt)                                                       |
| `--enable-chinese-filter` | Bật bộ lọc chỉ giữ lại tiếng Trung                         | (tắt)                                                       |
| `--strip-punctuation`     | Bỏ MỌI dấu câu (Unicode P*) khỏi text OCR — input sạch cho punctuate-srt; độc lập filter | (tắt)                            |
| `--ocr-model`             | Tên model trên Hugging Face                                | `Qwen/Qwen3-VL-8B-Instruct`                                 |
| `--device`                | Thiết bị xử lý (cuda/cpu)                                  | `cuda`                                                      |
| `--hf-token`              | Hugging Face Token                                         | (không dùng)                                                |
| `--batch-size`            | Batch size cho OCR batching                                | `8`                                                         |
| `--format`                | Định dạng output theo box (srt/txt)                        | `srt`                                                       |
| `--default-duration`      | Thời lượng mặc định mỗi subtitle                           | `3.0s`                                                      |
| `--min-duration`          | Thời lượng tối thiểu sau deduplicate                       | `1.0s`                                                      |
| `--max-duration`          | Thời lượng tối đa sau deduplicate                          | `7.0s`                                                      |
| `--no-deduplicate`        | Tắt gộp subtitle trùng lặp                                 | (tắt)                                                       |
| `--warn-english`          | Tạo file cảnh báo riêng nếu subtitle chứa tiếng Anh/số     | (tắt)                                                       |
| `--no-timestamp`          | Tắt timestamp (chỉ với format=txt)                         | (tắt)                                                       |
| `--config`                | Đường dẫn file cấu hình `.yaml`                            | (không dùng)                                                |

> **Mức ưu tiên Cấu hình**: CLI parameters có mức ưu tiên cao nhất, sau đó là tham số khai báo trong `--config`, và cuối cùng là Default values trong code.

## Python API

```python
from video_subtitle_extractor import VideoSubtitleExtractor
from video_subtitle_extractor.box_manager import OcrBox

# Định nghĩa vùng OCR (x, y, w, h tính bằng pixel)
boxes = [OcrBox(name="subtitle", x=0, y=800, w=1920, h=280)]

# Khởi tạo
extractor = VideoSubtitleExtractor(
    boxes=boxes,
    frame_interval=30,        # Mỗi 30 frame
    scene_threshold=1.5,      # Ngưỡng chuyển cảnh (phash)
    min_char_count=2,         # Tối thiểu 2 ký tự
    ocr_model="Qwen/Qwen3-VL-8B-Instruct",
    device="cuda"
)

# Trích xuất — tạo file {video_stem}_subtitle.srt
result = extractor.extract("video.mp4", output_dir="./output")

print(f"Processing time: {result.processing_time:.2f}s")
for box_name, count in result.subtitles_count.items():
    print(f"  Box '{box_name}': {count} subtitles → {result.output_paths[box_name]}")
```

### Batch processing

```python
from video_subtitle_extractor import VideoSubtitleExtractor
from video_subtitle_extractor.box_manager import OcrBox

boxes = [OcrBox(name="subtitle", x=0, y=800, w=1920, h=280)]
extractor = VideoSubtitleExtractor(boxes=boxes)

results = extractor.extract_from_directory(
    input_dir="./videos",
    output_dir="./subtitles"
)

for result in results:
    print(f"{result.video_path}: {result.subtitles_count}")
```

## Cấu trúc module

```
video_subtitle_extractor/
├── __init__.py                  # Package exports
├── extractor.py                 # VideoSubtitleExtractor (multi-box, phash scene detection)
├── native_video_extractor.py    # NativeVideoSubtitleExtractor (native pipeline Qwen3-VL)
├── frame_processor.py           # Frame sampling, ROI crop, scene detection
├── chinese_filter.py            # Lọc text tiếng Trung
├── subtitle_writer.py           # Xuất file SRT/TXT, deduplicate
├── box_manager.py               # OcrBox, BoxState, parse_boxes_file
├── text_isolator.py             # TextIsolationConfig, mask watermark trước OCR
├── video_source.py              # Giải mã video (hỗ trợ AV1/HEVC)
└── ocr/
    ├── __init__.py
    ├── base.py                  # BaseOCR interface
    ├── qwen3vl.py               # Qwen3VLOCR — backend duy nhất
    └── factory.py               # create_ocr_backend()
```

## Workflow

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
│   Video     │───►│  Frame Sampling  │───►│  ROI Crop   │
└─────────────┘    └──────────────────┘    └─────────────┘
                                                │
                                                ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
│   Output    │◄───│  Chinese Filter  │◄───│  Qwen3-VL   │
│   (SRT)     │    │  (tiếng Trung)   │    │    OCR      │
└─────────────┘    └──────────────────┘    └─────────────┘
```

```mermaid
flowchart LR
    A[Video input] --> B[Frame sampling]
    B --> C[Scene-change check]
    C --> D[Crop ROI subtitle]
    D --> E[CV Pre-filter]
    E --> F[OCR]
    F --> G[Filter Chinese text]
    G --> H[Build subtitle timeline]
    H --> I[Deduplicate]
    I --> J[Write SRT/TXT]
```

## 1) Luồng chạy từ CLI

- Entry nằm ở [`main()`](cli/video_ocr.py:190), parse tham số tại [`parse_args()`](cli/video_ocr.py:42).
- Tạo extractor tại [`VideoSubtitleExtractor.__init__()`](video_subtitle_extractor/extractor.py:70) với các tham số frame interval, ROI, scene threshold, device, format output.
- Chạy pipeline chính qua [`VideoSubtitleExtractor.extract()`](video_subtitle_extractor/extractor.py:231).

## 2) Bước xử lý frame (tối ưu trước OCR)

Trong [`FrameProcessor.extract_frames()`](video_subtitle_extractor/frame_processor.py:208), từng frame đi qua [`FrameProcessor.process_frame()`](video_subtitle_extractor/frame_processor.py:154):

- Sampling bằng [`FrameProcessor.should_process_frame()`](video_subtitle_extractor/frame_processor.py:72): chỉ lấy mỗi N frame.
- Scene detection bằng [`FrameProcessor.detect_scene_change()`](video_subtitle_extractor/frame_processor.py:112): so sánh grayscale giữa frame trước và hiện tại, dùng mean diff > threshold.
- Crop vùng subtitle bằng [`FrameProcessor.crop_roi()`](video_subtitle_extractor/frame_processor.py:84): mặc định lấy phần đáy (85%→100% chiều cao).
- Timestamp của subtitle được tính từ frame_number / fps.

## 3) OCR bằng Qwen3-VL

Trong [`VideoSubtitleExtractor.load_ocr_model()`](video_subtitle_extractor/extractor.py:148), load model Qwen3-VL rồi gọi OCR từng ROI bằng [`VideoSubtitleExtractor.ocr_image()`](video_subtitle_extractor/extractor.py:186).

Sau OCR:

- Text được lọc bằng [`ChineseFilter.filter_text()`](video_subtitle_extractor/chinese_filter.py:124).
- Nếu text hợp lệ, tạo subtitle entry với end_time là timestamp frame kế tiếp (hoặc default duration cho dòng cuối) trong [`VideoSubtitleExtractor.extract()`](video_subtitle_extractor/extractor.py:231).

## 4) Ghi file output

- Ghi SRT qua [`SubtitleWriter.write_srt()`](video_subtitle_extractor/subtitle_writer.py:208) hoặc TXT qua [`SubtitleWriter.write_txt()`](video_subtitle_extractor/subtitle_writer.py:259).
- Trước khi ghi có deduplicate liên tiếp bằng [`SubtitleWriter.deduplicate()`](video_subtitle_extractor/subtitle_writer.py:119), giúp gộp các dòng OCR trùng nhau theo thời gian.

## 5) OCR backend: Qwen3-VL

Pipeline dùng `Qwen3VLOCR` (trong [`video_subtitle_extractor/ocr/qwen3vl.py`](video_subtitle_extractor/ocr/qwen3vl.py)) làm backend duy nhất. Factory [`create_ocr_backend()`](video_subtitle_extractor/ocr/factory.py) khởi tạo thẳng `Qwen3VLOCR` với các tham số `model_name`, `device`, `max_new_tokens`, `min_pixels`, `max_pixels`.

## 6) Tham chiếu tài liệu

Tóm lại pipeline: tối ưu frame → OCR ROI subtitle bằng Qwen3-VL → lọc tiếng Trung → gộp trùng → xuất SRT/TXT.

## Tối ưu hiệu suất

### 1. Frame Sampling

| frame_interval | Mô tả              | Độ chính xác |
| -------------- | ------------------ | ------------ |
| 30             | Mỗi 1 giây (30fps) | Cao          |
| 60             | Mỗi 2 giây         | Trung bình   |
| 90             | Mỗi 3 giây         | Thấp         |

### 2. ROI Cropping

Vị trí ROI phụ thuộc vào loại video:

| Loại video    | roi_y_start | Ghi chú              |
| ------------- | ----------- | -------------------- |
| Phim điện ảnh | 0.85-0.90   | Subtitle ở dưới cùng |
| TV series     | 0.80-0.85   | Có thể cao hơn       |
| Variety show  | 0.75-0.85   | Thay đổi nhiều       |
| Short video   | 0.70-0.80   | Cần test             |

### 3. Scene Detection

| scene_threshold | Mô tả                 |
| --------------- | --------------------- |
| 10-20           | Nhạy, nhiều frame hơn |
| 30 (mặc định)   | Cân bằng              |
| 40-50           | Ít nhạy, ít frame hơn |

## Yêu cầu phần cứng

| Thành phần | Tối thiểu (Qwen3-VL-8B) | Khuyến nghị      |
| ---------- | ----------------------- | ---------------- |
| GPU        | NVIDIA 15GB VRAM        | NVIDIA 24GB+     |
| RAM        | 16GB                    | 32GB+            |
| Storage    | 10GB (model cache)      | SSD              |

> Qwen3-VL-8B yêu cầu ~15GB VRAM (xem `NATIVE_OCR_MIN_VRAM_GB` trong `tests/video_ocr/conftest.py`). Colab A100/L4 đáp ứng yêu cầu này.

## Troubleshooting

### Lỗi: "Failed to spawn: extract-subtitles"

```bash
# Cài lại project ở editable mode
uv pip install -e .

# Hoặc chạy trực tiếp file Python
uv run python cli/video_ocr.py video.mp4
```

### Lỗi: "CUDA out of memory"

```bash
# Giảm batch size
uv run extract-subtitles video.mp4 --batch-size 4

# Hoặc sử dụng CPU
uv run extract-subtitles video.mp4 --device cpu
```

### Lỗi: "No Chinese subtitles found"

1. Kiểm tra video có subtitle không
2. Điều chỉnh ROI: `--roi-start 0.80`
3. Giảm scene threshold: `--scene-threshold 20`
4. Tăng frame sampling: `--frame-interval 15`

### Lỗi: "Cannot open video"

```bash
# Cài đặt lại OpenCV trong môi trường uv
uv pip install opencv-python --upgrade
```
