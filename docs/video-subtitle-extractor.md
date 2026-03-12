# Video Subtitle Extractor

Trích xuất subtitle tiếng Trung từ video sử dụng DeepSeek-OCR-2.

## Tính năng

- **Frame Sampling**: Lấy mỗi N frame để tối ưu hiệu suất
- **ROI Cropping**: Chỉ OCR vùng subtitle (thường ở dưới video)
- **Scene Detection**: Chỉ xử lý khi có chuyển cảnh
- **Chinese Filter**: Lọc chỉ giữ text tiếng Trung
- **Deduplication**: Loại bỏ text trùng lặp liên tiếp
- **Multiple Output Formats**: SRT hoặc TXT

## Cài đặt

### 1. Cài đặt dependencies cơ bản

```bash
pip install opencv-python pyyaml numpy
```

### 2. Cài đặt DeepSeek-OCR-2

```bash
# Khi DeepSeek-OCR-2 được release
pip install deepseek-ocr

# Hoặc từ GitHub
pip install git+https://github.com/deepseek-ai/DeepSeek-OCR-2.git
```

### 3. Cài đặt PyTorch (cho GPU)

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision
```

## Sử dụng

### Cơ bản

```bash
# Trích xuất subtitle từ video
python main_extract.py video.mp4

# Output mặc định: video_chinese.srt
```

### Với các tùy chọn

```bash
# Chỉ định file output
python main_extract.py video.mp4 -o subtitles.srt

# Frame sampling mỗi 60 frames (nhanh hơn)
python main_extract.py video.mp4 --frame-interval 60

# Điều chỉnh vùng ROI (subtitle ở dưới hơn)
python main_extract.py video.mp4 --roi-start 0.9

# Sử dụng CPU thay vì GPU
python main_extract.py video.mp4 --device cpu

# Output format TXT
python main_extract.py video.mp4 --format txt
```

### Batch mode

```bash
# Xử lý tất cả video trong thư mục
python main_extract.py --input-dir ./videos --output-dir ./subtitles
```

### Sử dụng config file

```bash
python main_extract.py video.mp4 --config config/extractor_config.yaml
```

## Tham số

| Tham số             | Mặc định | Mô tả                              |
| ------------------- | -------- | ---------------------------------- |
| `--frame-interval`  | 30       | Số frame bỏ qua giữa mỗi lần xử lý |
| `--roi-start`       | 0.85     | Vị trí bắt đầu ROI (0-1)           |
| `--roi-end`         | 1.0      | Vị trí kết thúc ROI (0-1)          |
| `--scene-threshold` | 30.0     | Ngưỡng phát hiện chuyển cảnh       |
| `--min-chars`       | 2        | Số ký tự Trung tối thiểu           |
| `--device`          | cuda     | Thiết bị OCR (cuda/cpu)            |
| `--format`          | srt      | Format output (srt/txt)            |

## Python API

```python
from video_subtitle_extractor import VideoSubtitleExtractor

# Khởi tạo
extractor = VideoSubtitleExtractor(
    frame_interval=30,        # Mỗi 30 frame
    roi_y_start=0.85,         # Vùng subtitle từ 85% chiều cao
    scene_threshold=30.0,     # Ngưỡng chuyển cảnh
    min_char_count=2,         # Tối thiểu 2 ký tự Trung
    device="cuda"             # Sử dụng GPU
)

# Trích xuất
result = extractor.extract("video.mp4", "output.srt")

print(f"Extracted {result.subtitles_count} subtitles")
print(f"Processing time: {result.processing_time:.2f}s")
```

### Batch processing

```python
from video_subtitle_extractor import VideoSubtitleExtractor

extractor = VideoSubtitleExtractor()

# Xử lý tất cả video trong thư mục
results = extractor.extract_from_directory(
    input_dir="./videos",
    output_dir="./subtitles"
)

for result in results:
    print(f"{result.video_path}: {result.subtitles_count} subtitles")
```

## Cấu trúc module

```
video_subtitle_extractor/
├── __init__.py           # Package exports
├── extractor.py          # Main VideoSubtitleExtractor class
├── frame_processor.py    # Frame sampling, ROI, scene detection
├── chinese_filter.py     # Lọc text tiếng Trung
└── subtitle_writer.py    # Xuất file SRT/TXT
```

## Workflow

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
│   Video     │───►│  Frame Sampling  │───►│  ROI Crop   │
└─────────────┘    └──────────────────┘    └─────────────┘
                                                │
                                                ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
│   Output    │◄───│  Chinese Filter  │◄───│  DeepSeek   │
│   (SRT)     │    │  (tiếng Trung)   │    │    OCR      │
└─────────────┘    └──────────────────┘    └─────────────┘
```

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

| Thành phần | Tối thiểu       | Khuyến nghị      |
| ---------- | --------------- | ---------------- |
| GPU        | NVIDIA 6GB VRAM | NVIDIA 8GB+ VRAM |
| RAM        | 8GB             | 16GB+            |
| Storage    | 5GB             | SSD              |

## Troubleshooting

### Lỗi: "CUDA out of memory"

```bash
# Giảm batch size
python main_extract.py video.mp4 --batch-size 4

# Hoặc sử dụng CPU
python main_extract.py video.mp4 --device cpu
```

### Lỗi: "No Chinese subtitles found"

1. Kiểm tra video có subtitle không
2. Điều chỉnh ROI: `--roi-start 0.80`
3. Giảm scene threshold: `--scene-threshold 20`
4. Tăng frame sampling: `--frame-interval 15`

### Lỗi: "Cannot open video"

```bash
# Cài đặt lại OpenCV
pip install --upgrade opencv-python
```

## License

MIT License
