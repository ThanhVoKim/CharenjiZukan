# Project Journal

## 2026-03-12: Tạo Video Subtitle Extractor với DeepSeek-OCR-2

### Yêu cầu

Người dùng muốn tạo module trích xuất subtitle tiếng Trung từ video với các tính năng:

1. Sử dụng DeepSeek-OCR-2 cho OCR
2. Frame Sampling (mỗi 30 frame)
3. ROI Cropping (chỉ vùng subtitle)
4. Scene Detection (phát hiện chuyển cảnh)
5. Lọc chỉ tiếng Trung, loại bỏ tiếng Anh

### Thay đổi

1. **Tạo [`video_subtitle_extractor/`](../video_subtitle_extractor/) - Package chính**:
   - [`__init__.py`](../video_subtitle_extractor/__init__.py) - Package exports
   - [`frame_processor.py`](../video_subtitle_extractor/frame_processor.py) - Xử lý frame video
   - [`chinese_filter.py`](../video_subtitle_extractor/chinese_filter.py) - Lọc text tiếng Trung
   - [`subtitle_writer.py`](../video_subtitle_extractor/subtitle_writer.py) - Xuất file SRT/TXT
   - [`extractor.py`](../video_subtitle_extractor/extractor.py) - Main VideoSubtitleExtractor class

2. **Tạo [`config/extractor_config.yaml`](../config/extractor_config.yaml)** - File cấu hình

3. **Tạo [`main_extract.py`](../main_extract.py)** - Entry point CLI

4. **Cập nhật [`pyproject.toml`](../pyproject.toml)**:
   - Thêm dependencies: opencv-python, pyyaml
   - Thêm script: `extract-subtitles`
   - Bump version: 0.1.0 → 0.2.0

5. **Tạo [`docs/video-subtitle-extractor.md`](../docs/video-subtitle-extractor.md)** - Documentation

### Kiến trúc

```
video_subtitle_extractor/
├── __init__.py           # Package exports
├── extractor.py          # Main VideoSubtitleExtractor class
├── frame_processor.py    # Frame sampling, ROI, scene detection
├── chinese_filter.py     # Lọc text tiếng Trung
└── subtitle_writer.py    # Xuất file SRT/TXT
```

### Workflow

```
Video → Frame Sampling → Scene Detection → ROI Cropping → DeepSeek-OCR-2 → Chinese Filter → SRT Output
```

### Sử dụng

```bash
# Cơ bản
python main_extract.py video.mp4

# Với tùy chọn
python main_extract.py video.mp4 --frame-interval 60 --roi-start 0.9

# Batch mode
python main_extract.py --input-dir ./videos --output-dir ./subtitles
```

### Trạng thái

- ✅ Hoàn thành code structure
- ⏳ Cần tích hợp DeepSeek-OCR-2 khi library available
- ⏳ Cần test với video thực

---

## 2026-03-12: Tạo module logging chung (utils/logger.py)

### Yêu cầu

Người dùng muốn có một module logging chung để:

1. Tất cả file code sử dụng cùng một cấu hình logging
2. Dễ dàng quản lý và thay đổi logging level
3. Hỗ trợ Google Colab với `setup_colab_logging()`

### Thay đổi

1. **Tạo [`utils/logger.py`](../utils/logger.py)** - Module logging chung:
   - `setup_logging(level, log_file, format_string)` - Cấu hình logging
   - `get_logger(name)` - Lấy logger với tên module
   - `setup_colab_logging(verbose)` - Cấu hình cho Google Colab

2. **Tạo [`utils/__init__.py`](../utils/__init__.py)** - Package init

3. **Cập nhật các file sử dụng logging**:
   - [`speed_rate.py`](../speed_rate.py) - Import từ `utils.logger`
   - [`translate_srt.py`](../translate_srt.py) - Thêm `setup_logging()` tại entry point
   - [`tts_srt.py`](../tts_srt.py) - Thêm `setup_logging()` tại entry point

4. **Tạo [`docs/logging-guide.md`](../docs/logging-guide.md)** - Hướng dẫn chi tiết:
   - Giới thiệu logging vs print
   - Khi nào dùng print, khi nào dùng logging
   - Ví dụ cho từng module
   - Best practices
   - Troubleshooting

5. **Tạo [`plans/logging-guide.md`](../plans/logging-guide.md)** - Plan thiết kế

### Khuyến nghị sử dụng

| Loại thông báo              | Nên dùng  | Lý do                          |
| --------------------------- | --------- | ------------------------------ |
| Khởi tạo module (load-time) | `print()` | Luôn hiện, không cần cấu hình  |
| Progress trong Colab        | `print()` | Đơn giản, dễ thấy              |
| Thông tin xử lý             | `logging` | Có thể tắt/bật theo level      |
| Cảnh báo (warning)          | `logging` | Format chuẩn, có thể filter    |
| Lỗi (error)                 | `logging` | Cần timestamp, có thể ghi file |
| Debug chi tiết              | `logging` | Tắt được khi production        |

### Trạng thái

- ✅ Hoàn thành
- ⏳ Cần test trên Colab

---

## 2026-03-11: Fix logging rubberband/pyrubberband detection ở module-level

### Vấn đề

Khi chạy `tts_srt.py`, không thấy log thông báo về rubberband/pyrubberband detection như mong đợi:

```
[SpeedRate] ⚠️ rubberband binary có, nhưng pyrubberband lib chưa cài → fallback FFmpeg atempo
[SpeedRate] 💡 Cài đặt: pip install pyrubberband
```

### Nguyên nhân

- Code logging ở module-level trong [`speed_rate.py`](../speed_rate.py:70-74) và [`speed_rate.py`](../speed_rate.py:196-206) chạy **TRƯỚC** khi `logging.basicConfig()` được gọi trong `tts_srt.py`
- Khi import `speed_rate` module, logger `"srt_translator"` chưa có handler → messages bị mất

### Giải pháp

Thay thế `_safe_log()` bằng `print()` ở module-level code:

- [`speed_rate.py:71-74`](../speed_rate.py:71) - Log rubberband binary detection
- [`speed_rate.py:200-206`](../speed_rate.py:200) - Log pyrubberband library detection

### Trạng thái

- ✅ Đã fix
- ⏳ Cần test lại trên Colab

---

## 2026-03-11: Fix lỗi output không có extension trong tts_srt.py

### Vấn đề

Khi chạy `tts_srt.py` với `--output /content/output` (thư mục không có extension), FFmpeg báo lỗi:

```
Command '['ffmpeg', '-y', '-i', '/content/transvideo/tmp/sts-3_1773205367/_target.wav', '-b:a', '192k', '/content/output']' returned non-zero exit status 1.
```

### Nguyên nhân

- `output_file = "/content/output"` không có extension
- `out_ext = Path(output_file).suffix.lower()` trả về `""` (rỗng)
- Code nhảy vào nhánh `else` và gọi FFmpeg với output là thư mục
- FFmpeg không thể ghi vào thư mục → lỗi

### Giải pháp

Sửa [`tts_srt.py:233-243`](../tts_srt.py:233) để tự động thêm `.wav` nếu output không có extension:

```python
if not out_ext:
    output_file = output_file + ".wav"
    out_ext = ".wav"
```

### Trạng thái

- ✅ Đã fix
- ⏳ Cần test lại trên Colab

---

## 2026-03-11: Thêm hỗ trợ uv cho Google Colab

### Yêu cầu

Người dùng muốn sử dụng `uv` trên Google Colab để:

1. Tự động quản lý dependencies (không cần `!pip install` thủ công)
2. Chạy script trực tiếp mà không cần khai báo biến `PROJ`

### Thay đổi

1. **Tạo [`pyproject.toml`](../pyproject.toml)** - Khai báo dependencies và cấu hình project:
   - google-genai, tenacity (cho translate_srt.py)
   - edge-tts, pydub, pyrubberband, soundfile, aiohttp, numpy (cho tts_srt.py)

2. **Tạo [`docs/colab-usage.md`](../docs/colab-usage.md)** - Hướng dẫn chi tiết sử dụng trên Colab:
   - Cài đặt uv và clone project
   - Hướng dẫn sử dụng translate_srt.py và tts_srt.py
   - Workflow hoàn chỉnh (ví dụ dịch video Trung → Nhật)
   - Cách dùng truyền thống (không có uv)

3. **Cập nhật docstring trong [`tts_srt.py`](../tts_srt.py)** - Rút gọn, tham chiếu đến docs/colab-usage.md

4. **Cập nhật docstring trong [`translate_srt.py`](../translate_srt.py)** - Rút gọn, tham chiếu đến docs/colab-usage.md

### Cách dùng mới trên Colab

```colab
# Cài đặt
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os; os.environ['PATH'] += ':/root/.local/bin'
!git clone https://github.com/your-repo/videocolab.git /content/videocolab
%cd /content/videocolab
!apt-get install -y rubberband-cli

# Chạy script (uv tự động cài dependencies)
!uv run translate_srt.py --input video.srt --keys "AIza..."
!uv run tts_srt.py --input video_vi.srt --voice vi-VN-HoaiMyNeural
```

### Trạng thái

- ✅ Hoàn thành
- ⏳ Cần test trên Colab
