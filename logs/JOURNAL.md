# Project Journal

## 2026-03-12: Tổ chức lại cấu trúc project

### Yêu cầu

Người dùng muốn tổ chức lại cấu trúc project để rõ ràng hơn, bao gồm:

1. Tạo thư mục `cli/` cho các CLI modules
2. Tạo thư mục `prompts/` cho prompt templates
3. Tạo thư mục `tests/` cho unit tests
4. Tạo file `README.md`

### Thay đổi cấu trúc

**Cấu trúc mới:**

```
CharenjiZukan/
├── cli/                    # CLI modules (MỚI)
│   ├── __init__.py
│   ├── mute_srt.py
│   ├── speed_rate.py
│   ├── translate_srt.py
│   └── tts_srt.py
├── prompts/                # Prompt templates (MỚI)
│   └── gemini.txt
├── tests/                  # Unit tests (MỚI)
│   ├── __init__.py
│   └── test_srt_parser.py
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   └── srt_parser.py
├── docs/
├── logs/
├── plans/
├── translator.py           # Core module (giữ nguyên)
├── tts_edgetts.py          # Engine module (giữ nguyên)
├── pyproject.toml          # Cập nhật
└── README.md               # MỚI
```

### Các file đã di chuyển

| File cũ            | File mới               |
| ------------------ | ---------------------- |
| `mute_srt.py`      | `cli/mute_srt.py`      |
| `translate_srt.py` | `cli/translate_srt.py` |
| `tts_srt.py`       | `cli/tts_srt.py`       |
| `speed_rate.py`    | `cli/speed_rate.py`    |
| `gemini.txt`       | `prompts/gemini.txt`   |

### Cập nhật imports

Các file CLI đã cập nhật `PROJECT_ROOT` để import đúng từ project root:

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
```

### Cập nhật pyproject.toml

- Thêm `pytest` vào dev-dependencies
- Cập nhật script paths: `cli.translate_srt:main`, `cli.tts_srt:main`, `cli.mute_srt:main`
- Thêm script `mute` và `test`

### Trạng thái

- ✅ Hoàn thành

---

## 2026-03-12: Tạo module mute_srt.py - Mute audio từ file mute.srt

### Yêu cầu

Người dùng có audio chứa 2 ngôn ngữ (bình luận + video gốc trích dẫn), khiến WhisperAI không chính xác. Cần tool để:

1. Đánh dấu thủ công các đoạn cần mute trong file `mute.srt`
2. Tự động thay thế bằng silence (giữ nguyên độ dài audio)
3. Output audio mới tối ưu cho WhisperX (WAV 16kHz mono)

### Thay đổi

1. **Tạo [`utils/srt_parser.py`](../utils/srt_parser.py)** - SRT parser module chung:
   - `parse_srt(content)` - Parse SRT từ string
   - `parse_srt_file(file_path)` - Parse SRT từ file
   - `ts_to_ms(ts)` - Chuyển timestamp sang milliseconds
   - `segments_to_srt(segments)` - Chuyển segments thành SRT format

2. **Cập nhật [`tts_srt.py`](../tts_srt.py)** - Refactor để import `parse_srt` từ `utils/srt_parser`

3. **Cập nhật [`utils/__init__.py`](../utils/__init__.py)** - Export các hàm từ srt_parser

4. **Tạo [`mute_srt.py`](../mute_srt.py)** - Module chính:
   - CLI: `uv run mute_srt.py --input video.mp4 --mute mute.srt`
   - Output mặc định: `<input>_muted.wav` (WAV 16kHz mono)
   - Thay thế các đoạn được đánh dấu bằng silence
   - Giữ nguyên độ dài audio

5. **Tạo [`plans/mute-audio-feature.md`](../plans/mute-audio-feature.md)** - Plan chi tiết

### Quy ước file mute.srt

```
<video_name>mute.srt
```

Format:

```srt
1
00:01:24,233 --> 00:01:27,566
[MUTE] Đoạn video gốc được trích dẫn
```

### Workflow tích hợp

```
Video gốc → Tạo mute.srt thủ công → mute_srt.py → Audio WAV muted → WhisperX → Subtitle chính xác
```

### Trạng thái

- ✅ Hoàn thành code
- ⏳ Cần test với file audio/video thực

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
