# Project Journal

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
