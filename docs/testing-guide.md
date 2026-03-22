# Hướng dẫn Thiết kế và Viết Test — CharenjiZukan

## Mục lục

1. [Quy tắc cốt lõi & Cấu trúc dự án (TL;DR)](#1-quy-tắc-cốt-lõi--cấu-trúc-dự-án-tldr)
2. [Triết lý kiến trúc test — 4 Layers](#2-triết-lý-kiến-trúc-test--4-layers)
3. [Phân loại feature và cách tiếp cận](#3-phân-loại-feature-và-cách-tiếp-cận)
4. [Taxonomy Tag — Quy tắc gán nhãn](#4-taxonomy-tag--quy-tắc-gán-nhãn)
5. [Hardware Checking — Detect GPU và phân loại](#5-hardware-checking--detect-gpu-và-phân-loại)
6. [Tạo Sample Data cho Test](#6-tạo-sample-data-cho-test)
7. [Cấu trúc chuẩn của một file test](#7-cấu-trúc-chuẩn-của-một-file-test)
8. [Hướng dẫn chi tiết `test_matrix.yaml`](#8-hướng-dẫn-chi-tiết-test_matrixyaml)
9. [Hướng dẫn đọc kết quả từ `run_colab_tests.py`](#9-hướng-dẫn-đọc-kết-quả-từ-run_colab_testspy)
10. [Checklist trước khi submit test mới](#10-checklist-trước-khi-submit-test-mới)

---

## 1. Quy tắc cốt lõi & Cấu trúc dự án (TL;DR)

Phần này tóm tắt 10 quy tắc tuyệt đối không được vi phạm khi viết test. Tham chiếu link ở mỗi quy tắc để xem hướng dẫn và code mẫu chi tiết.

### 1.1. 10 Rules Bắt Buộc

- **R1 — 4-Layer Architecture:** Mọi feature phức tạp bắt buộc tổ chức thành 4 layer (Unit, Component, Integration, Real Model). L1-L3 phải pass mà không cần GPU. ([Xem Mục 2](#2-triết-lý-kiến-trúc-test--4-layers))
- **R2 — Naming Convention:** Bắt buộc đặt tên class theo prefix `TestLayer1_`, `TestLayer2_`... để hỗ trợ CLI filtering. ([Xem Mục 2](#đặt-tên-class-theo-layer))
- **R3 — No Sample Data in Git:** Không commit file media (.mp4, .wav). Mọi data test phải được tạo runtime qua pytest fixtures với `scope` phù hợp. ([Xem Mục 6](#6-tạo-sample-data-cho-test))
- **R4 — Lazy Imports:** Phải dùng `pytest.importorskip()` cho các thư viện nặng (cv2, torch, pydub) ở top-level. Không dùng `try/except` thông thường. ([Xem Mục 7](#7-cấu-trúc-chuẩn-của-một-file-test))
- **R5 — Hardware Validation:** Logic check GPU/VRAM phải dùng fixture từ `tests/conftest.py`, tuyệt đối không viết inline bằng `torch.cuda` trong test method. ([Xem Mục 5](#5-hardware-checking--detect-gpu-và-phân-loại))
- **R6 — Mandatory Mocking (L3):** Layer 3 bắt buộc phải mock hoàn toàn `_load_model()` và `_infer()`. Tuyệt đối không load model thật ở Layer 3. ([Xem Mục 3.5](#35-feature-yêu-cầu-ai-model-lớn--8gb-vram))
- **R7 — test_matrix.yaml Sync:** Tạo test mới phải cập nhật `test_matrix.yaml`. Mỗi layer 1 entry riêng biệt với timeout thực tế. ([Xem Mục 8](#8-hướng-dẫn-chi-tiết-test_matrixyaml))
- **R8 — Strict Tagging:** Chỉ sử dụng các tag đã quy định (`unit`, `integration`, `ffmpeg`, `gpu`, `gpu_small`, `native_ocr`, `demucs`). ([Xem Mục 4](#4-taxonomy-tag--quy-tắc-gán-nhãn))
- **R9 — No Hardcoded Secrets:** HuggingFace token và các API key bắt buộc đọc từ biến môi trường (Environment variables). ([Xem Mục 5.3](#53-biến-môi-trường-kiểm-soát-threshold))
- **R10 — CJK Font Rendering:** Khi test Layer 4 cần OCR chữ Trung/Nhật/Hàn, không dùng `cv2.putText()` (không hỗ trợ CJK), bắt buộc dùng Pillow. ([Xem Mục 3.3](#33-feature-xử-lý-video-opencv--ffmpeg))

### 1.2. Cấu trúc thư mục Testing

Toàn bộ tài nguyên testing được tổ chức với `testing-guide.md` làm nguồn sự thật duy nhất:

```text
CharenjiZukan/
├── run_colab_tests.py          ← Runner script (project root, KHÔNG di chuyển)
├── tests/
│   ├── conftest.py             ← Fixtures dùng chung (hardware check, skip logic)
│   ├── test_matrix.yaml        ← Cấu hình chạy test (tooling config)
│   ├── test_data/
│   │   └── .gitkeep            ← Thư mục rỗng, runtime tự sinh data vào đây
│   ├── test_<module_name>.py   ← Các file test áp dụng kiến trúc 4 Layers
│   └── ...
└── docs/
    └── testing-guide.md        ← Tài liệu chi tiết này
```

---

## 2. Triết lý kiến trúc test — 4 Layers

Mỗi feature phức tạp (đặc biệt các feature liên quan đến AI/GPU) **bắt buộc** được test theo 4 layer độc lập. Lý do: Layer trên có thể pass mà không cần layer dưới, giúp bạn debug chính xác vấn đề nằm ở đâu.

```
Layer 4 — Real Model          │ GPU + VRAM + model download
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│ Mục đích: xác nhận chất lượng AI thực tế
          ↑ phụ thuộc         │
Layer 3 — Pipeline Integration│ Synthetic video + Mocked model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│ Mục đích: toàn bộ extract() pipeline đúng
          ↑ phụ thuộc         │
Layer 2 — Component           │ Synthetic video, không model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│ Mục đích: frame sampling, crop ROI, batch split
          ↑ phụ thuộc         │
Layer 1 — Unit                │ Thuần Python, không I/O
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│ Mục đích: từng hàm nhỏ hoạt động đúng
```

**Khi Layer 3 pass nhưng Layer 4 fail**: Lỗi nằm ở chất lượng OCR hoặc prompt, không phải pipeline code.  
**Khi Layer 2 pass nhưng Layer 3 fail**: Lỗi nằm ở logic ghép pipeline, không phải video processing.  
**Khi Layer 1 pass nhưng Layer 2 fail**: Lỗi nằm ở integration với file thật, không phải logic thuần.

### Đặt tên class theo Layer

```python
class TestLayer1_TimestampParser:   ...  # Layer 1
class TestLayer2_FrameSampling:     ...  # Layer 2
class TestLayer3_FullPipeline:      ...  # Layer 3
class TestLayer4_RealModelOCR:      ...  # Layer 4
```

Tên prefix `TestLayer1_`, `TestLayer2_`, v.v. cho phép lọc theo `-k "Layer1"` trong pytest và `keyword` trong `test_matrix.yaml`.

---

## 3. Phân loại feature và cách tiếp cận

### 3.1 Feature không xử lý video (Pure Logic)

**Ví dụ**: `srt_parser.py`, `ass_utils.py`, `merge_srt.py`, `translate_srt.py`

Đây là các feature hoạt động hoàn toàn trên dữ liệu văn bản. Không cần GPU, không cần FFmpeg, không cần file media.

**Cách test**:

```
Layer 1 (Unit)        → Test từng hàm với input/output thuần Python
Layer 2 (Integration) → Test với file SRT/ASS thật (dùng tmp_path fixture)
Layer 3              → Không cần (không có pipeline phức tạp)
Layer 4              → Không cần (không có AI model)
```

**Sample data**: Tạo nội dung SRT/ASS inline trong test hoặc dùng `tmp_path`:

```python
@pytest.fixture()
def sample_srt_file(tmp_path: Path) -> Path:
    content = """1
00:00:01,000 --> 00:00:03,500
你好世界

2
00:00:05,000 --> 00:00:08,000
这是测试字幕
"""
    srt = tmp_path / "sample.srt"
    srt.write_text(content, encoding="utf-8")
    return srt
```

**Tags trong `test_matrix.yaml`**: `unit`

---

### 3.2 Feature xử lý audio (FFmpeg)

**Ví dụ**: `mute_srt.py`, `extract_srt.py`, `demucs_audio.py`, `media_speed.py` (audio mode)

Các feature này cần FFmpeg trong PATH và có thể cần `pyrubberband`.

**Cách test**:

```
Layer 1 (Unit)        → Test argument parser, logic tính toán timestamp, naming convention
Layer 2 (Component)   → Test với file WAV ngắn tổng hợp tạo bằng pydub/numpy
Layer 3 (Integration) → Test toàn bộ pipeline với file WAV thật
Layer 4              → Không cần (không có AI model)
```

**Tạo sample audio** không cần file thật:

```python
@pytest.fixture(scope="module")
def synthetic_wav_path(tmp_path_factory) -> Path:
    """Tạo file WAV 3 giây silence bằng pydub."""
    pydub = pytest.importorskip("pydub", reason="pydub chưa cài")
    from pydub import AudioSegment

    tmp_dir = tmp_path_factory.mktemp("audio_data")
    wav_path = tmp_dir / "sample_3s.wav"

    silence = AudioSegment.silent(duration=3000, frame_rate=44100)
    silence.set_channels(2).export(str(wav_path), format="wav")
    return wav_path
```

**Skip khi không có FFmpeg**:

```python
import shutil
pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg"),
    reason="FFmpeg không có trong PATH"
)
```

**Tags**: `unit` (Layer 1), `integration ffmpeg` (Layer 2-3)

---

### 3.3 Feature xử lý video (OpenCV + FFmpeg)

**Ví dụ**: `video_ocr.py` (frame sampling), `media_speed.py` (video mode), `native_video_extractor.py`

**Cách test**:

```
Layer 1 (Unit)        → Test logic thuần: crop coordinates, batch split, timestamp parse
Layer 2 (Component)   → Test với video tổng hợp tạo bằng cv2.VideoWriter
Layer 3 (Integration) → Test pipeline với mocked model + video tổng hợp
Layer 4 (Real Model)  → Test với model thật + GPU + video tổng hợp
```

**Tạo video tổng hợp**: Xem chi tiết tại [Mục 6.2](#62-tạo-synthetic-video-opencv).

**Lưu ý quan trọng về font CJK**: `cv2.putText()` **không hỗ trợ chữ CJK** (Trung/Nhật/Hàn). Nếu Layer 4 cần OCR nhận ra chữ thật, phải dùng Pillow:

```python
# ❌ Sai — cv2.putText không render được chữ Trung
cv2.putText(frame, "你好世界", ...)

# ✅ Đúng — dùng Pillow với font CJK
from PIL import Image, ImageDraw, ImageFont
pil_frame = Image.fromarray(frame)
draw = ImageDraw.Draw(pil_frame)
# font = ImageFont.truetype("path/to/NotoSansCJK.ttf", size=40)
draw.text((380, 1020), "你好世界", fill=(255, 255, 255))  # font=font nếu có
frame = np.array(pil_frame)
```

**Tags**: `unit` (L1), `integration` (L2-L3), `gpu` (L4)

---

### 3.4 Feature yêu cầu AI Model nhỏ (< 4GB VRAM)

**Ví dụ**: Các embedding model, classifier nhỏ

**Threshold VRAM**: `4` GB  
**Tags**: `unit` (L1), `integration` (L2-L3), `gpu_small` (L4)

```python
# conftest.py
@pytest.fixture(scope="module")
def skip_if_no_small_gpu():
    _require_gpu(min_vram_gb=float(os.getenv("MIN_VRAM_GB", "4")))
```

---

### 3.5 Feature yêu cầu AI Model lớn (>= 8GB VRAM)

**Ví dụ**: `video_subtitle_extractor` (DeepSeek-OCR-2), `native_video_extractor` (Qwen3-VL-8B)

Đây là category quan trọng nhất và phức tạp nhất trong dự án.

**Nguyên tắc bắt buộc**: Layer 1, 2, 3 **KHÔNG** được phụ thuộc GPU. Nếu test nào trong L1-L3 cần GPU để pass → đó là test được thiết kế sai, cần refactor.

**Cách mock model trong Layer 3**:

```python
# Mocked model: không load model, không dùng GPU
def fake_load_model(self):
    self._model_loaded = True  # Chỉ đánh dấu là đã load

infer_responses = iter(["[00:01.00 --> 00:03.00] 你好世界\n"])

def fake_infer(self, messages):
    return next(infer_responses, "")

monkeypatch.setattr(extractor, "_load_model", fake_load_model)
monkeypatch.setattr(extractor, "_infer", fake_infer)
```

**Threshold VRAM**:

- DeepSeek-OCR-2: `8` GB
- Qwen3-VL-8B: `15` GB
- Qwen3-VL-30B: `55` GB

**Tags**: `unit` (L1), `integration` (L2-L3), `gpu` + tên model (L4)  
**Env var**: `NATIVE_OCR_MIN_VRAM_GB`, `TEST_OCR_MODEL`

---

## 4. Taxonomy Tag — Quy tắc gán nhãn

Mỗi entry trong `test_matrix.yaml` phải có ít nhất 1 tag. Dưới đây là toàn bộ tag hợp lệ và ý nghĩa của chúng:

| Tag           | Ý nghĩa                                             | Phụ thuộc hardware     |
| ------------- | --------------------------------------------------- | ---------------------- |
| `unit`        | Test thuần logic, không I/O, không external process | Không có               |
| `integration` | Test với file thật hoặc subprocess bên ngoài        | Python dependencies    |
| `ffmpeg`      | Cần `ffmpeg` trong PATH                             | FFmpeg binary          |
| `gpu`         | Cần CUDA GPU với VRAM theo `NATIVE_OCR_MIN_VRAM_GB` | CUDA GPU ≥ 8GB         |
| `gpu_small`   | Cần CUDA GPU nhưng VRAM nhỏ hơn (< 4GB)             | CUDA GPU ≥ 4GB         |
| `native_ocr`  | Liên quan đến Native Video OCR pipeline (Qwen3-VL)  | Tùy layer              |
| `demucs`      | Liên quan đến Demucs audio separation               | GPU hoặc CPU ≥ 4 cores |

**Quy tắc gán nhiều tag**: Một entry có thể và nên có nhiều tag khi phù hợp:

```yaml
# ✅ Đúng: integration test cần FFmpeg
tags: ["integration", "ffmpeg"]

# ✅ Đúng: GPU test cho native OCR
tags: ["gpu", "native_ocr"]

# ❌ Sai: tag mâu thuẫn — unit test không cần GPU
tags: ["unit", "gpu"]
```

**Quy trình chạy theo tag trên Colab**:

```bash
# Colab không GPU → chỉ chạy unit và integration
python run_colab_tests.py --tags unit
python run_colab_tests.py --tags unit integration

# Colab có GPU T4 (16GB) → thêm gpu tests
python run_colab_tests.py --tags unit integration gpu

# Chạy tất cả có liên quan đến native OCR
python run_colab_tests.py --tags native_ocr
```

---

## 5. Hardware Checking — Detect GPU và phân loại

### 5.1 Các fixture trong `tests/conftest.py`

Không bao giờ viết hardware check trực tiếp trong file test. Tất cả logic detect phần cứng phải đặt trong `tests/conftest.py` dưới dạng fixture:

```python
# tests/conftest.py

def _require_gpu(min_vram_gb: float = 8.0) -> None:
    """
    Hàm helper nội bộ. Gọi pytest.skip() nếu GPU không đủ.
    Dùng bên trong fixture, không dùng trực tiếp trong test.
    """
    try:
        import torch
    except ImportError:
        pytest.skip("torch chưa cài đặt")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU không khả dụng")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram_gb < min_vram_gb:
        pytest.skip(f"VRAM {vram_gb:.1f}GB < {min_vram_gb}GB yêu cầu")


@pytest.fixture(scope="module")
def require_gpu_15gb():
    """Skip nếu GPU VRAM < 15GB (Qwen3-VL-8B)."""
    _require_gpu(float(os.getenv("NATIVE_OCR_MIN_VRAM_GB", "15")))
    yield


@pytest.fixture(scope="module")
def require_gpu_8gb():
    """Skip nếu GPU VRAM < 8GB (DeepSeek-OCR-2)."""
    _require_gpu(float(os.getenv("MIN_VRAM_GB", "8")))
    yield
```

### 5.2 Cách dùng fixture trong Layer 4

```python
# Cách 1: Dùng pytest.mark.skipif (evaluated at collection time)
_GPU_OK, _GPU_REASON = _check_gpu(min_vram_gb=15.0)

@pytest.mark.gpu
@pytest.mark.skipif(not _GPU_OK, reason=f"GPU: {_GPU_REASON}")
class TestLayer4_RealModelOCR:
    ...


# Cách 2: Dùng fixture (evaluated at run time, linh hoạt hơn)
@pytest.mark.gpu
class TestLayer4_RealModelOCR:
    def test_something(self, require_gpu_15gb):  # fixture được inject
        ...
```

**Khuyến nghị dùng Cách 1** (skipif decorator) cho Layer 4 class: Toàn bộ class bị skip rõ ràng ngay từ đầu, pytest output sạch hơn.

### 5.3 Biến môi trường kiểm soát threshold

Biến môi trường cho phép điều chỉnh threshold mà không cần sửa code:

| Biến                     | Mặc định                    | Ý nghĩa                             |
| ------------------------ | --------------------------- | ----------------------------------- |
| `NATIVE_OCR_MIN_VRAM_GB` | `15`                        | VRAM tối thiểu cho Qwen3-VL         |
| `MIN_VRAM_GB`            | `8`                         | VRAM tối thiểu chung cho GPU tests  |
| `TEST_OCR_MODEL`         | `Qwen/Qwen3-VL-8B-Instruct` | Model dùng trong Layer 4            |
| `HF_TOKEN`               | (empty)                     | HuggingFace token để download model |
| `CUDA_VISIBLE_DEVICES`   | (all)                       | Chỉ định GPU index                  |

Đặt trong `env` section của `test_matrix.yaml`:

```yaml
- name: "Native Video OCR — Layer 4"
  env:
    NATIVE_OCR_MIN_VRAM_GB: "15"
    CUDA_VISIBLE_DEVICES: "0"
    TEST_OCR_MODEL: "Qwen/Qwen3-VL-8B-Instruct"
```

---

## 6. Tạo Sample Data cho Test

**Nguyên tắc**: **Không commit file media** (video, audio) vào git. Tất cả sample data phải được tạo runtime trong fixtures.

### 6.1 Tạo Synthetic Audio (pydub)

```python
@pytest.fixture(scope="module")
def synthetic_wav_path(tmp_path_factory) -> Path:
    """
    WAV 3 giây, 44100Hz stereo.
    Nội dung: silence đơn giản — đủ để test FFmpeg processing.
    scope="module" → tạo 1 lần cho cả file test.
    """
    pydub = pytest.importorskip("pydub")
    from pydub import AudioSegment

    tmp_dir = tmp_path_factory.mktemp("audio")
    path = tmp_dir / "sample_3s.wav"
    silence = AudioSegment.silent(duration=3000, frame_rate=44100)
    silence.set_channels(2).export(str(path), format="wav")
    return path
```

### 6.2 Tạo Synthetic Video (OpenCV)

```python
@pytest.fixture(scope="module")
def synthetic_video_path(tmp_path_factory) -> Path:
    """
    Video 1920x1080, 30fps, 75 giây.
    Chứa chữ trắng tại ROI subtitle (y=984, h=80) theo từng khoảng thời gian.

    Lưu ý:
    - Dùng cv2.putText() → chỉ support ASCII/Latin. Đủ dùng cho test pipeline.
    - Layer 4 cần OCR thật → cần Pillow + font CJK thực sự.
    - scope="module" → render 1 lần, tái dùng cho tất cả test trong file.
    """
    cv2 = pytest.importorskip("cv2")

    SUBTITLES = [
        (1.0,  3.5,  "Subtitle Line One"),    # ASCII cho Layer 2-3
        (5.0,  8.0,  "Subtitle Line Two"),
        (62.0, 65.0, "Batch Two Subtitle"),   # Batch 2 (sau 60s)
    ]

    tmp_dir = tmp_path_factory.mktemp("video")
    path = tmp_dir / "synthetic_test.mp4"
    W, H, FPS = 1920, 1080, 30
    total_frames = int(75 * FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (W, H))

    for frame_no in range(total_frames):
        ts = frame_no / FPS
        frame = np.full((H, W, 3), 20, dtype=np.uint8)  # Dark background

        for start, end, text in SUBTITLES:
            if start <= ts < end:
                # ROI từ boxesOCR.txt: subtitle 370 984 1180 80
                cv2.putText(frame, text, (380, 1040),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                           (255, 255, 255), 2, cv2.LINE_AA)
                break
        writer.write(frame)

    writer.release()
    assert path.exists() and path.stat().st_size > 10_000
    return path
```

### 6.3 Tạo Synthetic Video với chữ CJK (Pillow) — cho Layer 4

```python
@pytest.fixture(scope="module")
def synthetic_video_cjk_path(tmp_path_factory) -> Path:
    """
    Video với chữ CJK thật để test OCR accuracy trong Layer 4.
    Yêu cầu font NotoSansCJK được cài trên hệ thống.
    Nếu không có font → fallback về ASCII.
    """
    cv2 = pytest.importorskip("cv2")
    from PIL import Image, ImageDraw, ImageFont

    SUBTITLES_CJK = [
        (1.0,  3.5,  "你好世界"),
        (5.0,  8.0,  "这是测试字幕"),
        (62.0, 65.0, "跨越批次字幕"),
    ]

    # Tìm font CJK
    font_candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/content/fonts/NotoSansCJK-Regular.ttc",  # Colab custom path
    ]
    font_path = next((p for p in font_candidates if Path(p).exists()), None)
    font = ImageFont.truetype(font_path, size=50) if font_path else None

    # ... (vẽ frame tương tự fixture trên nhưng dùng Pillow)
```

### 6.4 Tạo file SRT/ASS tạm

```python
@pytest.fixture()
def sample_srt_path(tmp_path: Path) -> Path:
    """
    Dùng scope mặc định (function) → tạo mới cho mỗi test.
    Phù hợp khi test có thể modify file.
    """
    content = "1\n00:00:01,000 --> 00:00:03,500\n你好世界\n\n"
    path = tmp_path / "test.srt"
    path.write_text(content, encoding="utf-8")
    return path
```

### 6.5 Tạo file prompt template tạm

```python
@pytest.fixture(scope="module")
def prompt_file_path(tmp_path_factory) -> Path:
    tmp_dir = tmp_path_factory.mktemp("prompts")
    path = tmp_dir / "test_prompt.txt"
    path.write_text(
        "Extract subtitles.\n{previous_context}\nNow extract:\n",
        encoding="utf-8"
    )
    return path
```

---

## 7. Cấu trúc chuẩn của một file test

Mọi file test trong dự án **bắt buộc** tuân theo cấu trúc này:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_<module_name>.py
===========================
Mô tả ngắn về feature đang được test.

Cấu trúc layers (điều chỉnh theo feature):
  Layer 1 — Unit Tests          (không cần GPU/FFmpeg/video)
  Layer 2 — Component Tests     (cần file thật hoặc synthetic)
  Layer 3 — Pipeline Integration (cần mocked model)
  Layer 4 — Real Model Tests    (cần GPU, đánh dấu @pytest.mark.gpu)

Cách chạy từng layer:
    pytest tests/test_<module>.py -v -k "Layer1"
    pytest tests/test_<module>.py -v -k "Layer2"
    pytest tests/test_<module>.py -v -k "Layer3"
    NATIVE_OCR_MIN_VRAM_GB=15 pytest tests/test_<module>.py -v -k "Layer4"
"""

import os
import sys
from pathlib import Path
from typing import List

import pytest

# ── Project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Lazy imports ─────────────────────────────────────────────────────
# Dùng pytest.importorskip cho dependency nặng (cv2, torch, pydub)
cv2 = pytest.importorskip("cv2", reason="pip install opencv-python")


# ═════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# Đặt tất cả fixture dùng chung ở đây, trước các class test.
# ═════════════════════════════════════════════════════════════════════

# [Fixtures ở đây]


# ═════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestLayer1_<ComponentName>:
    """Mô tả component đang test."""

    def test_<case_name>(self):
        ...


# ═════════════════════════════════════════════════════════════════════
# LAYER 2 — COMPONENT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestLayer2_<ComponentName>:
    ...


# ═════════════════════════════════════════════════════════════════════
# LAYER 3 — PIPELINE INTEGRATION
# ═════════════════════════════════════════════════════════════════════

class TestLayer3_<PipelineName>:
    ...


# ═════════════════════════════════════════════════════════════════════
# LAYER 4 — REAL MODEL TESTS (GPU Required)
# ═════════════════════════════════════════════════════════════════════

_GPU_OK, _GPU_REASON = _check_gpu_available(
    float(os.getenv("NATIVE_OCR_MIN_VRAM_GB", "15"))
)

@pytest.mark.gpu
@pytest.mark.skipif(not _GPU_OK, reason=f"GPU: {_GPU_REASON}")
class TestLayer4_<ModelName>:
    ...
```

### Quy tắc đặt tên

| Thành phần  | Format                              | Ví dụ                                      |
| ----------- | ----------------------------------- | ------------------------------------------ |
| File test   | `test_<module_name>.py`             | `test_native_video_ocr_pipeline.py`        |
| Class test  | `TestLayer{N}_{ComponentName}`      | `TestLayer2_FrameSampling`                 |
| Method test | `test_<what_is_being_tested>`       | `test_timestamps_increase_monotonically`   |
| Fixture     | `<noun>_path` hoặc `<noun>_fixture` | `synthetic_video_path`, `mocked_extractor` |

### Quy tắc về scope fixture

| Scope                 | Khi nào dùng                                    | Ví dụ                                        |
| --------------------- | ----------------------------------------------- | -------------------------------------------- |
| `function` (mặc định) | Fixture có thể bị modify bởi test               | `tmp_path`, file có thể bị ghi đè            |
| `class`               | Fixture dùng chung trong 1 class, có thể modify | Real model được load 1 lần cho class Layer 4 |
| `module`              | Fixture tốn kém, read-only                      | Video file tổng hợp, WAV file                |
| `session`             | Fixture rất tốn kém, toàn session               | Không dùng trong dự án này                   |

---

## 8. Hướng dẫn chi tiết `test_matrix.yaml`

File nằm tại: `tests/test_matrix.yaml`

### 8.1 Schema đầy đủ

```yaml
tests:
  - name: "Tên hiển thị — phải unique và mô tả đủ" # BẮT BUỘC
    file: "tests/test_file.py" # BẮT BUỘC, đường dẫn từ project root
    keyword: "Layer1 or Layer2" # Tùy chọn — map sang -k flag của pytest
    markers: ["unit", "slow"] # Tùy chọn — map sang -m flag của pytest
    env: # Tùy chọn — merge với env Colab hiện tại
      VARIABLE_NAME: "value"
    pytest_args: ["-v", "-s"] # Tùy chọn — cờ pytest bổ sung
    timeout_sec: 120 # Tùy chọn — timeout (giây), mặc định: 120
    enabled: true # Tùy chọn — bật/tắt entry, mặc định: true
    tags: ["unit"] # BẮT BUỘC — ít nhất 1 tag
```

### 8.2 Chi tiết từng tham số

#### `name` — BẮT BUỘC

Tên entry hiển thị trong console output và tên file report khi fail. Phải đủ mô tả để đọc report mà không cần mở file test.

```yaml
# ✅ Tốt: rõ ràng, đủ thông tin
name: "Native Video OCR — Layer 3: Full Pipeline (Mocked Model)"

# ❌ Kém: quá chung
name: "Test 1"
```

#### `file` — BẮT BUỘC

Đường dẫn tương đối từ project root. **Không** dùng đường dẫn tuyệt đối.

```yaml
file: "tests/test_native_video_ocr_pipeline.py"
```

#### `keyword`

Map sang `pytest -k "..."`. Dùng để chạy subset của file test.

```yaml
# Chỉ chạy Layer 1
keyword: "Layer1"

# Chỉ chạy các test thuộc Layer 2 hoặc Layer 3
keyword: "Layer2 or Layer3"

# Chỉ chạy test cụ thể theo tên method
keyword: "test_timestamps_increase_monotonically"

# Kết hợp: Layer1 nhưng không phải PromptBuilder tests
keyword: "Layer1 and not PromptBuilder"
```

**Khi nào bỏ trống `keyword`**: Khi muốn chạy toàn bộ file test.

#### `markers`

Map sang `pytest -m "..."`. Dùng khi test có decorator `@pytest.mark.xxx`.

```yaml
markers: ["gpu"]            # -m "gpu"
markers: ["unit", "fast"]   # -m "unit and fast"
```

**Lưu ý**: `markers` trong YAML là để filter test theo decorator, khác với `tags` là để filter entry trong YAML. Hai thứ này độc lập nhau.

#### `env`

Merge với environment Colab hiện tại. Các key ở đây sẽ **ghi đè** (không merge) giá trị Colab nếu trùng tên.

```yaml
env:
  NATIVE_OCR_MIN_VRAM_GB: "15" # Chỉ chạy khi VRAM >= 15GB
  CUDA_VISIBLE_DEVICES: "0" # Chỉ dùng GPU đầu tiên
  TEST_OCR_MODEL: "Qwen/Qwen3-VL-8B-Instruct"
  HF_TOKEN: "" # Không hardcode token — dùng secret Colab
```

**Quan trọng**: Không bao giờ hardcode token/key vào đây. Token phải được set trong Colab Secrets và available trong environment.

#### `pytest_args`

Cờ pytest bổ sung. `run_colab_tests.py` tự động thêm `--tb=short --no-header` nên không cần thêm lại.

```yaml
# Hiện log realtime (print statements trong test)
pytest_args: ["-v", "-s"]

# Dừng ngay khi có test đầu tiên fail
pytest_args: ["-v", "-x"]

# Chỉ chạy test đã fail lần trước
pytest_args: ["-v", "--lf"]
```

#### `timeout_sec`

Timeout tổng cho toàn bộ file test (sau khi lọc keyword/marker). Khi hết thời gian, process bị kill và kết quả là `TIMEOUT`.

```yaml
# Unit tests: nhanh, dùng timeout ngắn
timeout_sec: 30

# Integration tests với audio/video
timeout_sec: 120

# Tests có FFmpeg processing nặng
timeout_sec: 300

# Layer 4 với real model load + inference
timeout_sec: 600
```

**Hướng dẫn đặt timeout**: Thử chạy thủ công trước, ghi lại thời gian thực, nhân 2-3 lần.

#### `enabled`

```yaml
enabled: true   # Mặc định — test sẽ chạy
enabled: false  # Tạm tắt — test bị skip với log "SKIPPED (disabled)"
```

Dùng `enabled: false` khi: test đang develop chưa xong, test có bug chưa fix, môi trường đặc biệt cần (ví dụ cần GPU 40GB chưa có).

#### `tags`

Nhãn để lọc trong `run_colab_tests.py --tags`. Xem [Mục 4](#4-taxonomy-tag--quy-tắc-gán-nhãn).

### 8.3 Ví dụ hoàn chỉnh cho một feature mới

Khi thêm test cho feature `merge_video.py` mới, thêm vào `test_matrix.yaml`:

```yaml
# tests/test_matrix.yaml

# ──────────────────────────────────────────────────────────
# MERGE VIDEO — cli/merge_video.py
# ──────────────────────────────────────────────────────────

- name: "Merge Video — Layer 1: Argument Parser & Validation"
  file: "tests/test_merge_video.py"
  keyword: "Layer1"
  timeout_sec: 30
  pytest_args: ["-v"]
  tags: ["unit"]
  enabled: true

- name: "Merge Video — Layer 2: FFmpeg Command Building"
  file: "tests/test_merge_video.py"
  keyword: "Layer2"
  timeout_sec: 120
  pytest_args: ["-v", "-s"]
  tags: ["integration", "ffmpeg"]
  enabled: true

- name: "Merge Video — Layer 3: Full Merge Pipeline (Synthetic Files)"
  file: "tests/test_merge_video.py"
  keyword: "Layer3"
  timeout_sec: 300
  pytest_args: ["-v", "-s"]
  tags: ["integration", "ffmpeg"]
  enabled: true
```

---

## 9. Hướng dẫn đọc kết quả từ `run_colab_tests.py`

File nằm tại: `run_colab_tests.py` (project root)

### 9.1 Đọc output console

Mỗi test entry tạo ra block output sau:

```
[3/8] ⏳ Native Video OCR — Layer 3: Full Pipeline (Mocked Model)
  File   : tests/test_native_video_ocr_pipeline.py
  Filter : -k "Layer3"
  Tags   : integration, native_ocr
────────────────────────────────────────────────────────────
│ [stderr sẽ được prefix bằng │]
tests/test_native_video_ocr_pipeline.py::TestLayer3_FullPipeline::test_returns_extraction_result PASSED
tests/test_native_video_ocr_pipeline.py::TestLayer3_FullPipeline::test_model_loaded_exactly_once PASSED
...
────────────────────────────────────────────────────────────
✅ [PASSED] Native Video OCR — Layer 3: Full Pipeline (Mocked Model) (28.4s)
```

**Các trạng thái kết quả**:

| Icon | Status          | Ý nghĩa                 | Action                                        |
| ---- | --------------- | ----------------------- | --------------------------------------------- |
| ✅   | `PASSED`        | Tất cả test pass        | Không cần làm gì                              |
| ❌   | `FAILED`        | Có test fail            | Xem report `.md`                              |
| 💥   | `ERROR`         | Lỗi trước khi test chạy | Xem stderr trong report                       |
| ⏰   | `TIMEOUT`       | Process bị kill         | Tăng `timeout_sec` hoặc debug vòng lặp vô hạn |
| 🔍   | `NO_COLLECTION` | Không collect được test | Kiểm tra `keyword` filter                     |
| ⏭️   | `ALL_SKIPPED`   | Tất cả bị skip          | Thường do thiếu hardware — bình thường        |

### 9.2 Đọc và sử dụng file báo cáo fail

Khi có test fail, file `.md` được tạo tại `test_reports/`:

```
test_reports/
├── failed_native_video_ocr_layer_3_full_pipeline_mocked_model.md
└── failed_srt_parser_unit_tests.md
```

**Cấu trúc file báo cáo**:

```markdown
# ❌ Test Report: Native Video OCR — Layer 3

| Thông tin | Giá trị                                 |
| --------- | --------------------------------------- |
| Thời gian | 2026-03-22 14:30:00                     |
| File test | tests/test_native_video_ocr_pipeline.py |
| Kết quả   | FAILED (exit code: 1)                   |

## 1. Lệnh đã chạy ← Copy để chạy lại thủ công

## 2. Biến môi trường ← Context đầy đủ

## 3. Pytest Summary ← Pass/fail count nhanh

## 4. Full Output ← Toàn bộ pytest output

## 5. Stderr/Traceback ← Traceback chi tiết

## 6. Gợi ý cho AI Agent ← Câu hỏi gợi ý để phân tích
```

**Cách gửi cho AI Agent**: Attach file `.md` này vào chat và yêu cầu Agent phân tích phần "Stderr/Traceback" và đề xuất fix.

---

## 10. Checklist trước khi submit test mới

Trước khi thêm test file mới vào repository, xác nhận tất cả các mục sau:

**Cấu trúc file**:

- [ ] File đặt tại `tests/test_<module_name>.py`
- [ ] Có docstring mô tả các layers và cách chạy
- [ ] Import `PROJECT_ROOT` và thêm vào `sys.path`
- [ ] Dùng `pytest.importorskip()` cho dependency nặng (cv2, torch, pydub)
- [ ] Fixtures có `scope` phù hợp (`module` cho file tổng hợp, `function` cho file có thể bị modify)

**Layer 1 (Unit)**:

- [ ] Không có I/O với filesystem
- [ ] Không gọi subprocess hay external binary
- [ ] Không import torch, cv2, hay model weights
- [ ] Pass trong < 10 giây

**Layer 2 (Component)**:

- [ ] Sample data tạo runtime trong fixture, không commit vào git
- [ ] Fixture tổng hợp dùng `scope="module"` để tránh render lại nhiều lần
- [ ] Có assertion kiểm tra shape/size của output

**Layer 3 (Integration)**:

- [ ] Model và inference được mock hoàn toàn
- [ ] Test không require GPU
- [ ] Test offset timestamp của batch 2+
- [ ] Test output file được tạo đúng naming convention

**Layer 4 (Real Model)**:

- [ ] Có `@pytest.mark.gpu` decorator
- [ ] Có `@pytest.mark.skipif(not _GPU_OK, reason=...)`
- [ ] Dùng `scope="class"` cho fixture load model (tránh load lại)
- [ ] Đọc `HF_TOKEN` từ environment, không hardcode

**test_matrix.yaml**:

- [ ] Thêm entry cho từng layer riêng biệt
- [ ] `name` đủ mô tả, unique
- [ ] `keyword` map đúng tên class
- [ ] `tags` phản ánh đúng hardware requirements
- [ ] `timeout_sec` phù hợp (đã test thực tế)
- [ ] `enabled: true` cho L1-L3, cân nhắc `enabled: false` cho L4 nếu chưa có GPU

---

_Document này được maintain cùng với `tests/test_matrix.yaml` và `run_colab_tests.py`. Khi thêm tag mới hoặc thay đổi convention, cập nhật cả 3 file._
