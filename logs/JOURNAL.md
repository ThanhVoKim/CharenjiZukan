# Project Journal

## 2026-03-21: Kế hoạch tích hợp Qwen3-VL OCR backend (Architectural Plan)

### Yêu cầu

- Xây dựng kế hoạch chi tiết để tích hợp thêm backend OCR dùng Qwen3-VL, làm lựa chọn thay thế cho DeepSeek-OCR-2.
- Model mục tiêu bắt buộc: `Qwen/Qwen3-VL-8B-Thinking`.
- Bổ sung phương án chọn giữa mode "nhanh" và "chính xác" qua chọn model (`Instruct` vs `Thinking`) thay vì tắt thinking bằng cờ nội bộ.

### Quyết định kiến trúc

1. **Tách riêng module OCR theo hướng OOP**:
   - Tổ chức mới dưới `video_subtitle_extractor/ocr/` gồm: `base.py`, `factory.py`, `deepseek.py`, `qwen3vl.py`, `__init__.py`.
   - Dùng `BaseOCR` (abstract class) để chuẩn hóa interface `recognize()`/`recognize_batch()`.

2. **Áp dụng Factory cho backend OCR**:
   - `create_ocr_backend()` trong `ocr/factory.py` tự chọn backend theo `ocr.model`.
   - `extractor.py` không phụ thuộc trực tiếp vào `DeepSeekOCR`, chỉ gọi qua factory.

3. **Qwen3-VL implementation notes bắt buộc**:
   - Dùng `AutoModelForImageTextToText`.
   - Thiết lập batch generation đúng cho tokenizer: `padding_side='left'`.
   - Khi xử lý vision: truyền tường minh `image_patch_size=16` vào `process_vision_info`.
   - Tắt resize tại processor (`do_resize=False`) để tránh resize 2 lần.
   - Với model Thinking, strip block `<think>...</think>` trước khi đưa text qua pipeline lọc.

4. **Chiến lược phụ thuộc môi trường**:
   - Không thêm `qwen-vl-utils` vào dependencies cố định của project.
   - Cài thủ công khi cần dùng Qwen, phiên bản khuyến nghị: `qwen-vl-utils==0.0.14`.
   - Ghi chú rõ trong docs về tương thích:
     - DeepSeek-OCR-2: `transformers==4.45.2`
     - Qwen3-VL: `transformers>=4.57.0`

### Trạng thái hiện tại

- ✅ Đã hoàn thành tài liệu kế hoạch chi tiết tại `plans/qwen3vl-integration.md`.
- ✅ Đã phản ánh toàn bộ feedback kỹ thuật vào plan (OOP structure, factory wiring, model-selection strategy, hardware test fixtures).
- ✅ Đã hoàn thành implementation: tạo module `ocr`, tách `BaseOCR`, thêm class `Qwen3VLOCR` sử dụng `transformers` và `qwen-vl-utils`, thiết lập wiring `create_ocr_backend()` trong `extractor.py`.
- ✅ Đã cập nhật tham số `--qwen-max-new-tokens`, `--qwen-min-pixels` và `--qwen-max-pixels` qua CLI và file YAML config để quản lý VRAM.
- ✅ Cập nhật tài liệu Colab với luồng cấu hình thư viện tùy ý giữa hai models (deepseek vs qwen).

### Tác động luồng hệ thống

- Thiết kế mới giữ nguyên data flow hiện có trong extractor (frame sampling → scene detection → selective OCR → filtering → writing output), chỉ thay lớp backend OCR phía dưới qua strategy/factory.
- Kế hoạch đã đối chiếu logic workflow tổng thể để tránh phá vỡ luồng hiện tại của module trích xuất subtitle.

### Bước tiếp theo đề xuất

1. Khởi chạy integration tests và end-to-end tests cho Qwen3-VL mode trên GPU để đảm bảo VRAM và padding alignment hoạt động ổn định.
2. Xóa các file rác (nếu có) do quá trình caching.

---

## 2026-03-20: Thêm Computer Vision Pre-filtering (OpenCV Canny)

### Yêu cầu

- Giảm ảo giác OCR khi frame/ROI không có chữ thật (model vẫn sinh text rác).
- Giảm số lần gọi OCR không cần thiết để tăng tốc và ổn định timeline subtitle.
- Cho phép cấu hình tính năng tiền lọc qua CLI/YAML theo cơ chế precedence hiện có.

### Thay đổi

1. **Cập nhật [`video_subtitle_extractor/frame_processor.py`](../video_subtitle_extractor/frame_processor.py)**:
   - Thêm hàm [`has_text_content()`](../video_subtitle_extractor/frame_processor.py) dùng Canny edge để ước lượng ROI có khả năng chứa text.
   - Tính `edge_density` và so với `min_edge_density`; có guard cho ngưỡng Canny không hợp lệ.
   - Fail-open: nếu prefilter gặp lỗi, vẫn cho OCR chạy để tránh mất subtitle thật.

2. **Cập nhật [`video_subtitle_extractor/extractor.py`](../video_subtitle_extractor/extractor.py)**:
   - Mở rộng constructor với 4 tham số: `cv_prefilter`, `cv_min_edge_density`, `cv_edge_low`, `cv_edge_high`.
   - Chèn bước CV prefilter vào vòng lặp extraction sau scene-change, trước khi push vào queue OCR.
   - Bổ sung metadata output để lưu cấu hình CV prefilter đã dùng.

3. **Cập nhật CLI trong [`cli/video_ocr.py`](../cli/video_ocr.py)**:
   - Thêm các cờ CLI: `--cv-prefilter`, `--cv-min-edge-density`, `--cv-edge-low`, `--cv-edge-high`.
   - Wiring vào hàm `get_param()` theo precedence: CLI > YAML > Default.

4. **Cập nhật cấu hình [`config/extractor_config.yaml`](../config/extractor_config.yaml)**:
   - Thêm section `cv_prefilter` với keys:
     - `enabled`
     - `min_edge_density`
     - `edge_low_threshold`
     - `edge_high_threshold`

5. **Cập nhật tài liệu**:
   - [`docs/colab-guide.md`](../docs/colab-guide.md): thêm ví dụ lệnh có `--cv-prefilter` + các tham số edge; bổ sung bảng tham số.
   - [`docs/video-subtitle-extractor.md`](../docs/video-subtitle-extractor.md): bổ sung tham số CV prefilter và cập nhật workflow diagram với bước CV Pre-filter.

### Trạng thái

- ✅ Hoàn thành tích hợp CV prefilter trong runtime.
- ✅ Hoàn thành đồng bộ CLI/YAML/docs.
- ⏳ Đề xuất bước tiếp theo: benchmark trên 1 video thật để tune `min_edge_density` theo từng style subtitle.

---

## 2026-03-20: Cải thiện Deduplication và thêm Warn English (Video Subtitle Extractor)

### Yêu cầu

- Giải quyết triệt để lỗi overlapping timestamps và lỗi nhận diện nhấp nháy/lặp từ do sai số dao động mạnh của DeepSeek-VL-2.
- Loại bỏ các câu mô tả ảo giác rác của AI như `（图片中没有可见的文字内容）`.
- Bổ sung cờ `--warn-english` để giúp người dùng dò tìm chữ cái tiếng Anh/số bị lọt vào video output, xuất thành file `.txt` độc lập.

### Thay đổi

1. **Cập nhật [`video_subtitle_extractor/subtitle_writer.py`](../video_subtitle_extractor/subtitle_writer.py)**:
   - Hạ `similarity_threshold` xuống `0.70` và sử dụng `difflib.SequenceMatcher` nhằm merge các kết quả OCR bị dao động lớn (thay đổi tới 30% nội dung).
   - Thêm logic khóa cứng (clamp) `end_time` không được phép vượt qua `start_time` của entry theo sau, vá triệt để bug lồng chéo timestamp.
   - Viết mới hàm `generate_english_warnings()` xuất kết quả những câu subtitle có ký tự `[a-zA-Z0-9]`.

2. **Cập nhật [`video_subtitle_extractor/deepseek_ocr.py`](../video_subtitle_extractor/deepseek_ocr.py)**:
   - Thêm danh sách đen (blacklist) `empty_phrases` với các mô tả quen thuộc của AI khi frame trống (vd: `图片中没有可见的文字`, `no visible text`).

3. **Cập nhật Config, CLI và Docs**:
   - Thêm flag `--warn-english` vào parser của [`cli/video_ocr.py`](../cli/video_ocr.py).
   - Thêm `warn_english: false` vào khóa `output` của file [`config/extractor_config.yaml`](../config/extractor_config.yaml).
   - Cập nhật [`docs/colab-guide.md`](../docs/colab-guide.md) và [`docs/video-subtitle-extractor.md`](../docs/video-subtitle-extractor.md) với tham số mới.

### Trạng thái

- ✅ Hoàn thành code fix bug và feature mới.
- ✅ Cập nhật toàn bộ tài liệu và cấu hình liên quan.

---

## 2026-03-19: Loại bỏ tham số keep_numbers khỏi Chinese Filter

### Yêu cầu

Loại bỏ hoàn toàn tham số `keep_numbers` do logic hiện tại không phản ánh đúng ý nghĩa bật/tắt và có thể gây lặp ký tự số Hán khi bật.

### Thay đổi

1. **Cập nhật code runtime**:
   - Xóa `keep_numbers` khỏi constructor của `ChineseFilter` trong `video_subtitle_extractor/chinese_filter.py`.
   - Xóa `CHINESE_NUMBER_PATTERN` và nhánh nối số Hán trong `extract_chinese()`.
   - Xóa `keep_numbers` khỏi constructor của `VideoSubtitleExtractor` và phần wiring tại `cli/video_ocr.py`.

2. **Cập nhật cấu hình và test**:
   - Xóa key `chinese_filter.keep_numbers` trong `config/extractor_config.yaml`.
   - Cập nhật fixture config trong `tests/test_extractor_config.py` để bỏ key `keep_numbers`.

3. **Cập nhật tài liệu**:
   - Xóa tham số CLI `--keep-numbers` khỏi bảng tham số ở `docs/video-subtitle-extractor.md` và `docs/colab-guide.md`.

### Trạng thái

- ✅ Hoàn thành loại bỏ `keep_numbers` khỏi code, config, test và docs người dùng.
- ✅ Giữ nguyên tương thích luồng xử lý còn lại (không thay đổi cơ chế bật/tắt Chinese filter).

### Bước tiếp theo đề xuất

- Chạy regression test tối thiểu cho module extractor config/CLI để xác nhận không còn tham chiếu `keep_numbers`.

---

## 2026-03-19: Đồng bộ Config Video Subtitle Extractor

### Yêu cầu

Người dùng nhận thấy các tham số cấu hình trong `config/extractor_config.yaml` và CLI (`docs/colab-guide.md`) chưa đồng bộ với constructor của `VideoSubtitleExtractor`. Cần xây dựng cơ chế merge cấu hình với mức ưu tiên rõ ràng: `code defaults < YAML < CLI`, đồng thời chuẩn hóa schema YAML.

### Thay đổi

1. **Chuẩn hóa schema trong [`config/extractor_config.yaml`](../config/extractor_config.yaml)**:
   - Loại bỏ các keys chưa có runtime (như `logging`, `performance`, `confidence_threshold`, v.v.) để file YAML phản ánh chính xác 100% behavior hiện tại.
   - Thêm phần cấu hình output: `default_duration`, `min_duration`, `max_duration`, `deduplicate`, `include_timestamp`.
   - Thêm tham số còn thiếu như `min_scene_frames`.

2. **Mở rộng CLI arguments trong [`cli/video_ocr.py`](../cli/video_ocr.py)**:
   - Đặt `default=argparse.SUPPRESS` cho toàn bộ tham số CLI để dễ phân biệt user có truyền hay không.
   - Thêm các cờ CLI mới: `--min-scene-frames`, `--keep-numbers`, `--ocr-model`, `--default-duration`, `--min-duration`, `--max-duration`, `--no-deduplicate`, `--no-timestamp`.

3. **Cơ chế Merge Cấu Hình (Precedence)**:
   - Thiết kế hàm `get_param(cli_name, yaml_path, default_val)` trong `cli/video_ocr.py` để lấy giá trị theo thứ tự ưu tiên: CLI > YAML > Default.
   - Áp dụng `get_param` cho toàn bộ tham số khi khởi tạo `VideoSubtitleExtractor`.
   - Box configuration giờ cũng tuân theo: `CLI --boxes-file > YAML roi.boxes_file > YAML roi.boxes (inline) > Default fallback`.

4. **Wiring Writer Parameters**:
   - Cập nhật vòng lặp khởi tạo writer trong `cli/video_ocr.py` để inject `min_duration`, `max_duration` từ cấu hình vào `extractor.writers`.
   - Cập nhật hàm `extract()` trong `extractor.py` để đọc và truyền `include_timestamp`, `deduplicate` khi ghi file TXT/SRT.

5. **Cập nhật tài liệu**:
   - [`docs/video-subtitle-extractor.md`](../docs/video-subtitle-extractor.md) và [`docs/colab-guide.md`](../docs/colab-guide.md): Bổ sung toàn bộ tham số mới vào bảng tham số và ghi chú rõ ràng về Precedence (CLI > Config YAML > Default).

6. **Test hồi quy**:
   - Tạo file [`tests/test_extractor_config.py`](../tests/test_extractor_config.py) chứa các mock tests để kiểm chứng cơ chế precedence giữa CLI, YAML và defaults.

### Trạng thái

- ✅ Hoàn thành code (đã wiring, merge cấu hình)
- ✅ Cập nhật tài liệu và tạo YAML chuẩn
- ✅ Viết test module (sẵn sàng sử dụng khi fix đủ môi trường)

---

## 2026-03-18: Cải tiến Video Subtitle Extractor hỗ trợ Multi-box OCR

### Yêu cầu

Nâng cấp `VideoSubtitleExtractor` để theo dõi và trích xuất subtitle từ nhiều vùng ảnh (ROI boxes) độc lập trong cùng một video, thay vì chỉ 1 vùng duy nhất bằng tỷ lệ y-axis như trước.
Yêu cầu hỗ trợ file cấu hình text cho các box và tối ưu hiệu năng OCR (chỉ gọi OCR khi vùng ảnh bị thay đổi).

### Thay đổi

1. **Tạo [`video_subtitle_extractor/box_manager.py`](../video_subtitle_extractor/box_manager.py)**
   - Quản lý cấu trúc dữ liệu `OcrBox` (tọa độ x, y, w, h) và `BoxState`.
   - Hàm `parse_boxes_file()` đọc cấu hình box từ file `boxesOCR.txt`.

2. **Cập nhật [`video_subtitle_extractor/frame_processor.py`](../video_subtitle_extractor/frame_processor.py)**
   - Hàm `crop_roi` sử dụng tọa độ tuyệt đối x, y, w, h từ `OcrBox`.
   - Hàm `detect_scene_change_for_box` phát hiện thay đổi trên từng vùng ảnh cắt ra thay vì toàn bộ frame. Sử dụng Hash-based MD5 cho fast-check trước khi dùng pixel diff.

3. **Cập nhật [`video_subtitle_extractor/extractor.py`](../video_subtitle_extractor/extractor.py)**
   - `VideoSubtitleExtractor` giờ nhận `boxes: List[OcrBox]` thay vì `roi_y_start/end`.
   - Vòng lặp `extract()` duyệt qua từng box trên mỗi frame, duy trì trạng thái độc lập (`state.prev_roi`, `state.entries`).
   - Implement Selective OCR: Nhóm các box bị thay đổi hình ảnh vào 1 batch rồi gọi OCR 1 lần, bỏ qua box không đổi (kéo dài `end_time`).
   - Khởi tạo nhiều `SubtitleWriter` để xuất ra các file SRT riêng rẽ cho từng box (`video_subtitle.srt`, `video_note.srt`).

4. **Cập nhật [`cli/video_ocr.py`](../cli/video_ocr.py)**
   - Thêm tham số `--boxes-file` (mặc định: `assets/boxesOCR.txt`).
   - Hỗ trợ in log xuất kết quả nhiều file riêng rẽ.

5. **Cập nhật [`config/extractor_config.yaml`](../config/extractor_config.yaml)**
   - Đổi cấu trúc section `roi` thành dùng `boxes_file` hoặc `boxes` array (x, y, w, h).

### Trạng thái

- ✅ Hoàn thành code Multi-box extraction logic
- ✅ Hoàn thành tích hợp CLI và config
- ⏳ Cần test hiệu suất thực tế trên Colab

---

## 2026-03-17: Tích hợp Video Subtitle Extractor vào Workflow (Bước 2)

### Yêu cầu

Thay thế WhisperX STT bằng `video_subtitle_extractor` (DeepSeek-OCR-2) trong Bước 2 của workflow để trích xuất subtitle trực tiếp từ video. Đồng thời, đảo logic của tham số lọc tiếng Trung để mặc định nhận diện tất cả các ngôn ngữ.

### Thay đổi

1. **Cập nhật `cli/video_ocr.py` và `extractor.py`**:
   - Xóa tham số `--disable-chinese-filter`.
   - Thêm tham số `--enable-chinese-filter`.
   - Đổi logic mặc định: `disable_chinese_filter=True` (nhận tất cả ngôn ngữ).

2. **Cập nhật `docs/workflow.md`**:
   - Thay đổi Bước 2 từ WhisperX STT sang OCR Subtitle Extraction.
   - Cập nhật input từ `audio_muted.wav` thành `video.mp4`.
   - Cập nhật diagram Mermaid và bảng trạng thái module.

3. **Cập nhật `docs/colab-guide.md`**:
   - Thay thế hướng dẫn cài đặt WhisperX bằng hướng dẫn cài đặt DeepSeek-OCR-2 và PyTorch CUDA.
   - Cập nhật lệnh chạy Bước 2 sử dụng `extract-subtitles`.

4. **Cập nhật `pyproject.toml`**:
   - Kích hoạt dependency `deepseek-ocr>=1.0.0`.

5. **Cập nhật `docs/video-subtitle-extractor.md`**:
   - Thêm tham số `--enable-chinese-filter` vào bảng tài liệu.

### Trạng thái

- ✅ Hoàn thành cập nhật code và tài liệu.

---

## 2026-03-13: Tạo module Media Speed - Thay đổi tốc độ media (Bước 7)

### Yêu cầu

Tạo module để thay đổi tốc độ media (video, audio, SRT, ASS) cho Bước 7 trong workflow. Module cần hỗ trợ cả slow down (`speed < 1.0`) và speed up (`speed > 1.0`).

### Thay đổi

1. **Tạo [`utils/media_utils.py`](../utils/media_utils.py)** - Module tái sử dụng:
   - `detect_media_type(path)` - Nhận dạng loại file
   - `scale_time_ms(ms, speed)` - Scale milliseconds
   - `check_rubberband_available()` - Kiểm tra rubberband binary
   - `stretch_audio_rubberband(input_path, output_path, speed)` - Time-stretch audio bằng rubberband
   - `stretch_audio_atempo(input_path, output_path, speed)` - Fallback FFmpeg atempo
   - `stretch_audio(input_path, output_path, speed)` - Auto-select method
   - `change_video_speed(input_path, output_path, speed, keep_audio=True)` - Thay đổi tốc độ video
   - `scale_srt_timestamps(input_path, output_path, speed)` - Scale SRT timestamps
   - `scale_ass_timestamps(input_path, output_path, speed)` - Scale ASS timestamps
   - `parse_ass_timestamp_to_ms(ts)`, `ms_to_ass_timestamp(ms)` - ASS timestamp helpers
   - `get_default_output_path(input_path, speed)` - Tạo tên output mặc định

2. **Tạo [`cli/media_speed.py`](../cli/media_speed.py)** - CLI wrapper:
   - Hỗ trợ 4 loại media: video, audio, srt, ass
   - Auto-detect loại file từ extension
   - Output naming tự động: `*_slow.*` hoặc `*_fast.*`
   - CLI arguments: `--input`, `--output`, `--speed`, `--type`, `--no-keep-audio`, `--verbose`

3. **Tạo [`tests/conftest.py`](../tests/conftest.py)** - Pytest fixtures dùng chung:
   - `skip_if_weak_hardware` - Skip tests nếu hardware yếu
   - `check_ffmpeg_available` - Kiểm tra FFmpeg
   - `check_rubberband_available` - Kiểm tra rubberband
   - `check_audio_stretch_dependencies` - Combined check

4. **Tạo [`tests/test_media_utils.py`](../tests/test_media_utils.py)** - Unit tests:
   - Test type detection
   - Test time scaling
   - Test ASS timestamp conversion
   - Test SRT/ASS scaling
   - Test output naming

5. **Tạo [`tests/test_media_speed.py`](../tests/test_media_speed.py)** - CLI tests:
   - Test parser
   - Test main function
   - Integration tests với hardware check

6. **Cập nhật [`utils/__init__.py`](../utils/__init__.py)** - Export media_utils functions

7. **Cập nhật [`pyproject.toml`](../pyproject.toml)** - Thêm script entry: `media-speed`

8. **Cập nhật [`docs/workflow.md`](../docs/workflow.md)** - Đánh dấu Bước 7 hoàn thành

### Tính năng chính

| Loại  | Xử lý            | Method                                           |
| ----- | ---------------- | ------------------------------------------------ |
| Video | Video + Audio    | FFmpeg setpts + rubberband/atempo                |
| Audio | Time-stretch     | rubberband (pitch-preserving) hoặc FFmpeg atempo |
| SRT   | Scale timestamps | Python (giữ nguyên text)                         |
| ASS   | Scale timestamps | Python (chỉ Dialogue lines)                      |

### Output naming mặc định

| Speed         | Output pattern |
| ------------- | -------------- |
| `speed < 1.0` | `*_slow.*`     |
| `speed > 1.0` | `*_fast.*`     |
| `speed = 1.0` | `*_copy.*`     |

### Ví dụ sử dụng

```bash
# Slow down video 0.65x
uv run cli/media_speed.py --input video.mp4 --speed 0.65

# Speed up audio 1.5x
uv run cli/media_speed.py -i audio.wav -s 1.5

# Scale SRT timestamps
uv run cli/media_speed.py -i subtitle.srt -s 0.65

# Video without audio
uv run cli/media_speed.py -i video.mp4 -s 0.65 --no-keep-audio
```

### Dependencies

- **FFmpeg** - Bắt buộc cho video/audio processing
- **rubberband-cli** - Tùy chọn, cho audio stretching chất lượng cao (giữ pitch)
- **pyrubberband** - Python wrapper cho rubberband
- **soundfile** - Đọc/ghi audio files

### Trạng thái

- ✅ Hoàn thành code
- ✅ Hoàn thành unit tests
- ✅ Cập nhật documentation

---

## 2026-03-13: Tạo module Demucs Audio - Tách voice/background từ audio

### Yêu cầu

Tạo CLI module [`cli/demucs_audio.py`](../cli/demucs_audio.py) để tách voice khỏi background music sử dụng Demucs AI model.

### Thay đổi

1. **Tạo [`cli/demucs_audio.py`](../cli/demucs_audio.py)** - CLI module:
   - `check_hardware_requirements()` - Kiểm tra GPU/CPU requirements
   - `check_demucs_installed()` - Kiểm tra demucs đã cài đặt
   - `get_device()` - Auto-detect device (cuda/cpu)
   - `separate_audio()` - Tách audio thành sources
   - CLI arguments: `--input`, `--output`, `--model`, `--stems`, `--keep`, `--device`, `--verbose`

2. **Cập nhật [`pyproject.toml`](../pyproject.toml)**:
   - Thêm dependencies: `demucs>=4.0.0`, `torch>=2.0.0`, `torchaudio>=2.0.0`
   - Thêm script entry point: `demucs-audio`

3. **Tạo [`tests/test_demucs_audio.py`](../tests/test_demucs_audio.py)** - Unit tests:
   - Test hardware check
   - Test CLI arguments
   - Skip test nếu hardware yếu (không GPU, CPU < 4 cores)

4. **Cập nhật [`docs/workflow.md`](../docs/workflow.md)**:
   - Đánh dấu Demucs là ✅ Hoàn thành
   - Thêm chi tiết options và ví dụ sử dụng

### CLI Arguments

| Argument  | Default  | Mô tả                                                          |
| --------- | -------- | -------------------------------------------------------------- |
| `--stems` | 2        | Số nguồn tách: 2 (vocals+bgm) hoặc 4 (drums/bass/other/vocals) |
| `--keep`  | bgm      | Giữ lại: `bgm` hoặc `vocals`                                   |
| `--model` | htdemucs | Model: htdemucs, htdemucs_ft, mdx, mdx_extra                   |

### Ví dụ sử dụng

```bash
# Mặc định: 2-stems, output bgm (remove vocals)
uv run cli/demucs_audio.py --input audio_muted.wav

# 2-stems, output vocals (remove bgm)
uv run cli/demucs_audio.py --input audio_muted.wav --keep vocals

# 4-stems, output bgm
uv run cli/demucs_audio.py --input audio_muted.wav --stems 4
```

### Hardware Requirements

- **GPU**: Khuyến nghị CUDA GPU
- **CPU**: Tối thiểu 4 cores nếu không có GPU
- **RAM**: Tối thiểu 8GB, khuyến nghị 16GB
- **Disk**: ~1GB cho model weights

### Trạng thái

- ✅ Hoàn thành code
- ✅ Hoàn thành unit tests
- ✅ Cập nhật documentation

---

## 2026-03-13: Tạo module Merge SRT - Merge 2 file SRT theo timestamp

### Yêu cầu

Module merge 2 file SRT (`subtitle_commentary.srt` + `subtitle_quoted.srt`) thành 1 file hoàn chỉnh (`subtitle_merged.srt`), sắp xếp theo timestamp.

### Thay đổi

1. **Tạo [`cli/merge_srt.py`](../cli/merge_srt.py)** - CLI module:
   - `check_overlap(segments)` - Kiểm tra overlapping segments
   - `merge_srt_segments(commentary_segments, quoted_segments, check_overlaps)` - Merge logic
   - `merge_srt_files(commentary_path, quoted_path, output_path, check_overlaps)` - File operations
   - CLI arguments: `--commentary`, `--quoted`, `--output`, `--no-check-overlap`, `--verbose`

2. **Cập nhật [`docs/workflow.md`](../docs/workflow.md)**:
   - Đánh dấu Merge SRT là ✅ Hoàn thành

### Logic Merge

```
1. Parse subtitle_commentary.srt → list segments
2. Parse subtitle_quoted.srt → list segments
3. Merge 2 lists thành 1 list
4. Sort merged list theo start_time
5. Check overlaps (nếu bật) → log error với icon ❌
6. Đánh số lại line (1, 2, 3, ...)
7. Export thành subtitle_merged.srt
```

### Ví dụ sử dụng

```bash
# Cơ bản - output mặc định là subtitle_merged.srt
uv run cli/merge_srt.py --commentary subtitle_commentary.srt --quoted subtitle_quoted.srt

# Với output tùy chỉnh
uv run cli/merge_srt.py -c commentary.srt -q quoted.srt -o merged.srt

# Bỏ qua check overlap
uv run cli/merge_srt.py -c commentary.srt -q quoted.srt --no-check-overlap
```

### Dependencies

- `utils/srt_parser.py` - Parse SRT file (`parse_srt_file`, `segments_to_srt`)
- `utils/logger.py` - Logging

### Unit Tests

Tạo [`tests/test_merge_srt.py`](../tests/test_merge_srt.py) với 15 test cases:

- `TestCheckOverlap`: 5 tests (no overlap, with overlap, multiple overlaps, empty, single)
- `TestMergeSrtSegments`: 7 tests (basic, empty commentary, empty quoted, both empty, same timestamp, gap, overlap detection)
- `TestMergeSrtFiles`: 3 tests (basic merge, empty commentary, file not found)

### Trạng thái

- ✅ Hoàn thành code
- ✅ Hoàn thành unit tests (15/15 passed)

---

## 2026-03-13: Tạo module SRT to ASS - Chuyển đổi subtitle sang ASS format

### Yêu cầu

Module chuyển đổi file SRT (subtitle) sang file ASS (Advanced SubStation Alpha) để overlay note lên video.

### Thay đổi

1. **Tạo [`utils/ass_utils.py`](../utils/ass_utils.py)** - ASS utilities module:
   - `srt_timestamp_to_ass(timestamp)` - Convert SRT timestamp → ASS timestamp
   - `ass_timestamp_to_srt(timestamp)` - Convert ASS timestamp → SRT timestamp
   - `normalize_newlines(text)` - Chuẩn hóa newlines thành `\N` (ASS line break)
   - `wrap_text(text, max_chars)` - Ngắt dòng text nếu quá max_chars
   - `create_dialogue_line(start, end, text, style)` - Tạo dòng Dialogue ASS
   - `parse_ass_file(file_path)` - Parse file ASS thành dict
   - `write_ass_file(output_path, ...)` - Ghi file ASS từ components
   - `convert_srt_segments_to_ass_dialogues(segments, ...)` - Convert SRT segments → ASS dialogues

2. **Tạo [`cli/srt_to_ass.py`](../cli/srt_to_ass.py)** - CLI module:
   - `load_template(template_path)` - Load ASS template file
   - `convert_srt_to_ass(srt_path, output_path, ...)` - Main conversion function
   - CLI arguments: `--input`, `--template`, `--output`, `--max-chars`, `--style`, `--verbose`

3. **Tạo [`tests/test_ass_utils.py`](../tests/test_ass_utils.py)** - Unit tests:
   - Test timestamp conversion (SRT ↔ ASS)
   - Test text wrapping
   - Test newline normalization
   - Test dialogue line creation
   - Test ASS file parsing/writing
   - Test full conversion

4. **Cập nhật [`utils/__init__.py`](../utils/__init__.py)** - Export ass_utils functions

### Format conversion

| SRT            | ASS          |
| -------------- | ------------ |
| `00:01:24,233` | `0:01:24.23` |
| `HH:MM:SS,mmm` | `H:MM:SS.cc` |
| milliseconds   | centiseconds |

### Ví dụ sử dụng

```bash
# Cơ bản
uv run cli/srt_to_ass.py --input note_translated.srt --output note_overlay.ass

# Với template tùy chỉnh
uv run cli/srt_to_ass.py -i input.srt -t custom.ass -o output.ass

# Với max chars tùy chỉnh
uv run cli/srt_to_ass.py -i input.srt --max-chars 20 -o output.ass
```

### Dependencies

- `utils/srt_parser.py` - Parse SRT file
- `utils/logger.py` - Logging
- `assets/sample.ass` - Default ASS template

### Trạng thái

- ✅ Hoàn thành code
- ⏳ Cần test với file SRT thực

---

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

3. **Tạo [`cli/video_ocr.py`](../cli/video_ocr.py)** - Entry point CLI

4. **Cập nhật [`pyproject.toml`](../pyproject.toml)**:
   - Thêm dependencies: opencv-python, pyyaml
   - Thêm script: `extract-subtitles`
   - Bump version: 0.1.0 → 0.2.0

5. **Tạo [`docs/video-subtitle-extractor.md`](../docs/video-subtitle-extractor.md)** - Documentation

### Kiến trúc

```

video_subtitle_extractor/
├── **init**.py # Package exports
├── extractor.py # Main VideoSubtitleExtractor class
├── frame_processor.py # Frame sampling, ROI, scene detection
├── chinese_filter.py # Lọc text tiếng Trung
└── subtitle_writer.py # Xuất file SRT/TXT

```

### Workflow

```

Video → Frame Sampling → Scene Detection → ROI Cropping → DeepSeek-OCR-2 → Chinese Filter → SRT Output

````

### Sử dụng

```bash
# Cơ bản
python cli/video_ocr.py video.mp4

# Với tùy chọn
python cli/video_ocr.py video.mp4 --frame-interval 60 --roi-start 0.9

# Batch mode
python cli/video_ocr.py --input-dir ./videos --output-dir ./subtitles
````

### Trạng thái

<<<<<<< HEAD

- ✅ Hoàn thành code
- # ⏳ Cần test với file audio/video thực
- ✅ Hoàn thành code structure
- ⏳ Cần tích hợp DeepSeek-OCR-2 khi library available
- ⏳ Cần test với video thực
  > > > > > > > DeepSeek-OCR-2

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

---

## 2026-03-13: Cập nhật Workflow 9 bước xử lý Video 2 ngôn ngữ

### Yêu cầu

Người dùng muốn xây dựng workflow hoàn chỉnh để xử lý video có 2 ngôn ngữ (bình luận + video gốc trích dẫn), bao gồm:

1. Input: video.mp4, mute.srt, onlyEng.srt
2. Xử lý audio: mute và extract các đoạn
3. Speech-to-Text với WhisperX
4. Merge subtitle từ 2 nguồn
5. Dịch subtitle
6. Demucs remove voice
7. Slow down 0.65x
8. TTS
9. Merge video
10. Speed up 1.2x

### Thay đổi

1. **Cập nhật [`docs/workflow.md`](../docs/workflow.md)** với workflow đầy đủ:
   - Naming convention cho các file output
   - Diagram Mermaid mô tả luồng xử lý
   - Chi tiết từng bước với input/output
   - Logic merge subtitle (Bước 3)
   - Bảng trạng thái các module

2. **Naming Convention mới:**

   | Tên file                  | Mô tả                                        |
   | ------------------------- | -------------------------------------------- |
   | `audio_muted.wav`         | Audio đã mute các đoạn trong mute.srt        |
   | `audio_extracted.wav`     | Audio chỉ chứa các đoạn được extract         |
   | `subtitle_commentary.srt` | Subtitle cho phần bình luận (từ WhisperX)    |
   | `subtitle_quoted.srt`     | Subtitle cho phần video trích dẫn (thủ công) |
   | `subtitle_merged.srt`     | Subtitle merge từ commentary + quoted        |
   | `subtitle_translated.srt` | Subtitle đã dịch                             |
   | `audio_bgm.wav`           | Audio background (Demucs remove voice)       |
   | `video_slow.mp4`          | Video slow 0.65x                             |
   | `video_slow_final.mp4`    | Video slow đã ghép audio                     |
   | `video_final.mp4`         | Video hoàn chỉnh cuối cùng                   |

3. **Thay đổi naming (2026-03-13):**
   - `subtitle_muted.srt` → `subtitle_commentary.srt` (rõ nghĩa hơn: subtitle cho phần bình luận)
   - `onlyEng.srt` → `subtitle_quoted.srt` (rõ nghĩa hơn: subtitle cho phần video trích dẫn)

4. **Module cần tạo:**

   | Module        | File                  | Trạng thái |
   | ------------- | --------------------- | ---------- |
   | Extract Audio | `cli/extract_srt.py`  | ❌ Cần tạo |
   | Merge SRT     | `cli/merge_srt.py`    | ❌ Cần tạo |
   | Demucs        | `cli/demucs_audio.py` | ❌ Cần tạo |
   | Slow Down     | `cli/slow_media.py`   | ❌ Cần tạo |
   | Merge Video   | `cli/merge_video.py`  | ❌ Cần tạo |
   | Speed Up      | `cli/speed_video.py`  | ❌ Cần tạo |

### Trạng thái

- ✅ Cập nhật docs/workflow.md
- ⏳ Cần implement các module còn thiếu

---

## 2026-03-13: Tạo module extract_srt.py và audio_utils.py

### Yêu cầu

Triển khai Bước 1b của workflow: Extract audio - giữ lại CHỈ các đoạn được đánh dấu trong mute.srt, các đoạn khác thành silence.

### Thay đổi

1. **Tạo [`utils/audio_utils.py`](../utils/audio_utils.py)** - Module xử lý audio dùng chung:
   - `load_audio()` - Load audio từ file audio/video
   - `export_audio()` - Export audio ra file WAV (mặc định 16kHz mono cho WhisperX)
   - `create_silence()` - Tạo silence segment

2. **Cập nhật [`utils/__init__.py`](../utils/__init__.py)** - Export các hàm từ audio_utils

3. **Tạo [`cli/extract_srt.py`](../cli/extract_srt.py)** - Module extract audio:
   - `apply_extract()` - Giữ lại CHỈ các đoạn trong mute.srt
   - CLI interface tương tự mute_srt.py
   - Output mặc định: `<input>_extracted.wav`

4. **Refactor [`cli/mute_srt.py`](../cli/mute_srt.py)** - Sử dụng `utils/audio_utils.py` thay vì code riêng

5. **Cập nhật [`docs/workflow.md`](../docs/workflow.md)**:
   - Đánh dấu Extract Audio là ✅ Hoàn thành
   - Thêm module Audio Utils vào bảng module

### So sánh mute_srt.py vs extract_srt.py

| Module    | mute_srt.py                    | extract_srt.py                             |
| --------- | ------------------------------ | ------------------------------------------ |
| Chức năng | Mute các đoạn TRONG mute.srt   | Giữ lại CHỈ các đoạn TRONG mute.srt        |
| Output    | Audio có silence tại đoạn mute | Audio chỉ có đoạn được extract             |
| Use case  | Phần bình luận (WhisperX STT)  | Phần video trích dẫn (ghép vào video cuối) |

### Ví dụ

**mute.srt:**

```srt
1
00:00:10,000 --> 00:00:20,000
[MUTE] Video trích dẫn
```

**Audio gốc (60s):**

```
[00:00-00:10] Bình luận → [00:10-00:20] Trích dẫn → [00:20-00:60] Bình luận
```

**audio_muted.wav (mute_srt.py):**

```
[00:00-00:10] Bình luận → [00:10-00:20] SILENCE → [00:20-00:60] Bình luận
```

**audio_extracted.wav (extract_srt.py):**

```
[00:00-00:10] SILENCE → [00:10-00:20] Trích dẫn → [00:20-00:60] SILENCE
```

### Trạng thái

- ✅ Hoàn thành extract_srt.py
- ✅ Hoàn thành audio_utils.py
- ✅ Refactor mute_srt.py
- ✅ Cập nhật docs/workflow.md

---

## 2026-03-13: Thêm Bước 4 Note Processing vào Workflow

### Yêu cầu

Người dùng muốn thêm bước xử lý file note vào workflow:

- Input thêm file `note_source.srt` - danh sách note hiển thị trên video
- Translate note → tạo `note_translated.srt`
- Convert SRT → ASS với template → tạo `note_overlay.ass`
- ASS cũng cần slow down 0.65x và merge vào video

### Thay đổi

1. **Cập nhật [`docs/workflow.md`](../docs/workflow.md)**:
   - Thêm Naming Convention cho note files:
     - `note_source.srt` - File note gốc (chưa dịch)
     - `note_translated.srt` - File note đã dịch
     - `note_overlay.ass` - File ASS để overlay note lên video
     - `note_overlay_slow.ass` - File ASS đã slow down 0.65x

   - Thêm Bước 4: Note Processing
     - 4a. Translate Note - dùng `translate_srt.py`
     - 4b. SRT to ASS - cần tạo `cli/srt_to_ass.py`

   - Cập nhật các bước sau:
     - Bước 5: Translate (trước là Bước 4)
     - Bước 6: Demucs (trước là Bước 5)
     - Bước 7: Slow Down - thêm xử lý `note_overlay.ass`
     - Bước 8: TTS (trước là Bước 7)
     - Bước 9: Merge Video - thêm `--note-ass` parameter
     - Bước 10: Speed Up (trước là Bước 9)

   - Thêm module cần tạo:
     - `cli/srt_to_ass.py` - Chuyển SRT → ASS với template
     - `utils/ass_utils.py` - Xử lý ASS format

2. **Logic SRT to ASS**:
   - Sử dụng `assets/sample.ass` làm template
   - Giữ nguyên style, thay thế Start, End, Text
   - Tự động ngắt dòng nếu quá 14 ký tự Nhật
   - Giữ nguyên các xuống dòng có sẵn với `\N`

3. **Cập nhật workflow diagram** với Bước 4 Note Processing

### Workflow mới (10 bước)

```
B1: Audio Processing (mute + extract)
B2: Speech-to-Text (WhisperX)
B3: Merge Subtitle
B4: Note Processing (translate + SRT→ASS)  [MỚI]
B5: Translate
B6: Demucs Voice Removal
B7: Slow Down 0.65x (thêm note_overlay.ass)
B8: TTS
B9: Merge Video (thêm note_overlay_slow.ass)
B10: Speed Up 1.2x
```

### Trạng thái

- ✅ Cập nhật docs/workflow.md
- ⏳ Cần tạo `cli/srt_to_ass.py`
- ⏳ Cần tạo `utils/ass_utils.py`

---

## 2026-03-16: Tích hợp Video Subtitle Extractor với uv + virtual environment

### Yêu cầu

Người dùng muốn chạy `video_subtitle_extractor` bằng `uv`/`!uv` với virtual environment và cập nhật tài liệu hướng dẫn.

### Thay đổi

1. **Cập nhật [`pyproject.toml`](../pyproject.toml)**:
   - Thêm script entrypoint: `extract-subtitles = "cli.video_ocr:main"`
   - Mở rộng package discovery để bao gồm `video_subtitle_extractor*`

2. **Cập nhật [`docs/video-subtitle-extractor.md`](../docs/video-subtitle-extractor.md)**:
   - Bổ sung hướng dẫn setup `uv venv` cho local
   - Bổ sung hướng dẫn `!uv` cho Google Colab
   - Chuẩn hóa ví dụ chạy bằng `uv run extract-subtitles`
   - Bổ sung fallback `uv run python cli/video_ocr.py ...`
   - Thêm troubleshooting cho lỗi spawn CLI khi chưa cài editable package

3. **Đảm bảo tương thích flow hiện tại**:
   - Giữ nguyên workflow xử lý frame → OCR → lọc Chinese → ghi subtitle
   - Không thay đổi logic core trong các module extractor hiện hữu

### Trạng thái

- ✅ Có thể chạy bằng `uv run extract-subtitles ...`
- ✅ Có thể dùng `!uv` trên Colab với virtual environment
- ⏳ DeepSeek-OCR-2 runtime thực tế vẫn phụ thuộc package upstream (hiện code vẫn có mock fallback)

## 2026-03-21 - Rename OCR entrypoint to cli/video_ocr.py

- Đổi tên và di chuyển entrypoint OCR sang `cli/video_ocr.py`.
- Cập nhật tham chiếu trong pyproject/docs/plans/tests/logs sang đường dẫn mới.
- Cập nhật bootstrap path trong cli/video_ocr.py để import module từ project root.

## 2026-03-21: Rename OCR CLI command to video-ocr

### Change

- Updated pyproject script from `extract-subtitles` to `video-ocr` targeting `cli.video_ocr:main`.
- Updated OCR command examples in `docs/colab-guide.md` to use `video-ocr`.

### Status

- Completed command name synchronization for OCR CLI.
