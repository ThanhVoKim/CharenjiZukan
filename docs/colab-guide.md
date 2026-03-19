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
| `hf_token`     | `hf_...`                                | Hugging Face Access Token    |

**Cách tạo GitHub Personal Access Token:**

1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Tick `repo` (để truy cập private repo)
4. Copy token và thêm vào Secrets

**Cách tạo Gemini API Key:**

1. Truy cập [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy key và thêm vào Secrets

**Cách tạo Hugging Face Token:**

1. Truy cập [Hugging Face Settings - Tokens](https://huggingface.co/settings/tokens)
2. Click "Create new token"
3. Chọn quyền Read
4. Copy token và thêm vào Secrets

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

# Cài đặt project ở chế độ editable (để sử dụng CLI commands)
!uv pip install -e .

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

!apt install libcudnn8 libcudnn8-dev -y

# Set environment variables
%env MPLBACKEND=agg
%env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
%env LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/
```

### 1.4. Cài đặt môi trường cho DeepSeek-OCR-2 (tùy chọn - cho trích xuất phụ đề cứng)

DeepSeek-OCR-2 là mô hình AI trên Hugging Face, không phải là một package Python thông thường. Mã nguồn và weights của mô hình sẽ được tự động tải về thông qua thư viện `transformers` khi chạy script.

Các thư viện nền (như `transformers`, `torch`, `einops`, `PyMuPDF`) đã được cấu hình trong `pyproject.toml` và sẽ tự động cài đặt khi chạy `!uv pip install -e .`. Tuy nhiên, bạn cần cài thêm `flash-attn` để tăng tốc xử lý:

```colab
# Cài đặt Flash Attention (yêu cầu cho DeepSeek-OCR-2)
!uv pip install -p .venv/bin/python flash-attn==2.7.3 --no-build-isolation
```

---

## 2. Các script chính

### 2.1. Mute Audio (mute-srt)

Dùng khi audio có 2 ngôn ngữ (ví dụ: bình luận + video gốc trích dẫn). Thay thế các đoạn được đánh dấu bằng silence, giữ nguyên độ dài audio.

#### Mute audio nhanh

```colab
!uv run mute-srt --input /content/video.mp4 --mute /content/mute.srt
```

#### Đầy đủ tham số

```colab
!uv run mute-srt \
    --input       /content/video.mp4 \
    --mute        /content/mute.srt \
    --output      /content/audio_muted.wav \
    --sample-rate 16000 \
    --verbose
```

#### Tham số

| Tham số           | Mô tả                                | Mặc định               |
| ----------------- | ------------------------------------ | ---------------------- |
| `--input`, `-i`   | File audio/video đầu vào             | (bắt buộc)             |
| `--mute`, `-m`    | File mute.srt chứa các đoạn cần mute | (bắt buộc)             |
| `--output`, `-o`  | File audio đầu ra                    | `<input>_muted.wav`    |
| `--sample-rate`   | Sample rate output                   | `16000` (cho WhisperX) |
| `--verbose`, `-v` | Hiển thị log chi tiết                | (tắt)                  |

#### File mute.srt format

Tạo file `mute.srt` đánh dấu các đoạn cần mute:

```srt
1
00:01:24,233 --> 00:01:27,566
[MUTE] Đoạn video gốc được trích dẫn

2
00:05:30,000 --> 00:05:45,500
[MUTE] Đoạn ngôn ngữ thứ hai
```

> **Lưu ý:** Text trong file mute.srt không quan trọng, chỉ cần timestamp đúng format SRT.

---

### 2.2. Extract Audio (extract-srt)

Ngược với mute-srt: Giữ lại CHỈ các đoạn được đánh dấu trong mute.srt, các đoạn khác thành silence.

#### Extract audio nhanh

```colab
!uv run extract-srt --input /content/video.mp4 --mute /content/mute.srt
```

#### Đầy đủ tham số

```colab
!uv run extract-srt \
    --input       /content/video.mp4 \
    --mute        /content/mute.srt \
    --output      /content/audio_extracted.wav \
    --sample-rate 16000 \
    --verbose
```

#### Tham số

| Tham số           | Mô tả                                   | Mặc định                |
| ----------------- | --------------------------------------- | ----------------------- |
| `--input`, `-i`   | File audio/video đầu vào                | (bắt buộc)              |
| `--mute`, `-m`    | File mute.srt chứa các đoạn cần extract | (bắt buộc)              |
| `--output`, `-o`  | File audio đầu ra                       | `<input>_extracted.wav` |
| `--sample-rate`   | Sample rate output                      | `16000` (cho WhisperX)  |
| `--verbose`, `-v` | Hiển thị log chi tiết                   | (tắt)                   |

---

### 2.3. Merge SRT (merge-srt)

Merge 2 file SRT thành 1 file hoàn chỉnh, sắp xếp theo timestamp.

#### Merge nhanh

```colab
!uv run merge-srt \
    --commentary /content/video_subtitle.srt \
    --quoted     /content/subtitle_quoted.srt
```

#### Đầy đủ tham số

```colab
!uv run merge-srt \
    --commentary       /content/video_subtitle.srt \
    --quoted           /content/subtitle_quoted.srt \
    --output           /content/subtitle_merged.srt \
    --no-check-overlap \
    --verbose
```

#### Tham số

| Tham số              | Mô tả                                       | Mặc định              |
| -------------------- | ------------------------------------------- | --------------------- |
| `--commentary`, `-c` | File SRT chứa subtitle phần bình luận       | (bắt buộc)            |
| `--quoted`, `-q`     | File SRT chứa subtitle phần video trích dẫn | (bắt buộc)            |
| `--output`, `-o`     | File SRT output                             | `subtitle_merged.srt` |
| `--no-check-overlap` | Bỏ qua kiểm tra overlapping segments        | (mặc định check)      |
| `--verbose`, `-v`    | Hiển thị log chi tiết                       | (tắt)                 |

---

### 2.4. Dịch SRT (translate-srt)

#### Dịch nhanh (với Secrets)

```colab
from google.colab import userdata
gemini_key = userdata.get('gemini_key')

!uv run translate-srt \
    --input /content/video.srt \
    --keys  "{gemini_key}"
```

#### Đầy đủ tham số

```colab
from google.colab import userdata
gemini_key = userdata.get('gemini_key')

!uv run translate-srt \
    --input     /content/video.srt \
    --output    /content/video_ja.srt \
    --lang      "Japanese" \
    --keys      "{gemini_key}" \
    --model     "gemini-2.5-flash" \
    --prompt    /content/prompts/gemini.txt \
    --batch     30 \
    --budget    24576 \
    --wait      0.5 \
    --no-context \
    --verbose
```

#### Tham số

| Tham số           | Mô tả                                      | Mặc định             |
| ----------------- | ------------------------------------------ | -------------------- |
| `--input`, `-i`   | File .srt đầu vào                          | (bắt buộc)           |
| `--keys`, `-k`    | Gemini API key(s), phân cách bằng dấu phẩy | (bắt buộc)           |
| `--output`, `-o`  | File .srt đầu ra                           | `<input>_<lang>.srt` |
| `--lang`, `-l`    | Ngôn ngữ đích (tên tiếng Anh đầy đủ)       | `Vietnamese`         |
| `--model`, `-m`   | Model Gemini                               | `gemini-2.5-flash`   |
| `--prompt`        | Đường dẫn tới file prompt gemini.txt       | `prompts/gemini.txt` |
| `--batch`, `-b`   | Số dòng dịch mỗi lần                       | `30`                 |
| `--budget`        | Thinking budget tokens (0 để tắt)          | `24576`              |
| `--wait`          | Giây chờ giữa mỗi batch                    | `0`                  |
| `--no-context`    | Tắt global context                         | (mặc định bật)       |
| `--verbose`, `-v` | Hiển thị log chi tiết                      | (tắt)                |

---

### 2.5. SRT to ASS (srt-to-ass)

Chuyển file SRT thành file ASS để overlay lên video.

#### Chuyển đổi nhanh

```colab
!uv run srt-to-ass --input /content/note_translated.srt
```

#### Đầy đủ tham số

```colab
!uv run srt-to-ass \
    --input     /content/note_translated.srt \
    --output    /content/note_overlay.ass \
    --template  /content/CharenjiZukan/assets/sample.ass \
    --max-chars 14 \
    --style     NoteStyle \
    --verbose
```

#### Tham số

| Tham số            | Mô tả                                   | Mặc định            |
| ------------------ | --------------------------------------- | ------------------- |
| `--input`, `-i`    | File SRT đầu vào                        | (bắt buộc)          |
| `--output`, `-o`   | File ASS đầu ra                         | `<input>.ass`       |
| `--template`, `-t` | File ASS template                       | `assets/sample.ass` |
| `--max-chars`      | Số ký tự tối đa mỗi dòng (tự động ngắt) | `14`                |
| `--style`          | Tên style trong ASS                     | `NoteStyle`         |
| `--verbose`, `-v`  | Hiển thị log chi tiết                   | (tắt)               |

---

### 2.6. Text-to-Speech (tts-srt)

#### Xem danh sách giọng

```colab
# Giọng tiếng Việt
!uv run tts-srt --list-voices vi

# Giọng tiếng Nhật
!uv run tts-srt --list-voices ja
```

#### TTS nhanh

```colab
!uv run tts-srt \
    --input /content/video_ja.srt \
    --voice ja-JP-KeitaNeural
```

#### TTS với autorate (tự động nén giọng)

```colab
!uv run tts-srt \
    --input    /content/video_ja.srt \
    --output   /content/video_ja.wav \
    --voice    ja-JP-KeitaNeural \
    --rate     +5% \
    --autorate
```

#### Đầy đủ tham số

```colab
!uv run tts-srt \
    --input      /content/video_ja.srt \
    --output     /content/video_ja.wav \
    --voice      ja-JP-KeitaNeural \
    --rate       +10% \
    --volume     +0% \
    --pitch      +0Hz \
    --autorate \
    --max-speed  100.0 \
    --concurrent 10 \
    --cache      /content/cache_tts \
    --proxy      http://127.0.0.1:7890 \
    --verbose
```

#### Tham số

| Tham số          | Mô tả                           | Mặc định                  |
| ---------------- | ------------------------------- | ------------------------- |
| `--input`, `-i`  | File .srt đầu vào               | (bắt buộc)                |
| `--voice`, `-v`  | Tên giọng EdgeTTS               | (bắt buộc)                |
| `--output`, `-o` | File audio đầu ra (.wav/.mp3)   | `output/<input_stem>.wav` |
| `--rate`         | Tốc độ giọng (vd: +10%, -5%)    | `+0%`                     |
| `--volume`       | Âm lượng (vd: +20%)             | `+0%`                     |
| `--pitch`        | Cao độ (vd: +50Hz)              | `+0Hz`                    |
| `--autorate`     | Tự động nén audio khớp slot SRT | (tắt)                     |
| `--max-speed`    | Giới hạn tốc độ nén tối đa      | `100.0`                   |
| `--concurrent`   | Số request EdgeTTS song song    | `10`                      |
| `--cache`        | Thư mục cache audio tạm         | `tmp/<stem>_<ts>/`        |
| `--proxy`        | Proxy URL                       | (không dùng)              |
| `--list-voices`  | Liệt kê giọng EdgeTTS           | (không dùng)              |
| `--verbose`      | Bật logging debug               | (tắt)                     |

---

### 2.7. Demucs Audio Separation (demucs-audio)

Tách voice/background từ audio sử dụng AI model Demucs.

#### Tách background music (mặc định)

```colab
!uv run demucs-audio --input /content/audio_muted.wav
```

#### Tách vocals (giữ lại giọng nói)

```colab
!uv run demucs-audio --input /content/audio.wav --keep vocals
```

#### Đầy đủ tham số

```colab
!uv run demucs-audio \
    --input   /content/audio_muted.wav \
    --output  /content/audio_bgm.wav \
    --model   htdemucs \
    --keep    bgm \
    --bitrate 128k \
    --device  cuda \
    --verbose
```

#### Ví dụ nâng cao với `--keep`

```colab
# Chỉ lấy source "other" (index 2)
!uv run demucs-audio --input audio.wav --keep 2 --output other.mp3

# Lấy drums + bass (index 0,1)
!uv run demucs-audio --input audio.wav --keep 0,1 --output drums_bass.m4a

# Lấy drums + bass + other (index 0-2)
!uv run demucs-audio --input audio.wav --keep 0-2 --output bgm.mp3

# Lấy tất cả trừ vocals
!uv run demucs-audio --input audio.wav --keep 0-2 --output no_vocals.mp3
```

#### Tham số

| Tham số           | Mô tả                                               | Mặc định          |
| ----------------- | --------------------------------------------------- | ----------------- |
| `--input`, `-i`   | File audio đầu vào                                  | (bắt buộc)        |
| `--output`, `-o`  | File audio đầu ra (.wav, .mp3, .m4a, .aac)          | `<input>_bgm.wav` |
| `--model`, `-m`   | Model Demucs: htdemucs, htdemucs_ft, mdx, mdx_extra | `htdemucs`        |
| `--keep`, `-k`    | Sources giữ lại (xem bảng dưới)                     | `bgm`             |
| `--bitrate`, `-b` | Bitrate cho MP3/M4A output                          | `192k`            |
| `--device`, `-d`  | Device: cuda, cuda:0, cpu                           | auto-detect       |
| `--verbose`, `-v` | Hiển thị log chi tiết                               | (tắt)             |

#### Tham số `--keep` (chọn sources)

Demucs tách audio thành 4 sources với index:

| Index | Source | Mô tả     |
| ----- | ------ | --------- |
| 0     | drums  | Trống     |
| 1     | bass   | Bass      |
| 2     | other  | Nhạc khác |
| 3     | vocals | Giọng hát |

**Cách sử dụng `--keep`:**

```colab
# Presets
--keep bgm      # drums + bass + other (mặc định)
--keep vocals   # chỉ giọng hát
--keep drums    # chỉ trống
--keep bass     # chỉ bass
--keep other    # chỉ nhạc khác

# Index đơn lẻ
--keep 2        # chỉ other
--keep 3        # chỉ vocals

# Nhiều index (phẩy)
--keep 0,2      # drums + other
--keep 1,2      # bass + other

# Range index (gạch nối)
--keep 0-2      # drums + bass + other (tương đương bgm)
--keep 1-3      # bass + other + vocals
```

#### Output formats

```colab
# WAV (mặc định - chất lượng cao, file lớn)
--output bgm.wav

# MP3 (nén, file nhỏ hơn)
--output bgm.mp3 --bitrate 128k

# M4A/AAC (tốt cho video)
--output bgm.m4a --bitrate 192k
```

#### Lưu ý quan trọng

- Demucs yêu cầu audio **stereo (2 channels)** và **44.1kHz+** để có chất lượng tốt
- Nếu input là mono, script sẽ tự động convert sang stereo
- Nên dùng audio chất lượng cao thay vì audio 16kHz mono từ WhisperX
- WAV có chất lượng cao nhất nhưng file lớn; MP3/M4A nhỏ hơn nhưng mất chất lượng

---

### 2.8. Media Speed (media-speed)

Thay đổi tốc độ media (video, audio, SRT, ASS). Hỗ trợ cả slow down và speed up.

#### Slow down video 0.65x

```colab
!uv run media-speed --input /content/video.mp4 --speed 0.65
```

#### Slow down audio

```colab
!uv run media-speed --input /content/audio.wav --speed 0.65
```

#### Scale SRT timestamps

```colab
!uv run media-speed --input /content/subtitle.srt --speed 0.65
```

#### Scale ASS timestamps

```colab
!uv run media-speed --input /content/note_overlay.ass --speed 0.65
```

#### Đầy đủ tham số

```colab
!uv run media-speed \
    --input          /content/video.mp4 \
    --output         /content/video_slow.mp4 \
    --speed          0.65 \
    --type           auto \
    --no-keep-audio \
    --verbose
```

#### Tham số

| Tham số           | Mô tả                                   | Mặc định                               |
| ----------------- | --------------------------------------- | -------------------------------------- |
| `--input`, `-i`   | File input (video, audio, SRT, ASS)     | (bắt buộc)                             |
| `--output`, `-o`  | File output                             | `<input>_slow.*` hoặc `<input>_fast.*` |
| `--speed`, `-s`   | Hệ số tốc độ (< 1.0: slow, > 1.0: fast) | `0.65`                                 |
| `--type`, `-t`    | Loại file: auto, video, audio, srt, ass | `auto` (auto-detect)                   |
| `--no-keep-audio` | Không giữ audio trong video output      | (mặc định giữ audio)                   |
| `--verbose`, `-v` | Hiển thị log chi tiết                   | (tắt)                                  |

---

### 2.9. Trích xuất phụ đề cứng Multi-Box (extract-subtitles)

Trích xuất phụ đề (hardsub) trực tiếp từ khung hình video sử dụng mô hình DeepSeek-OCR-2, hỗ trợ nhiều vùng box độc lập.

#### Trích xuất nhanh (với Secrets)

```colab
from google.colab import userdata
hf_token = userdata.get('hf_token')

!uv run extract-subtitles /content/video.mp4 \
    --boxes-file /content/CharenjiZukan/assets/boxesOCR.txt \
    --hf-token "{hf_token}"
```

#### Đầy đủ tham số

```colab
from google.colab import userdata
hf_token = userdata.get('hf_token')

!uv run extract-subtitles /content/video.mp4 \
    --boxes-file /content/CharenjiZukan/assets/boxesOCR.txt \
    --output-dir /content \
    --frame-interval 30 \
    --scene-threshold 30.0 \
    --min-chars 2 \
    --device cuda \
    --hf-token "{hf_token}" \
    --format srt \
    --enable-chinese-filter
```

#### File `boxesOCR.txt`

```text
subtitle 370 930 1180 140
```

Mỗi dòng gồm `box_name x y w h`.

#### Tham số

| Tham số                   | Mô tả                                                 | Mặc định              |
| ------------------------- | ----------------------------------------------------- | --------------------- |
| `input_video`             | File video đầu vào                                    | (bắt buộc)            |
| `--boxes-file`            | File cấu hình các vùng OCR theo format `name x y w h` | `assets/boxesOCR.txt` |
| `--output-dir`            | Thư mục output cho các file theo box                  | cùng thư mục video    |
| `--frame-interval`        | Số frame bỏ qua giữa mỗi lần xử lý                    | `30`                  |
| `--scene-threshold`       | Ngưỡng phát hiện chuyển cảnh cho từng box             | `30.0`                |
| `--min-chars`             | Số ký tự tối thiểu để ghi nhận                        | `2`                   |
| `--device`                | Thiết bị xử lý (cuda/cpu)                             | `cuda`                |
| `--hf-token`              | Hugging Face Token                                    | (không dùng)          |
| `--format`                | Định dạng output theo box (srt/txt)                   | `srt`                 |
| `--enable-chinese-filter` | Bật bộ lọc chỉ giữ lại tiếng Trung                    | (tắt)                 |

#### Output theo từng box

Ví dụ input là `/content/video.mp4` với 2 box `subtitle` và `note`, output sẽ là:

- `/content/video_subtitle.srt`
- `/content/video_note.srt`

---

## 3. Workflow đầy đủ theo docs/workflow.md

### Bước 1: Audio Processing

#### 1a. Mute Audio

```colab
!uv run mute-srt \
    --input       /content/video.mp4 \
    --mute        /content/mute.srt \
    --output      /content/audio_muted.wav \
    --sample-rate 16000
```

#### 1b. Extract Audio

```colab
!uv run extract-srt \
    --input       /content/video.mp4 \
    --mute        /content/mute.srt \
    --output      /content/audio_extracted.wav \
    --sample-rate 16000
```

### Bước 2: Trích xuất phụ đề

Tùy thuộc vào loại video, bạn có thể chọn một trong hai cách:

#### Cách 2A: Speech-to-Text với WhisperX (cho video có giọng đọc rõ ràng)

Chuyển audio thành file subtitle .srt:

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

```colab
!uv run whisperx "/content/audio_muted.wav" \
  --model large-v2 \
  --language zh \
  --align_model jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn \
  --device cuda \
  --compute_type float16 \
  --batch_size 16 \
  --output_dir "/content/output"
```

#### Cách 2B: Trích xuất phụ đề cứng với DeepSeek-OCR-2 (cho video có hardsub)

Nếu video có sẵn phụ đề cứng (hardsub) tiếng Trung trên màn hình:

```colab
from google.colab import userdata
hf_token = userdata.get('hf_token')

!uv run extract-subtitles /content/video.mp4 \
    --boxes-file /content/CharenjiZukan/assets/boxesOCR.txt \
    --output-dir /content \
    --frame-interval 30 \
    --hf-token "{hf_token}" \
    --enable-chinese-filter

# Dùng box chính cho merge
# /content/video_subtitle.srt
```

### Bước 3: Merge Subtitle

```colab
!uv run merge-srt \
    --commentary /content/video_subtitle.srt \
    --quoted     /content/subtitle_quoted.srt \
    --output     /content/subtitle_merged.srt
```

### Bước 4: Note Processing

#### 4a. Translate Note

```colab
from google.colab import userdata
gemini_key = userdata.get('gemini_key')

!uv run translate-srt \
    --input  /content/note_source.srt \
    --output /content/note_translated.srt \
    --lang   "Japanese" \
    --keys   "{gemini_key}"
```

#### 4b. Convert SRT to ASS

```colab
!uv run srt-to-ass \
    --input     /content/note_translated.srt \
    --output    /content/note_overlay.ass \
    --template  /content/CharenjiZukan/assets/sample.ass \
    --max-chars 14 \
    --style     NoteStyle
```

### Bước 5: Translate Subtitle

```colab
from google.colab import userdata
gemini_key = userdata.get('gemini_key')

!uv run translate-srt \
    --input  /content/subtitle_merged.srt \
    --output /content/subtitle_translated.srt \
    --lang   "Japanese" \
    --keys   "{gemini_key}"
```

### Bước 6: Demucs Voice Removal

```colab
!uv run demucs-audio \
    --input  /content/audio_muted.wav \
    --output /content/audio_bgm.wav \
    --model  htdemucs \
    --stems  2 \
    --keep   bgm
```

### Bước 7: Slow Down 0.65x

```colab
# Video
!uv run media-speed --input /content/video.mp4 --speed 0.65

# Audio extracted
!uv run media-speed --input /content/audio_extracted.wav --speed 0.65

# Audio BGM
!uv run media-speed --input /content/audio_bgm.wav --speed 0.65

# Subtitle
!uv run media-speed --input /content/subtitle_translated.srt --speed 0.65

# Note overlay
!uv run media-speed --input /content/note_overlay.ass --speed 0.65
```

### Bước 8: TTS

```colab
!uv run tts-srt \
    --input    /content/subtitle_slow_translated.srt \
    --output   /content/audio_slow_tts.wav \
    --voice    ja-JP-KeitaNeural \
    --rate     +5% \
    --autorate
```

---

## 4. Cách dùng truyền thống (không có uv)

Nếu không muốn dùng uv, bạn có thể cài đặt thủ công:

```colab
!pip install google-genai tenacity edge-tts pydub pyrubberband soundfile aiohttp -q
!apt-get install -y rubberband-cli

# Sử dụng Secrets cho API key
from google.colab import userdata
gemini_key = userdata.get('gemini_key')

# Chạy trực tiếp với python (đường dẫn từ thư mục project)
!python cli/translate_srt.py --input video.srt --keys "{gemini_key}"
!python cli/tts_srt.py --input video_vi.srt --voice vi-VN-HoaiMyNeural
```

---

## 5. Xử lý sự cố

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
!uv run tts-srt --input video.srt --voice ja-JP-KeitaNeural \
    --proxy http://127.0.0.1:7890
```

### Output không có extension

Script sẽ tự động thêm `.wav` nếu output không có extension.

### Lỗi "Failed to spawn: mute-srt"

Chạy lệnh sau để cài đặt package:

```colab
%cd /content/CharenjiZukan
!uv pip install -e .
```

Hoặc chạy trực tiếp file Python:

```colab
!uv run python cli/mute_srt.py --input video.mp4 --mute mute.srt
```

---

## 7. Lưu ý quan trọng

1. **Cài đặt project**: Sau khi clone, cần chạy `!uv pip install -e .` để cài đặt project ở chế độ editable, cho phép sử dụng các CLI commands (`mute-srt`, `translate-srt`, `tts-srt`, `extract-subtitles`).

2. **rubberband-cli**: Cần cài đặt bằng `apt-get` vì đây là binary hệ thống, không phải Python package. Dùng cho time-stretch audio chất lượng cao.

3. **API Keys**: Sử dụng Google Colab Secrets để bảo mật API keys. Không hardcode token vào code.

4. **Multi-Box OCR**: File `boxesOCR.txt` phải đúng format `name x y w h`, mỗi box một dòng.

5. **Output Multi-Box**: `extract-subtitles` xuất nhiều file theo mẫu `<video_stem>_<box_name>.srt` hoặc `.txt`, không còn dùng `--output` file đơn.

6. **Autorate**: Khi bật `--autorate`, audio sẽ được nén/giãn để khớp với thời lượng slot trong file SRT.

7. **Gemini API Key**: Sử dụng cú pháp `--keys "{gemini_key}"` với biến từ `userdata.get('gemini_key')`.

8. Trình tự chạy lại để xác thực fix:

```colab
!uv cache clean
!uv sync --reinstall
!uv pip install -e .
```
