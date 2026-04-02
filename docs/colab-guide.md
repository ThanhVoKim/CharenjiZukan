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

!pip install -q pyyaml pytest

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

### 1.5. Cài đặt Qwen3-VL OCR (tùy chọn)

Nếu muốn dùng Qwen3-VL thay cho DeepSeek-OCR-2 (có tốc độ chậm hơn nhưng đọc chính xác hơn, đặc biệt khi dùng bản Thinking), cần cài đặt thủ công do yêu cầu phiên bản `transformers` khác với DeepSeek:

> **⚠️ Lưu ý phiên bản transformers:**
>
> - DeepSeek-OCR-2: yêu cầu `transformers==4.45.2`
> - Qwen3-VL: yêu cầu `transformers>=4.57.0`
>   Hai model **không thể dùng chung** một phiên bản `transformers` cùng lúc.

```colab
# Nâng transformers cho Qwen3-VL
!uv pip install --upgrade "transformers>=4.57.0"

# Cài qwen-vl-utils (phiên bản khuyến nghị)
!uv pip install qwen-vl-utils==0.0.14
```

**Chạy với Qwen3-VL (Bước 2):**

```colab
# Đọc nhanh (Instruct)
!uv run video-ocr /content/video.mp4 --ocr-model Qwen/Qwen3-VL-8B-Instruct --device cuda

# Đọc chính xác với suy luận (Thinking)
!uv run video-ocr /content/video.mp4 --ocr-model Qwen/Qwen3-VL-8B-Thinking --device cuda
```

---

## 2. Các script chính

### 2.0 Speech-to-Text với WhisperX (cho video có giọng đọc rõ ràng)

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

#### Dịch nhanh (Gemini - Mặc định)

```colab
from google.colab import userdata
gemini_key = userdata.get('gemini_key')

!uv run translate-srt \
    --input /content/video.srt \
    --keys  "{gemini_key}"
```

#### Dịch với OpenAI-Compatible (DeepSeek)

```colab
!uv run translate-srt \
    --input /content/video.srt \
    --provider openai \
    --provider-config config/openai_compat_translate.yaml \
    --keys "sk-deepseek-xxx" \
    --lang "Japanese"
```

#### Dịch với Vertex AI (Application Default Credentials)

```colab
!uv run translate-srt \
    --input /content/video.srt \
    --provider vertexai \
    --provider-config config/vertexai_translate.yaml \
    --lang "Japanese"
```

#### Đầy đủ tham số

```colab
from google.colab import userdata
gemini_key = userdata.get('gemini_key')

!uv run translate-srt \
    --input     /content/video.srt \
    --output    /content/video_ja.srt \
    --provider  gemini \
    --lang      "Japanese" \
    --keys      "{gemini_key}" \
    --model     "gemini-3-flash-preview" \
    --prompt    /content/prompts/gemini.txt \
    --batch     30 \
    --budget    24576 \
    --wait      0.5 \
    --no-context \
    --verbose
```

#### Tham số

| Tham số             | Mô tả                                   | Mặc định                 |
| ------------------- | --------------------------------------- | ------------------------ |
| `--input`, `-i`     | File .srt đầu vào                       | (bắt buộc)               |
| `--provider`, `-p`  | Provider (gemini/openai/vertexai)       | `gemini`                 |
| `--provider-config` | Đường dẫn file config YAML              | (tuỳ provider)           |
| `--base-url`        | Override base URL cho OpenAI provider   | `None`                   |
| `--keys`, `-k`      | API key(s), phân cách bằng dấu phẩy     | (bắt buộc gemini/openai) |
| `--output`, `-o`    | File .srt đầu ra                        | `<input>_<lang>.srt`     |
| `--lang`, `-l`      | Ngôn ngữ đích (tên tiếng Anh đầy đủ)    | `Vietnamese`             |
| `--model`, `-m`     | Model Gemini (nếu dùng gemini provider) | `gemini-3-flash-preview` |
| `--prompt`          | Đường dẫn tới file prompt gemini.txt    | `prompts/gemini.txt`     |
| `--batch`, `-b`     | Số dòng dịch mỗi lần                    | `30`                     |
| `--budget`          | Thinking budget tokens (Gemini only)    | `24576`                  |
| `--wait`            | Giây chờ giữa mỗi batch                 | `0`                      |
| `--max-chars`       | Số ký tự tối đa mỗi dòng (0 = tắt)      | `0`                      |
| `--no-context`      | Tắt global context                      | (mặc định bật)           |
| `--verbose`, `-v`   | Hiển thị log chi tiết                   | (tắt)                    |

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

| Tham số              | Mô tả                           | Mặc định                  |
| -------------------- | ------------------------------- | ------------------------- |
| `--input`, `-i`      | File .srt đầu vào               | (bắt buộc)                |
| `--voice`, `-v`      | Tên giọng EdgeTTS               | (bắt buộc)                |
| `--output`, `-o`     | File audio đầu ra (.wav/.mp3)   | `output/<input_stem>.wav` |
| `--rate`             | Tốc độ giọng (vd: +10%, -5%)    | `+0%`                     |
| `--volume`           | Âm lượng (vd: +20%)             | `+0%`                     |
| `--pitch`            | Cao độ (vd: +50Hz)              | `+0Hz`                    |
| `--autorate`         | Tự động nén audio khớp slot SRT | (tắt)                     |
| `--max-speed`        | Giới hạn tốc độ nén tối đa      | `100.0`                   |
| `--concurrent`       | Số request EdgeTTS song song    | `10`                      |
| `--cache`            | Thư mục cache audio tạm         | `tmp/<stem>_<ts>/`        |
| `--no-strip-silence` | Tắt cắt silence ở đuôi mỗi clip | (tắt)                     |
| `--silence-thresh`   | Ngưỡng dBFS coi là silence      | `-50`                     |
| `--proxy`            | Proxy URL                       | (không dùng)              |
| `--list-voices`      | Liệt kê giọng EdgeTTS           | (không dùng)              |
| `--verbose`          | Bật logging debug               | (tắt)                     |

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

### 2.9. Trích xuất phụ đề cứng Multi-Box (video-ocr)

Trích xuất phụ đề (hardsub) trực tiếp từ khung hình video sử dụng mô hình DeepSeek-OCR-2, hỗ trợ nhiều vùng box độc lập.

#### Trích xuất nhanh (với Secrets)

```colab
from google.colab import userdata
hf_token = userdata.get('hf_token')

!uv run video-ocr /content/video.mp4 \
    --boxes-file /content/CharenjiZukan/assets/boxesOCR.txt \
    --hf-token "{hf_token}"
```

#### Đầy đủ tham số

```colab
from google.colab import userdata
hf_token = userdata.get('hf_token')

!uv run video-ocr /content/video.mp4 \
    --boxes-file /content/CharenjiZukan/assets/boxesOCR.txt \
    --output-dir /content \
    --frame-interval 3 \
    --scene-threshold 1.5 \
    --min-scene-frames 3 \
    --phash-threshold 4 \
    --noise-threshold 25 \
    --cv-prefilter \
    --cv-min-edge-density 0.03 \
    --cv-edge-low 50 \
    --cv-edge-high 150 \
    --min-chars 2 \
    --device cuda \
    --warn-english \
    --save-minify-txt \
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

| Tham số                   | Mô tả                                                                            | Mặc định                                                    |
| ------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `input_video`             | File video đầu vào (hoặc directory nếu dùng --input-dir)                         | (bắt buộc)                                                  |
| `--boxes-file`            | File cấu hình các vùng OCR theo format `name x y w h`                            | `assets/boxesOCR.txt` (nếu có config yaml thì config > cli) |
| `--output-dir`            | Thư mục output cho các file theo box                                             | cùng thư mục video                                          |
| `--frame-interval`        | Số frame bỏ qua giữa mỗi lần xử lý                                               | `30`                                                        |
| `--scene-threshold`       | Ngưỡng phần trăm thay đổi pixel trên tổng diện tích box để phát hiện chuyển cảnh | `1.5`                                                       |
| `--min-scene-frames`      | Số frame tối thiểu giữa 2 lần chuyển cảnh để tránh nhiễu                         | `3`                                                         |
| `--phash-threshold`       | Ngưỡng Hamming distance cho perceptual hash                                      | `4`                                                         |
| `--noise-threshold`       | Ngưỡng loại bỏ nhiễu nén video khi so sánh pixel                                 | `25`                                                        |
| `--cv-prefilter`          | Bật tiền lọc OpenCV để bỏ qua ROI không có dấu hiệu chữ                          | (tắt)                                                       |
| `--cv-min-edge-density`   | Ngưỡng mật độ cạnh tối thiểu cho CV prefilter                                    | `0.03`                                                      |
| `--cv-edge-low`           | Ngưỡng thấp Canny edge detector                                                  | `50`                                                        |
| `--cv-edge-high`          | Ngưỡng cao Canny edge detector                                                   | `150`                                                       |
| `--min-chars`             | Số ký tự tối thiểu để ghi nhận                                                   | `2`                                                         |
| `--no-scene-detection`    | Tắt bỏ tính năng Scene detection (tương đương threshold=0)                       | (tắt)                                                       |
| `--enable-chinese-filter` | Bật bộ lọc chỉ giữ lại tiếng Trung                                               | (tắt)                                                       |
| `--no-punctuation`        | Không giữ dấu câu tiếng Trung (khi bật filter)                                   | (tắt)                                                       |
| `--ocr-model`             | Tên model Hugging Face (DeepSeek-OCR-2 hoặc Qwen3-VL)                            | `deepseek-ai/DeepSeek-OCR-2`                                |
| `--qwen-max-new-tokens`   | [Chỉ Qwen3-VL] Số token tối đa sinh ra                                           | `256`                                                       |
| `--qwen-min-pixels`       | [Chỉ Qwen3-VL] Pixel blocks tối thiểu (ảnh hưởng VRAM)                           | `256`                                                       |
| `--qwen-max-pixels`       | [Chỉ Qwen3-VL] Pixel blocks tối đa (ảnh hưởng VRAM)                              | `1280`                                                      |
| `--device`                | Thiết bị xử lý (cuda/cpu)                                                        | `cuda`                                                      |
| `--hf-token`              | Hugging Face Token                                                               | (không dùng)                                                |
| `--batch-size`            | Batch size cho OCR batching                                                      | `8`                                                         |
| `--format`                | Định dạng output theo box (srt/txt)                                              | `srt`                                                       |
| `--default-duration`      | Thời lượng mặc định mỗi subtitle                                                 | `3.0s`                                                      |
| `--min-duration`          | Thời lượng tối thiểu sau deduplicate                                             | `1.0s`                                                      |
| `--max-duration`          | Thời lượng tối đa sau deduplicate                                                | `7.0s`                                                      |
| `--no-deduplicate`        | Tắt gộp subtitle trùng lặp                                                       | (tắt)                                                       |
| `--warn-english`          | Tạo file cảnh báo riêng nếu subtitle chứa tiếng Anh/số                           | (tắt)                                                       |
| `--save-minify-txt`       | Lưu file `<video>_script.txt` thuần văn bản, mỗi câu 1 dòng                      | (tắt)                                                       |
| `--no-timestamp`          | Tắt timestamp (chỉ với format=txt)                                               | (tắt)                                                       |
| `--config`                | Đường dẫn file cấu hình `.yaml`                                                  | (không dùng)                                                |

> **Mức ưu tiên Cấu hình**: CLI parameters có mức ưu tiên cao nhất, sau đó là tham số khai báo trong `--config`, và cuối cùng là Default values.

#### Output theo từng box

Ví dụ input là `/content/video.mp4` với 2 box `subtitle` và `note`, output sẽ là:

- `/content/video_subtitle.srt`
- `/content/video_note.srt`

#### Hướng dẫn chỉnh ngưỡng

| Tình huống                       | scene_threshold | phash_threshold |
| :------------------------------- | :-------------- | :-------------- |
| Box nhỏ, subtitle thay đổi nhanh | 1.0             | 3               |
| Box lớn, subtitle thay đổi chậm  | 0.5             | 5               |
| Video nhiều nhiễu/hiệu ứng       | 2.0             | 6               |
| Mặc định cân bằng                | 1.5             | 4               |

---

### 2.10. Native Video Subtitle Extractor (video-ocr-native)

CLI này dùng Qwen3-VL Native Video mode để xử lý subtitle theo từng batch video (mặc định 60 giây), giữ context theo chiến lược multi-turn giữa các batch.

#### Chạy nhanh (khuyến nghị cho Colab)

```colab
from google.colab import userdata
hf_token = userdata.get('hf_token')

!uv run video-ocr-native /content/video.mp4 \
    --hf-token "{hf_token}" \
    --device cuda
```

#### Chạy đầy đủ tham số chính

```colab
from google.colab import userdata
hf_token = userdata.get('hf_token')

!uv run video-ocr-native /content/video.mp4 \
    --config /content/CharenjiZukan/config/native_video_ocr_config.yaml \
    --boxes-file /content/CharenjiZukan/assets/boxesOCR.txt \
    --output-dir /content \
    --prompt-file /content/CharenjiZukan/prompts/native_video_ocr_prompt.txt \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --device cuda \
    --hf-token "{hf_token}" \
    --frame-interval 6 \
    --batch-duration 60 \
    --sample-fps 5.0 \
    --max-new-tokens 2048 \
    --total-pixels 20971520 \
    --min-pixels 65536 \
    --max-frames 2048 \
    --warn-english \
    --save-minify-txt \
    --verbose
```

#### Tham số chính

| Tham số             | Mô tả                                                           | Mặc định                              |
| ------------------- | --------------------------------------------------------------- | ------------------------------------- |
| `video`             | Đường dẫn video input                                           | (bắt buộc)                            |
| `--config`          | YAML config cho native pipeline                                 | `config/native_video_ocr_config.yaml` |
| `--boxes-file`      | File ROI dạng `name x y w h`                                    | theo config (`assets/boxesOCR.txt`)   |
| `--output-dir`      | Thư mục output                                                  | cùng thư mục video                    |
| `--prompt-file`     | Prompt template cho native extraction                           | `prompts/native_video_ocr_prompt.txt` |
| `--model`           | Model Qwen3-VL                                                  | `Qwen/Qwen3-VL-8B-Instruct`           |
| `--device`          | Thiết bị chạy model                                             | `cuda`                                |
| `--hf-token`        | Hugging Face token                                              | (không dùng)                          |
| `--frame-interval`  | Lấy mẫu 1 frame mỗi N frame                                     | `6`                                   |
| `--batch-duration`  | Số giây mỗi batch video                                         | `60.0`                                |
| `--sample-fps`      | FPS khai báo cho frame-list native video                        | `5.0`                                 |
| `--max-new-tokens`  | Số token output tối đa mỗi batch                                | `2048`                                |
| `--total-pixels`    | Giới hạn tổng pixel video input                                 | `20971520`                            |
| `--min-pixels`      | Giới hạn pixel tối thiểu                                        | `65536`                               |
| `--max-frames`      | Giới hạn số frame mỗi batch                                     | `2048`                                |
| `--warn-english`    | Lưu file cảnh báo English/number                                | (tắt)                                 |
| `--save-minify-txt` | Lưu file script thuần văn bản (mỗi câu 1 dòng, không timestamp) | (tắt)                                 |
| `-v`, `--verbose`   | Tăng chi tiết log                                               | (tắt)                                 |
| `--quiet`           | Chỉ in lỗi                                                      | (tắt)                                 |

#### Output files

Với input `/content/video.mp4`:

- `/content/video_native.srt` (luôn tạo)
- `/content/video_native_script.txt` (khi bật `--save-minify-txt`)
- `/content/video_subtitle_english_warnings.txt` (khi bật `--warn-english`)

---

### 2.11. TTS-Video Sync Pipeline (sync-video)

CLI `sync-video` dùng pipeline `sync_engine` để đồng bộ video + TTS theo timeline subtitle (gồm 5 phase: phân tích timeline, xử lý video chunks, ghép audio, remap timestamps, render final).

#### Chạy nhanh

```colab
!uv run sync-video \
    --video /content/video.mp4 \
    --subtitle /content/subtitle_translated.srt \
    --tts-voice ja-JP-KeitaNeural \
    --output-dir /content/output_sync
```

#### Chạy đầy đủ tham số

```colab
!uv run sync-video \
    --video /content/video.mp4 \
    --subtitle /content/subtitle_translated.srt \
    --tts-voice ja-JP-KeitaNeural \
    --mute /content/mute.srt \
    --note-overlay-png /content/note-overlay.png \
    --note-overlay-ass /content/note_overlay.ass \
    --black-bg /content/black-background.png \
    --ambient /content/ambient.mp3 \
    --slow-cap 0.5 \
    --output-dir /content/output_sync \
    --output-name video_synced \
    --no-hardsub \
    --workers 4 \
    --no-gpu \
    --subtitle-fontname "Noto Sans CJK JP" \
    --subtitle-fontsize 22 \
    --subtitle-color "&H00EEF5FF" \
    --subtitle-margin-v 6 \
    --note-max-chars 15
```

#### Tham số

| Tham số               | Mô tả                                                     | Mặc định             |
| --------------------- | --------------------------------------------------------- | -------------------- |
| `--video`             | File video gốc (`.mp4`, `.mkv`)                           | (bắt buộc)           |
| `--subtitle`          | File subtitle `.srt` đầy đủ (bao gồm cả vùng mute nếu có) | (bắt buộc)           |
| `--tts-voice`         | Giọng đọc EdgeTTS (ví dụ: ja-JP-KeitaNeural)              | `ja-JP-KeitaNeural`  |
| `--mute`              | File mute `.srt` cho vùng quoted (không TTS)              | (không dùng)         |
| `--note-overlay-png`  | Ảnh PNG tĩnh nền note                                     | (không dùng)         |
| `--note-overlay-ass`  | File ASS text cho note overlay                            | (không dùng)         |
| `--black-bg`          | Ảnh dải đen nền note (tự tạo nếu không truyền)            | (không dùng)         |
| `--ambient`           | Nhạc nền ambient cho toàn bộ video                        | `assets/ambient.mp3` |
| `--slow-cap`          | Giới hạn tốc độ video thấp nhất (cap cho stretch)         | `0.5`                |
| `--output-dir`        | Thư mục output                                            | `./sync_output/`     |
| `--output-name`       | Tên base cho tất cả file output                           | `video_synced`       |
| `--no-hardsub`        | Bỏ render MP4 hardsub, chỉ xuất các file đã remap         | (tắt)                |
| `--workers`           | Số worker FFmpeg chạy song song khi xử lý chunk video     | `4`                  |
| `--no-gpu`            | Dùng `libx264` thay `h264_nvenc` (CPU mode)               | (tắt)                |
| `--keep-tmp`          | Giữ lại thư mục tạm chứa các chunks video để debug        | (tắt)                |
| `--subtitle-fontname` | Font subtitle dùng khi burn hardsub                       | `Noto Sans CJK JP`   |
| `--subtitle-fontsize` | Cỡ chữ subtitle                                           | `22`                 |
| `--subtitle-color`    | Màu chữ subtitle (ASS hex format)                         | `&H00EEF5FF`         |
| `--subtitle-margin-v` | Margin dọc subtitle (px)                                  | `6`                  |
| `--note-max-chars`    | Số ký tự tối đa mỗi dòng khi wrap text ASS note           | `15`                 |

#### Quy ước input/output quan trọng

- Chương trình tự động sinh audio theo `--tts-voice`.
- Khi chạy đủ pipeline (không bật `--no-hardsub`), output chính bao gồm:
  - `<output-name>.mp4`
  - `<output-name>_tts_synced.srt`
  - `<output-name>_synced.srt`
- Output tùy chọn nếu có input tương ứng:
  - `<output-name>_mute_synced.srt` (khi có `--mute`)
  - `<output-name>_note_synced.ass` (khi có `--note-overlay-ass`)

---

## 3. Chạy test trên Google Colab với `run_colab_tests.py`

File nằm tại: `run_colab_tests.py` (project root)

### 3.1 Cú pháp đầy đủ

```colab
!python run_colab_tests.py [OPTIONS]

OPTIONS:
  --matrix PATH       File test_matrix.yaml (mặc định: tests/test_matrix.yaml)
  --tags TAG [TAG...] Lọc entry theo tags (OR logic: khớp bất kỳ tag nào)
  --name SUBSTR       Lọc entry theo tên (substring, case-insensitive)
  --reports-dir DIR   Thư mục lưu báo cáo fail (mặc định: test_reports/)
  --list              Hiển thị danh sách test sẽ chạy, không chạy thật
```

### 3.2 Các trường hợp sử dụng thường gặp

**Xem danh sách không chạy**:

```colab
!python run_colab_tests.py --list
!python run_colab_tests.py --tags gpu --list
```

**Chạy toàn bộ tests nhanh (không GPU)**:

```colab
!python run_colab_tests.py --tags unit
!python run_colab_tests.py --tags unit integration
```

**Chạy tests cần FFmpeg**:

```colab
!python run_colab_tests.py --tags ffmpeg
!python run_colab_tests.py --tags integration ffmpeg
```

**Chạy test 1 hàm**:

```colab
!python -m pytest tests/test_translation_providers.py::TestLayer4_RealAPIs::test_vertexai_real_api
```

**Chạy tất cả tests liên quan 1 feature**:

```colab
!python run_colab_tests.py --name "Native Video"
!python run_colab_tests.py --name "SRT Parser"
```

**Chạy toàn bộ** (enabled tests):

```colab
!python run_colab_tests.py
```

**Dùng file matrix khác** (khi có nhiều môi trường):

```colab
!python run_colab_tests.py --matrix tests/test_matrix_ci.yaml
```

### 3.3 Chạy toàn bộ test trong một file

Khi một entry trong [`tests/test_matrix.yaml`](tests/test_matrix.yaml) **bỏ trống `keyword`**, script [`run_colab_tests.py`](tests/run_colab_tests.py) sẽ **không thêm cờ `-k`** và `pytest` sẽ collect toàn bộ test trong file đó.

#### Ví dụ entry trong `test_matrix.yaml`

```yaml
- name: "Native Video OCR — All Layers"
  file: "tests/test_native_video_ocr_pipeline.py"
  # keyword: bỏ trống hoàn toàn
  timeout_sec: 900 # Tổng timeout của cả 4 layers cộng lại
  pytest_args: ["-v", "-s"]
  tags: ["unit", "integration", "gpu", "native_ocr"]
  enabled: true
```

> `timeout_sec` ở đây là timeout tổng cho toàn bộ các layer trong cùng một file.

#### Chạy trực tiếp bằng pytest (không qua `run_colab_tests.py`)

```bash
python -m pytest tests/test_native_video_ocr_pipeline.py -v
```

### 3.4 Quy trình làm việc trên Google Colab

#### 3.4.1 Workflow chuẩn khi develop một feature mới

```text
Bước 1: Viết code feature
Bước 2: Viết file test (4 layers) + thêm vào test_matrix.yaml
Bước 3: Chạy Layer 1 → fix cho đến khi pass
Bước 4: Chạy Layer 2 → fix cho đến khi pass
Bước 5: Chạy Layer 3 → fix cho đến khi pass
Bước 6: (Khi có GPU) Chạy Layer 4 → confirm chất lượng AI
```

```colab
# Cell: Layer 1 và 2 (không cần GPU)
!python run_colab_tests.py --tags unit integration --name "TÊN FEATURE"

# Cell: Layer 3 (không cần GPU)
!python run_colab_tests.py --name "TÊN FEATURE" --name "Layer 3"

# Cell: Nếu có lỗi, xem report
!ls test_reports/
!cat "test_reports/failed_*.md"
```

#### 3.4.2 Workflow debug khi có fail

```colab
# Bước 1: Xem report tóm tắt
!ls -la test_reports/*.md

# Bước 2: Đọc report chi tiết (hoặc download file .md để gửi AI)
import subprocess
result = subprocess.run(["cat", "test_reports/failed_xxx.md"], capture_output=True, text=True)
print(result.stdout[:5000])  # Print 5000 chars đầu

# Bước 3: Chạy lại command từ mục "1. Lệnh đã chạy" trong report
!python -m pytest tests/test_native_video_ocr_pipeline.py -k "Layer3" -v --tb=long -s

# Bước 4: Chạy 1 test duy nhất để isolate lỗi
!python -m pytest tests/test_native_video_ocr_pipeline.py::TestLayer3_FullPipeline::test_entries_count -v --tb=long
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
!python cli/tts_srt.py --input video_vi.srt --voice ja-JP-KeitaNeural
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

1. **Cài đặt project**: Sau khi clone, cần chạy `!uv pip install -e .` để cài đặt project ở chế độ editable, cho phép sử dụng các CLI commands (`mute-srt`, `translate-srt`, `tts-srt`, `video-ocr`).

2. **rubberband-cli**: Cần cài đặt bằng `apt-get` vì đây là binary hệ thống, không phải Python package. Dùng cho time-stretch audio chất lượng cao.

3. **API Keys**: Sử dụng Google Colab Secrets để bảo mật API keys. Không hardcode token vào code.

4. **Multi-Box OCR**: File `boxesOCR.txt` phải đúng format `name x y w h`, mỗi box một dòng.

5. **Output Multi-Box**: `video-ocr` xuất nhiều file theo mẫu `<video_stem>_<box_name>.srt` hoặc `.txt`, không còn dùng `--output` file đơn.

6. **Autorate**: Khi bật `--autorate`, audio sẽ được nén/giãn để khớp với thời lượng slot trong file SRT.

7. **Gemini API Key**: Sử dụng cú pháp `--keys "{gemini_key}"` với biến từ `userdata.get('gemini_key')`.

8. Trình tự chạy lại để xác thực fix:

```colab
!uv cache clean
!uv sync --reinstall
!uv pip install -e .
```
