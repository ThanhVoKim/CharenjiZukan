# Sync Video Guide

Hướng dẫn đầy đủ về flow `sync-video` và schema `render_config.json`.

## Mục lục

1. [Tổng quan flow sync-video](#1-tổng-quan-flow-sync-video)
2. [Schema `render_config.json`](#2-schema-render_configjson)
3. [Cấu hình `llm_metadata`](#3-cấu-hình-llm_metadata)
4. [Output paths](#4-output-paths)
5. [Task-file override](#5-task-file-override)
6. [Fail policy](#6-fail-policy)

---

## 1. Tổng quan flow sync-video

CLI `sync-video` đồng bộ video với TTS audio và subtitle, gồm 6 phase:

```
Phase 0: Auto Generate TTS
    ↓
Phase 1: Analysis (classify blocks, compute speeds, build timeline)
    ↓
Phase 2: Video Processing (split + stretch + concat chunks)
    ↓
Phase 2.5: BGM Extraction (optional, nếu extract_bgm=true)
    ↓
Phase 3: Audio Assembly (mix TTS + original audio + ambient + BGM)
    ↓
Phase 4: Recalculate Timestamps (SRT + ASS output)
    ↓
Phase 5: Final Render (hardsub video với FFmpeg)
    ↓
Phase 6: LLM Metadata (post-render, nếu llm_metadata.enabled=true)
```

### Flow LLM Metadata chi tiết

```
parse_srt_file(args.subtitle)
    ↓
prepare_llm_metadata_input()  ← GHI FILE .txt NGAY SAU PARSE
    ├── Kiểm tra llm_metadata.enabled
    ├── Resolve debug_input_path từ config
    ├── Gọi write_segments_to_flat_text() → ghi raw text phẳng
    └── Trả về Path | None
    ↓
... Phase 0 → Phase 5 ...
    ↓
render_final_video() hoàn tất
    ↓
if metadata_input_path:
    run_llm_metadata_task(input_text_path=...)
        ├── Kiểm tra enabled
        ├── execute_llm_metadata_task()
        │   ├── Đọc input_text_path
        │   ├── Resolve output_path
        │   ├── Tạo provider qua llm_ai.task_runner
        │   └── Gọi run_generic_text_task()
        └── Xử lý fail_policy
```

---

## 2. Schema `render_config.json`

File mặc định: `assets/default_render_config.json`

### 2.1 `resolution`

```json
{
  "resolution": {
    "width": 1920,
    "height": 1080
  }
}
```

| Key      | Type | Default | Mô tả                   |
| -------- | ---- | ------- | ----------------------- |
| `width`  | int  | 1920    | Chiều rộng output video |
| `height` | int  | 1080    | Chiều cao output video  |

### 2.2 `watermark_img`

```json
{
  "watermark_img": {
    "enabled": false,
    "path": "assets/CharenjiZukan-watermark.png",
    "position": "top_left",
    "scale": 0.15
  }
}
```

| Key        | Type  | Default    | Mô tả                                                          |
| ---------- | ----- | ---------- | -------------------------------------------------------------- |
| `enabled`  | bool  | false      | Bật/tắt watermark ảnh                                          |
| `path`     | str   | —          | Đường dẫn tới ảnh watermark                                    |
| `position` | str   | "top_left" | Vị trí: top_left, top_right, bottom_left, bottom_right, center |
| `scale`    | float | 0.15       | Tỉ lệ so với chiều rộng video                                  |

### 2.3 `watermark_text`

```json
{
  "watermark_text": {
    "enabled": false,
    "text": "",
    "font_size": 24,
    "color": "white",
    "position": "bottom_right"
  }
}
```

### 2.4 `black_strip`

```json
{
  "black_strip": {
    "enabled": false,
    "height": 60
  }
}
```

### 2.5 `subtitles`

```json
{
  "subtitles": {
    "enabled": true,
    "style": {
      "font_size": 24,
      "font_color": "&H00FFFFFF",
      "outline_color": "&H00000000",
      "outline": 2,
      "shadow": 1,
      "alignment": 2,
      "margin_l": 30,
      "margin_r": 30,
      "margin_v": 50
    }
  }
}
```

| Key                   | Type | Default      | Mô tả                    |
| --------------------- | ---- | ------------ | ------------------------ |
| `enabled`             | bool | true         | Bật/tắt hard-sub         |
| `style.font_size`     | int  | 24           | Cỡ chữ subtitle          |
| `style.font_color`    | str  | "&H00FFFFFF" | Màu chữ (ASS format)     |
| `style.outline_color` | str  | "&H00000000" | Màu viền                 |
| `style.outline`       | int  | 2            | Độ dày viền              |
| `style.shadow`        | int  | 1            | Độ sâu bóng              |
| `style.alignment`     | int  | 2            | Căn lề (2=bottom-center) |
| `style.margin_l`      | int  | 30           | Lề trái                  |
| `style.margin_r`      | int  | 30           | Lề phải                  |
| `style.margin_v`      | int  | 50           | Lề dọc                   |

### 2.6 `note_overlay`

```json
{
  "note_overlay": {
    "enabled": false
  }
}
```

### 2.7 `audio_mix`

```json
{
  "audio_mix": {
    "tts_volume": 1.0,
    "original_volume": 0.3,
    "ambient_volume": 0.15,
    "bgm_volume": 0.4
  }
}
```

### 2.8 `audio_separator`

```json
{
  "audio_separator": {
    "extract_bgm": false,
    "extract_vocals": false,
    "mdxc_params": {
      "model": "MDX23C-InstVoc_HQ",
      "device": "cuda"
    }
  }
}
```

### 2.9 `video_encoding`

```json
{
  "video_encoding": {
    "codec": "h264",
    "quality": ["-crf", "18"],
    "preset": "medium"
  }
}
```

---

## 3. Cấu hình `llm_metadata`

```json
{
  "llm_metadata": {
    "enabled": false,
    "task_config": "config/llm_tasks/seo_metadata.yaml",
    "provider": null,
    "provider_config": null,
    "model": null,
    "keys": null,
    "system_prompt": null,
    "temperature": null,
    "max_tokens": null,
    "request_timeout": null,
    "provider_overrides": {},
    "input": {
      "write_debug_input": false,
      "debug_input_filename_template": "{video_stem}_metadata_input.txt"
    },
    "output": {
      "directory_policy": "/",
      "filename_template": "{video_stem}_metadata.md"
    },
    "fail_policy": "warn"
  }
}
```

### 3.1 Các key chính

| Key           | Type | Default                              | Mô tả                                                                               |
| ------------- | ---- | ------------------------------------ | ----------------------------------------------------------------------------------- |
| `enabled`     | bool | false                                | Bật/tắt bước tạo metadata sau render                                                |
| `task_config` | str  | "config/llm_tasks/seo_metadata.yaml" | YAML config cho generic LLM task                                                    |
| `fail_policy` | str  | "warn"                               | `warn` → log warning, không fail pipeline; `raise`/`error`/`fail` → raise exception |

### 3.2 Provider overrides

Các key provider (có thể đặt trực tiếp trong `llm_metadata` hoặc trong `provider_overrides`):

| Key               | Type  | Mô tả                                  |
| ----------------- | ----- | -------------------------------------- |
| `provider`        | str   | Provider LLM: gemini, openai, vertexai |
| `provider_config` | str   | Đường dẫn YAML config provider         |
| `model`           | str   | Tên model                              |
| `keys`            | str   | API key(s), phân cách bằng dấu phẩy    |
| `system_prompt`   | str   | System prompt                          |
| `temperature`     | float | Nhiệt độ sampling                      |
| `max_tokens`      | int   | Max output tokens                      |
| `request_timeout` | int   | Timeout request (giây)                 |

### 3.3 `input`

| Key                             | Type | Default                            | Mô tả                                      |
| ------------------------------- | ---- | ---------------------------------- | ------------------------------------------ |
| `write_debug_input`             | bool | false                              | Ghi raw text input ra file `.txt` để debug |
| `debug_input_filename_template` | str  | "{video_stem}\_metadata_input.txt" | Tên file debug input                       |

### 3.4 `output`

| Key                 | Type | Default                     | Mô tả                                                              |
| ------------------- | ---- | --------------------------- | ------------------------------------------------------------------ |
| `directory_policy`  | str  | "/"                         | `/` = thư mục chứa video input; hoặc đường dẫn tuyệt đối/tương đối |
| `filename_template` | str  | "{video_stem}\_metadata.md" | Tên file output metadata                                           |

### 3.5 Template variables

Các biến có thể dùng trong `filename_template` và `debug_input_filename_template`:

| Variable         | Ví dụ (video = `content/a/b.mp4`)               |
| ---------------- | ----------------------------------------------- |
| `{video_stem}`   | `b`                                             |
| `{video_name}`   | `b.mp4`                                         |
| `{video_suffix}` | `.mp4`                                          |
| `{output_name}`  | Giá trị `--output-name` (default: `video_stem`) |

---

## 4. Output paths

### 4.1 `directory_policy: "/"`

Thư mục output = thư mục chứa video input (KHÔNG phải filesystem root).

Ví dụ:

- Input video: `content/episodes/ep01.mp4`
- `filename_template`: `{video_stem}_metadata.md`
- → Output: `content/episodes/ep01_metadata.md`

### 4.2 `directory_policy` là đường dẫn

Nếu `directory_policy` không phải `"/"`, nó được xử lý như đường dẫn:

- Tuyệt đối: dùng trực tiếp
- Tương đối: resolve từ PROJECT_ROOT

---

## 5. Task-file override

Khi dùng `--task-file`, mỗi task JSON có thể chứa key `llm_metadata` để override cấu hình từ `render_config.json`.

### 5.1 Boolean override

```json
{
  "input": "video.mp4",
  "subtitle": "video.srt",
  "llm_metadata": false
}
```

→ Tắt LLM metadata cho task này (bất kể `render_config.json` bật hay không).

### 5.2 Object override (deep merge)

```json
{
  "input": "video.mp4",
  "subtitle": "video.srt",
  "llm_metadata": {
    "enabled": true,
    "output": {
      "filename_template": "{output_name}_seo.md"
    }
  }
}
```

→ Merge sâu vào `render_config.llm_metadata`. Các key không được override giữ nguyên giá trị từ `render_config.json`.

---

## 6. Fail policy

| Policy                     | Hành vi                                                   |
| -------------------------- | --------------------------------------------------------- |
| `warn` (default)           | Log warning, trả về `None`, pipeline tiếp tục bình thường |
| `raise` / `error` / `fail` | Raise exception, pipeline dừng với lỗi                    |

Ví dụ:

```json
{
  "llm_metadata": {
    "enabled": true,
    "fail_policy": "raise"
  }
}
```

---

## 7. Kiến trúc module

```
cli/sync_video.py          ← Entrypoint CLI (pyproject.toml [project.scripts])
    ↓ import
sync_engine/llm_metadata.py ← Orchestration LLM metadata
    ↓ import
llm_ai/task_runner.py       ← Provider creation logic (dùng chung)
llm_ai/tasks/generic_text_task.py ← Generic LLM text task runner
utils/srt_parser.py         ← SRT parsing + write_segments_to_flat_text()
```

Nguyên tắc:

- `cli/` chỉ chứa entrypoint command được định nghĩa trong `pyproject.toml` `[project.scripts]`
- `sync_engine/` phụ thuộc vào `llm_ai/`, không phụ thuộc ngược vào `cli/`
- `llm_ai/` là thư viện LLM độc lập, không phụ thuộc vào `cli/` hay `sync_engine/`
- `utils/` là thư viện tiện ích thuần, không phụ thuộc vào các package khác
