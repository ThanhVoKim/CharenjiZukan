# Sync Video Guide

Hướng dẫn đầy đủ về flow `sync-video` và schema `render_config.json`.

## Mục lục

1. [Tổng quan flow sync-video](#1-tổng-quan-flow-sync-video)
2. [Schema `render_config.json`](#2-schema-render_configjson)
3. [Cấu hình `image_overlay`](#3-cấu-hình-image_overlay)
4. [Cấu hình `forced_alignment_subtitle`](#4-cấu-hình-forced_alignment_subtitle)
5. [Cấu hình `llm_metadata`](#5-cấu-hình-llm_metadata)
6. [Output paths](#6-output-paths)
7. [Task-file override](#7-task-file-override)
8. [Fail policy](#8-fail-policy)
9. [Kiến trúc module](#9-kiến-trúc-module)

---

## 1. Tổng quan flow sync-video

CLI `sync-video` đồng bộ video với TTS audio và subtitle, gồm 7 phase:

```
Phase 0: Auto Generate TTS
    ↓
Phase 1: Analysis (classify blocks, compute speeds, build timeline)
    ↓
Phase 2: Video Processing (split + stretch + concat chunks; nung black_strip vào video_stretched)
    ↓
Phase 2.5: Optional Global BGM Extraction (nếu audio_policies.global_bgm là whole_video hoặc exclude_mute)
    ↓
Phase 3: Audio Assembly (main concat + ambient overlay + BGM overlay theo audio_policies)
    ↓
Phase 3.5: Forced Alignment Subtitle (optional, nếu forced_alignment_subtitle.enabled=true)
    ↓
Phase 4: Recalculate Timestamps + Image Overlay static image + Dynamic Note Overlay ASS
    ↓
Phase 5: Final Render (hardsub video với FFmpeg)
    ↓
Phase 6: LLM Metadata (post-render, nếu llm_metadata.enabled=true)
```

### Flow TTS audio processing chi tiết (qwen / qwen_custom)

Áp dụng cho provider `qwen`, `qwen_custom`, `edge`.

```
model.generate_*()
    ↓
_postprocess(include_pad=False)
    ├── clean_tail=true  → librosa.trim(top_db)   [cắt silence/garbage 2 đầu]
    │                    → fade-out tail + fade-in head (fade_ms ms)
    └── clean_tail=false → bỏ qua trim, chỉ fade
    ↓
sf.write(dubb-N.wav)     [ghi wav không có padding]
    ↓                    [sau khi ghi xong TẤT CẢ clips trong batch]
apply_speed_scale()      [atempo chain, chỉ chạy nếu speed_scale > 1.0]
    ↓
_pad_file(pre, post)     [thêm silence cố định: pre_phoneme_length đầu, post_phoneme_length cuối]
```

**Lý do thứ tự `speedup → pad`:** padding là silence cố định sau dub, không nên bị nén theo
`speed_scale`. Nếu pad trước speedup, `post_phoneme_length=0.1s` với `speed_scale=1.3` → padding
thực tế chỉ còn ~0.077s. Với thứ tự mới, padding luôn đúng giá trị cấu hình.

---

### Flow Forced Alignment Subtitle chi tiết

```
assemble_audio_track() hoàn tất → mixed_audio.wav
    ↓
if forced_alignment_subtitle.enabled:
    run_forced_alignment_subtitle(timeline, subtitle_segments, mute_segments)
        ├── Dựng clip list từ timeline: zip tts_only ↔ TTS TimelineSegment
        │     ↓ Mỗi clip: {audio_path=dubb-N.wav, text, offset_ms=new_start, audio_speed}
        ├── execute_forced_alignment_clips()  ← Load aligner 1 lần, loop batch clip
        │     ├── aligner.align(audio=[clip], text=[line])  ← ~vài giây/clip, không OOM
        │     ├── word_ms + offset_ms / audio_speed → timestamp tuyệt đối trên timeline cuối
        │     ├── merge_punctuation() + segment_words_to_subtitles() per clip
        │     └── try/finally: del aligner + clear_vram()  ← LUÔN giải phóng GPU
        ├── Dòng vùng mute / clip thiếu → recalculate_segments() (remap timeline)
        ├── Gộp aligned + remap, sort theo thời gian
        └── ghi <output-name>_synced.srt (đầy đủ, không sót dòng mute)
    ↓
    Nếu alignment thất bại + fail_policy=warn:
        → Fallback sang recalculate_srt() (remap timestamp toàn bộ)
    ↓
Phase 4 tiếp tục bình thường
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

### Flow Image Overlay static image chi tiết

```text
Phase 2 cập nhật timeline stretch thực tế
    ↓
Phase 4 nếu image_overlay.enabled=true
    ├── Đọc --image-overlay-srt theo timeline video gốc
    ├── Text mỗi block SRT = basename ảnh, không có extension
    ├── Resolve static image asset trong --image-overlay-dir
    ├── Remap start/end bằng cùng mapping với subtitle remap
    ├── Optional ghi <output-name>_image_overlay_synced.srt để debug
    └── Truyền ImageOverlayEvent vào renderer
        ↓
Phase 5 render theo layer order:
Base video → Black strip → [Tuber] → Image overlay static image → Note overlay → Watermark img → Watermark text → Subtitle
```

> **Layer order & black_strip (cập nhật):**
>
> - `black_strip` **không còn** ghép ở Phase 5 mà được **nung thẳng vào `video_stretched.mp4` ở Phase 2** (gấp chung vào encode batch — **không thêm lần encode nào**). Vì vậy strip luôn nằm **dưới** mọi layer khác, kể cả tuber overlay (tuber composite seek vào `video_stretched.mp4` nên đắp lên trên strip). Final render tự động `skip_layers={"black_strip"}` để tránh nung 2 lần.
> - Strip phủ **full width video gốc** theo mặc định (probe width source, không hardcode 1920). Nếu `black_strip.scale_width` > width video → tự kẹp về full width. `scale_height` theo config; `x` mặc định căn giữa `(W-w)/2`, `y` theo config.
> - Thứ tự ghép các layer còn lại trong Phase 5 điều khiển bằng `render_config.layer_order` (xem mục 2.x). Đổi thứ tự = sửa mảng này, không cần đổi code.
> - **Tuber overlay** là layer baked vào base (xem `docs/tuber-overlay-guide.md`); vị trí z-order của nó cố định **trên black_strip, dưới image/note/watermark/subtitle**, không nằm trong `layer_order`.

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

### 2.1b `layer_order`

Mảng điều khiển **thứ tự ghép layer** ở Phase 5 (final render). Render lần lượt từ dưới lên trên theo thứ tự trong mảng (phần tử đầu = dưới cùng, phần tử cuối = trên cùng). Thiếu key này → dùng thứ tự mặc định bên dưới.

```json
{
  "layer_order": [
    "black_strip",
    "image_overlay",
    "note_overlay",
    "watermark_img",
    "watermark_text",
    "subtitles"
  ]
}
```

| Giá trị hợp lệ   | Layer                                     |
| ---------------- | ----------------------------------------- |
| `black_strip`    | Dải đen (đã nung ở Phase 2 — xem ghi chú) |
| `image_overlay`  | Ảnh tĩnh full-screen theo SRT             |
| `note_overlay`   | Hộp ghi chú động (ASS box)                |
| `watermark_img`  | Watermark ảnh                             |
| `watermark_text` | Watermark chữ (drawtext)                  |
| `subtitles`      | Hard-sub SRT                              |

> - Giá trị lạ trong mảng → log warning và bỏ qua.
> - `black_strip` thực tế đã nung vào base ở Phase 2 nên Phase 5 tự skip; giữ nó trong `layer_order` chỉ để tài liệu hóa vị trí z-order (dưới cùng). Muốn đổi vị trí strip cần đổi cách nung ở Phase 2, không chỉ sửa `layer_order`.
> - **Tuber overlay** không nằm trong `layer_order` (baked vào base, luôn trên strip / dưới các layer còn lại).

### 2.2 `watermark_img`

```json
{
  "watermark_img": {
    "enabled": false,
    "path": "assets/CharenjiZukan-watermark.png",
    "width": null,
    "x": "1680",
    "y": "39"
  }
}
```

| Key       | Type        | Default | Mô tả                                                                                                                                                                                           |
| --------- | ----------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enabled` | bool        | false   | Bật/tắt watermark ảnh                                                                                                                                                                           |
| `path`    | str         | —       | Đường dẫn tới ảnh watermark                                                                                                                                                                     |
| `width`   | int \| null | null    | Chiều rộng watermark (px). `null`/bỏ trống/`<=0` → **giữ nguyên width gốc của ảnh**. Số > 0 → scale theo width đó, **height tự suy theo aspect ratio** (FFmpeg `scale=width:-1`) nên không méo. |
| `x`       | str         | —       | Tọa độ X overlay (biểu thức FFmpeg, vd `1680` hoặc `W-w-40`)                                                                                                                                    |
| `y`       | str         | —       | Tọa độ Y overlay (biểu thức FFmpeg)                                                                                                                                                             |

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
    "enabled": true,
    "path": "assets/black-background.jpg",
    "scale_width": "1920",
    "scale_height": "94",
    "x": "(main_w-overlay_w)/2",
    "y": "968"
  }
}
```

| Key            | Type | Default    | Mô tả                                                                                                  |
| -------------- | ---- | ---------- | ------------------------------------------------------------------------------------------------------ |
| `enabled`      | bool | false      | Bật/tắt dải đen                                                                                        |
| `path`         | str  | —          | Đường dẫn ảnh dải (vd nền đen)                                                                         |
| `scale_width`  | str  | full width | Chiều rộng dải. **Mặc định = full width video gốc**; nếu giá trị > width video → tự kẹp về full width. |
| `scale_height` | str  | 94         | Chiều cao dải (px)                                                                                     |
| `x`            | str  | `(W-w)/2`  | Tọa độ X (mặc định căn giữa)                                                                           |
| `y`            | str  | `968`      | Tọa độ Y                                                                                               |

> **Quan trọng:** `black_strip` được **nung vào `video_stretched.mp4` ở Phase 2** (gấp chung vào encode batch — không thêm lần encode). Hệ quả:
>
> - Strip luôn nằm **dưới** tuber overlay và mọi layer Phase 5.
> - Strip phủ **full width** bất kể độ phân giải source (probe width video gốc, không hardcode).
> - Final render (`sync-video` và `tuber-repair`) tự `skip_layers={"black_strip"}` để tránh nung 2 lần.

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

### 2.6 `image_overlay`

`image_overlay` dùng một SRT riêng để điều khiển static image full-screen. Timestamp trong SRT này thuộc timeline video gốc và được remap theo timeline stretch ở Phase 4 trước khi render. PNG vẫn là định dạng khuyến nghị khi cần alpha/transparent, nhưng resolver mặc định chấp nhận nhiều định dạng ảnh tĩnh.

```json
{
  "image_overlay": {
    "enabled": false,
    "mode": "srt_fullscreen_static_image",
    "file_ext": "auto",
    "fit": "stretch_to_output",
    "x": "0",
    "y": "0",
    "opacity": 1.0,
    "missing_policy": "warn",
    "keep_intermediate_srt": false,
    "direct_overlay_max_events": 200,
    "command_line_max_chars": 25000,
    "render_strategy": "auto"
  }
}
```

| Key                         | Type  | Default                       | Mô tả                                                                                                     |
| --------------------------- | ----- | ----------------------------- | --------------------------------------------------------------------------------------------------------- |
| `enabled`                   | bool  | false                         | Bật/tắt image overlay                                                                                     |
| `mode`                      | str   | `srt_fullscreen_static_image` | Phase hiện tại hỗ trợ SRT → static image full-screen                                                      |
| `file_ext`                  | str   | `auto`                        | `auto` resolve các định dạng ảnh tĩnh hỗ trợ; vẫn có thể đặt `.png`, `.jpg`... để ép một extension cụ thể |
| `fit`                       | str   | `stretch_to_output`           | Scale ảnh về đúng width/height output; có thể méo nếu asset sai aspect ratio                              |
| `x`, `y`                    | str   | `0`, `0`                      | Tọa độ overlay trong FFmpeg                                                                               |
| `opacity`                   | float | 1.0                           | Opacity toàn bộ image overlay                                                                             |
| `missing_policy`            | str   | `warn`                        | `warn` bỏ qua asset thiếu; `raise` dừng pipeline                                                          |
| `keep_intermediate_srt`     | bool  | false                         | Giữ `<output-name>_image_overlay_synced.srt` để debug timestamp đã remap                                  |
| `direct_overlay_max_events` | int   | 200                           | Ngưỡng event để `auto` chuyển từ direct filter graph sang script                                          |
| `command_line_max_chars`    | int   | 25000                         | Ngưỡng an toàn command-line, hữu ích trên Windows                                                         |
| `render_strategy`           | str   | `auto`                        | `auto`, `direct`, `script`; `intermediate` mới là stub và chưa được nối pipeline                          |

Khi `file_ext="auto"`, resolver tìm theo thứ tự ưu tiên: `.png`, `.jpg`, `.jpeg`, `.jfif`, `.gif`, `.webp`, `.bmp`, `.tif`, `.tiff`, `.avif`, `.heic`, `.heif`, `.jp2`, `.j2k`, `.jxl`, `.tga`, `.svg`, `.ico`. Text SRT vẫn luôn là basename không có extension. Nếu nhiều file cùng basename tồn tại, `missing_policy="warn"` chọn file theo thứ tự ưu tiên và log warning; `missing_policy="raise"` dừng pipeline để tránh chọn nhầm asset.

### 2.7 `note_overlay`

`note_overlay` sử dụng mode `dynamic_ass_box`: pipeline remap timestamp ASS, sau đó sinh một file ASS cuối gồm nền hộp bằng drawing (`NoteBox`) và text (`NoteText`). Không còn dùng PNG nền cố định; key legacy `png_path` chỉ được nhận diện để warning deprecation.

```json
{
  "note_overlay": {
    "enabled": true,
    "mode": "dynamic_ass_box",
    "default_layout": "top_left",
    "font": {
      "fontname": "Noto Sans CJK JP",
      "font_path": "assets/NotoSansCJKsc-VF.ttf",
      "font_size": 42,
      "line_spacing": 1.25,
      "primary_color": "&H00FFFFFF"
    },
    "layouts": {
      "top_left": {
        "anchor": "top_left",
        "margin_x": 80,
        "margin_y": 100,
        "width": 680,
        "height": 260,
        "padding_left": 32,
        "padding_right": 32,
        "padding_top": 28,
        "padding_bottom": 36,
        "height_safety_margin": 10,
        "background_color": "&HCC000000"
      },
      "bottom_right": {
        "anchor": "bottom_right",
        "margin_x": 80,
        "margin_y": 180,
        "width": 720,
        "height": 300,
        "padding_left": 32,
        "padding_right": 32,
        "padding_top": 28,
        "padding_bottom": 40,
        "height_safety_margin": 10,
        "background_color": "&HCC000000"
      }
    }
  }
}
```

| Key                                  | Type        | Default                       | Mô tả                                                                                      |
| ------------------------------------ | ----------- | ----------------------------- | ------------------------------------------------------------------------------------------ |
| `enabled`                            | bool        | false                         | Bật/tắt note overlay. Nếu bật nhưng không truyền `--note-overlay-ass`, pipeline tự bỏ qua. |
| `mode`                               | str         | `dynamic_ass_box`             | Mode chính thức. `png_legacy`/`png_path` chỉ còn warning deprecation.                      |
| `default_layout`                     | str         | `top_left`                    | Layout fallback khi ASS `Name` rỗng hoặc không khớp.                                       |
| `font.font_path`                     | str \| null | `assets/NotoSansCJKsc-VF.ttf` | Font dùng để đo pixel bằng Pillow; nếu load fail sẽ fallback heuristic.                    |
| `font.font_size`                     | int         | 42                            | Font size mặc định; preset có thể override.                                                |
| `font.line_spacing`                  | float       | 1.25                          | Line height tính bằng `font_size * line_spacing`.                                          |
| `layouts.<key>.anchor`               | str         | `top_left`                    | `top_left`, `top_right`, `bottom_left`, `bottom_right`, `center`.                          |
| `layouts.<key>.x/y`                  | int \| null | null                          | Nếu cả `x` và `y` có giá trị, dùng tọa độ tuyệt đối thay anchor.                           |
| `layouts.<key>.width`                | int         | 640                           | Chiều rộng box.                                                                            |
| `layouts.<key>.height`               | int         | 0                             | Chiều cao tối thiểu; box tự mở rộng nếu text dài hơn.                                      |
| `layouts.<key>.padding_*`            | int         | —                             | Padding text bên trong box.                                                                |
| `layouts.<key>.height_safety_margin` | int         | 10                            | Cộng thêm vào required height để hạn chế lệch giữa Pillow và libass.                       |
| `layouts.<key>.background_color`     | str         | `&HCC000000`                  | Màu nền ASS `&HAABBGGRR`.                                                                  |

Quy ước input:

- Với ASS nguồn: field `Name`/Actor trong mỗi `Dialogue` là layout key, ví dụ `Dialogue: ...,NoteStyle,bottom_right,...`.
- Với SRT nguồn: dùng `srt-to-ass`; dòng text đầu tiên của block nhiều dòng có thể là layout key. Dòng key không render ra video.
- Block SRT chỉ có một dòng text trùng layout key vẫn được coi là body text để tránh nuốt chữ.

Output khi có `--note-overlay-ass`:

- `<output-name>_note_overlay.ass` — file ASS cuối có `NoteBox` và `NoteText`.
- `<output-name>_note_synced.ass` — file trung gian chỉ được giữ khi bật `--keep-tmp` hoặc `note_overlay.keep_intermediate_ass=true`.

### 2.8 `audio_mix`

`audio_mix` là nơi cấu hình gain cho các layer audio trong Phase 3. TTS non-Voicevox và mute chunks được áp volume khi chuẩn hóa từng chunk; ambient/BGM vẫn chỉ áp gain ở final mix để tránh nhân volume hai lần.

```json
{
  "audio_mix": {
    "non_voicevox_tts_volume": 1.75,
    "mute_audio_volume": 1.0,
    "ambient_volume": 0.03,
    "bgm_volume": 1.0
  }
}
```

| Key                       | Type  | Default | Mô tả                                                                                                        |
| ------------------------- | ----- | ------- | ------------------------------------------------------------------------------------------------------------ |
| `non_voicevox_tts_volume` | float | `1.75`  | Gain cho TTS provider không bắt đầu bằng `voicevox` (ví dụ Edge/Qwen). Voicevox/Voicevox Nemo bỏ qua key này |
| `mute_audio_volume`       | float | `1.0`   | Gain cho audio của segment `mute` khi policy là `original`, `vocals` hoặc `instrumental`                     |
| `ambient_volume`          | float | `0.03`  | Gain cuối cùng cho ambient overlay ở final mix                                                               |
| `bgm_volume`              | float | `1.0`   | Gain cuối cùng cho global BGM overlay ở final mix                                                            |

> Voicevox family dùng `volume_scale` trong `config/tts_config.yaml`; render config không can thiệp vào volume của Voicevox/Voicevox Nemo.

### 2.9 `audio_policies`

`audio_policies` là schema chính thức mới để điều khiển audio behavior trong [`cli/sync_video.py`](cli/sync_video.py) và [`sync_engine/audio_assembler.py`](sync_engine/audio_assembler.py).

```json
{
  "audio_policies": {
    "global_bgm": "off",
    "mute_audio": "original",
    "ambient": "exclude_mute"
  }
}
```

| Key          | Giá trị hợp lệ                                  | Default        | Mô tả                                                                                                                                       |
| ------------ | ----------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `global_bgm` | `off`, `whole_video`, `exclude_mute`            | `off`          | Bật/tắt global BGM overlay. `whole_video` phát trên toàn timeline; `exclude_mute` vẫn build track đúng timing nhưng mute ở các đoạn `mute`. |
| `mute_audio` | `original`, `vocals`, `instrumental`, `silence` | `original`     | Quyết định audio của chính các segment `mute`: giữ audio gốc, chỉ vocals, chỉ instrumental, hoặc silence.                                   |
| `ambient`    | `off`, `whole_video`, `exclude_mute`            | `exclude_mute` | Điều khiển ambient overlay độc lập với mute audio policy.                                                                                   |

Ví dụ nhanh:

```json
{
  "audio_policies": {
    "global_bgm": "exclude_mute",
    "mute_audio": "original",
    "ambient": "whole_video"
  }
}
```

- `global_bgm=exclude_mute`: track BGM được sync theo timeline final rồi apply volume mask ở final mix.
- `mute_audio=instrumental`: nếu đã có [`bgm_path`](sync_engine/audio_assembler.py:513) từ global BGM extraction, runtime sẽ ưu tiên reuse track này cho mute chunks để tránh chạy separator trùng.
- `ambient=exclude_mute`: ambient preprocessing chỉ loop/trim + mask 0/1; volume cuối cùng vẫn lấy từ `audio_mix.ambient_volume`.

### 2.10 `audio_separator`

`audio_separator` giờ chỉ còn chứa tuning cho separator runtime. Hai key cũ `extract_bgm` và `extract_vocals` được xem là deprecated compatibility keys; nếu xuất hiện cùng `audio_policies`, runtime sẽ warning và ưu tiên `audio_policies`.

```json
{
  "audio_separator": {
    "mdxc_params": {
      "model": "MDX23C-InstVoc_HQ",
      "device": "cuda"
    }
  }
}
```

### 2.11 `video_encoding`

`sync-video` hiện ép toàn bộ lệnh FFmpeg render video trong Phase 2 và Phase 5 dùng HEVC NVENC cố định:

```bash
-c:v hevc_nvenc -preset p4 -tune hq -cq 28
```

Block `video_encoding` trong render config chỉ còn vai trò mô tả/tương thích cấu hình cũ; runtime không dùng block này để chọn codec hoặc fallback CPU.

```json
{
  "video_encoding": {
    "codec": "hevc_nvenc",
    "preset": "p4",
    "tune": "hq",
    "quality": ["-cq", "28"]
  }
}
```

### 2.12 `forced_alignment_subtitle`

```json
{
  "forced_alignment_subtitle": {
    "enabled": false,
    "model_path": null,
    "device": null,
    "dtype": null,
    "attn_implementation": null,
    "language": "English",
    "max_chars": 42,
    "min_chars": 0,
    "split_on_comma": true,
    "offset_seconds": 0.24,
    "keep_tts_synced_debug": false,
    "fail_policy": "warn",
    "batch_size": 16
  }
}
```

| Key                     | Type        | Default     | Mô tả                                                                               |
| ----------------------- | ----------- | ----------- | ----------------------------------------------------------------------------------- |
| `enabled`               | bool        | false       | Bật/tắt bước forced alignment subtitle sau Phase 3                                  |
| `model_path`            | str \| null | null        | Đường dẫn model HuggingFace hoặc local. null → `"Qwen/Qwen3-ForcedAligner-0.6B"`    |
| `device`                | str \| null | null        | Device map cho model. null → `"cuda:0"`                                             |
| `dtype`                 | str \| null | null        | Tên dtype (`"bfloat16"`, `"float16"`, `"float32"`). null → `torch.bfloat16`         |
| `attn_implementation`   | str \| null | null        | Attention implementation (vd `"sdpa"`). null → dùng default của model               |
| `language`              | str         | `"English"` | Ngôn ngữ cho forced alignment                                                       |
| `max_chars`             | int         | 42          | Số ký tự tối đa mỗi subtitle block. Nếu tổng chars ≤ max_chars → không ngắt         |
| `min_chars`             | int         | 0           | Số ký tự tối thiểu mỗi block. 0 = không có giới hạn tối thiểu                       |
| `split_on_comma`        | bool        | true        | Cho phép ngắt subtitle tại dấu phẩy                                                 |
| `offset_seconds`        | float       | 0.24        | Offset (giây) dịch chuyển timestamp subtitle so với audio                           |
| `keep_tts_synced_debug` | bool        | false       | Giữ file `<name>_tts_synced.srt` (remap) để debug, ngay cả khi alignment thành công |
| `fail_policy`           | str         | `"warn"`    | `warn` → fallback sang remap SRT; `raise`/`error`/`fail` → dừng pipeline            |
| `batch_size`            | int         | 16          | Số clip TTS align đồng thời trong một lần gọi `aligner.align()`. Giảm nếu OOM       |

**Lưu ý quan trọng về segmentation:**

- Nếu tổng số ký tự của cả câu ≤ `max_chars`, câu đó **không được ngắt** thành 2 block.
- `min_chars = 0` có nghĩa là không có giới hạn tối thiểu — cho phép block rất ngắn (vd chỉ chứa 1 từ).
- Khi `enabled = true` và alignment thành công, file `<output-name>_synced.srt` sẽ chứa kết quả forced alignment thay vì remap timestamp.
- Khi `enabled = true` nhưng alignment thất bại với `fail_policy = "warn"`, hệ thống tự động fallback sang `recalculate_srt()` (remap timestamp) như cũ.

**Cơ chế per-clip (từ phiên bản mới):**

Pipeline align **từng clip TTS `dubb-N.wav`** (đã sinh ở Phase 0) thay vì cả file `mixed_audio.wav`. Mỗi clip chỉ vài giây nên không bao giờ chạm giới hạn ~5 phút của `Qwen3ForcedAligner`, cho phép chạy video dài tùy ý. Dòng phụ đề trong vùng **mute** (không có clip TTS) tự động được remap timeline rồi gộp vào `_synced.srt` — không sót dòng nào. Giảm `batch_size` nếu gặp OOM khi align nhiều clip cùng lúc.

---

## 3. Cấu hình `image_overlay`

### 3.1 Input SRT và thư mục ảnh tĩnh

Bật `image_overlay.enabled=true` trong render config, sau đó truyền thêm hai CLI args:

```bash
uv run sync-video \
  --video content/video.mp4 \
  --subtitle content/subtitle.srt \
  --image-overlay-srt content/image_overlay.srt \
  --image-overlay-dir content/image_overlays \
  --render-config assets/default_render_config.json
```

Quy ước SRT:

```srt
1
00:00:01,000 --> 00:00:03,500
frame_intro

2
00:00:05,000 --> 00:00:08,000
callout_01
```

Với `file_ext="auto"`, pipeline sẽ resolve basename sang file ảnh tĩnh đầu tiên tìm được theo thứ tự ưu tiên. Ví dụ hợp lệ:

- `content/image_overlays/frame_intro.png`
- `content/image_overlays/callout_01.webp`
- `content/image_overlays/diagram.JPG`

Text SRT chỉ được dùng làm basename; không truyền extension, slash, backslash hoặc path traversal. Nếu nhiều dòng text, pipeline chỉ dùng dòng đầu tiên làm key và log warning. PNG vẫn là định dạng khuyến nghị khi cần transparent/alpha channel.

### 3.2 Timestamp remap

Image overlay SRT luôn bám theo timeline video gốc. Sau Phase 2, pipeline dùng cùng mapping stretch với subtitle để remap từng event. Vì vậy forced alignment subtitle không làm lệch overlay: forced alignment chỉ thay nội dung/timestamp subtitle hardsub, còn image overlay vẫn theo video stretch timeline.

Nếu bật `keep_intermediate_srt=true` hoặc truyền `--keep-tmp`, pipeline ghi thêm:

```text
<output-name>_image_overlay_synced.srt
```

File này giúp kiểm tra timestamp overlay sau remap.

### 3.3 Layer order và kích thước ảnh

Renderer burn layer theo thứ tự cố định:

```text
Base video → Image overlay static image full-screen → Note overlay → Black strip → Watermark → Subtitle
```

Phase đầu chỉ hỗ trợ `fit=stretch_to_output`. Nên export ảnh đúng kích thước output, ví dụ 1920x1080 hoặc 1080x1920. Nếu cần video bên dưới vẫn thấy được, nên dùng PNG/WebP có alpha channel. Nếu asset sai aspect ratio, FFmpeg vẫn scale về output resolution nên ảnh có thể bị kéo méo.

### 3.4 Strategy render

| Strategy       | Khi dùng                                                                                             |
| -------------- | ---------------------------------------------------------------------------------------------------- |
| `direct`       | Số event ít, filter graph ngắn; renderer truyền trực tiếp qua `-filter_complex`                      |
| `script`       | Số event lớn hoặc command-line quá dài; renderer ghi graph ra file và dùng `-filter_complex_script`  |
| `auto`         | Mặc định; tự chuyển sang `script` khi vượt `direct_overlay_max_events` hoặc `command_line_max_chars` |
| `intermediate` | Chưa triển khai; hiện chỉ có stub để dành cho phase tối ưu sau này                                   |

Renderer deduplicate static image theo absolute path. Nếu cùng một ảnh xuất hiện nhiều lần trong SRT, FFmpeg chỉ load asset đó một lần và dùng `split=N` để tái sử dụng cho nhiều event.

---

## 4. Cấu hình `forced_alignment_subtitle`

### 3.1 Tổng quan

Forced alignment subtitle sử dụng model `Qwen3ForcedAligner` để căn chỉnh chính xác từng từ trong transcript với timestamp audio. Kết quả thay thế SRT remap (recalculate) thông thường, cho phép subtitle đồng bộ chính xác với audio.

Pipeline align **từng clip TTS `dubb-N.wav`** (Phase 0 đã sinh, 1:1 mỗi dòng phụ đề) thay vì toàn bộ `mixed_audio.wav`. Lợi ích:

- Mỗi clip vài giây → không bao giờ OOM, chạy được video dài bất kỳ (model chỉ chịu ~5 phút/lần).
- Aligner nghe clip TTS sạch (không lẫn BGM/ambient) → chính xác hơn.
- Dòng phụ đề vùng **mute** (không có clip TTS) tự remap timeline rồi gộp vào SRT — không sót dòng.
- GPU được giải phóng hoàn toàn sau khi align (`try/finally clear_vram`) → không ảnh hưởng các bước NVENC phía sau.

**Điều kiện chạy:**

- `forced_alignment_subtitle.enabled = true` trong `render_config.json`
- Phase 0 (TTS) đã hoàn tất → các file `tts_clips/dubb-N.wav` tồn tại
- Phase 1 (Analysis) đã hoàn tất → `timeline` sẵn sàng để offset word timing

---

## 5. Cấu hình `llm_metadata`

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

### 5.1 Các key chính

| Key           | Type | Default                              | Mô tả                                                                               |
| ------------- | ---- | ------------------------------------ | ----------------------------------------------------------------------------------- |
| `enabled`     | bool | false                                | Bật/tắt bước tạo metadata sau render                                                |
| `task_config` | str  | "config/llm_tasks/seo_metadata.yaml" | YAML config cho generic LLM task                                                    |
| `fail_policy` | str  | "warn"                               | `warn` → log warning, không fail pipeline; `raise`/`error`/`fail` → raise exception |

### 5.2 Provider overrides

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

### 5.3 `input`

| Key                             | Type | Default                            | Mô tả                                      |
| ------------------------------- | ---- | ---------------------------------- | ------------------------------------------ |
| `write_debug_input`             | bool | false                              | Ghi raw text input ra file `.txt` để debug |
| `debug_input_filename_template` | str  | "{video_stem}\_metadata_input.txt" | Tên file debug input                       |

### 5.4 `output`

| Key                 | Type | Default                     | Mô tả                                                              |
| ------------------- | ---- | --------------------------- | ------------------------------------------------------------------ |
| `directory_policy`  | str  | "/"                         | `/` = thư mục chứa video input; hoặc đường dẫn tuyệt đối/tương đối |
| `filename_template` | str  | "{video_stem}\_metadata.md" | Tên file output metadata                                           |

### 5.5 Template variables

Các biến có thể dùng trong `filename_template` và `debug_input_filename_template`:

| Variable         | Ví dụ (video = `content/a/b.mp4`)               |
| ---------------- | ----------------------------------------------- |
| `{video_stem}`   | `b`                                             |
| `{video_name}`   | `b.mp4`                                         |
| `{video_suffix}` | `.mp4`                                          |
| `{output_name}`  | Giá trị `--output-name` (default: `video_stem`) |

---

## 6. Output paths

### 6.1 `directory_policy: "/"`

Thư mục output = thư mục chứa video input (KHÔNG phải filesystem root).

Ví dụ:

- Input video: `content/episodes/ep01.mp4`
- `filename_template`: `{video_stem}_metadata.md`
- → Output: `content/episodes/ep01_metadata.md`

### 6.2 `directory_policy` là đường dẫn

Nếu `directory_policy` không phải `"/"`, nó được xử lý như đường dẫn:

- Tuyệt đối: dùng trực tiếp
- Tương đối: resolve từ PROJECT_ROOT

---

## 7. Task-file override

Khi dùng `--task-file`, mỗi task JSON có thể chứa key `llm_metadata` để override cấu hình từ `render_config.json`.

### 7.1 Boolean override

```json
{
  "input": "video.mp4",
  "subtitle": "video.srt",
  "llm_metadata": false
}
```

→ Tắt LLM metadata cho task này (bất kể `render_config.json` bật hay không).

### 7.2 Object override (deep merge)

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

### 7.3 Image overlay task fields

Mỗi task batch có thể truyền overlay riêng:

```json
{
  "input": "video.mp4",
  "subtitle": "video.srt",
  "image_overlay_srt": "overlays/video_overlay.srt",
  "image_overlay_dir": "overlays/images",
  "output": "out/video_synced.mp4"
}
```

Hai field này chỉ có hiệu lực khi `image_overlay.enabled=true` trong render config. Nếu config tắt, CLI sẽ log info và bỏ qua overlay dù task có truyền đường dẫn.

---

## 8. Fail policy

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

## 9. Kiến trúc module

```
cli/sync_video.py                        ← Entrypoint CLI (pyproject.toml [project.scripts])
    ↓ import
sync_engine/image_overlay.py             ← Parse/resolve/remap image overlay SRT + static image events
sync_engine/forced_alignment_subtitle.py ← Orchestration forced alignment subtitle
    ↓ import
utils/asr_subtitle_utils.py              ← Shared ASR subtitle logic (merge punctuation, segment, write SRT)
utils/text_segmenter.py                  ← Smart segmentation algorithm
    ↓ import
sync_engine/llm_metadata.py             ← Orchestration LLM metadata
sync_engine/note_overlay_layout.py      ← Expand dynamic ASS note overlay
    ↓ import
llm_ai/task_runner.py                    ← Provider creation logic (dùng chung)
llm_ai/tasks/generic_text_task.py        ← Generic LLM text task runner
utils/srt_parser.py                      ← SRT parsing + write_segments_to_flat_text()
```

Nguyên tắc:

- `cli/` chỉ chứa entrypoint command được định nghĩa trong `pyproject.toml` `[project.scripts]`
- `sync_engine/` phụ thuộc vào `llm_ai/` và `utils/`, không phụ thuộc ngược vào `cli/`
- `utils/` là thư viện tiện ích thuần, không phụ thuộc vào các package khác — chứa logic ASR/subtitle dùng chung bởi `sync_engine/` và `cli/`
- `llm_ai/` là thư viện LLM độc lập, không phụ thuộc vào `cli/` hay `sync_engine/`
