# Hướng dẫn sử dụng Tuber Overlay (Pre-render mode)

Dùng `--tuber-config` trong `sync-video` để kích hoạt overlay nhân vật ảo PNGTuber
lên video output.

**V2 — Pre-render mode (default):** Character frames được bake sẵn bằng Python/PIL
(port thuật toán warp từ `remotion_tuber/`). Runtime thuần Python + FFmpeg, nhanh
hơn Remotion ~100×, phù hợp video dài 2-3h.

**V3 (hiện tại):** Bỏ `build_group_base` — composite seek trực tiếp vào `video_stretched.mp4`
(giảm 8→5 encode). Song song prerender + group composite (`performance.maxWorkers`).
Resume skip-done bằng hash (`resume.skipDone`). Mode `"hybrid"` cho miệng mượt hơn.

**V1 — Remotion mode (legacy):** Giữ nguyên code tham khảo `remotion_tuber/`,
nhưng runtime ưu tiên pre-render.

## Kiến trúc (V2 — Pre-render)

```
sync_video (Python)
  → tuber_config: load & validate config JSON
  → tuber_manifest: build render groups từ timeline, export manifest + mouthEvents
  → tuber_prerender: pre-render body×mouth frames (1 lần, offline)
  → tuber_mouth_events: analyze TTS audio → mouthEvents per segment
  → tuber_overlay: copy pre-rendered frames → composite overlay (FFmpeg) → concat
  → render_final_video (với video đã có tuber overlay)
```

**V1 (Remotion) — fallback khi chưa pre-render:**
```
  → prepare_assets (npm chromakey body)
  → render-groups (npm bundle Remotion → renderFrames → PNG)
  → composite overlay (FFmpeg) → concat
```

> ⚠️ **Cập nhật đồng bộ tài liệu:** Mọi thay đổi về cấu hình JSON, tham số CLI, flow pipeline hay quy tắc mới (ví dụ thêm key config, sửa retry logic, đổi output structure...) phải được cập nhật đồng thời trong cả hai file: [`docs/tuber-overlay-guide.md](docs/tuber-overlay-guide.md) (hướng dẫn sử dụng) và [`remotion_tuber/README.md`](remotion_tuber/README.md) (tài liệu kỹ thuật subproject). Config là mặt phân cách (interface) giữa Python và TypeScript — nếu chỉ sửa một bên, tài liệu sẽ mâu thuẫn, gây nhầm lẫn cho người đọc hoặc Agent sau này.

## Cách dùng nhanh

```bash
# Bật tuber overlay
uv run sync-video \
    --video /content/video.mp4 \
    --subtitle /content/subtitle_ja.srt \
    --tuber-config assets/tuber_overlay_config.json
```

```bash
# Late repair: render tuber muộn sau khi fallback non-tuber
uv run tuber-repair --tuber-root tuber-output/<job>/tuber
```

## Cấu trúc thư mục output

```
tuber-output/<job>/tuber/
├── run_manifest.json          # Runtime manifest cho repair
├── media/
│   ├── base_video_stretched.mp4
│   ├── final_audio_mixed.wav
│   └── video_stretched_with_tuber.mp4
├── groups/
│   ├── group_0001/
│   │   ├── group_manifest.json
│   │   ├── status.json           # done/failed/skipped + inputHash (V3)
│   │   ├── overlay_frames/       # (tạm, bị xóa nếu artifactPolicy.overlayFrames=safe)
│   │   └── video_with_tuber.mp4  # V3: seek từ video_stretched (không còn base.mp4)
│   └── group_0002/...
├── final_render_inputs/          # (chỉ khi artifactPolicy.mode=repairable)
│   ├── final_render_manifest.json
│   ├── subtitle_synced.srt
│   ├── render_config.json
│   └── ...
└── logs/
    ├── render_driver.log
    ├── prepare_assets.log
    └── debug_frames/             # (chỉ khi debug.frameOutput.enabled=true)
        └── group_0001/
            ├── overlay_000000.png     # N frame đầu group
            ├── overlay_000001.png
            ├── composited_000000.png  # tương ứng trong video output
            ├── composited_000001.png
            └── boundary.json         # metadata: groupStartFrame, fps, margin
```

---

## Tham số JSON config (`tuber_overlay_config.json`)

### `enabled`

- **Kiểu:** `boolean`
- **Mặc định:** `false`
- Bật/tắt toàn bộ tuber overlay. `false` → `sync-video` chạy như cũ, không overlay.

### `outputDir`

- **Kiểu:** `string`
- **Mặc định:** `"tuber-output"`
- Thư mục gốc chứa output tuber. Path tương đối tính từ project root.

### `remotion` — cấu hình Remotion subproject

> **Chỉ bắt buộc khi `overlay.mode` = `"remotion"` hoặc `"auto"`.**
> Khi `overlay.mode = "prerender"`, section này có thể bỏ qua.

| Key             | Kiểu     | Mặc định                     | Mô tả                                                    |
| --------------- | -------- | ---------------------------- | -------------------------------------------------------- |
| `projectDir`    | `string` | `"remotion_tuber"`           | Đường dẫn subproject Remotion. Bắt buộc (nếu dùng Remotion). |
| `compositionId` | `string` | `"TuberOverlay"`             | Composition ID trong Root.tsx.                           |
| `entryPoint`    | `string` | `"src/index.ts"`             | Entry point cho Remotion bundle.                         |
| `renderDriver`  | `string` | `"scripts/render-groups.ts"` | Script render driver (bundle once → renderFrames/group). |

### `asset` — asset PNGTuber

| Key            | Kiểu     | Mặc định                          | Mô tả                                                                        |
| -------------- | -------- | --------------------------------- | ---------------------------------------------------------------------------- |
| `assetDir`     | `string` | `"assets/pngtuber/nike_loop_fix"` | Thư mục asset chứa `mouth_track.json`, mouth sprites, body source. Bắt buộc. |
| `assetId`      | `string` | (tên thư mục asset)               | ID dùng trong `public/pngtuber/<id>/`. Bỏ trống để lấy từ `assetDir`.        |
| `mouthTrack`   | `string` | `"mouth_track.json"`              | File tracking mouth quad (theo frame). Bắt buộc.                             |
| `mouthSprites` | `object` | `{closed, half, open}`            | Path tới 3 sprite mouth PNG (relative từ `assetDir`). Bắt buộc.              |
| `bodySource`   | `string` | `"loop_mouthless_h264.mp4"`       | Video body loop không miệng, nền màu đặc. Bắt buộc.                          |
| `chromakey`    | `object` | `{}`                              | Tham số chromakey cho body source.                                           |

#### `asset.prerender` — pre-render character frames

| Key            | Kiểu     | Mặc định | Mô tả                                                                                                                      |
| -------------- | -------- | -------- | -------------------------------------------------------------------------------------------------------------------------- |
| `characterDir` | `string` | `null`   | Thư mục chứa pre-rendered frames + `prerender_manifest.json`. `null` → dùng `assetDir/prerendered` (tạo tự động nếu thiếu). |

#### `asset.chromakey` — tham số chromakey

| Key          | Kiểu     | Mặc định      | Mô tả                                                                            |
| ------------ | -------- | ------------- | -------------------------------------------------------------------------------- |
| `color`      | `string` | (auto-detect) | Màu nền cần key, dạng `0xRRGGBB`. Bỏ trống → auto-dò từ 4 góc frame đầu.         |
| `similarity` | `number` | `0.10`        | FFmpeg `chromakey` similarity (0–1). Tăng để bắt nhiều sắc thái của màu nền hơn. |
| `blend`      | `number` | `0.10`        | FFmpeg `chromakey` blend (0–1). Tăng để làm mềm viền alpha.                      |

> Máy hiện tại (asset nike_loop_fix): nền green `~0x08A702`, nên đặt `color: "0x08A702"`, `similarity: 0.12`, `blend: 0.1`.

### `character` — vị trí & kích thước nhân vật

| Key            | Kiểu     | Mặc định       | Mô tả                                                                                                   |
| -------------- | -------- | -------------- | ------------------------------------------------------------------------------------------------------- |
| `left`         | `number` | (ratio)        | Pixel từ trái khung hình. Không đặt → dùng `leftRatio`.                                                 |
| `top`          | `number` | (ratio)        | Pixel từ trên khung hình. Không đặt → dùng `topRatio`.                                                  |
| `width`        | `number` | (ratio)        | Chiều rộng ô character (px). **Width ưu tiên**: height tự suy = width / aspect để giữ tỉ lệ, không méo. |
| `height`       | `number` | (suy từ width) | Chiều cao ô character (px). **Chỉ dùng khi không có `width`**.                                          |
| `leftRatio`    | `number` | `0.6`          | Vị trí trái theo tỉ lệ (0–1) × composition width.                                                       |
| `topRatio`     | `number` | `0.3`          | Vị trí trên theo tỉ lệ (0–1) × composition height.                                                      |
| `widthRatio`   | `number` | (không dùng)   | Width theo tỉ lệ × composition width.                                                                   |
| `heightRatio`  | `number` | `0.6`          | Height theo tỉ lệ × composition height. Chỉ dùng khi không có `width`/`widthRatio`.                     |
| `clipInset`    | `string` | `"0px"`        | CSS `inset()` để cắt viền ngoài ô (mask). `"0px"` = không clip.                                         |
| `positionJson` | `null`   | `null`         | **Chưa dùng (V2)** — chỗ dành cho dữ liệu vị trí/transform theo thời gian.                              |

**Quy tắc kích thước (V1):**

| Manifest cho               | width                | height                                              |
| -------------------------- | -------------------- | --------------------------------------------------- |
| có `width`/`widthRatio`    | dùng giá trị đó      | **= width / aspect** (height/heightRatio bị bỏ qua) |
| chỉ `height`/`heightRatio` | = height × aspect    | dùng giá trị đó                                     |
| không cho gì               | = (0.6 · H) × aspect | 0.6 · H                                             |

`aspect = mouth_track.width / mouth_track.height` (mặc định 1920/1080 = 16:9). Thường chỉ cần đặt `width`, height tự suy.

### `grouping` — chia group render

| Key           | Kiểu     | Mặc định | Mô tả                                                                 |
| ------------- | -------- | -------- | --------------------------------------------------------------------- |
| `maxGroupSec` | `number` | `300`    | Duration tối đa 1 group (giây). Group không bao giờ cắt giữa segment. |
| `paddingSec`  | `number` | `0`      | **Không dùng ở V1** (padding = 0).                                    |

### `mouth` — chế độ miệng

| Key            | Kiểu      | Mặc định      | Mô tả                                                                                                                                                                     |
| -------------- | --------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mode`         | `string`  | `"cue"`       | `"cue"` V1 — dùng `hasTts`. `"amplitude"` V2 — RMS per frame → state. `"hybrid"` V3 — RMS gate (nói/im) + debounce bằng `cadenceMs` để ổn định trạng thái half↔open. |
| `silenceDb`    | `number`  | `-40.0`       | Ngưỡng dB coi là im lặng. Dưới ngưỡng → `closed`.                                                                                                                        |
| `minSilenceMs` | `number`  | `200`         | Bỏ qua khoảng im lặng ngắn hơn (tránh miệng nhấp nháy).                                                                                                                  |
| `cadenceMs`    | `number`  | `150`         | **Chỉ dùng ở mode `"hybrid"`**: debounce — trạng thái non-closed (half↔open) phải giữ tối thiểu `cadenceMs` ms trước khi chuyển tiếp. Chuyển về `closed` (silence) không bị giới hạn. |
| `mouthStates`  | `[string]`| `[...3]`      | Danh sách mouth states theo thứ tự từ nhỏ đến lớn (mở rộng được).                                                                                                        |

#### So sánh `mouth.mode`

| Mode          | Cách hoạt động                                                                                             | Khi nào dùng                                                      |
| ------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `"cue"`       | V1 legacy: `hasTts=true` → miệng mở suốt segment (kể cả silent margin)                                    | Backward compat, không dùng TTS WAV                               |
| `"amplitude"` | Phân tích RMS per frame → tự do chuyển state mỗi frame; có thể nhấp nháy khi biên độ dao động nhanh       | Khi muốn khẩu hình bám sát audio thật (nhiễu được chấp nhận)     |
| `"hybrid"`    | RMS gate xác định đoạn nói/im; trong đoạn nói dùng debounce `cadenceMs` → ổn định hơn, ít nhấp nháy hơn  | **Khuyến nghị** cho PNGTuber production; cân bằng tự nhiên/mượt  |

### `overlay` — format overlay và mode render

| Key      | Kiểu     | Mặc định         | Mô tả                                                                                                     |
| -------- | -------- | ---------------- | --------------------------------------------------------------------------------------------------------- |
| `format` | `string` | `"png_sequence"` | `"png_sequence"` — PNG alpha sequence từng frame.                                                         |
| `mode`   | `string` | `"auto"`         | Kiểm soát engine render: `"remotion"` / `"prerender"` / `"auto"`. Xem bảng dưới.                          |

#### `overlay.mode` — lựa chọn engine render

| Giá trị       | Hành vi                                                                                                       |
| ------------- | ------------------------------------------------------------------------------------------------------------- |
| `"remotion"`  | Luôn dùng Remotion (Node + Chromium). Section `remotion` **bắt buộc**.                                        |
| `"prerender"` | Luôn dùng pre-render Python/PIL. Nếu `prerender_manifest.json` chưa có → **tự động chạy** `prerender_character()`. Section `remotion` **không cần**. |
| `"auto"`      | Auto-detect: nếu `prerender_manifest.json` tồn tại → pre-render; ngược lại → Remotion.                        |

### `artifactPolicy` — chính sách giữ/xóa artifact

| Key                 | Kiểu     | Mặc định       | Mô tả                                                                                   |
| ------------------- | -------- | -------------- | --------------------------------------------------------------------------------------- |
| `mode`              | `string` | `"repairable"` | `"repairable"` — giữ đủ artifact để repair sau fallback.                                |
| `overlayFrames`     | `string` | `"safe"`       | `"safe"` — xóa overlay frames sau khi composite+validate OK. `"keep"` — giữ.            |
| `finalRenderInputs` | `string` | `"keep"`       | `"keep"` — giữ `final_render_inputs/` để repair. `"delete"` — không hỗ trợ repair muộn. |
| `logs`              | `string` | `"keep"`       | `"keep"` — giữ log render driver/ffmpeg.                                                |
| `failedGroups`      | `string` | `"keep"`       | `"keep"` — giữ artifact group fail để debug. `"delete"` — xóa sau fallback.             |

### `repair` — cấu hình repair

| Key                   | Kiểu     | Mặc định        | Mô tả                                                        |
| --------------------- | -------- | --------------- | ------------------------------------------------------------ |
| `defaultOutputSuffix` | `string` | `"_with_tuber"` | Suffix cho output repair (vd `video_synced_with_tuber.mp4`). |

### `retry` — chính sách retry

| Key             | Kiểu     | Mặc định                 | Mô tả                                                                                                    |
| --------------- | -------- | ------------------------ | -------------------------------------------------------------------------------------------------------- |
| `retryAttempts` | `number` | `3`                      | Số lần retry mỗi group (render + composite + validate).                                                  |
| `onExhausted`   | `string` | `"render_without_tuber"` | Hành vi khi hết retry: chỉ hỗ trợ `"render_without_tuber"` — fallback render final video không có tuber. |

### `performance` — tăng tốc song song (V3)

| Key          | Kiểu     | Mặc định | Mô tả                                                                                          |
| ------------ | -------- | -------- | ---------------------------------------------------------------------------------------------- |
| `maxWorkers` | `number` | `2`      | Số worker song song cho prerender_character (bake body×mouth) và composite groups. NVENC thường giới hạn ~2-3 session. |

### `resume` — skip-done / re-render (V3)

| Key        | Kiểu      | Mặc định | Mô tả                                                                                                                                   |
| ---------- | --------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `skipDone` | `boolean` | `true`   | `true`: so hash input → skip group done + hash khớp + output hợp lệ (resume save). `false`: xóa sạch groups/ + prerendered/, dựng lại. |

### `debug` — debug frame output (V3)

Config: `debug.frameOutput.enabled` / `debug.frameOutput.marginFrames`.

| Key                          | Kiểu      | Mặc định | Mô tả                                                                                         |
| ---------------------------- | --------- | -------- | --------------------------------------------------------------------------------------------- |
| `frameOutput.enabled`        | `boolean` | `false`  | Dump overlay + composited frames quanh boundary vào `logs/debug_frames/`. Chi phí 0 khi false. |
| `frameOutput.marginFrames`   | `number`  | `3`      | Số frame ở **start** và **end** của mỗi group cần dump (tổng = 2 × marginFrames / group).    |

**Output khi enabled:** `logs/debug_frames/{group_id}/overlay_{n}.png`, `composited_{n}.png`, `boundary.json` (groupStartFrame, fps, margin). Dùng để soi lệch frame tại điểm nối giữa các group.

### `validation`

| Key                    | Kiểu      | Mặc định | Mô tả                                                              |
| ---------------------- | --------- | -------- | ------------------------------------------------------------------ |
| `validateDoneOutputs`  | `boolean` | `true`   | Validate group output (file size, duration) trước khi coi là done. |
| `durationToleranceSec` | `number`  | `0.1`    | Dung sai duration (giây) khi validate.                             |
| `minOutputBytes`       | `number`  | `1024`   | File size tối thiểu để coi output hợp lệ.                          |

---

## Config mẫu đầy đủ

### Prerender mode (khuyến nghị — không cần Remotion/Node)

```json
{
  "enabled": true,
  "outputDir": "tuber-output",
  "asset": {
    "assetDir": "assets/pngtuber/nike_loop_fix",
    "assetId": "nike_loop_fix",
    "mouthTrack": "mouth_track.json",
    "mouthSprites": {
      "closed": "mouth/closed.png",
      "half": "mouth/half.png",
      "open": "mouth/open.png"
    },
    "bodySource": "loop_mouthless_h264.mp4",
    "chromakey": {
      "color": "0x08A702",
      "similarity": 0.12,
      "blend": 0.1
    },
    "prerender": {
      "characterDir": null
    }
  },
  "character": {
    "left": 1280,
    "top": 360,
    "width": 512,
    "clipInset": "0px"
  },
  "grouping": {
    "maxGroupSec": 300,
    "paddingSec": 0
  },
  "mouth": {
    "mode": "hybrid",
    "silenceDb": -40.0,
    "minSilenceMs": 200,
    "cadenceMs": 150,
    "mouthStates": ["closed", "half", "open"]
  },
  "overlay": {
    "format": "png_sequence",
    "mode": "prerender"
  },
  "artifactPolicy": {
    "mode": "repairable"
  },
  "repair": {
    "defaultOutputSuffix": "_with_tuber"
  },
  "retry": {
    "retryAttempts": 3,
    "onExhausted": "render_without_tuber"
  },
  "performance": {
    "maxWorkers": 2
  },
  "resume": {
    "skipDone": true
  },
  "debug": {
    "frameOutput": {
      "enabled": false,
      "marginFrames": 3
    }
  }
}
```

### Remotion mode (legacy — cần Node.js + Remotion)

```json
{
  "enabled": true,
  "outputDir": "tuber-output",
  "remotion": {
    "projectDir": "remotion_tuber",
    "compositionId": "TuberOverlay",
    "entryPoint": "src/index.ts",
    "renderDriver": "scripts/render-groups.ts"
  },
  "asset": {
    "assetDir": "assets/pngtuber/nike_loop_fix",
    "mouthTrack": "mouth_track.json",
    "mouthSprites": {
      "closed": "mouth/closed.png",
      "half": "mouth/half.png",
      "open": "mouth/open.png"
    },
    "bodySource": "loop_mouthless_h264.mp4",
    "chromakey": { "color": "0x08A702", "similarity": 0.12, "blend": 0.1 }
  },
  "character": { "left": 1280, "top": 360, "width": 512, "clipInset": "0px" },
  "grouping": { "maxGroupSec": 300, "paddingSec": 0 },
  "mouth": { "mode": "cue" },
  "overlay": { "format": "png_sequence", "mode": "remotion" },
  "artifactPolicy": { "mode": "repairable" },
  "repair": { "defaultOutputSuffix": "_with_tuber" },
  "retry": { "retryAttempts": 3, "onExhausted": "render_without_tuber" }
}
```

> **Pre-render auto-run:** Khi `overlay.mode = "prerender"` và `prerender_manifest.json`
> chưa tồn tại, pipeline **tự động chạy** `prerender_character()` (cần `body-transparent/`
> đã qua chromakey, `mouth/`, `mouth_track.json`). Lần sau có manifest → dùng luôn, skip pre-render.

## CLI `tuber-repair`

Sau khi `sync-video` chạy với `artifactPolicy.mode=repairable` và tuber fail → fallback render non-tuber,
có thể chạy repair để render tuber overlay muộn:

```bash
uv run tuber-repair --tuber-root tuber-output/<job>/tuber
# output: tuber-output/<job>/video_synced_with_tuber.mp4
```

Tham số:

| Tham số        | Mô tả                                                                    |
| -------------- | ------------------------------------------------------------------------ |
| `--tuber-root` | **(bắt buộc)** Đường dẫn `tuberRoot` (thư mục chứa `run_manifest.json`). |
| `--output`     | Override path final output (mặc định: `<job>/<name>_with_tuber.mp4`).    |

Input bắt buộc trong `tuberRoot`:

- `run_manifest.json`
- `media/base_video_stretched.mp4`
- `media/final_audio_mixed.wav`
- `final_render_inputs/final_render_manifest.json`

Input tùy chọn (nếu có ở all-in run):

- `final_render_inputs/subtitle_synced.srt`
- `final_render_inputs/note_overlay_final.ass`
- `final_render_inputs/image_overlay_events.json`
- `final_render_inputs/render_config.json`

---

## Retry logic

Retry áp dụng cho **mỗi group**, trong cùng lần chạy `sync-video`. Logic vòng lặp:

```
với mỗi group, attempt = 0 → retryAttempts:
  1. render overlay (batch fail → re-render riêng group đó, không tốn attempt)
  2. composite FFmpeg (overlay lên base, encode HEVC NVENC)
  3. validate (file size, duration)
  → thành công: thoát vòng lặp
  → fail: attempt += 1, lặp lại từ (1)
sau khi tất cả group pass:
  4. concat copy group video → video_stretched_with_tuber.mp4
```

- **Concatenation KHÔNG nằm trong retry** — nó chạy 1 lần sau khi tất cả group đã pass validate.
- `attempt` chỉ tăng khi composite hoặc validate raise exception. Batch render fail → re-render riêng là "soft fallback", không tính là retry.
- Hết `retryAttempts` → `TuberOverlayError` → fallback `render_without_tuber` (final video KHÔNG có tuber, pipeline chính không bị ảnh hưởng).
