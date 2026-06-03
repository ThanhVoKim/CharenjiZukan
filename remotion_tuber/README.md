# remotion_tuber

Renderer overlay **MotionPNGTuber** (Remotion) cho pipeline `sync_video`.
Chỉ render **PNG overlay alpha theo group** từ `group_manifest.json`. Pipeline
Python/FFmpeg vẫn là chính; subproject này không biết gì về TTS/stretch/audio.

## Vai trò trong flow
```
Python (sync_video) → dựng group base + group_manifest.json
   → gọi scripts/render-groups.ts  (bundle once → renderFrames/group)
   → overlay PNG alpha cho từng group
Python → FFmpeg composite overlay lên group base → concat → video_stretched_with_tuber.mp4
```

## Cài đặt
```bash
cd remotion_tuber
npm install            # Remotion sẽ tải Chrome headless ở lần render đầu
```

## Chuẩn bị asset (B1 + B2)
Copy asset vào `public/` + key nền màu đặc của body → transparent PNG sequence:
```bash
npm run prepare-assets                  # auto-dò màu nền từ 4 góc frame đầu rồi key
# tuỳ chọn:
npm run prepare-assets -- --color 0x08A702 --similarity 0.12 --blend 0.10   # ghi đè màu key
npm run prepare-assets -- --no-key      # bỏ chromakey (asset đã alpha / chỉ extract)
```
Kết quả: `public/pngtuber/<assetId>/{mouth_track.json, mouth/*.png, body-transparent/frame-NNN.png}`.

> Asset mặc định `loop_mouthless_h264.mp4` có **nền màu đặc** (green tự nhiên `~0x08A702`, KHÔNG phải
> green chuẩn `0x00FF00`). `prepare-assets` **tự dò màu nền** từ 4 góc frame đầu rồi chromakey → alpha,
> nên không cần truyền `--color`. Truyền `--color` khi nền không đồng nhất hoặc auto-dò sai.

## Render overlay
```bash
npm run render-groups -- test-manifests/group_synthetic.json
# nhiều group:
npm run render-groups -- /abs/group_0001/group_manifest.json /abs/group_0002/group_manifest.json
```
Output: `out/<groupId>/overlay_frames/frame_*.png` (hoặc `manifest.overlayDir` nếu có).
Mỗi group in 1 dòng `__TUBER_RENDER_RESULT__={...}` để Python parse; exit≠0 nếu fail.

## Debug bằng tay
```bash
npm run studio        # mở Remotion Studio (cần truyền inputProps để thấy nội dung)
```

## Hợp đồng inputProps (Python ↔ Remotion)
Driver truyền `{ manifest, mouthTrack, assetId }`:
- `manifest`: nội dung `group_manifest.json` (fps/width/height/groupStartFrame/segments/character...).
- `mouthTrack`: nội dung `mouth_track.json` của asset (driver đọc từ `public/`).
- `assetId`: tên thư mục asset trong `public/pngtuber/`.

`Root.tsx::calculateMetadata` lấy **động** `fps/width/height/durationInFrames` từ `manifest`
(B3/B4 — không hardcode 1920×1080/30). `staticFile()` resolve asset từ `public/` (B2).

## Cấu hình `character` (vị trí + kích thước)
`resolveCharacterBox` trong `TuberOverlay.tsx` suy ô character ra px. Ưu tiên px tường minh
(`left/top/width/height`), nếu không thì dùng ratio theo composition (`leftRatio/topRatio/widthRatio/heightRatio`).

**Kích thước — `width` ưu tiên (giữ tỉ lệ, không méo):**

| Manifest cho | width | height |
|---|---|---|
| có `width`/`widthRatio` | dùng giá trị đó | **= width / aspect** (height/heightRatio **bị bỏ qua**) |
| chỉ `height`/`heightRatio` | = height × aspect | dùng giá trị đó |
| không cho gì | = (0.6·H)×aspect | 0.6 · H |

`aspect = mouth_track.width / mouth_track.height` (asset hiện tại 1920/1080 = 16:9). Body render
`objectFit:'fill'` (kéo giãn lấp đầy ô) nên ô PHẢI đúng aspect để không méo — vì vậy **thường chỉ cần
đặt `width`**, `height` tự suy. Đặt cả hai thì `height` bị bỏ qua.

- `clipInset`: CSS `clipPath: inset(<clipInset>)` áp cho cả body + canvas. `"0px"` = không clip; giá
  trị khác (vd `"10px"`, `"5% 8%"`) = **che/cắt viền ngoài** ô (mask), KHÔNG resize nội dung.
- `positionJson`: **chưa dùng (V2)** — chỗ dành cho dữ liệu vị trí/transform theo thời gian; hiện để `null`, bị bỏ qua.

## Map quyết định plan
- **B1** green-screen→alpha: `scripts/prepare-assets.ts` + `body-transparent` + `<Img staticFile>`.
- **B2** asset trong public: `prepare-assets` copy vào `public/`, component dùng `staticFile`.
- **B3/B4** tổng quát res/fps: `calculateMetadata` đọc manifest.
- **B5** group base do Python dựng; subproject chỉ render overlay, Python composite.
- **M1** mouth mode `cue` (chưa đọc audio); amplitude/hybrid để V2.
