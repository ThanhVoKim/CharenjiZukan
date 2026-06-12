# Hướng dẫn Text Isolation — Lọc watermark/overlay khỏi OCR phụ đề

Tài liệu này hướng dẫn **cắt mẫu**, chạy `tools/calibrate_text_isolation.py`, và **đọc/dùng
các tham số** mà nó trả về cho `TextIsolationConfig`.

## 1. Tính năng giải quyết vấn đề gì

OCR phụ đề (`video-ocr`) nhận **mọi chữ** trong vùng ROI — kể cả **watermark / text overlay**
của creator. Khi overlay **di chuyển vào đè lên dải phụ đề**, lọc theo vùng/ngôn ngữ không tách
được.

Quan sát then chốt: watermark/overlay thường có **opacity < 70%** → pixel bị trộn với nền
(`hiển_thị = α·text + (1−α)·nền`, α < 0.7). Hệ quả **không phụ thuộc màu**:

- Tương phản / gradient bị nhân với α → biên mờ, cạnh yếu.
- Không có viền (stroke) tối sắc như phụ đề cứng.
- Màu bị blend lệch khỏi màu phụ đề gốc.

`text_isolator` dùng các đặc trưng này để **giữ glyph phụ đề đặc, xóa glyph watermark mờ** ở mức
từng connected-component, **trước khi** đưa ảnh vào OCR. Thuần OpenCV, deterministic, không model.

> **Mặc định TẮT.** Chỉ bật cho video có watermark text (`--isolate-text`). Video sạch chạy như cũ.

## 2. Quy tắc cắt mẫu (BẮT BUỘC)

Ngưỡng chỉ chính xác nếu mẫu phản ánh đúng pixel lúc chạy thật:

1. **Cắt ở ĐỘ PHÂN GIẢI GỐC của video (1:1 pixel).** Không screenshot phóng to, không resize.
   `min_contrast` và `min_component_area` phụ thuộc scale pixel → sai scale = sai ngưỡng.
2. **Ôm sát chữ nhưng chừa vài pixel nền xung quanh.** Cần nền để đo tương phản/viền. Crop khít
   100% ruột chữ sẽ mất tín hiệu.
3. **Lưu PNG (lossless).** Tránh JPG (artifact nén làm nhiễu phép đo).
4. **~5–10 ảnh mỗi thư mục.**

### Bỏ mẫu vào thư mục nào

| Loại crop | Thư mục | Vai trò |
| --- | --- | --- |
| Phụ đề sạch (trắng + màu) | `--subtitle-samples` | Đặc trưng cần **GIỮ** |
| **Phụ đề bị watermark ĐÈ CHỒNG** | `--subtitle-samples` | Ca khó nhất — xác nhận mask không ăn mất phụ đề ở vùng overlap |
| **Chỉ** watermark/overlay (không có phụ đề) | `--watermark-samples` | "Chữ ký pixel" cần **XÓA** |

## 3. Chạy script hiệu chỉnh

```bash
uv run python tools/calibrate_text_isolation.py \
    --subtitle-samples ./samples/subtitle/ \
    --watermark-samples ./samples/watermark/ \
    --subtitle-colors "white,#FFD700" \
    --out ./samples/text_isolation_config.json
```

Kết quả:
- `text_isolation_config.json` — đủ tham số `TextIsolationConfig`.
- `preview/<tên>_before_after.png` — ảnh ghép `[gốc | đã mask]` từng mẫu.
- Bảng tóm tắt console: tương phản 2 lớp + **độ tách bạch** + ngưỡng đề xuất.

> **Luôn soi ảnh preview**: phụ đề (kể cả dấu câu, nét mảnh chữ Hán) phải còn nguyên; watermark
> phải thành đen. Ngưỡng số chỉ là điểm khởi đầu — mắt người là trọng tài cuối.

### Định dạng màu cho `--subtitle-colors`

Nhiều màu ngăn bằng `,` hoặc `;`. Mỗi màu là: tên (`white`, `yellow`, `gold`...), hex (`#FFD700`,
`#FFF`), hoặc bộ ba RGB (`255:215:0`, `255-215-0`). Một bộ `r,g,b` đơn cũng nhận: `"255,255,255"`.

## 4. Đọc & dùng từng tham số

Năm tham số trong JSON, kèm **độ tin cậy** (mức nên tin con số script đề xuất):

### `min_contrast` (int) — **độ tin cậy: MẠNH (feature lõi)**
Ngưỡng morphological gradient. Component có **tương phản đỉnh < ngưỡng** bị coi là mờ → **xóa**.
Đây là cơ chế chính tách watermark mờ khỏi phụ đề đặc.
- **Tăng** → siết chặt hơn, xóa được watermark "đậm" hơn, nhưng có thể ăn vào phụ đề nét mảnh.
- **Giảm** → giữ nhiều hơn, an toàn cho phụ đề nhưng lọt watermark.
- Chỉ chỉnh khi preview cho thấy lọt/ăn nhầm. Tin con số đề xuất nếu **độ tách bạch TỐT**.

### `color_tolerance` (int) — **độ tin cậy: MẠNH**
Sai số khoảng cách màu (Lab) quanh `subtitle_colors`. Pixel lệch màu quá ngưỡng bị loại.
- **Tăng** → nhận dải màu rộng hơn (kể cả biến thể sáng/tối của màu phụ đề), nhưng dễ lọt overlay
  cùng tông.
- **Giảm** → kén màu hơn, loại overlay khác tông tốt hơn, nhưng có thể mất pixel rìa chữ.
- Nếu phụ đề có **nhiều màu**, liệt kê đủ trong `--subtitle-colors` thay vì nới `color_tolerance` to.

### `stroke_max_luminance` (int) — **độ tin cậy: KHÁ (cần phụ đề có viền)**
Pixel độ sáng ≤ ngưỡng này được coi là **viền tối**. Component phải có viền tối lân cận mới được giữ
(khi `require_stroke=True`).
- **Tăng** → coi nhiều pixel là viền hơn (nới lỏng) → giữ nhiều hơn.
- **Giảm** → chỉ viền rất tối mới tính → chặt hơn.
- Vô hiệu khi `require_stroke=False`.

### `require_stroke` (bool) — **độ tin cậy: KHUYẾN NGHỊ**
Bật/tắt kiểm tra viền. Script đề xuất `True` nếu mẫu phụ đề có tỉ lệ pixel tối đáng kể.
- Phụ đề **có viền đen** (đa số phim TQ) → để `True`, đây là lớp lọc mạnh.
- Phụ đề **không viền** (chữ phẳng) → đặt `False` (CLI: `--no-require-stroke`), nếu không sẽ xóa nhầm
  phụ đề.

### `min_component_area` (int) — **độ tin cậy: YẾU (chỉ diệt nhiễu)**
Diện tích (pixel) tối thiểu để giữ một component. **KHÔNG dùng để loại watermark** — watermark cùng
cỡ phụ đề, đặt cao sẽ **xóa luôn phụ đề và dấu câu**.
- Việc duy nhất: dọn đốm nhiễu 1–3px do nén/rìa mask.
- **Đặt nhỏ.** 1080p thường an toàn ở **5–15**; nghi ngờ thì để **0–5**. Sai về phía nhỏ.
- Nhiễu nhiều → **siết `min_contrast`** chứ đừng nâng cái này lên cao.
- Cảnh báo CJK: chữ Hán có nét mảnh + dấu câu (。、) rất nhỏ → ngưỡng quá tay sẽ cắt mất chúng.

### Đọc "độ tách bạch" trong báo cáo
Bảng console in `P10_phụ_đề − P90_watermark` cho tương phản:
- `> 20` → **TỐT** ✅: hai lớp tách rõ, tin ngưỡng đề xuất.
- `0 → 20` → **HẸP** ⚠️: tách được nhưng mong manh, kiểm preview kỹ.
- `≤ 0` → **CHỒNG LẤN** ❌: opacity không đủ phân biệt — cân nhắc thêm color gate chặt hơn, hoặc
  watermark này không tách được bằng opacity (cần hướng khác).

## 5. Áp dụng khi chạy `video-ocr`

### Cách A — nạp file JSON đã hiệu chỉnh (khuyến nghị)
```bash
uv run video-ocr video.mp4 \
    --isolate-text \
    --isolate-config ./samples/text_isolation_config.json \
    --subtitle-colors "white,#FFD700"
```

### Cách B — truyền tay từng tham số
```bash
uv run video-ocr video.mp4 \
    --isolate-text \
    --subtitle-colors "white,#FFD700" \
    --color-tolerance 40 \
    --subtitle-min-contrast 45 \
    --stroke-max-luminance 80 \
    --min-component-area 8
# Phụ đề không viền: thêm --no-require-stroke
```

### Cách C — YAML config
```yaml
text_isolation:
  enabled: true
  config_file: ./samples/text_isolation_config.json   # tùy chọn
  subtitle_colors: ["white", "#FFD700"]
  color_tolerance: 40
  min_contrast: 45
  stroke_max_luminance: 80
  min_component_area: 8
  require_stroke: true
```

Thứ tự ưu tiên: **CLI > YAML > JSON-calibrate > default**.

## 6. Lưu ý hiệu năng & thiết kế

- Mask chạy trên crop ROI nhỏ → cỡ mili-giây, **< 1%** so với OCR GPU.
- Mask chạy **trước** CV prefilter: frame chỉ-watermark → ROI trống → prefilter skip OCR →
  **giảm số lần gọi model**. Tổng thể thường **nhanh hơn**.
- Scene detection vẫn chạy trên ROI **gốc** (không mask) để bắt đúng thời điểm phụ đề đổi.

## 7. Giới hạn đã biết

- **Watermark đặc, đục (opacity cao), cùng màu, đè khít dòng phụ đề**: opacity masking không tách
  được (độ tách bạch ❌). Cần hướng khác (text detection + lọc hình học/tracking vị trí) — ngoài
  phạm vi hiện tại.
- Tham số phụ thuộc layout từng creator → nên hiệu chỉnh lại khi đổi nguồn video.
