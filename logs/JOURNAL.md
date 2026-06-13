# Project Journal

## 2026-06-13: Tối ưu token LLM — context cache đo được + thinking_level per-task + Responses anchor-fork

### Tóm tắt

Giải bài toán "đốt token" khi `use_full_context: true` cho video dài (hàng nghìn block). Kết luận
nghiên cứu: **Agent Engine Sessions KHÔNG giúp** (gửi lại history mỗi lượt). C��ng cụ đúng là **context
caching** — vốn đã có cho Vertex nhưng (1) translation dùng `provider_chain` nên bị vô hiệu, (2) không
có telemetry để biết cache có hit. Đồng thời thêm điều khiển `thinking_level` (Gemini 3) per-task.

### Thay đổi

- **`llm_ai/providers/vertexai.py`**: `_build_generate_config` pop `thinking_level` (low/medium/high) →
  `ThinkingConfig`, loại `thinking_budget` để tránh lỗi 400 (Gemini 3). Thêm `_capture_telemetry` đọc
  `usage_metadata` (`cached_content_token_count`...) → `last_telemetry_record`.
- **`llm_ai/provider_chain.py`**: `FallbackLLMProvider` giờ **tự quản global context**.
  `set_global_context` trả `True` (caller bỏ inline); `_ensure_context_on_active()` set context cho
  provider khi active (cache nếu được, prepend inline nếu không) — re-apply khi fallover.
  `apply_provider_chain_entry_overrides` deep-merge `generation_config` (thay vì replace).
- **`llm_ai/providers/openai.py`**: thêm `set_global_context` cho Responses-API profile → tạo **anchor
  R0** (`store: true`) gửi full context 1 lần; mọi batch **fork từ R0 cố định** (biến mới
  `_anchor_response_id`, KHÔNG sequential-chaining để tránh phình token). Error-recovery: R0 invalid →
  tạo lại anchor 1 lần. Nhánh chat_completions giữ nguyên.
- **`llm_ai/task_runner.py`**: `apply_provider_overrides` deep-merge `generation_config` từ task config.
- **`translation/batching.py`**: thêm `CacheTelemetryAccumulator` cộng dồn token qua batch.
- **`translation/srt_translator.py`** + **`punctuation/srt_punctuator.py`**: ghi nhận telemetry mỗi
  batch, log 1 dòng tổng kết `🧠 Cache: X/Y prompt tokens hit (Z%)`.
- **Configs**: `srt_translation.yaml` đảo chain → **Vertex primary** (thinking_level: low) +
  **freemodel_dev fallback** (Responses API). `punctuation_restoration.yaml` thêm `thinking_level: low`.
- **Tests**: `tests/translation/test_translation_providers.py` — sửa import stale
  (`translation.vertexai_provider` → `llm_ai.providers.vertexai`); thêm Layer1 (telemetry accumulator,
  thinking_level, chain-aware context) + Layer3 (Responses anchor fork-from-R0). Matrix entries hiện có
  đã bao qua keyword Layer1/Layer3.

### Lý do

OpenAI prompt cache là _giảm giá best-effort_ (token vẫn bị tính, evict 5-10 phút idle) — KHÁC Vertex
`CachedContent` (bỏ token khỏi prompt, ~90% off, TTL deterministic). Anchor-fork: cache discount hết
hạn KHÔNG cần gửi lại full_context — fork cùng R0 tự re-warm; tạo R0 mới là strictly worse. Vì vậy
chọn Vertex làm primary cho translation (tiết kiệm chắc chắn), freemodel_dev Responses làm fallback.
`cached_tokens` telemetry chỉ để **quan sát**, không phải trigger gửi lại context.

### Pending / Next

- E2E trên 1 video dài thật: xác nhận `cached_content_token_count`/`cached_tokens` > 0 và so token
  Console Billing trước/sau.
- Tuỳ chọn follow-up: dùng `prompt_cache_key` ổn định cho OpenAI chat_completions profile khi backend
  hỗ trợ (`supports_prompt_cache_key`).
- `google-api-core` + `openai` đã được cài vào `.venv` local để chạy test (CPU-only, an toàn).

---

## 2026-06-13: Dọn config + môi trường flow OCR (align-srt CLI thuần, dedup punctuation, .venv chính)

### Tóm tắt

Refactor theo review người dùng — 3 điểm: (1) `align-srt` đứng riêng nên **bỏ đọc YAML config**, dùng
**CLI args thuần + `--task-file`** (đúng idiom repo); (2) **dedup** config punctuation — giữ
`config/llm_tasks/punctuation_restoration.yaml` làm **SSOT**, bỏ key trùng khỏi config video-ocr;
(3) `align-srt` chạy ở **`.venv` chính** như `sync-video` (cần `qwen-asr` + `audio-separator` cài ở
`.venv` chính — cùng bộ deps forced-align của sync-video; KHÔNG cần `flash-attn`).

### Thay đổi

- **`cli/align_srt.py`**: xoá `_load_yaml` + arg `--config`; knob forced-alignment thành **default của
  argparse** (`_ALIGN_DEFAULTS`); `resolve_align_cfg` build từ args; vocal-sep preset qua
  `--separator-preset` (default `vocal_extraction`), bỏ section `vocal_separation:`. Toggle tách vẫn
  `--no-separate`/`--vocals`.
- **`cli/video_ocr.py`**: thêm 2 CLI args `--punctuate` (mặc định TẮT) + `--punctuation-task-config`;
  `run_punctuation_phase(output_paths, args, config, format)` đọc enabled/task_config từ CLI
  (CLI>YAML>default), mọi tham số LLM (language/batch_size/use_full_context/provider) lấy **chỉ từ
  task_cfg**. Bỏ các key trùng trong section `punctuation:`.
- **`docs/colab-guide.md`**: Mục 2.0c — `align-srt` chạy `!uv run align-srt` ở `.venv` chính (gỡ
  `.venv-qwen3asr` + cài audio-separator vào venv đó); gỡ `forced_alignment:`/`vocal_separation:` khỏi
  `flow.yaml`; cập nhật bảng tham số (bỏ `--config`, thêm `--separator-preset`, `--no-split-on-comma`,
  `--punctuate`/`--punctuation-task-config`).

### Lý do

align-srt là CLI độc lập, repo không có master-config-per-CLI → CLI args đủ và nhất quán. Punctuation
config trùng lặp → SSOT = llm_tasks yaml (như `srt_translation.yaml`). align-srt dùng chung lõi
`utils/forced_aligner.py` (`from qwen_asr import Qwen3ForcedAligner`, `attn_implementation=None`) nên
**cần `qwen-asr`** (không cần `flash-attn` — aligner không gọi flash_attention_2) — y hệt forced-align
của sync-video; chọn chạy ở `.venv` chính (cài qwen-asr + audio-separator ở đó) cho nhất quán với
sync-video, thay vì `.venv-qwen3asr`.

---

## 2026-06-13: OCR-centric source SRT — Punctuation (LLM) + Forced-alignment timing

### Tóm tắt

Tách trách nhiệm tạo SRT nguồn: **text từ OCR** (ground-truth), **dấu câu từ LLM** (vertexAI),
**timing + ngắt block từ Qwen3-ForcedAligner**. Thay cho qwen3-asr (text+timing đều do ASR → text dễ sai).
Các bước CLI rời, dùng chung 1 file YAML config; aligner tự tách vocal từ video.

### Flow

```
video-ocr <video> --config flow.yaml
  → <stem>_<box>.srt  + (nếu punctuation.enabled) _punct.srt + _flat.txt
align-srt <stem>_<box>_flat.txt --video <video> --config flow.yaml
  → tự tách vocal (reuse audio-separator) → _aligned.srt (text OCR + timing align)
```

### Thay đổi

- **`utils/forced_aligner.py`** (MỚI): tách lõi `load_forced_aligner` + `execute_forced_alignment`
  (trung lập, nhận `align_cfg` dict) ra khỏi `sync_engine/`. Dùng chung sync-video + align-srt.
- **`sync_engine/forced_alignment_subtitle.py`**: chỉ còn glue render_config
  (`_resolve_aligner_config`, `run_forced_alignment_subtitle`), re-export lõi. **sync_video không đổi.**
- **`punctuation/srt_punctuator.py`** (MỚI): `restore_punctuation_srt` chạy BATCH (mô phỏng
  srt_translator: parse_srt + translation/batching + render_batch_prompt). Validator
  `_content_signature` (strip Unicode P\* + whitespace) chống ảo giác — chỉ thêm dấu, không đổi chữ;
  lệch → BatchIntegrityError → retry → giữ nguyên block gốc. `flatten_srt_to_text` nối 1 dòng
  (CJK không space / Latin có space) cho aligner.
- **`cli/align_srt.py`** (MỚI) + script `align-srt`: nhận `--video`, tự trích audio (ffmpeg) +
  tách vocal (reuse `cli/audio_separator.separate_audio`, preset vocal_extraction), align → SRT.
  Đọc section `forced_alignment` trong --config (CLI > YAML > default). Hỗ trợ `--vocals`, `--no-separate`, `--task-file`.
- **`cli/video_ocr.py`**: thêm `run_punctuation_phase` (tuỳ chọn, bật bằng `punctuation.enabled`
  trong --config YAML); provider dựng qua `llm_ai.task_runner.create_task_provider`.
- **`config/llm_tasks/punctuation_restoration.yaml`** + **`prompts/llm_tasks/punctuation_restoration.txt`**
  (MỚI): vertexai; prompt ràng buộc đa ngôn ngữ + block-là-mảnh-câu + chỉ thêm dấu + giữ số dòng (`<PUNCT_TEXT>`).
- **`pyproject.toml`**: đăng ký `align-srt`, thêm `punctuation*` vào packages.find.
- **Tests**: `tests/punctuation/test_srt_punctuator.py` (9 test L1/L2, mock provider — PASS local);
  cập nhật patch target trong `tests/sync_engine/test_forced_alignment_subtitle.py` sang
  `utils.forced_aligner.*` (25 test PASS). Thêm 2 entry `test_matrix.yaml`.

### Quyết định thiết kế

- **Không dùng `llm-task`/`generic_text_task`** cho punctuation (gửi nguyên file 1 call, không batch,
  không kiểm tra toàn vẹn) → dùng SRT-batch như translate (chống drift theo số block + char-preservation).
- **Validator trung lập ngôn ngữ** (Unicode category P\*) thay vì hardcode bộ dấu CJK.
- **align-srt là CLI riêng** (không gọi từ video-ocr) — người dùng chạy 2 bước rời, dễ debug.
- Aligner cần text **phẳng 1 dòng** (`merge_punctuation` nuốt `\n`) → có bước flatten riêng.

### Trạng thái

- ✅ Lõi + 2 CLI/phase + config/prompt + test L1/L2 (local PASS) + matrix + journal.
- ⏳ Pending: Layer 3/4 (align-srt + tách vocal) cần GPU + `qwen-asr` + `audio-separator` → chạy Colab;
  real-API test punctuation (vertexAI) trên dữ liệu OCR thật để tinh chỉnh prompt + `max_chars` tiếng Trung.

---

## 2026-06-12: Text Isolation — Lọc watermark/overlay mờ khỏi OCR phụ đề

### Tóm tắt

Thêm tính năng tách phụ đề lời thoại khỏi watermark/text-overlay (opacity < 70%) **trước khi OCR**,
dựa trên opacity/color masking thuần OpenCV. Giải quyết ca creator có overlay động đè lên dải phụ đề.
Mặc định TẮT — chỉ bật cho video có watermark.

### Nguyên lý

Watermark mờ (α<0.7) bị blend với nền → giảm tương phản/gradient, mất viền tối, lệch màu. `text_isolator`
quyết định GIỮ/XÓA ở mức từng connected-component: color gate (màu chỉ định, Lab) + contrast gate
(morph gradient ~ opacity) + stroke gate (viền tối) + min_component_area (chỉ diệt nhiễu). Deterministic,
không model → không hallucinate.

### Thay đổi

- **`video_subtitle_extractor/text_isolator.py`** (MỚI): `TextIsolationConfig` + `isolate_subtitle_text`
  - `parse_color_spec` (tên/hex/RGB).
- **`tools/calibrate_text_isolation.py`** (MỚI): script hiệu chỉnh giám sát 2 thư mục mẫu
  (`--subtitle-samples`, `--watermark-samples`), xuất JSON config + ảnh preview before/after + báo cáo
  độ tách bạch. Tham số suy ra: min_contrast/color_tolerance (mạnh), stroke_max_luminance/require_stroke
  (khá), min_component_area (yếu, chỉ diệt nhiễu — P5×0.5).
- **`video_subtitle_extractor/extractor.py`**: thêm param `text_isolation`; mask ảnh TRƯỚC CV prefilter
  (frame chỉ-watermark → ROI trống → skip OCR, tiết kiệm GPU); scene-detect vẫn trên ROI gốc.
- **`cli/video_ocr.py`**: flags `--isolate-text`, `--isolate-config`, `--subtitle-colors`,
  `--color-tolerance`, `--subtitle-min-contrast`, `--stroke-max-luminance`, `--min-component-area`,
  `--no-require-stroke`. Ưu tiên CLI > YAML > JSON-calibrate > default.
- **Tests**: `tests/video_ocr/test_text_isolator.py` (Layer1 unit parse_color_spec/config — verified
  PASS; Layer2 component masking — chạy trên Colab vì cần cv2). Thêm 2 entry vào `test_matrix.yaml`.
- **Docs**: `docs/text-isolation-guide.md` — quy tắc cắt mẫu + cách đọc/dùng 5 tham số + giới hạn.

### Quyết định thiết kế

- **Bỏ lọc thời gian (`min_persistence_frames`)**: watermark trôi chậm sống LÂU (không bị loại), phụ đề
  câu ngắn sống ngắn (bị loại nhầm) → phản tác dụng. Loại hẳn.
- Màu phụ đề **do người dùng chỉ định** (không tự dò), hỗ trợ cả chữ trắng lẫn chữ màu.
- VLM tự phân loại subtitle: loại (dễ ảo giác).

### Trạng thái

- ✅ Module + script + tích hợp + CLI + test + doc hoàn tất. Layer1 test PASS local.
- ⏳ Pending: chạy Layer2 trên Colab (cần cv2); người dùng cung cấp bảng màu thực tế + crop mẫu để
  hiệu chỉnh ngưỡng; xác nhận chất lượng mask qua ảnh preview.
- 🔮 Tương lai (nếu watermark đặc đè khít dòng phụ đề): text detection + lọc hình học/tracking vị trí.

## 2026-06-12: Phase 1 — Gỡ hoàn toàn path Remotion, prerender là path duy nhất

### Tóm tắt

Xóa subproject `remotion_tuber/` (Node/Remotion) và toàn bộ code chết liên quan. Path "prerender" (Python/PIL/FFmpeg) trở thành path render duy nhất (TODO dòng 351 DONE ✓).

### Thay đổi

- **`sync_engine/tuber_overlay.py`**: Xóa `_run_render_driver`, `_which_npm`, `prepare_assets`, `composite_group` (V1), `build_group_base`. Bỏ param `use_prerender`/`project_dir`/`do_prepare_assets` khỏi `render_and_composite_groups`, `render_groups_to_video`, `prepare_groups_and_base`. Đơn giản hóa `run_tuber_flow_all_in` — loại bỏ nhánh `overlay_mode=="remotion"`.
- **`sync_engine/tuber_config.py`**: Xóa `_REMOTION_REQUIRED_KEYS`, `remotion_project_dir()`. `overlay_mode` giờ luôn trả về `"prerender"`.
- **`sync_engine/tuber_manifest.py`**: Bỏ param `remotion` + block `"remotion"` khỏi `build_run_manifest`.
- **`cli/tuber_repair.py`**: Raise `TuberOverlayError` nếu thiếu prerenderManifest (không còn fallback Remotion). Bỏ đọc `run_manifest["remotion"]`.
- **Config**: Xóa block `"remotion": {...}` khỏi `assets/charenjizukan_tuber_overlay_config.json` và `assets/tuber_overlay_config.json`.
- **Tests**: Xóa `tests/sync_engine/test_tuber_remotion_validation.py`. Viết lại `TestLayer3_RetryAndCleanup` mock `_pipe_prerender_frames`. Đổi env `REMOTION_TUBER_E2E` → `PRERENDER_E2E` trong `test_tuber_prerender.py`. Xóa 3 entry "Tuber Remotion" khỏi `test_matrix.yaml`.
- **Docs**: Dọn `docs/tuber-overlay-guide.md` + `docs/colab-guide.md` — bỏ mọi hướng dẫn Remotion/npm.
- **`remotion_tuber/`**: `git rm -r` (xóa 17 file).

### Trạng thái

- ✅ Prerender là path duy nhất, không còn dead code Remotion.
- ✅ `overlay_mode` config cũ `"remotion"`/`"auto"` được coi như `"prerender"` (backward-compat cảnh báo).
- ⏳ Phase 2: gom `sync_engine/tuber_*.py` → package `sync_engine/tuber/` (commit 2).

---

## 2026-06-11: Local testing — venv `uv` cô lập (CPU-only), sửa runner dùng `sys.executable`

### Bối cảnh

Cần chạy test ở máy local (Windows, Python global 3.13) mà không đụng Python global và không kéo package GPU (chúng được test trên Colab).

### Đã làm

- Tạo `.venv` bằng `uv venv --python 3.12 .venv` (khớp Colab `requires-python >= 3.12`).
- Cài CPU-only vào `.venv`: `uv pip install -e .` + `pytest pyyaml pydub numpy`. KHÔNG dùng `--system`, KHÔNG cài group `[qwen-tts]` (torch/flash-attn). Đã verify `Location` trỏ `.venv` và không có package GPU; global Python sạch.
- **Sửa `run_colab_tests.py::_build_pytest_cmd`**: `["python", ...]` → `[sys.executable, ...]` để subprocess pytest chạy đúng interpreter `.venv` thay vì rơi vào python PATH khác (uv cpython).
- **Sửa `run_colab_tests.py`**: thêm `_safe_report_name()` (sanitize ký tự Windows-invalid `< > : " / \ | ? *`) → hết crash `OSError [Errno 22]` khi tên report chứa `>`. Thêm `PYTHONIOENCODING=utf-8`/`PYTHONUTF8=1` vào env subprocess → test in emoji/CJK (vd `📄`) không crash cp1252 trên Windows.
- **Sửa `pyproject.toml`**: thêm `tts*` vào `packages.find.include` (trước đây thiếu → `import tts.edgetts` fail khi không có cwd trên path). Reinstall editable.
- Sửa các test stale: `test_extractor_config.py` (thêm `importorskip("cv2")`), `test_ass_utils.py` (2 test sai expectation), `test_tts_edgetts.py` (import `cli.tts_srt`→`cli.tts`, skip Layer 3 dùng API `run_tts` cũ, cập nhật `test_strip_silence_only_tail` theo logic "strip cả hai đầu" hiện tại).
- **Xóa Demucs** (đã bị thay bằng audio-separator): bỏ 2 entry trong `test_matrix.yaml` trỏ file đã xoá `tests/cli/test_demucs_audio.py`; cập nhật wording trong `audio_assembler.py`, `colab-guide.md`, `testing-guide.md`. Giữ nguyên 2 file `plans/` (bản ghi lịch sử, theo yêu cầu user).
- Cập nhật `docs/testing-guide.md`: thêm **Mục 0** ở đầu file (hướng dẫn `uv` + `.venv`, cấm `--system`, cấm package GPU local) để Agent scan nắm ngay.

### Kết quả

- `python run_colab_tests.py --tags unit` → **35 passed, 0 failed**, 1 all-skipped (Translation L2 cần API), 3 no-collection (cần `cv2`/opencv — cố ý KHÔNG cài local).

### Next steps đề xuất

- Khi cần chạy L4/gpu hoặc các test cần `cv2`: chạy trên Colab (venv Colab kế thừa torch/cv2).
- Cân nhắc viết test thay thế cho audio-separator (chỗ Demucs cũ để trống).

## 2026-06-11: qwen3_asr — expandable_segments + `--max-new-tokens` (mặc định 1024) giảm OOM VRAM audio dài

### Bối cảnh

Trên L4 Colab (22GB), transcribe video ~1h thấy VRAM tăng tịnh tiến gần chạm trần, suýt OOM ở `--batch-size 8` và OOM hẳn ở `16`. Câu hỏi: đây là hành vi mặc định của Qwen3-ASR hay code thiếu clear VRAM.

### Nguyên nhân gốc

Vì CLI gọi `transcribe(..., return_time_stamps=True)` nên `qwen_asr` ép chunk audio theo `MAX_FORCE_ALIGN_INPUT_SECONDS = 180s` (3 phút) → video 1h ≈ 20 chunk, gom batch theo `max_inference_batch_size` (= `--batch-size`). Thư viện KHÔNG `empty_cache()`/`del` tensor/giải phóng KV-cache giữa các batch; PyTorch giữ lại reserved memory → nhìn `nvidia-smi` thấy leo dần tới đỉnh rồi plateau. **Không phải leak** (text trả về string, timestamp merge bằng list — đều ở CPU); đỉnh VRAM ∝ `batch_size × max_new_tokens`. Batch 16 = 16 chunk×3phút đồng thời → KV-cache gấp đôi → vượt 22GB. `clear_vram()` đặt ở `finally` chỉ chạy SAU cả file, vô dụng trong-file và không chèn được vào vòng lặp chunk nội bộ thư viện.

### Thay đổi

1. `cli/qwen3_asr.py`:
   - `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")` ngay đầu module (TRƯỚC khi torch khởi tạo CUDA allocator) → giảm phân mảnh, trị đúng kiểu "leo tới trần rồi OOM". Dùng `setdefault` để tôn trọng giá trị user set sẵn.
   - Bỏ hardcode `max_new_tokens=4096` → tham số `--max-new-tokens` (mặc định **1024**), xuyên suốt: chữ ký `run_batch_transcribe` → `from_pretrained` → arg → lời gọi trong `main()`. (Official transformers example dùng 256; chunk 3 phút không cần tới 4096, cap cao chỉ làm trần KV-cache khi 1 chunk lặp/hallucinate.)

### Lưu ý còn lại

- Chưa verify thật trên Colab L4 — cần chạy lại video 1h xác nhận batch 16 không còn OOM.
- `batch_size` vẫn là đòn bẩy chính cho đỉnh VRAM; `--max-new-tokens` là cap an toàn — nếu chunk dài bị cắt cụt text thì tăng lên.

### File thay đổi

- `cli/qwen3_asr.py`

---

## 2026-06-11: Watermark width + layer_order config + nung black_strip ở stretch (dưới tuber)

### Bối cảnh

Yêu cầu: (1) thêm `width` optional cho watermark_img (height auto theo aspect, mặc định width gốc); (2) sắp lại layer order `Base → Black strip → Tuber → Image → Note → Watermark → Subtitle`; (3) tuber phải nằm TRÊN black_strip. Vấn đề kiến trúc: tuber baked vào base TRƯỚC final render nên vốn là layer dưới cùng (dưới cả strip). Promote tuber thành alpha layer độc lập thì mất NVENC (NVENC không hỗ trợ alpha) + file ProRes/VP9 alpha cực lớn cho video 2-3h.

### Quyết định

Nung `black_strip` ngay ở **Phase 2 (stretch)** thay vì Phase 5 → gấp chung vào encode batch sẵn có, **0 encode thêm**, và strip tự nằm DƯỚI tuber (tuber composite seek vào `video_stretched.mp4` sau đó). Tuber giữ nguyên cơ chế baked. Các layer Phase 5 còn lại điều khiển thứ tự bằng `render_config.layer_order`.

### Thay đổi

1. `sync_engine/video_processor.py`: `build_ffmpeg_batch_cmd(..., strip, video_width)` + helper `_build_strip_overlay` (full-width clamp) + `_probe_video_width`. `process_video_chunks_parallel(..., strip)` probe width source rồi overlay strip lên concat trước khi encode. Strip phủ full width video gốc; `scale_width` config > width video → kẹp về full width.
2. `sync_engine/renderer.py`: refactor 6 layer (black_strip/image_overlay/note_overlay/watermark_img/watermark_text/subtitles) thành builder + dispatch theo `render_config["layer_order"]` (thiếu → default order). Thêm param `skip_layers`. Watermark_img: `width` > 0 → `scale={width}:-1` (height auto giữ aspect); null/absent/<=0 → giữ ảnh gốc.
3. `cli/sync_video.py`: helper `_resolve_black_strip`; Phase 2 truyền `strip=` vào stretch; Phase 5 `skip_layers={"black_strip"}` khi strip bật.
4. `cli/tuber_repair.py`: final render `skip_layers={"black_strip"}` (base promote đã chứa strip nung).
5. `assets/default_render_config.json`: thêm `"width": null` vào watermark_img + mảng `"layer_order"`.
6. Docs: `sync-video-guide.md` (sơ đồ layer, mục layer_order, watermark_img width, black_strip nung ở Phase 2), `tuber-overlay-guide.md` (z-order tuber trên strip).

### Layer order kết quả

`Base → Black strip (nung Phase 2) → Tuber (baked) → Image overlay → Note overlay → Watermark img → Watermark text → Subtitle`

### Lưu ý còn lại

- Resume hash tuber anchor theo video gốc + group_manifest, KHÔNG theo stretched → đổi black_strip config KHÔNG tự re-render tuber group (strip nằm dưới, không đổi hình tuber). Muốn ép thì `resume.skipDone=false`.
- Chưa verify FFmpeg thật trên Colab GPU — cần chạy pipeline thật để xác nhận strip hiển thị đúng vị trí dưới tuber. Test unit (80 passed) chỉ verify command/filter string + pipeline mock.
- Muốn tuber xen kẽ z-order tùy ý (vd trên image overlay) → phải tách alpha layer (mất NVENC + disk lớn), chưa làm.

### File thay đổi

- `sync_engine/video_processor.py`, `sync_engine/renderer.py`, `cli/sync_video.py`, `cli/tuber_repair.py`, `assets/default_render_config.json`, `docs/sync-video-guide.md`, `docs/tuber-overlay-guide.md`

---

## 2026-06-09: Tuber group cuối fail "Duration lệch" — clamp group theo frame THỰC của video_stretched

### Vấn đề

Video dài (1h30–1h40) fail ở Phase 5.0 tuber overlay, **chỉ group cuối** (hoặc vài group đuôi) fail với `Duration lệch: 286.067s vs expected 286.200s (tol 0.1s)` — hụt đúng ~4 frame (0.133s). Group cuối là group duy nhất chạm **EOF thật** của `base_video_stretched.mp4`.

### Nguyên nhân gốc

Manifest tính `groupStartFrame`/`renderDurationFrames` bằng tổng dồn `segment_output_frames()` (frame **lý thuyết**). Nhưng `build_ffmpeg_batch_cmd` dùng `trim=end_frame=N` — chỉ **chặn-trên**, không đảm bảo đủ N: segment chạm EOF nguồn ra thiếu vài frame. File `video_stretched.mp4` thật vì thế luôn ≤ lý thuyết, phần hụt dồn vào group cuối. Composite seek theo thời gian tuyệt đối → group cuối trim không đủ N → lệch tolerance 0.1s → fail → fallback render KHÔNG tuber.

### Thay đổi

**A. Fix gốc ở Phase 2 — pad tail (frame lý thuyết thành nguồn sự thật duy nhất thật sự):**

- `sync_engine/video_processor.py`: thêm hằng `_TAIL_PAD_SECONDS=2.0` + chèn `tpad=stop_mode=clone:stop_duration=2.0` NGAY TRƯỚC `trim=end_frame=N` trong cả `build_ffmpeg_batch_cmd` (Phase 2 dùng) và `build_ffmpeg_chunk_cmd` (legacy). Segment chạm EOF được clone frame cuối cho đủ N rồi trim chốt đúng N → `video_stretched.mp4` thật == frame lý thuyết == audio == sub. Segment thường (đủ/dư N) bị trim cắt bỏ tpad ngay, không encode, output không đổi → không phá test filter-string hiện có (tpad nằm giữa `fps=` và `trim=end_frame=N[vi]`, mọi substring assert vẫn còn).

**B. Lưới an toàn ở lớp tuber — clamp theo frame thật:**

1. `sync_engine/tuber_overlay.py`: thêm `probe_frame_count(video, fps)` (ưu tiên `nb_frames`, fallback `duration*fps`, KHÔNG `-count_frames` để tránh decode toàn bộ — đọc metadata header, tức thì, không GPU). `run_tuber_flow_all_in` đo frame thật của base stretched rồi truyền xuống; `prepare_groups_and_base` nhận `real_total_frames`.
2. `sync_engine/tuber_manifest.py`: `build_render_groups(..., real_total_frames=None)` + helper `_clamp_groups_to_real()` — cắt `group_end_frame` của group chứa EOF, bỏ group nằm hẳn sau EOF. Deficit > 2s → WARNING (nghi Phase 2 truncate thật, vd ca 1h30 hụt ~25 phút), ≤ 2s → INFO.
3. `tests/sync_engine/test_tuber_overlay_pipeline.py`: 4 unit test mới trong `TestLayer1_GroupBuilding`.

Sau (A), file thật khớp lý thuyết nên (B) hầu như không kích hoạt với rounding đuôi — chỉ còn là lưới bắt truncate bất thường.

### Lưu ý còn lại

- Ca 1h30 hụt ~25 phút là **truncate thật ở Phase 2** (một batch ra ngắn). tpad (A) chỉ bù ≤2s nên KHÔNG che được ca này → clamp (B) bắt + WARNING. Gốc vẫn nên thêm validate frame-count sau `_concat_chunks` (chỉ đang validate size) — việc riêng.
- Chưa verify FFmpeg thật trên Colab — cần chạy lại pipeline video 1h40 để xác nhận group cuối pass và `video_stretched` đủ frame.

### File thay đổi

- `sync_engine/video_processor.py`, `sync_engine/tuber_overlay.py`, `sync_engine/tuber_manifest.py`, `tests/sync_engine/test_tuber_overlay_pipeline.py`

---

## 2026-06-08: Fix chromakey — bỏ qua key khi bodySource đã có alpha sẵn

### Vấn đề

Body tuber overlay hiện ra bán trong suốt ("như ẩn như hiện", chỉ rõ mouth) dù user KHÔNG khai báo `chromakey` trong config. Hai nguyên nhân:

1. **Code luôn chromakey**: `extract_body_transparent()` không có đường tắt — khi `chroma_color=None` nó _auto-detect_ màu 4 góc rồi vẫn `chromakey=...`. Với nguồn đã trong suốt (hoặc H264 nền bị nướng đen), chromakey key luôn cả vùng tối của body → bán trong suốt.
2. **Hiểu nhầm H264**: `loop_mouthless_h264.mp4` không thể trong suốt thật — H264/.mp4 không mang kênh alpha. Muốn nền trong suốt phải dùng ProRes4444 `.mov` / VP9 `.webm` / PNG sequence.

### Thay đổi

1. `sync_engine/tuber_prerender.py`:
   - Thêm `_source_has_alpha()` (ffprobe pix_fmt → nhận diện yuva\*/rgba/...).
   - `extract_body_transparent()` thêm tham số `chromakey_enabled: Optional[bool]`. Tri-state: None=auto (bỏ qua chromakey nếu nguồn có alpha), False=luôn giữ alpha gốc (`vf=format=rgba`), True=luôn chromakey. Cảnh báo khi tắt chromakey nhưng nguồn không alpha (sẽ ra frame đặc).
2. `sync_engine/tuber_config.py`: property `chromakey_enabled` đọc `asset.chromakey.enabled` (tri-state).
3. `sync_engine/tuber_overlay.py`: `_auto_run_prerender()` truyền `config.chromakey_enabled` xuống.
4. `tests/sync_engine/test_tuber_overlay_pipeline.py`: 3 unit test cho `chromakey_enabled` (None/False/True).

### Hành động cho user

- Re-export body sang `.mov` (ProRes4444) / `.webm` (VP9 alpha) / PNG seq → code TỰ nhận alpha và bỏ qua chromakey. Hoặc đặt `asset.chromakey.enabled=false` để tắt tường minh.
- Còn dùng H264 green-screen thì khai `asset.chromakey.color` (vd `0x00FF00`) thay vì để auto-detect màu đen.

### Lưu ý còn lại

- Đường Remotion (`remotion_tuber/scripts/prepare-assets.ts`, mode `remotion`/`direct`) vẫn luôn chromakey — chưa sửa vì config hiện dùng `overlay.mode=prerender`. Cần port logic skip-alpha tương tự nếu sau này dùng Remotion với nguồn alpha.
- Chưa verify FFmpeg thật trên nguồn .mov/.webm (máy dev không có asset); cần chạy lại pipeline trên Colab để xác nhận frame body đặc/trong suốt đúng.

### File thay đổi

- `sync_engine/tuber_prerender.py`, `sync_engine/tuber_config.py`, `sync_engine/tuber_overlay.py`, `tests/sync_engine/test_tuber_overlay_pipeline.py`
- `docs/tuber-overlay-guide.md` (doc: key `asset.chromakey.enabled`, cảnh báo H264 không alpha, Phase 0 diagram)

---

## 2026-06-08: Prerender song song thật — ThreadPool → ProcessPoolExecutor (vượt GIL)

### Vấn đề

`prerender_character()` (bake body×mouth) đọc đúng `maxWorkers=4`, truyền đúng vào hàm, và _đã_ vào nhánh `ThreadPoolExecutor` — nhưng 705 frame vẫn mất ~48s như đơn luồng. **Nguyên nhân:** warp là CPU-bound Python thuần + PIL (`compute_affine`, `_invert_affine`, `apply_mouth_calibration`, dựng mask `ImageDraw.polygon`, `Image.transform/alpha_composite/paste`) → giữ **GIL** → thread không song song. (Thread chỉ hiệu quả ở composite groups / video chunks vì chúng gọi FFmpeg subprocess, nhả GIL.)

### Thay đổi (`sync_engine/tuber_prerender.py`)

1. Thêm cấp module: `_PRERENDER_CTX` + `_prerender_pool_init()` (nạp sprite + track 1 lần/worker) + `_prerender_body_worker(body_idx)` (render mọi mouth_state cho 1 body frame). Gom task theo **body_idx** → mở ảnh body 705→141 lần, mỗi worker chỉ nhận 1 int qua pickle.
2. Nhánh `max_workers>1`: đổi sang `ProcessPoolExecutor` với `mp_context=get_context("spawn")` (an toàn khi tiến trình cha `worker_task` đã spawn + init CUDA; child chỉ làm PIL). Tiến độ cộng dồn qua `as_completed` + mốc 100.
3. Nhánh `max_workers<=1`: giữ tuần tự (eager cache) — hành vi & output không đổi → test integration mặc định (`max_workers=1`) vẫn xanh.

### Lý do thiết kế

- `initargs` đều picklable (track dict, path str, list, int); worker/init/`warp_sprite_to_quad`/`apply_mouth_calibration` đều module-level → spawn re-import được.
- `_PRERENDER_CTX` reset trong mỗi child qua initializer → không rò trạng thái.

### Kiểm chứng

- `pytest tests/sync_engine/test_tuber_prerender.py` → 25 passed, 5 skipped (integration cần body-transparent/ chưa extract local; PIL không có trên máy dev nên nhánh process chỉ chạy thật trên Colab).
- **Còn phải verify trên Colab:** thời gian prerender giảm rõ + output PNG y hệt nhánh tuần tự (cùng thuật toán warp). Chưa chạy được nhánh process tại local do thiếu PIL.

### File thay đổi

- `sync_engine/tuber_prerender.py`

---

## 2026-06-08: V5 — Mouth vowel selection (spectral centroid, port ③ライブ実行)

### Tóm tắt

PNGTuber chọn khẩu hình chỉ theo biên độ (closed/half/open) nên 2 khẩu hình nguyên âm `e.png` (え) / `u.png` (う) không bao giờ được chọn. Port **Tầng 2** từ nút `③ライブ実行` của repo gốc [MotionPNGTuber](https://github.com/rotejin/MotionPNGTuber): chọn `e`/`u` bằng **spectral centroid** (FFT brightness) tại đỉnh sóng khi đang `open`. Đây là proxy độ sáng phổ (KHÔNG phải nhận diện phoneme thật): centroid thấp → `u`, cao → `e`.

### Thiết kế (offline, 2 tầng)

- **Tầng 1** (như cũ): RMS amplitude → per-frame level closed/half/open.
- **Tầng 2** (mới): tính centroid mỗi frame (`numpy.fft.rfft`, chuẩn hoá [0,1] theo Nyquist), ngưỡng `U_TH`/`E_TH` = percentile 20/80 của centroid các frame `open` trên toàn clip (offline → ổn định hơn live adaptive). Tại đỉnh sóng + cooldown → cập nhật khẩu hình sticky.
- **Backward-compatible tuyệt đối:** chỉ kích hoạt khi `mouth.mouthStates` có `e`/`u`. Cấu hình 3-state cũ, thiếu numpy, hoặc quá ít frame `open` → tự bỏ qua Tầng 2 (fail mềm), hành vi y hệt trước.

### Thay đổi chính

1. **`tuber_mouth_events.py`**: tách `_read_wav_samples`/`_samples_to_rms` (giữ `_read_wav_rms`); thêm `_frame_centroids` (numpy lazy+guarded), `_percentile`, `_select_vowel_shapes`, `_apply_vowel_selection`. `analyze_tts_amplitude` tái cấu trúc 3 bước (levels → vowel rewrite → collapse events) + nhận `mouth_states`, `peak_margin`, `min_vowel_interval_ms`, `vowel_low/high_percentile`.
2. **`tuber_config.py`**: accessor `mouth_peak_margin`, `mouth_min_vowel_interval_ms`, `mouth_vowel_low_pct`, `mouth_vowel_high_pct`.
3. **`tuber_overlay.py`**: `mouth_opts` truyền thêm `mouth_states` + vowel knobs; thêm `_prerender_is_stale()` → tự bake bổ sung frame `e`/`u` khi prerendered/ cũ thiếu state (không cần `resume.skipDone=false`).
4. **`pyproject.toml`**: khai báo tường minh `numpy>=1.24` (trước chỉ có gián tiếp qua imagehash/torch).
5. **Docs sync**: `docs/tuber-overlay-guide.md` (bảng `mouth` V5 + config mẫu 5 state + ghi rõ prerender-only) và `remotion_tuber/README.md` (Remotion path không hỗ trợ vowel).

### Resume / Hash

`compute_group_input_hash` đã băm `mouthEvents` → state e/u đổi → group tự re-render. Prerender staleness fix lo phần bake frame nguyên âm. Không sửa hash.

### Lưu ý

- Centroid là proxy thô → e/u đôi lúc lệch nguyên âm thật (đúng kỳ vọng, giống bản gốc).
- **Chỉ hỗ trợ ở `overlay.mode="prerender"`**; path Remotion (`mouthState.ts`) chưa có Tầng 2.
- Config mặc định `assets/tuber_overlay_config.json` GIỮ 3-state (asset `nike_loop_fix` chỉ có 3 sprite); ví dụ 5-state nằm trong docs.

### Tests

- `tests/sync_engine/test_tuber_mouth_events.py`: `TestLayer1_VowelSelection` (`_select_vowel_shapes`, `_percentile`, backward-compat 3-state), `TestLayer2_VowelFromWav` (WAV sin tần thấp→`u`/cao→`e`).
- `tests/test_matrix.yaml`: thêm entry Layer1/Layer2 vowel.

### File thay đổi

- `sync_engine/tuber_mouth_events.py`, `sync_engine/tuber_config.py`, `sync_engine/tuber_overlay.py`
- `pyproject.toml`, `docs/tuber-overlay-guide.md`, `remotion_tuber/README.md`
- `tests/sync_engine/test_tuber_mouth_events.py`, `tests/test_matrix.yaml`

---

## 2026-06-05: Fix resume.skipDone luôn miss — inputHash anchor sai vào video tái tạo

### Triệu chứng

Chạy `sync-video` lần 2 (cùng video/SRT/config), group đáng lẽ `skipped` nhưng status ra `done` → re-render lại từ đầu, resume vô tác dụng.

### Nguyên nhân gốc

`compute_group_input_hash()` (`tuber_status.py`) trộn `st_mtime_ns` của `video_stretched.mp4` (base video đã promote) vào hash. NHƯNG `sync-video` tái tạo `video_stretched.mp4` từ đầu mỗi lần chạy (stretch lại → mtime mới), `promote_media` copy lại file mới → hash lưu ở lần 1 KHÔNG BAO GIỜ khớp lần 2 → skip-check fail → re-render → status `done`.

(Ngay cả content-hash video stretched cũng không cứu được: NVENC không byte-deterministic giữa các lần encode.)

### Cách sửa

Đổi anchor của hash từ **intermediate tái tạo** sang **input ổn định**: video GỐC (`--video`). Mọi yếu tố stretch/mouth đã nằm trong `group_manifest` rồi, nên video gốc + manifest là đủ để định danh output 1 group, và video gốc không đổi giữa các lần chạy.

### Thay đổi chính

1. **`compute_group_input_hash(group_manifest, prerender_manifest, source_video)`** (`tuber_status.py`): param `stretched_video` → `source_video`; guard `None` (graceful, không raise).
2. **Đường truyền `source_video`**: `run_tuber_flow_all_in` truyền `Path(video_path)` → `render_groups_to_video` → `render_and_composite_groups` → 2 chỗ gọi hash trong `_process_one_group`.
3. **`run_manifest["sourceVideo"]`** (`tuber_manifest.py::build_run_manifest`): ghi path video gốc để repair tái lập đúng hash.
4. **`tuber_repair`**: đọc `run_manifest["sourceVideo"]`, fallback `baseVideo` nếu manifest cũ chưa có field (hash khác → re-render, an toàn).

### Tests

- `TestLayer1_GroupHash::test_hash_stable_when_intermediate_regenerated` — regression: cùng video gốc → hash khớp dù intermediate đổi mtime; đổi video gốc → hash đổi.
- `TestLayer1_GroupHash::test_hash_tolerates_missing_source_video` — `source_video=None` không raise.
- Sửa 2 test stale có sẵn (do default `direct` từ V4): `test_defaults` (assert `direct`), `test_pipe_cmd_no_overlay_frames_dir` (kiểm tra intent qua AST, bỏ docstring thay vì grep chuỗi).
- `tests/sync_engine/test_tuber_overlay_pipeline.py` + `test_tuber_repair.py`: 77 passed, 7 skipped.

### File thay đổi

- `sync_engine/tuber_status.py`, `sync_engine/tuber_overlay.py`, `sync_engine/tuber_manifest.py`, `cli/tuber_repair.py`
- `tests/sync_engine/test_tuber_overlay_pipeline.py`

---

## 2026-06-05: V4 — Direct RGB Pipe (bỏ I/O PNG sequence trung gian)

### Tóm tắt

V4 thêm `overlay.format: "direct"` (default) — pipe raw RGBA từ `prerendered/` cache thẳng vào FFmpeg stdin, bỏ hoàn toàn `overlay_frames/*.png` trung gian. Giữ `"png_sequence"` làm debug mode. Resume/skipDone không thay đổi.

### Thay đổi chính

1. **`overlay.format: "direct"` (default mới):** Thêm giá trị `"direct"` vào `overlay.format` accessor (`tuber_config.py`). Default đổi từ `"png_sequence"` → `"direct"`. Giá trị lạ → fallback `"direct"` + warning. Đã bỏ `pipeMode` riêng — gộp chung vào `format` để tránh 2 field cùng mô tả 1 thứ.

2. **`_pipe_prerender_frames()`** (`tuber_overlay.py`): Hàm mới thay `_build_prerender_frame_list` + `composite_group_from_stretched`. Đọc kích thước THẬT từ frame probe đầu tiên (PIL `.size`), dùng hybrid seek y hệt V3, bơm raw RGBA tuần tự vào `process.stdin`. Stderr ghi ra file log (không PIPE) để tránh deadlock buffer.

3. **`_make_mouth_lookup()`** (`tuber_overlay.py`): Tách hàm binary-search lookup mouth state ra module-level để dùng chung cho cả direct pipe và png_sequence path (DRY).

4. **Rẽ nhánh trong `_process_one_group()`**: Khi `use_prerender=True`, kiểm tra `overlay_format` → `"direct"` gọi `_pipe_prerender_frames`, `"png_sequence"` gọi `_build_prerender_frame_list` + `composite_group_from_stretched`. Fallback tự động: direct fail hết retry → thử png_sequence 1 lần cuối.

5. **Đường truyền `overlay_format`**: `run_tuber_flow_all_in` lấy `config.overlay_format` → `render_groups_to_video` → `render_and_composite_groups`. `tuber_repair` đọc `run_manifest["overlayFormat"]` (đã được ghi lúc all-in) với fallback `cfg.overlay_format`.

6. **Tests Layer 1 mới**: `TestLayer1_OverlayFormatConfig` (config accessor), `TestLayer1_DirectPipeCmd` (source inspection cmd + `_make_mouth_lookup`), `TestLayer1_RepairResolveFormat` (`test_tuber_repair.py`).

7. **Docs**: `tuber-overlay-guide.md` — thêm V4 summary, bảng `overlay.format` với 2 giá trị + debug workflow, note `overlay_frames/` chỉ tồn tại ở `png_sequence`. Sample config đổi sang `"direct"`.

### Đã bác bỏ

- **Split Alpha (2 HEVC video)**: 3 generation HEVC loss làm hỏng viền alpha. Không phù hợp production pipeline. Direct pipe đạt cùng lợi ích I/O mà lossless và 1 encode.

### File thay đổi

- `sync_engine/tuber_config.py`, `sync_engine/tuber_overlay.py`, `cli/tuber_repair.py`
- `assets/tuber_overlay_config.json`
- `tests/sync_engine/test_tuber_overlay_pipeline.py`, `tests/sync_engine/test_tuber_repair.py`, `tests/test_matrix.yaml`
- `docs/tuber-overlay-guide.md`

---

## 2026-06-05: V3 — Tối ưu hiệu năng + Resume cho pre-render pipeline

### Tóm tắt

V3 không thêm tính năng người dùng mới, tập trung vào hiệu năng, resume skip-done, sửa bug repair cho V2 prerender.

### Thay đổi chính

1. **Composite-seek (bỏ build_group_base):** Thay `build_group_base(video_gốc, trim+stretch lại)` bằng `composite_group_from_stretched(seek vào video_stretched.mp4, trim-by-frame, overlay 1 lần)`. Giảm từ 8 encode còn 5 encode (N=3 groups). Không thay đổi `render_final_video`.

2. **Fix tuber_repair.py cho V2 prerender:** Đọc `prerenderManifest` từ `run_manifest`, truyền `use_prerender`/`prerender_dir`/`prerender_manifest` xuống `render_groups_to_video`, bỏ hardcode `asset_id`/`nike_loop_fix`. Repair V2 không còn gọi npm/Remotion.

3. **Parallel prerender_character (`max_workers`):** `prerender_character()` nhận `max_workers: int`. Khi >1 dùng `ThreadPoolExecutor` cho loop body×mouth (170×3=510 frame độc lập). Body cache + sprite cache load trước ở main thread (read-only).

4. **Parallel groups (`max_workers`):** Vòng `for g in groups` trong `render_and_composite_groups` được bọc `ThreadPoolExecutor`. Retry per-group vẫn hoạt động. Sort kết quả theo group index. Mặc định 2 worker.

5. **Hash + skipDone (`resume.skipDone`):** `compute_group_input_hash()` = SHA-256(segments, renderStartFrame, renderDurationFrames, compOffsets, prerender outputSize, stretched_video mtime+size). Lưu vào `status.json["inputHash"]`. Khi `skipDone=true` + hash khớp + output validate OK → skip group (status=Skipped). Khi `skipDone=false` → xóa sạch groups/ + prerendered/ và dựng lại.

6. **Mode hybrid (mouth):** `analyze_tts_amplitude` nhận `mode` param. Hybrid: RMS gate (nói/im), debounce non-closed transitions bằng `cadenceMs` (half↔open phải chờ đủ cadence_frames). Silence luôn override → closed ngay. Thêm `"mode" key` vào `mouth_opts` chain.

7. **debugFrameOutput:** `debug.frameOutput.{enabled, marginFrames}`. Khi enabled, dump overlay + composited frames quanh boundary (start + end) vào `logs/debug_frames/{group_id}/` + `boundary.json`.

8. **Config mới:** 3 blocks trong `tuber_overlay_config.json`: `performance.maxWorkers`, `resume.skipDone`, `debug.frameOutput.{enabled, marginFrames}`.

### File thay đổi

- `sync_engine/tuber_overlay.py` — composite-seek (mới), parallel (sửa), skip-done, debug dump
- `sync_engine/tuber_prerender.py` — parallel prerender_character với max_workers
- `sync_engine/tuber_status.py` — compute_group_input_hash, inputHash field
- `sync_engine/tuber_config.py` — accessors: max_workers, resume_skip_done, debug_frame_output_enabled, debug_frame_margin
- `sync_engine/tuber_mouth_events.py` — mode hybrid debounce
- `cli/tuber_repair.py` — fix V2 prerender path (prerenderManifest, use_prerender, stretched_video)
- `assets/tuber_overlay_config.json` — blocks performance, resume, debug
- `tests/sync_engine/test_tuber_mouth_events.py` — Hybrid Layer 1
- `tests/sync_engine/test_tuber_overlay_pipeline.py` — CompositeSeek, Performance, Hash Layer 1
- `tests/sync_engine/test_tuber_repair.py` — MỚI: Repair Layer 1
- `tests/test_matrix.yaml` — 3 entry mới
- `docs/tuber-overlay-guide.md` — cập nhật

### Verification

- 60 Layer 1 tests pass (13 existing + 47 new/expanded)

---

## 2026-06-04: Bugfix — miệng mở trễ ~1s (bug gộp event \_merge_short_silence)

### Tóm tắt

- **Triệu chứng**: PNGTuber mở miệng trễ ~1s so với audio, xảy ra cả EdgeTTS lẫn Voicevox.
- **Chẩn đoán**: Sinh script `diag_mouth_onset.py` để in onset tại nhiều ngưỡng dB và events từ pipeline thật. Output trên `dubb-2.wav` (EdgeTTS) cho thấy onset thật = frame 5-6, nhưng `analyze_tts_amplitude()` trả `open` ở frame 39.
- **Root cause**: Bug trong `_merge_short_silence()` (tuber_mouth_events.py). Bước gộp consecutive same-state: `merged[-1]["frame"] = ev["frame"]` ghi đè frame onset sớm bằng frame muộn. Ví dụ: 2 đoạn `open` (7-34 và 39-44) quanh 1 `closed` ngắn 4 frame bị drop → khi gộp, onset `open` bị dời từ 7 → 39 (~1s). Đây là gốc rễ dùng chung cho CẢ hai engine.
- **Giả thuyết đã loại**: Voicevox lead-in (strip-silence). WAV onset ~0.17s, không phải 1s.
- **Fix**: Bỏ dòng ghi đè frame — giữ nguyên frame chuyển trạng thái sớm nhất khi gộp. Thêm `exclude_mute → whole_video` fallback khi không có mute segment (tường minh hoá policy).

### File thay đổi

- `sync_engine/tuber_mouth_events.py` — fix gộp event trong `_merge_short_silence()`
- `sync_engine/audio_assembler.py` — fallback exclude_mute → whole_video khi `has_mute_segment = False`
- `tests/sync_engine/test_tuber_mouth_events.py` — thêm regression test `test_merge_short_silence_keeps_early_onset_frame`
- `diag_mouth_onset.py` — script chẩn đoán onset miệng (giữ lại làm công cụ)

### Verification

Chạy diagnostic sau fix: `uv run python diag_mouth_onset.py --glob ".../dubb-2.wav"` → `open` ở frame ~6-7 (không còn 39). Tất cả 13 Layer1 unit test pass.

---

## 2026-06-04: V2 — Pre-render character + lip-sync amplitude

### Tóm tắt

- **Mục tiêu**: Giải quyết 2 vấn đề blocking V1 — miệng không khép khi im lặng, hiệu năng video dài.
- **Kết quả chính**: Port thuật toán affine warp từ TypeScript → Python/PIL, module `tuber_prerender.py` pre-render 170×N body×mouth, module `tuber_mouth_events.py` phân tích RMS amplitude TTS → mouthEvents chính xác đến frame.
- **Kiến trúc**: Pipeline thuần Python + FFmpeg khi có pre-render. Remotion code giữ nguyên làm reference, không gọi runtime. Config thêm `mouth.mode=amplitude`, `asset.prerender`, composite hỗ trợ offset cho character box.

### File tạo mới

| File                                                                        | Mô tả                                                                  |
| --------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| [`sync_engine/tuber_mouth_events.py`](../sync_engine/tuber_mouth_events.py) | Phân tích RMS amplitude TTS → mouthEvents [{frame, state}] per segment |
| [`sync_engine/tuber_prerender.py`](../sync_engine/tuber_prerender.py)       | Port mouthWarp.ts (affine 2-triangle), pre-render body×mouth → PNG     |

### File sửa

| File                                                                      | Thay đổi                                                                                                                   |
| ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| [`sync_engine/tuber_config.py`](../sync_engine/tuber_config.py)           | +mouth (mode, silenceDb, minSilenceMs, cadenceMs, mouthStates) +prerender config                                           |
| [`sync_engine/tuber_manifest.py`](../sync_engine/tuber_manifest.py)       | +compute_character_box, +mouthEvents build, +compWidth/compHeight/compOffset, +prerenderManifest trong run_manifest        |
| [`sync_engine/tuber_overlay.py`](../sync_engine/tuber_overlay.py)         | +composite offset_x/offset_y, +prerender path (\_build_prerender_frame_list, use_prerender param), +mouthEvents map lookup |
| [`assets/tuber_overlay_config.json`](../assets/tuber_overlay_config.json) | Config mới với mode=amplitude + prerender section                                                                          |
| [`docs/tuber-overlay-guide.md`](../docs/tuber-overlay-guide.md)           | Update architecture, mouth config, prerender docs                                                                          |

### Quyết định kiến trúc

1. **Pre-render là default**: Nếu `prerender_manifest.json` tồn tại → tự động dùng. Không cần config flag renderMode.
2. **Không CLI mới**: Pre-render qua `python -m sync_engine.tuber_prerender`. MouthEvents tích hợp sẵn vào `build_group_manifest()`.
3. **Backward compatible**: Manifest cũ không có mouthEvents → V1 legacy vẫn hoạt động.
4. **TTS amplitude → mouthEvents**: Đọc WAV trực tiếp từ TTS clip path có sẵn trong TimelineSegment. Không cần build speech_control_audio.wav riêng.
5. **Character box crop**: Pre-render output được crop từ 1920×1080 → kích thước ô character (vd 512×288) → composite với offset, giảm pixel xử lý ~14×.

### Trạng thái

- ✓ P1 (MouthEvents) + P2 (Pre-render) core code
- ✓ Config + docs
- ⬜ Test files: `test_tuber_mouth_events.py`, `test_tuber_prerender.py` (cần viết)
- ⬜ Smoke test Colab
- ⬜ Cleanup: xóa `remotion_tuber/` runtime calls khi V2 ổn định

---

## 2026-06-03: MotionPNGTuber — implement Python orchestration + test suite

### Tóm tắt

- **Mục tiêu**: Implement toàn bộ Python orchestration cho tuber overlay theo plan `motionpngtuber-remotion-prototype-plan.md`.
- **Kết quả chính**: 5 module Python mới (`tuber_config`, `tuber_manifest`, `tuber_artifacts`, `tuber_status`, `tuber_overlay`), CLI `tuber_repair`, tích hợp `--tuber-config` vào `sync_video`, sample config, test Layer 1-3 (46 passed, 7 skipped), docs mới.
- **Subproject `remotion_tuber/`**: Đã hoàn thiện trước đó (render driver, component, mouth warp, prepare-assets auto-detect màu nền `0x08A702`, width ưu tiên cho character).

### File tạo

- [`sync_engine/tuber_config.py`](../sync_engine/tuber_config.py) — Load/validate config, resolve layout (jobName sentinel, tuberRoot, artifactPolicy).
- [`sync_engine/tuber_manifest.py`](../sync_engine/tuber_manifest.py) — Build groups từ timeline (Phase F), export run/group manifest absolute paths (Phase H).
- [`sync_engine/tuber_artifacts.py`](../sync_engine/tuber_artifacts.py) — Promote media/final_render_inputs (Phase E), cleanup overlay_frames/failedGroups theo policy (Phase T), serialize/deserialize ImageOverlayEvent.
- [`sync_engine/tuber_status.py`](../sync_engine/tuber_status.py) — status.json per group (pending/running/done/failed/skipped).
- [`sync_engine/tuber_overlay.py`](../sync_engine/tuber_overlay.py) — Orchestration: prepare-assets, build group base (tái dùng `build_ffmpeg_batch_cmd` — B5), render driver, composite, validate, concat, retry, coordinator `run_tuber_flow_all_in`.
- [`cli/tuber_repair.py`](../cli/tuber_repair.py) — Late repair CLI: đọc run_manifest + final_render_inputs, render tuber muộn, gọi `render_final_video`.
- [`assets/tuber_overlay_config.json`](../assets/tuber_overlay_config.json) — Sample config với `chromakey: {color:"0x08A702", similarity:0.12, blend:0.1}`.
- [`docs/tuber-overlay-guide.md`](../docs/tuber-overlay-guide.md) — Hướng dẫn đầy đủ tham số JSON config + CLI repair + retry logic.
- [`tests/sync_engine/test_tuber_overlay_pipeline.py`](../tests/sync_engine/test_tuber_overlay_pipeline.py) — Test Layer 1 (frame math, group, config, status, serialize), Layer 2 (manifest export, artifact promote), Layer 3 (composite, validate, retry/cleanup mock). 46 passed, 7 skipped (thiếu GPU).

### File sửa

- [`cli/sync_video.py`](../cli/sync_video.py) — Thêm `--tuber-config` arg, load config + swap `stretched_video` → `video_stretched_with_tuber` + fallback ở Phase 5.
- [`pyproject.toml`](../pyproject.toml) — Thêm `tuber-repair = "cli.tuber_repair:main"`.
- [`tests/test_matrix.yaml`](../tests/test_matrix.yaml) — Thêm 3 entry Tuber Overlay Pipeline Layer 1/2/3.
- [`docs/colab-guide.md`](../docs/colab-guide.md) — Thêm `--tuber-config` vào bảng tham số sync-video + section 2.12 Tuber Overlay + CLI `tuber-repair`.

### Quyết định kiến trúc

1. **Width ưu tiên trong `resolveCharacterBox`**: Có `width` → suy `height = width / aspect(mouth_track)`, bỏ qua `height`. Không cần cờ `maintainAspect`. Giữ `objectFit:'fill'` để body + canvas đồng scale, miệng không lệch.
2. **Auto-detect màu nền chromakey**: `prepare-assets` lấy median 4 góc frame đầu làm key color, không hardcode `0x00FF00`. Asset hiện tại nền `~0x08A702`.
3. **Concatenation ngoài retry loop**: Concat `-c copy` chỉ chạy 1 lần sau khi tất cả group pass validate. Chỉ render + composite + validate mới có retry.
4. **`attempt` counter chỉ tăng khi composite/validate fail**: Batch render fail → re-render riêng group đó trong cùng attempt (soft fallback), không tính là retry.
5. **Dùng chung `detect_hevc_nvenc` (SSOT)**: Test skip GPU dùng `_GPU_OK = detect_hevc_nvenc()` khớp fixture `use_gpu` trong `tests/conftest.py`.

### Trạng thái hiện tại

- ✓ 5 module Python orchestration đã implement (tuber_config, tuber_manifest, tuber_artifacts, tuber_status, tuber_overlay).
- ✓ [`cli/tuber_repair.py`](../cli/tuber_repair.py) hoàn chỉnh với late repair flow.
- ✓ Tích hợp vào [`sync_video`](../cli/sync_video.py) (load config → resolve layout → build groups → base.mp4 → render → composite → validate → concat → fallback).
- ✓ Sample config [`assets/tuber_overlay_config.json`](../assets/tuber_overlay_config.json) có chromakey params.
- ✓ Test Layer 1-3 (46 passed, 7 skipped do thiếu GPU NVENC).
- ✓ [`docs/tuber-overlay-guide.md`](../docs/tuber-overlay-guide.md) hoàn chỉnh.
- ✓ [`docs/colab-guide.md`](../docs/colab-guide.md) đã cập nhật.
- ✓ Không hồi quy: [`test_tuber_remotion_validation.py`](../tests/sync_engine/test_tuber_remotion_validation.py) vẫn 6 passed.

### Pending / Next steps

- Smoke test end-to-end trên Colab với video thật 30-60s: `sync-video --tuber-config` → kiểm tra output có tuber overlay + alpha sạch.
- Khi có GPU: bật `enabled: true` trong config, chạy `REMOTION_TUBER_E2E=1` verify Layer 4 Remotion thật.

## 2026-06-01: Sync-video audio mix volume schema

### Tóm tắt

- **Mục tiêu**: Thêm schema volume rõ ràng cho Phase 3 audio assembly của [`sync-video`](../cli/sync_video.py): [`non_voicevox_tts_volume`](../sync_engine/audio_assembler.py:611) cho TTS non-Voicevox và [`mute_audio_volume`](../sync_engine/audio_assembler.py:612) cho audio trong mute chunks.
- **Kết quả chính**: [`compress_tts_clip()`](../sync_engine/audio_assembler.py:159) không còn hardcode boost `volume=1.75`; volume non-Voicevox lấy từ render config, còn Voicevox/Voicevox Nemo tiếp tục dùng `volume_scale` trong [`config/tts_config.yaml`](../config/tts_config.yaml).
- **Tác động workflow**: Backward compatibility được giữ bằng default runtime/config `non_voicevox_tts_volume=1.75`; `mute_audio_volume=1.0` mặc định không đổi loudness mute audio hiện tại.

### File sửa

- [`sync_engine/audio_assembler.py`](../sync_engine/audio_assembler.py) — Đọc volume mới từ `audio_mix_config`, áp gain cho non-Voicevox TTS và mute chunks (`original`, `vocals`, `instrumental`), bỏ qua render volume cho provider bắt đầu bằng `voicevox`.
- [`assets/default_render_config.json`](../assets/default_render_config.json) — Thêm `non_voicevox_tts_volume` và `mute_audio_volume` vào block `audio_mix` mặc định.
- [`docs/sync-video-guide.md`](../docs/sync-video-guide.md) — Cập nhật schema `audio_mix`, mô tả rõ Voicevox family dùng `volume_scale` riêng trong TTS config.
- [`tests/sync_engine/test_audio_assembler.py`](../tests/sync_engine/test_audio_assembler.py) — Thêm Layer 1 regression tests cho filter volume non-Voicevox, Voicevox ignore render volume, và truyền `mute_audio_volume` vào mute chunk.

### Verification

- `python -m compileall -q sync_engine/audio_assembler.py tests/sync_engine/test_audio_assembler.py && python -m pytest tests/sync_engine/test_audio_assembler.py -v` → `15 passed`.

### Trạng thái hiện tại

- ✓ Schema [`audio_mix`](../docs/sync-video-guide.md:324) đã có `non_voicevox_tts_volume`, `mute_audio_volume`, `ambient_volume`, `bgm_volume`.
- ✓ Voicevox/Voicevox Nemo không bị ảnh hưởng bởi render volume mới.
- ✓ Mute chunk volume có default `1.0` và chỉ đổi khi render config override.
- ✓ Compile và test mục tiêu cho [`tests/sync_engine/test_audio_assembler.py`](../tests/sync_engine/test_audio_assembler.py) đã pass.

### Pending / Next steps

- Không còn pending trong phạm vi thay đổi schema volume audio hiện tại.

## 2026-06-01: Fix pre-cut hybrid-copy keep topology collapse

### Tóm tắt

- **Mục tiêu**: Sửa lỗi [`hybrid-copy`](../cli/pre_cut_video.py:53) chỉ tạo 2 file `keep_*` trong khi cùng remove SRT với [`reencode-smooth`](../cli/pre_cut_video.py:54) tạo đúng 4 file.
- **Nguyên nhân**: Nhánh [`run_pre_cut()`](../utils/video_cutter.py:434) đang dùng [`expand_to_keyframes()`](../utils/video_cutter.py:209) để mở rộng **remove ranges** tới keyframe trước/sau. Khi GOP/keyframe thưa, các vùng xóa sau mở rộng bị overlap rồi bị [`normalize_and_merge()`](../utils/video_cutter.py:178) gộp lại, làm mất topology các đoạn keep ở giữa.
- **Kết quả chính**: [`hybrid-copy`](../utils/video_cutter.py:488) vẫn query keyframes để fail-fast nhưng không còn dùng keyframe expansion để thay đổi remove ranges; topology SRT sau safe margin/normalize là SSOT cho số lượng `keep_*`.

### File sửa

- [`utils/video_cutter.py`](../utils/video_cutter.py) — Đổi nhánh [`method == "hybrid-copy"`](../utils/video_cutter.py:488) để đặt `final_remove = normalized`, tránh merge các remove range chỉ vì keyframe expansion.
- [`tests/utils/test_video_cutter.py`](../tests/utils/test_video_cutter.py) — Thêm [`TestLayer2_HybridCopyKeepTopology`](../tests/utils/test_video_cutter.py:742) mô phỏng case 3 remove ranges thực tế và assert tạo đúng 4 `keep_*` parts; chỉnh temp cleanup tests để tạo fake part theo từng FFmpeg command.

### Verification

- `python -m compileall -q utils/video_cutter.py tests/utils/test_video_cutter.py && python -m pytest tests/utils/test_video_cutter.py -v` → `58 passed`.

### Trạng thái hiện tại

- ✓ [`hybrid-copy`](../utils/video_cutter.py:488) không còn collapse các keep ranges hợp lệ do mở rộng vùng xóa theo keyframe.
- ✓ Case 3 remove SRT ranges với safe margin `100ms` tạo 4 keep ranges/4 temp files trong regression test.
- ✓ [`reencode-smooth`](../utils/video_cutter.py:495) vẫn giữ logic snap-to-frame-grid riêng.

### Pending / Next steps

- Chạy lại [`pre-cut-video`](../cli/pre_cut_video.py:119) với video Colab thực tế để xác nhận số file `keep_*` khớp kỳ vọng end-to-end.

## 2026-06-01: HEVC NVENC runtime probe shared utility

### Tóm tắt

- **Mục tiêu**: Sửa false-positive khi phát hiện `hevc_nvenc`: FFmpeg có advertise encoder nhưng encode thật fail do thiếu CUDA/NVIDIA runtime (`libcuda.so.1`).
- **Kết quả chính**: [`utils.ffmpeg_probe.detect_hevc_nvenc()`](../utils/ffmpeg_probe.py:45) là SSOT dùng chung cho runtime probe 2 bước: kiểm tra `ffmpeg -encoders`, sau đó dummy encode 1 frame bằng đúng tham số production `hevc_nvenc -preset p4 -tune hq -cq 28`.
- **Tác động workflow**: [`sync_engine.video_processor.process_video_chunks_parallel()`](../sync_engine/video_processor.py:231), [`sync_engine.renderer.render_final_video()`](../sync_engine/renderer.py:199), [`utils.video_cutter.run_pre_cut()`](../utils/video_cutter.py:443) và fixture test dùng cùng detector, tránh drift giữa sync-video, renderer, pre-cut và tests.

### File sửa

- [`utils/ffmpeg_probe.py`](../utils/ffmpeg_probe.py) — Thêm shared runtime probe có cache, lưu lý do fail gần nhất qua [`utils.ffmpeg_probe.get_hevc_nvenc_unavailable_reason()`](../utils/ffmpeg_probe.py:33), reset cache cho tests và hằng số encoder chung.
- [`sync_engine/video_processor.py`](../sync_engine/video_processor.py) — Dùng shared detector/encoder args, fail-fast phase video chunks kèm chi tiết probe.
- [`sync_engine/renderer.py`](../sync_engine/renderer.py) — Import detector và encoder args trực tiếp từ shared utils để final render không phụ thuộc vào module video processor.
- [`utils/video_cutter.py`](../utils/video_cutter.py) — Bỏ detector local chỉ đọc encoder list; flow `reencode-smooth` dùng shared runtime probe và đưa reason vào RuntimeError.
- [`tests/conftest.py`](../tests/conftest.py) — Fixture `use_gpu` gọi shared detector thay vì duplicate dummy encode.
- [`tests/utils/test_ffmpeg_probe.py`](../tests/utils/test_ffmpeg_probe.py) — Thêm regression tests cho false-positive `ffmpeg -encoders`, cache success và thiếu FFmpeg.
- [`tests/sync_engine/test_video_processor.py`](../tests/sync_engine/test_video_processor.py), [`tests/sync_engine/test_concat_demuxer.py`](../tests/sync_engine/test_concat_demuxer.py), [`tests/sync_engine/test_sync_video_pipeline.py`](../tests/sync_engine/test_sync_video_pipeline.py) — Chuyển test/users sang detector shared utils.
- [`tests/test_matrix.yaml`](../tests/test_matrix.yaml) — Thêm entry Layer 1 cho FFmpeg probe shared utility.

### Verification

- `python -m compileall -q utils sync_engine tests && python -m pytest tests/utils/test_ffmpeg_probe.py tests/sync_engine/test_video_processor.py tests/utils/test_video_cutter.py tests/sync_engine/test_image_overlay.py -v` → `71 passed, 1 skipped`.
- `python -m pytest tests/sync_engine/test_sync_video_pipeline.py -v` → `0 collected / 1 skipped`, exit code `5` vì module pipeline bị skip toàn bộ trong môi trường hiện tại trước collection đầy đủ.

### Quyết định kiến trúc

1. **Runtime usability là SSOT trong utils**: [`ffmpeg -encoders`](../utils/ffmpeg_probe.py:63) chỉ là bước lọc nhanh; dummy encode mới quyết định `hevc_nvenc` usable hay không.
2. **Dùng đúng tham số production trong probe**: Probe dùng chung [`utils.ffmpeg_probe.HEVC_NVENC_VIDEO_ARGS`](../utils/ffmpeg_probe.py:17) để tránh pass giả với cấu hình khác.
3. **Cache theo process**: Kết quả probe được cache để tránh mỗi phase/test gọi FFmpeg dummy encode nhiều lần; tests có [`utils.ffmpeg_probe.reset_hevc_nvenc_probe_cache()`](../utils/ffmpeg_probe.py:38).
4. **Không đặt detector trong sync_engine**: Detector nằm ở [`utils/ffmpeg_probe.py`](../utils/ffmpeg_probe.py) để sync-video, final renderer, pre-cut CLI path và tests dùng chung.

### Trạng thái hiện tại

- ✓ False-positive NVENC do thiếu `libcuda.so.1` đã được chặn ở shared runtime probe.
- ✓ [`sync_engine/renderer.py`](../sync_engine/renderer.py) và [`utils/video_cutter.py`](../utils/video_cutter.py) không còn giữ logic detect riêng gây drift.
- ✓ Test Layer 1 cho video processor không phụ thuộc OpenCV/Numpy nhờ lazy import trong fixture Layer 2.
- ✓ Target compile/tests đã chạy pass; pipeline integration vẫn skip trong môi trường hiện tại do dependency/hardware.

### Pending / Next steps

- Chạy lại sync-video end-to-end trên máy có NVIDIA driver/CUDA runtime usable để xác nhận nhánh render thật không bị skip.
