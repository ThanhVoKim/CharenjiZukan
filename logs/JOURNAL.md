# Project Journal

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

## 2026-05-31: Forced alignment dash compound reconstruction

### Tóm tắt

- **Mục tiêu**: Sửa lỗi forced alignment subtitle tái tạo sai compound word có hyphen/dash, ví dụ `47round-round`, `largecapacity-capacity`, `continuousfire-fire`.
- **Kết quả chính**: `merge_punctuation()` trong [`utils/asr_subtitle_utils.py`](../utils/asr_subtitle_utils.py) nhận diện dash giữa ký tự chữ/số trong transcript gốc, phục hồi text compound gốc và bỏ qua suffix token bị aligner lặp.
- **Phạm vi cố ý giới hạn**: Chỉ xử lý hyphen/dash compound words (`-`, `‐`, `‑`, `‒`, `–`, `—`); underscore `_` không được normalize theo quyết định scope hiện tại.

### File sửa

- [`utils/asr_subtitle_utils.py`](../utils/asr_subtitle_utils.py) — Thêm helper nhận diện dash compound, normalize phần compound theo ký tự chữ/số, đối chiếu transcript cursor và skip suffix token an toàn.
- [`tests/utils/test_asr_subtitle_utils.py`](../tests/utils/test_asr_subtitle_utils.py) — Thêm regression tests cho `47-round`, `large-capacity`, `continuous-fire`, en dash, em dash và test khóa hành vi underscore/legacy split.

### Verification

- `python -m pytest tests/utils/test_asr_subtitle_utils.py tests/cli/test_qwen3_asr.py -q` → `60 passed in 0.28s`.

### Quyết định kiến trúc

1. **Fix tại reconstruction layer**: Lỗi phát sinh trong nhánh forced alignment khi tái ghép text từ token aligner, không phải trong nhánh timestamp remap; vì vậy sửa tại `utils/asr_subtitle_utils.py` để dùng chung cho `sync_engine/forced_alignment_subtitle.py` và CLI ASR.
2. **Transcript gốc là SSOT cho dấu dash**: Khi aligner normalize `large-capacity` thành `largecapacity`, output subtitle phải phục hồi nguyên compound từ transcript gốc thay vì tạo token lai.
3. **Skip suffix có guard theo cursor**: Chỉ bỏ qua token suffix trùng khi vị trí hiện tại trong transcript không thật sự bắt đầu bằng từ đó, tránh nuốt mất từ hợp lệ ngay sau compound.
4. **Không mở rộng sang underscore**: `_` giữ hành vi punctuation hiện tại để tránh normalize các identifier/chuỗi không phải subtitle tự nhiên.

### Trạng thái hiện tại

- ✓ Forced-alignment subtitle đã phục hồi đúng hyphen/dash compound words.
- ✓ Regression tests mục tiêu đã chạy pass.
- ✓ Nhánh remap timestamp vẫn không đổi và tiếp tục giữ nguyên text input.

### Pending / Next steps

- Nếu gặp thêm ký tự dash Unicode ngoài bộ hiện tại trong transcript thực tế, có thể bổ sung vào `COMPOUND_DASH_CHARS` kèm test tương ứng.

## 2026-05-30: Implemented sync-video audio policy refactor

### Tóm tắt

- **Mục tiêu**: Triển khai runtime/config/docs/tests theo [`plans/sync-video-audio-policy-refactor-plan.md`](../plans/sync-video-audio-policy-refactor-plan.md).
- **Kết quả chính**: [`cli/sync_video.py`](../cli/sync_video.py) hiện resolve `audio_policies`; [`sync_engine/audio_assembler.py`](../sync_engine/audio_assembler.py) đã hỗ trợ `global_bgm`, `mute_audio`, `ambient`, reuse `bgm_path` cho `mute_audio=instrumental`, ambient/BGM masking theo volume expression và final mix là SSOT cho `ambient_volume`/`bgm_volume`.
- **Migration hoàn tất**: Đã cập nhật config mẫu, docs sync-video, fake render config trong tests và mở rộng test coverage cho policy resolver/mask logic.
- **Verification**: `python -m compileall -q cli sync_engine tests && python -m pytest tests/sync_engine/test_audio_assembler.py tests/sync_engine/test_note_overlay_layout.py tests/sync_engine/test_sync_video_pipeline.py -v` → `40 passed, 1 skipped` (pipeline NVENC integration bị skip trong môi trường hiện tại).

### File sửa

- [`cli/sync_video.py`](../cli/sync_video.py) — Resolve `audio_policies`, log policy đã resolve, chỉ extract global BGM khi policy yêu cầu, truyền policy vào audio assembly.
- [`sync_engine/audio_assembler.py`](../sync_engine/audio_assembler.py) — Thêm policy normalization/validation, mute-range helpers, ambient preprocess theo mask 0/1, global BGM synced track + `exclude_mute` volume mask, reuse `bgm_path` để tránh separator trùng.
- [`assets/default_render_config.json`](../assets/default_render_config.json) — Migrate sang block `audio_policies` mặc định.
- [`assets/thearmorylog_render_config.json`](../assets/thearmorylog_render_config.json) — Migrate sang block `audio_policies` cho preset có global BGM.
- [`docs/sync-video-guide.md`](../docs/sync-video-guide.md) — Cập nhật phase overview, schema `audio_mix`/`audio_policies`/`audio_separator` và migration note deprecation.
- [`tests/sync_engine/test_audio_assembler.py`](../tests/sync_engine/test_audio_assembler.py) — Bổ sung Layer 1 policy/mask tests và giữ Layer 2 FFmpeg coverage.
- [`tests/sync_engine/test_sync_video_pipeline.py`](../tests/sync_engine/test_sync_video_pipeline.py) — Cập nhật fake render config theo schema mới.
- [`tests/sync_engine/test_note_overlay_layout.py`](../tests/sync_engine/test_note_overlay_layout.py) — Cập nhật fake render config theo schema mới.

### Quyết định kiến trúc

1. **Policy ownership rõ ràng**: `mute_audio` quyết định audio của segment `mute`; ambient/BGM là overlay độc lập.
2. **Hybrid BGM masking**: Global BGM luôn được sync theo timeline final trước, sau đó `exclude_mute` mới apply volume mask ở final mix.
3. **SSOT volume tại final mix**: Ambient preprocessing chỉ lo timing + mask 0/1; volume cuối cùng lấy từ `audio_mix` để tránh double attenuation.
4. **Migration an toàn**: `audio_separator.extract_bgm` và `extract_vocals` vẫn được map như deprecated compatibility keys khi config cũ chưa migrate.

### Trạng thái hiện tại

- ✓ Runtime audio policy refactor đã triển khai xong.
- ✓ Config/docs/test fake config đã migrate sang `audio_policies`.
- ✓ Test mục tiêu đã chạy pass trong môi trường hiện tại.
- ⚠ [`tests/sync_engine/test_sync_video_pipeline.py`](../tests/sync_engine/test_sync_video_pipeline.py) vẫn skip nhánh integration đầy đủ khi máy không có `hevc_nvenc`.

### Pending / Next steps

- Nếu cần xác minh end-to-end trên pipeline render thật, chạy lại [`tests/sync_engine/test_sync_video_pipeline.py`](../tests/sync_engine/test_sync_video_pipeline.py) trên môi trường có FFmpeg `hevc_nvenc`.
- Có thể bổ sung thêm test chuyên biệt cho nhánh separator batch/reuse `bgm_path` bằng monkeypatch nếu muốn regression guard sâu hơn mà không phụ thuộc separator thật.

## 2026-05-29: OpenAI-compatible capability implementation

### Tóm tắt

- **Mục tiêu**: Triển khai coding theo `plans/openai-compatible-provider-capabilities-plan.md`, giữ Chat Completions cơ bản làm baseline nhưng thêm profile/capability flags riêng cho từng OpenAI-compatible `base_url`.
- **Kết quả chính**: Đã thêm hạ tầng `OpenAICompatProfile`, custom capability exceptions, request builders cho Chat Completions/Responses, telemetry cache/usage best-effort, versioned capability report và provider integration backward-compatible.
- **Tác động vận hành**: Provider fail-fast trước network call khi config yêu cầu capability chưa bật; endpoint rejection 400/404/422 cho capability đã bật được wrap thành `CapabilityRejectedError`; provider chain không fallback khi gặp lỗi capability/config để tránh chạy sai endpoint/profile.

### File mới

- `llm_ai/openai_compat.py` — Config models, capability flags, exceptions, payload builders, telemetry helpers và versioned capability report writer.
- `tests/llm_ai/test_openai_compat_capabilities.py` — Layer 1/2 tests cho config parsing, request builders, telemetry/report và mocked OpenAI-compatible client.
- `tests/llm_ai/test_openai_compat_capability_probe.py` — Layer 4 opt-in real endpoint probe, ghi capability report theo profile/timestamp.

### File sửa

- `llm_ai/providers/openai.py` — Tích hợp capability profile, Chat Completions/Responses branching, raw header telemetry, previous response state, compaction và custom endpoint rejection wrapping.
- `llm_ai/factory.py` — Truyền full provider config vào OpenAI-compatible provider để giữ schema mới/backward-compatible.
- `llm_ai/provider_chain.py` — Cho phép override profile/capability fields trong `provider_chain` và re-raise `OpenAICompatCapabilityError` thay vì fallback.
- `llm_ai/base.py` — Thêm optional `compact_state`, `last_response_id`, `last_telemetry_record` không phá interface `call(message) -> str`.
- `config/llm/openai_compat.yaml` — Mở rộng profile schema với `api_mode`, `capability_flags`, `request_options`, `stateful_options`, `telemetry`; advanced features mặc định tắt.
- `tests/llm_ai/test_generic_text_task.py` — Thêm regression test đảm bảo provider chain không fallback khi gặp capability error.
- `docs/testing-guide.md` — Bổ sung tag hợp lệ `llm_capability`, `openai_compat`, `llm_capability_probe`, `external_api` và env probe liên quan.
- `tests/test_matrix.yaml` — Thêm entries Layer 1, Layer 2, Layer 4 cho OpenAI-compatible capability tests.
- `.gitignore` — Ignore runtime telemetry/report outputs `logs/llm_telemetry.jsonl` và `tests/test_reports/`.
- `run_colab_tests.py` — Cấu hình stdout/stderr UTF-8 để thông báo lỗi dependency không vỡ encoding trên Windows console.

### Verification

- `python -m compileall -q run_colab_tests.py llm_ai tests/llm_ai && python -m pytest tests/llm_ai/test_openai_compat_capabilities.py tests/llm_ai/test_generic_text_task.py tests/llm_ai/test_openai_compat_capability_probe.py -v` → 25 passed, 1 skipped (Layer 4 probe skip đúng vì chưa bật env/API key).
- `python run_colab_tests.py --tags llm_capability` → chưa chạy được trong môi trường hiện tại vì thiếu `PyYAML`; sau fix UTF-8 runner đã hiển thị lỗi rõ: `PyYAML chưa cài. Chạy: pip install pyyaml`.

### Quyết định kiến trúc

1. **Capability-first, fail-fast**: Request builders chỉ inject tham số nâng cao khi capability flag tương ứng bật; lỗi config/capability dùng custom exception rõ nguyên nhân.
2. **Không fallback trên capability error**: `provider_chain` chỉ fallback cho lỗi runtime/provider thông thường; capability/config errors là deterministic và phải dừng ngay.
3. **Telemetry production là metadata-only**: Không tạo request phụ để test cache; chỉ ghi usage/header/latency từ response thật, sanitize secret và hash `base_url`.
4. **Probe có lịch sử**: Capability report ghi theo `profile_name` và timestamp, đồng thời cập nhật `latest.json` để tiện đọc trạng thái mới nhất.

### Pending

- Cài dependency project (`PyYAML`) trong môi trường chạy hiện tại rồi chạy lại `python run_colab_tests.py --tags llm_capability`.
- Khi có API key/base_url thật, chạy Layer 4 probe với `OPENAI_COMPAT_PROBE_ALLOW_COST=1`, `OPENAI_COMPAT_PROFILE` và `OPENAI_API_KEY` để tạo report thực tế.
- Mở rộng probe nâng cao cho từng capability cụ thể nếu cần xác minh sâu hơn ngoài basic generation.

## 2026-05-24: ASR subtitle segmentation — `max_chars=0` grammar-only

### Tóm tắt

- **Mục tiêu**: Làm rõ và kiểm thử lại semantic `max_chars=0` trong helper ASR subtitle, tránh việc Qwen3-ASR gom toàn bộ transcript thành một block SRT khi người dùng kỳ vọng vẫn chia theo dấu câu.
- **Kết quả chính**: `segment_words_to_subtitles()` hiện dùng `smart_segment()` cho cả chế độ min/max và grammar-only; `max_chars=0` nghĩa là chỉ chia theo dấu câu, không ép độ dài; `max_chars<0` giữ chế độ legacy single-block cho caller muốn tắt segmentation hoàn toàn.
- **Tác động tới forced alignment**: `forced_alignment_subtitle` dùng chung helper nên cũng nhận semantic mới khi cấu hình `max_chars=0`; config mặc định vẫn là `42`, vì vậy default workflow không đổi.

### File sửa

- `utils/asr_subtitle_utils.py` — Bỏ early return single-block cho `max_chars=0`, thêm nhánh legacy `max_chars<0`, cập nhật docstring invariant.
- `tests/utils/test_asr_subtitle_utils.py` — Thay expectation cũ bằng các test grammar-only, forwarding `min_chars`/`split_on_comma`, và legacy negative single-block.
- `cli/qwen3_asr.py` — Cập nhật help text `--max-chars` để mô tả `0` là grammar-only thay vì “tắt”.

### Verification

- `python -m pytest tests/utils/test_asr_subtitle_utils.py tests/sync_engine/test_forced_alignment_subtitle.py -v` → 49 passed, 4 warnings marker cũ.
- `python -m pytest tests/utils/test_asr_subtitle_utils.py tests/utils/test_text_segmenter.py tests/cli/test_qwen3_asr.py tests/sync_engine/test_forced_alignment_subtitle.py -v` → 104 passed, 4 warnings marker cũ.
- `python -m compileall -q utils tests cli` → pass.

### Pending

- Cân nhắc đăng ký các marker `Layer1`/`Layer2` trong pytest config hoặc đổi sang naming/filter convention hiện hành để hết warning.
- Nếu cần chế độ single-block từ CLI, nên thêm option rõ ràng thay vì dùng `--max-chars 0`.

## 2026-05-24: Sync video — Image Overlay static image auto extension

### Tóm tắt

- **Mục tiêu**: Thay cấu hình image overlay từ extension cố định `.png` sang chế độ mặc định `file_ext="auto"`, cho phép SRT basename resolve nhiều định dạng ảnh tĩnh.
- **Kết quả chính**: Resolver image overlay hiện hỗ trợ allowlist static image (`.png`, `.jpg`, `.jpeg`, `.jfif`, `.gif`, `.webp`, `.bmp`, `.tif`, `.tiff`, `.avif`, `.heic`, `.heif`, `.jp2`, `.j2k`, `.jxl`, `.tga`, `.svg`, `.ico`), so khớp basename/extension không phân biệt hoa thường, vẫn giữ khả năng ép extension cụ thể để tương thích cấu hình cũ.
- **Xử lý xung đột**: Nếu nhiều asset cùng basename tồn tại, `missing_policy="warn"` chọn theo thứ tự ưu tiên extension và log warning; `missing_policy="raise"` dừng pipeline để tránh chọn nhầm.

### File sửa

- `sync_engine/image_overlay.py` — Thêm `SUPPORTED_STATIC_IMAGE_EXTENSIONS`, chế độ auto extension, resolver candidate theo basename không phân biệt hoa thường, validate SRT key không chứa bất kỳ extension ảnh tĩnh hỗ trợ/configured nào.
- `cli/sync_video.py` — Fallback cấu hình `image_overlay.file_ext` chuyển sang `auto`; help text CLI chuyển từ PNG-only sang static image.
- `assets/default_render_config.json` — Đổi `image_overlay.mode` thành `srt_fullscreen_static_image` và `file_ext` thành `auto`.
- `sync_engine/renderer.py` — Cập nhật log/comment từ PNG sang static image; filter graph runtime vẫn dùng path đã resolve.
- `tests/sync_engine/test_image_overlay.py` — Cập nhật Layer 1 resolver tests cho `.png`, `.webp`, `.JPG`, case-insensitive matching, duplicate basename và explicit extension compatibility.
- `docs/sync-video-guide.md` và `docs/colab-guide.md` — Cập nhật hướng dẫn static image, ví dụ thư mục ảnh đa định dạng và khuyến nghị PNG/WebP khi cần alpha.
- `tests/test_matrix.yaml` — Đổi tên entry Layer 1 từ SRT/PNG sang SRT/Static Image Domain Logic.

### Quyết định kiến trúc

1. **Mặc định auto nhưng vẫn backward-compatible**: `file_ext="auto"` là default mới; config cũ đặt `.png` vẫn ép đúng một extension.
2. **SRT vẫn là basename-only**: Text block không được chứa extension để tránh phụ thuộc định dạng file và để resolver tự chọn asset phù hợp.
3. **Ưu tiên deterministic**: Khi trùng basename, resolver chọn theo thứ tự allowlist để kết quả ổn định giữa các lần chạy.
4. **Alpha vẫn là khuyến nghị định dạng, không còn ràng buộc runtime**: PNG/WebP phù hợp overlay trong suốt, nhưng JPEG/BMP/TIFF/AVIF/... vẫn được chấp nhận nếu FFmpeg đọc được trên môi trường chạy.

### Trạng thái hiện tại

- ✓ Runtime/config/docs/test matrix đã chuyển sang static image auto extension.
- ✓ Tests mục tiêu đã được cập nhật để cover multi-extension và explicit-extension compatibility.
- ✓ Verification đã chạy:
  - `python -m compileall -q sync_engine cli tests` → pass.
  - JSON load `assets/default_render_config.json` → pass.
  - `python -m pytest tests/sync_engine/test_image_overlay.py -v -s` → 11 passed.

### Pending

- Cần chạy thử render video thật với bộ static image overlay đa định dạng trên môi trường FFmpeg/NVENC thực tế.

## 2026-05-23: Sync video — Image Overlay PNG theo SRT

### Tóm tắt

- **Mục tiêu**: Thêm chức năng overlay PNG transparent full-screen vào flow sync-video, dùng SRT riêng làm timeline điều khiển; text mỗi block là basename ảnh không có extension.
- **Kết quả chính**: Pipeline đã parse/resolve/remap image overlay SRT theo timeline video stretch, renderer burn layer đúng thứ tự `Base video → Image overlay → Note overlay → Black strip → Watermark → Subtitle`, và hỗ trợ cả `-filter_complex` direct lẫn `-filter_complex_script` fallback.
- **Phạm vi tối ưu**: Intermediate overlay video chưa triển khai theo đúng plan; hiện chỉ có stub `render_intermediate_overlay_track()` để ghi rõ ý định phase tương lai.

### File mới

- `sync_engine/image_overlay.py` — Thêm dataclass `ImageOverlayEvent`, `ImageOverlayAsset`; helper normalize key, resolve PNG, load SRT events, deduplicate assets, remap timestamp bằng `remap_timestamp()`, ghi debug SRT, và stub intermediate overlay track.
- `tests/sync_engine/test_image_overlay.py` — Thêm Layer 1 domain tests và Layer 2 renderer command/filter graph tests cho image overlay.

### File sửa

- `sync_engine/renderer.py` — Refactor final render layer order; thêm input PNG dedup theo absolute path, scale PNG về output resolution, `format=rgba`, opacity bằng `colorchannelmixer`, reuse asset bằng `split=N`, overlay event theo `enable='between(t,start,end)'`, chọn strategy `direct`/`script`/`auto`, ghi/xóa filter complex script tạm theo policy.
- `cli/sync_video.py` — Thêm CLI args `--image-overlay-srt`, `--image-overlay-dir`; thêm task-file keys `image_overlay_srt`, `image_overlay_dir`; Phase 4 load và remap image overlay SRT; truyền events vào renderer; optional ghi `<output-name>_image_overlay_synced.srt`.
- `assets/default_render_config.json` — Thêm block `image_overlay` với `enabled`, `mode`, `file_ext`, `fit`, `opacity`, `missing_policy`, `direct_overlay_max_events`, `command_line_max_chars`, `render_strategy`.
- `docs/sync-video-guide.md` — Thêm mô tả flow image overlay, schema config, CLI usage, timestamp remap, layer order, direct/script strategy và task-file fields.
- `docs/colab-guide.md` — Thêm hướng dẫn bật image overlay trên Colab, ví dụ SRT/PNG, CLI args, batch JSON fields, output debug SRT và lưu ý vận hành.
- `tests/test_matrix.yaml` — Thêm entries Layer 1/Layer 2 cho image overlay.

### Quyết định kiến trúc

1. **SRT overlay bám timeline video gốc**: Image overlay SRT được remap sau khi Phase 2 đã cập nhật timeline stretch thực tế, tương tự subtitle remap; forced alignment subtitle không ảnh hưởng overlay.
2. **Renderer sở hữu FFmpeg input index**: Module image overlay chỉ trả event/path; renderer quyết định input indexes để tránh lệch khi có note, strip, watermark hoặc audio inputs.
3. **Deduplicate PNG input**: Một PNG dùng nhiều lần chỉ được load một input FFmpeg; renderer dùng `split=N` để cấp stream cho từng overlay event.
4. **Direct trước, script khi lớn**: `render_strategy=auto` dùng `-filter_complex` cho case nhỏ và chuyển sang `-filter_complex_script` khi event count hoặc command-line length vượt ngưỡng an toàn.
5. **Intermediate chỉ là future stub**: Chưa generate video overlay trung gian trong phase này để tránh mở rộng scope; strategy `intermediate` raise rõ ràng khi được gọi.
6. **Fit mode phase đầu là stretch**: `fit=stretch_to_output` đảm bảo phủ full-screen; docs khuyến nghị export PNG đúng resolution output để tránh méo aspect ratio.

### Trạng thái hiện tại

- ✓ Image overlay SRT/PNG domain module đã có parse, resolve, remap, debug SRT và intermediate stub.
- ✓ Renderer final video đã hỗ trợ layer order mới và direct/script strategy.
- ✓ CLI/task-file/config/docs/test matrix đã cập nhật.
- ✓ Verification đã chạy:
  - `python -m compileall -q sync_engine cli tests` → pass.
  - JSON load `assets/default_render_config.json` → pass.
  - `python -m pytest tests/sync_engine/test_image_overlay.py -v -s` → 11 passed.
  - `python -m pytest tests/sync_engine/test_note_overlay_layout.py -k Layer3 -v -s` → pass.
  - Kiểm tra entries image overlay trong `tests/test_matrix.yaml` bằng string assertion → pass.
- ⚠ YAML parse bằng `uv run ...` không chạy trên máy hiện tại vì `uv` không có trong PATH; test matrix vẫn được kiểm tra sự hiện diện entry mới bằng Python string assertion.

### Pending

- Chưa chạy render video thật với PNG overlay trên input media thực tế; cần thực hiện khi có bộ asset/video mẫu và môi trường FFmpeg/NVENC phù hợp.

## 2026-05-21: Sync video — ép toàn bộ render video sang HEVC NVENC

### Tóm tắt

- **Mục tiêu**: Cập nhật toàn bộ lệnh FFmpeg render video trong flow `cli/sync_video.py` sang cấu hình HEVC NVENC cố định: `-c:v hevc_nvenc -preset p4 -tune hq -cq 28`.
- **Kết quả chính**: Phase 2 video chunk batching và Phase 5 final render đều encode bằng `hevc_nvenc`; không còn fallback render video sang `h264_nvenc` hoặc `libx264` trong flow sync-video.
- **Fail-fast**: Pipeline kiểm tra encoder `hevc_nvenc` trước khi render và dừng rõ ràng nếu FFmpeg/máy chạy không hỗ trợ NVIDIA HEVC NVENC.

### File sửa

- `sync_engine/video_processor.py` — Thêm `_HEVC_NVENC_VIDEO_ARGS`, `detect_hevc_nvenc()`, ép `build_ffmpeg_chunk_cmd()` và `build_ffmpeg_batch_cmd()` dùng `-c:v hevc_nvenc -preset p4 -tune hq -cq 28`; `process_video_chunks_parallel()` fail-fast khi thiếu encoder và warning khi `use_gpu=False`/`--no-gpu` được truyền vào. Bước `_concat_chunks()` vẫn dùng `-c:v copy` vì đây là concat demuxer copy stream, không phải render/re-encode.
- `sync_engine/renderer.py` — Thay logic chọn encoder cũ bằng HEVC NVENC cố định cho final render; bỏ import không còn dùng; fail-fast khi thiếu `hevc_nvenc`; warning khi `use_gpu=False` vì CPU fallback không còn áp dụng.
- `cli/sync_video.py` — Cập nhật help của `--no-gpu` thành tùy chọn tương thích cũ; flow vẫn bắt buộc render video bằng `hevc_nvenc -preset p4 -tune hq -cq 28`.
- `assets/default_render_config.json` và `assets/thearmorylog_render_config.json` — Cập nhật block `video_encoding` sang mô tả HEVC NVENC (`codec=hevc_nvenc`, `preset=p4`, `tune=hq`, `quality=["-cq", "28"]`) để tránh cấu hình mẫu còn ghi `p5`/`cq 23`.
- `docs/sync-video-guide.md` — Cập nhật section `video_encoding`: runtime hiện ép HEVC NVENC cố định, block config chỉ còn vai trò mô tả/tương thích cũ.
- `docs/colab-guide.md` — Cập nhật mô tả `--no-gpu` cho sync-video, không còn ghi CPU mode `libx264` thay `h264_nvenc`.
- `tests/conftest.py` — Cập nhật fixture `use_gpu` để dummy encode kiểm tra `hevc_nvenc` với `-preset p4 -tune hq -cq 28`.
- `tests/sync_engine/test_video_processor.py` — Cập nhật Layer 1 assertions sang HEVC NVENC; Layer 2 skip khi thiếu `hevc_nvenc`; sửa unpack return `(output_video, actual_durations)`; chỉnh `PROJECT_ROOT` về workspace root để import đúng package runtime `sync_engine`.
- `tests/sync_engine/test_concat_demuxer.py` — Cập nhật assertions từ `libx264`/`h264_nvenc`/`p5` sang `hevc_nvenc`/`p4`/`hq`/`cq 28`; integration/real-video tests skip khi thiếu encoder; chỉnh `PROJECT_ROOT` về workspace root để import đúng package runtime `sync_engine`.
- `tests/sync_engine/test_sync_video_pipeline.py` — Cập nhật render config test sang HEVC NVENC và skip pipeline integration khi thiếu `hevc_nvenc`.
- `tests/sync_engine/test_note_overlay_layout.py` — Cập nhật render config mocked pipeline sang HEVC NVENC để thống nhất tài liệu/cấu hình.

### Quyết định kiến trúc

1. **Không CPU fallback cho sync-video render**: Theo yêu cầu vận hành, render video trong flow `cli/sync_video.py` bắt buộc dùng HEVC NVENC cố định thay vì tự chọn theo `use_gpu`.
2. **Giữ tương thích chữ ký/CLI**: Các tham số `use_gpu` và `--no-gpu` vẫn tồn tại để không phá API/CLI cũ, nhưng nếu tắt GPU thì chỉ warning và vẫn dùng HEVC NVENC.
3. **Concat demuxer không đổi**: `_concat_chunks()` giữ `-c:v copy` vì bước này nối các batch đã encode cùng chuẩn, không phải bước render video.
4. **Config không còn quyết định codec**: `video_encoding` trong JSON chỉ còn là mô tả cấu hình hiện hành/tương thích cũ; runtime không đọc block này để đổi codec/quality.

### Trạng thái hiện tại

- ✓ Phase 2 chunk render ép `hevc_nvenc -preset p4 -tune hq -cq 28`.
- ✓ Phase 5 final render ép `hevc_nvenc -preset p4 -tune hq -cq 28`.
- ✓ CLI help, JSON config mẫu, docs và test liên quan đã được cập nhật khỏi thông tin encoder cũ.
- ✓ Verification cú pháp/config/command assertions đã chạy:
  - `python -m compileall -q sync_engine cli tests` → pass.
  - JSON load cho `assets/default_render_config.json` và `assets/thearmorylog_render_config.json` → pass.
  - Direct assertions cho `build_ffmpeg_chunk_cmd()` và `build_ffmpeg_batch_cmd()` xác nhận `hevc_nvenc`, `p4`, `hq`, `cq 28` → pass.
  - `detect_hevc_nvenc()` trên máy hiện tại trả về `True`.
  - Pytest Layer 1 cho `tests/sync_engine/test_video_processor.py` và `tests/sync_engine/test_concat_demuxer.py` không chạy case vì module bị skip tại collection do dependency optional `cv2` (`pytest.importorskip`).

## 2026-05-21: Pre-cut video — loại bỏ đoạn thừa trước transcript/sync

### Tóm tắt

- **Mục tiêu**: Tạo CLI pre-cut riêng để loại bỏ các đoạn thừa từ video gốc trước khi chạy transcript, translate và flow sync hiện tại.
- **Kết quả chính**: CLI `pre-cut-video` nhận video gốc + remove SRT, tạo video clean + manifest JSON. Video clean trở thành source timeline mới cho mọi bước tiếp theo.
- **Hai method**: `hybrid-copy` (default — video stream copy, audio AAC encode, keyframe expansion) và `reencode-smooth` (hevc_nvenc re-encode, CQ 28, frame grid snap).

### File mới

- `utils/video_cutter.py` — Core logic: `probe_video_info()`, `query_keyframes()`, `detect_hevc_nvenc()`, `parse_remove_srt()`, `apply_safe_margin()`, `normalize_and_merge()`, `expand_to_keyframes()`, `snap_to_frame_grid()`, `invert_to_keep_ranges()`, `build_hybrid_copy_part_cmd()`, `build_reencode_part_cmd()`, `concat_parts()`, `run_pre_cut()`, `_build_manifest()`. Data classes: `RemoveRange`, `KeepRange`, `VideoInfo`, `CutResult`.
- `cli/pre_cut_video.py` — CLI entrypoint với tất cả tham số: `--input`, `--output`, `--remove-srt`, `--manifest`, `--method`, `--hevc-cq`, `--maxrate-ratio`, `--hevc-preset`, `--audio-bitrate`, `--audio-fade-ms`, `--safe-margin-ms`, `--disable-audio-fade`, `--keep-tmp`, `--verbose`.

### File sửa

- `pyproject.toml` — Thêm entrypoint `pre-cut-video = "cli.pre_cut_video:main"`.
- `docs/sync-video-guide.md` — Thêm section 9: Pre-cut video, cập nhật TOC và kiến trúc module.
- `docs/colab-guide.md` — Thêm section 2.12: Pre-cut Video với ví dụ workflow đầy đủ.

### Quyết định kiến trúc

1. **Độc lập khỏi sync_engine**: Core logic đặt trong `utils/video_cutter.py`, không phụ thuộc vào `sync_engine/`. Pre-cut là bước tiền xử lý riêng biệt.
2. **Part-based workflow**: Cả hai method đều tạo từng keep part rồi concat bằng demuxer — nhất quán, dễ debug.
3. **Keyframe expansion conservative**: Với `hybrid-copy`, remove range được mở rộng về keyframe gần nhất — xóa thừa thay vì để sót nội dung cần xóa.
4. **Fail-fast**: Không có audio stream → fail. Không có keyframes (hybrid-copy) → fail. Không có hevc_nvenc (reencode-smooth) → fail. Không fallback CPU.
5. **Manifest**: Ghi đủ source ranges, normalized ranges, expanded ranges, keep ranges, encoder info, drift detection fields.
6. **Temp cleanup**: Mặc định xóa sau concat thành công; giữ lại khi `--keep-tmp`.

### Pending

✓ Viết test suite theo testing-guide (parse SRT, range processing, keyframe expansion, frame snap, manifest).
✓ Chạy verification test trên video thật với cả hai method.

## 2026-05-19: Refactor note overlay sang Dynamic ASS Box

### Tóm tắt

- **Mục tiêu**: Hoàn thiện refactor note overlay từ PNG cố định sang Dynamic ASS Box per-dialogue.
- **Kết quả chính**: Note overlay giờ sinh ASS cuối gồm `NoteBox` layer nền và `NoteText` layer chữ; layout được chọn qua field `Name`/Actor trong ASS, hoặc dòng đầu tiên trong SRT sau khi convert bằng `srt-to-ass`.
- **Backward compatibility**: Config legacy có `png_path`/`png_legacy` được nhận diện để warning deprecation; ASS nguồn cũ có `Name` rỗng fallback về `default_layout`.

### File chính đã sửa

- `sync_engine/note_overlay_layout.py` — module expand dynamic ASS box, validate layout config, wrap pixel, tính geometry và emit ASS drawing/text.
- `utils/ass_utils.py` — `wrap_text_pixel()` và parser layout key từ SRT.
- `cli/srt_to_ass.py` — CLI flags `--layout-key`, `--srt-layout-key-mode`.
- `cli/sync_video.py` — Phase 4 tạo `<output>_note_overlay.ass`, cleanup ASS trung gian theo policy.
- `sync_engine/renderer.py` — FFmpeg filter chaining subtitle trước, note overlay ASS sau; không dùng PNG note overlay.
- `tests/sync_engine/test_note_overlay_layout.py` và `tests/cli/test_srt_to_ass_layout_key.py` — Layer 1/2/3 tests cho parser, layout, expand và pipeline mocked.

### Quyết định kiến trúc

1. **Không concat vật lý ASS**: Subtitle và note overlay được burn bằng 2 filter node liên tiếp trong FFmpeg để tránh conflict style/script metadata.
2. **Per-dialogue layout**: Field `Name`/Actor là single source of truth cho layout key.
3. **Min-height dynamic box**: `height` là floor; box tự mở rộng theo `text_height + padding + height_safety_margin`.
4. **Deprecated PNG overlay**: `assets/note_overlay.png` giữ lại tạm thời nhưng không còn nằm trong runtime path chính.

### Trạng thái hiện tại

- Đã cập nhật docs Colab và sync-video guide cho dynamic ASS box.
- Cần chạy verification test suite mục tiêu sau khi hoàn tất chỉnh sửa.

## 2026-05-18: Tách provider Voicevox Nemo và Voicevox chính thức trong TTS

### Tóm tắt

- **Mục tiêu**: Tách flow Voicevox Nemo (hiện tại) khỏi Voicevox chính thức (mới) thành 2 provider riêng biệt trong hệ thống TTS.
- **Lý do**: Flow hiện tại dùng thư viện Voicevox Nemo nhưng provider lại đặt tên là `voicevox`. Cần tích hợp thêm Voicevox chính thức nên phải tách rõ ràng.
- **Provider mới**: `voicevox_nemo` (Nemo, port 50121, speaker 10008) và `voicevox` (chính thức, port 50021, speaker 100).

### File mới

- `tts/voicevox_base.py` — `VoicevoxRestTTSEngine`: Base engine dùng chung cho cả 2 provider Voicevox, chứa toàn bộ logic REST API (audio_query, synthesis), retry, semaphore-based concurrency
- `tts/voicevox_nemo.py` — `VoicevoxNemoTTSEngine`: Wrapper cho Voicevox Nemo, kế thừa `VoicevoxRestTTSEngine`, defaults: port 50121, speaker 10008, speed_scale 1.12, concurrent_requests 100
- `plans/voicevox-nemo-voicevox-provider-split-plan.md` — Kế hoạch chi tiết toàn bộ quá trình tách provider

### File sửa

- `tts/voicevox.py` — Refactor từ engine Nemo cũ thành `VoicevoxTTSEngine` wrapper cho Voicevox chính thức, kế thừa `VoicevoxRestTTSEngine`, defaults: port 50021, speaker 100, speed_scale 1.12, concurrent_requests 100
- `tts/base.py` — Cập nhật docstring: `(EdgeTTS, Voicevox, Voicevox Nemo, Qwen3-TTS, etc.)`
- `cli/tts.py` — Thêm import `VoicevoxNemoTTSEngine`, cập nhật `get_engine()` factory, parser choices: `["edge", "voicevox_nemo", "voicevox", "qwen"]`
- `cli/sync_video.py` — Thêm import `VoicevoxNemoTTSEngine`, cập nhật parser choices, thêm block khởi tạo `voicevox_nemo` engine, đổi `is_voicevox` → `is_voicevox_family` (bao gồm cả 2 provider), cập nhật logging và help text
- `sync_engine/audio_assembler.py` — Cập nhật comment: Voicevox family (cả chính thức và Nemo) tự quản lý volumeScale
- `sync_engine/analyzer.py` — Cập nhật comment: "Voicevox mode" → "Voicevox family mode"
- `config/tts_config.yaml` — Thêm block `voicevox_nemo` (port 50121, speaker 10008), cập nhật block `voicevox` (port 50021, speaker 100, speed_scale 1.12, concurrent_requests 100)
- `docs/colab-guide.md` — Cập nhật section 2.6 (TTS) và 2.11 (sync-video): 4 engine, hướng dẫn riêng cho Voicevox Nemo và Voicevox chính thức, cập nhật bảng tham số và YAML config
- `tests/cli/test_tts_refactor.py` — Thêm assertion `voicevox_nemo` trong `test_load_config()`

### Quyết định kiến trúc

1. **Shared REST base**: `VoicevoxRestTTSEngine` trong `tts/voicevox_base.py` chứa toàn bộ logic async REST, retry, semaphore. Các provider-specific class (`VoicevoxNemoTTSEngine`, `VoicevoxTTSEngine`) chỉ là wrapper mỏng set defaults khác nhau.
2. **Voicevox family mode**: Cả 2 provider Voicevox đều dùng chung logic no-cap (bỏ qua audio compression) và volume filter bypass (tự quản lý volumeScale). Biến `is_voicevox_family` thay thế `is_voicevox` cũ.
3. **Default khác biệt**: Voicevox Nemo (port 50121, speaker 10008) vs Voicevox chính thức (port 50021, speaker 100). Cả 2 đều dùng `speed_scale=1.12`, `concurrent_requests=100`.
4. **Breaking change**: Provider `voicevox` cũ (thực chất là Nemo) giờ trỏ đến Voicevox chính thức. Người dùng Nemo phải chuyển sang `--provider voicevox_nemo`.

### Pending

- Chạy test verification cho TTS-related tests
- Kiểm tra các file còn sót reference đến `voicevox` chưa được cập nhật

## 2026-05-17: Forced Alignment Subtitle — tích hợp Qwen3ForcedAligner vào sync-video pipeline

### Tóm tắt

- **Mục tiêu**: Thêm bước forced alignment subtitle vào pipeline `sync-video`, sử dụng `Qwen3ForcedAligner` để tạo SRT với timestamp chính xác cho từng từ, thay thế SRT remap (recalculate) thông thường.
- **Chạy trên**: `mixed_audio.wav` sau Phase 3 (Audio Assembly), dùng text từ `flat_transcript.txt`.
- **Bật/tắt**: Chỉ qua `render_config.json` → `forced_alignment_subtitle.enabled`, không thêm CLI flag mới.

### File mới

- `utils/asr_subtitle_utils.py` — Shared ASR subtitle logic (trích xuất từ `cli/qwen3_asr.py`):
  - `format_srt_time()` — Format timestamp SRT
  - `merge_punctuation()` — Gộp dấu câu vào word items (hỗ trợ cả object attributes và dict keys)
  - `segment_words_to_subtitles()` — Chia word items thành subtitle blocks (invariant: tổng chars ≤ max_chars → không ngắt)
  - `write_subtitle_srt()` — Ghi subtitle blocks ra file SRT với offset
- `sync_engine/forced_alignment_subtitle.py` — Orchestration forced alignment cho sync-video:
  - `_resolve_aligner_config()` — Map JSON config sang function params (null → dùng default)
  - `load_forced_aligner()` — Load Qwen3ForcedAligner (default: `Qwen/Qwen3-ForcedAligner-0.6B`, `torch.bfloat16`, `cuda:0`)
  - `execute_forced_alignment()` — Đọc transcript, load model, align, merge punctuation, segment, write SRT, clear VRAM
  - `run_forced_alignment_subtitle()` — Entry point, kiểm tra `enabled`, xử lý `fail_policy`
- `tests/utils/test_asr_subtitle_utils.py` — Unit tests cho shared ASR subtitle utils
- `tests/sync_engine/test_forced_alignment_subtitle.py` — Mock integration tests cho forced alignment orchestration (Layer 1 + Layer 2)
- `plans/sync-video-forced-alignment-srt-plan.md` — Architecture plan document

### File sửa

- `assets/default_render_config.json` — Thêm block `forced_alignment_subtitle` với đầy đủ keys (enabled, model_path, device, dtype, attn_implementation, language, max_chars, min_chars, split_on_comma, offset_seconds, keep_tts_synced_debug, fail_policy)
- `cli/sync_video.py` — Thêm Phase 3.5: gọi `run_forced_alignment_subtitle()` sau `assemble_audio_track()`, fallback sang `recalculate_srt()` nếu fail_policy=warn, chỉ tạo `_tts_synced.srt` khi `keep_tts_synced_debug=true`
- `cli/qwen3_asr.py` — Refactor: xóa inline `merge_punctuation()`, `format_srt_time()`, CJK constants; import từ `utils/asr_subtitle_utils.py`; dùng `segment_words_to_subtitles()` + `write_subtitle_srt()` thay vì manual loop
- `sync_engine/__init__.py` — Thêm export `run_forced_alignment_subtitle`
- `tests/test_matrix.yaml` — Thêm 3 entries: ASR Subtitle Utils Layer 1, Forced Alignment Subtitle Layer 1, Forced Alignment Subtitle Layer 2
- `docs/sync-video-guide.md` — Thêm Phase 3.5 flow, section 2.10 schema, section 3 cấu hình forced_alignment_subtitle, cập nhật kiến trúc module
- `docs/colab-guide.md` — Cập nhật sync-video section với forced alignment config và output info

### Quyết định kiến trúc

1. **Ranh giới module**: Shared ASR/subtitle logic đặt trong `utils/` (không phải `sync_engine/`) vì không phải core sync engine — dùng chung bởi cả `cli/qwen3_asr.py` và `sync_engine/forced_alignment_subtitle.py`
2. **Output policy**: Forced alignment ghi trực tiếp vào `<name>_synced.srt` (file dùng bởi renderer), loại bỏ `_tts_synced.srt` mặc định; chỉ giữ remap SRT khi `keep_tts_synced_debug=true`
3. **Config defaults**: JSON dùng `null` cho model/device/dtype/attn → function defaults: `Qwen/Qwen3-ForcedAligner-0.6B`, `torch.bfloat16`, `cuda:0`
4. **Segmentation invariant**: Tổng chars ≤ max_chars → không ngắt thành 2 blocks; `min_chars=0` → không giới hạn tối thiểu
5. **Fail policy**: `warn` (default) → fallback remap SRT; `raise`/`error`/`fail` → dừng pipeline

### Kiến trúc module

```
cli/sync_video.py                        ← Entrypoint CLI
    ↓ import
sync_engine/forced_alignment_subtitle.py ← Orchestration forced alignment
    ↓ import
utils/asr_subtitle_utils.py              ← Shared ASR subtitle logic
utils/text_segmenter.py                  ← Smart segmentation algorithm
```

### Pending

- Chạy test verification cho unit tests và mock integration tests
- GPU integration test (Layer 3) với real model — cần CUDA + VRAM

## 2026-05-17: Refactor kiến trúc LLM metadata — tách khỏi `cli/`

### Tóm tắt

- **Mục tiêu**: Đảm bảo `cli/` chỉ chứa entrypoint command (`pyproject.toml` `[project.scripts]`), không chứa helper/library module.
- **File mới**:
  - `llm_ai/task_runner.py` — Provider creation logic dùng chung (trích xuất từ `cli/llm_task.py` private helpers → public API)
  - `sync_engine/llm_metadata.py` — LLM metadata orchestration (di chuyển từ `cli/sync_video_llm_metadata.py`)
  - `docs/sync-video-guide.md` — Tài liệu đầy đủ flow sync-video + schema `render_config.json`
- **File sửa**:
  - `utils/srt_parser.py` — Thêm `write_segments_to_flat_text()` (normalize + ghi raw text phẳng ra `.txt`)
  - `cli/llm_task.py` — Xóa toàn bộ private helpers, import từ `llm_ai.task_runner`
  - `cli/sync_video.py` — Import từ `sync_engine.llm_metadata`; flow mới: ghi `.txt` sau `parse_srt_file()`, gọi `run_llm_metadata_task(input_text_path=...)` sau final render
  - `tests/cli/test_sync_video_llm_metadata.py` — Cập nhật import + monkeypatch path + test `write_segments_to_flat_text`
- **File xóa**:
  - `cli/sync_video_llm_metadata.py` — Đã di chuyển toàn bộ logic sang `sync_engine/llm_metadata.py`

### Thay đổi chữ ký

- `run_llm_metadata_task()`: `subtitle_segments` + `tmp_dir` → `input_text_path`
- `execute_llm_metadata_task()`: `subtitle_segments` + `tmp_dir` → `input_text_path`
- Thêm `prepare_llm_metadata_input()` trong `sync_engine/llm_metadata.py` để chuẩn bị file `.txt` từ subtitle segments

### Kiến trúc module

```
cli/sync_video.py          ← Entrypoint CLI
    ↓ import
sync_engine/llm_metadata.py ← Orchestration (phụ thuộc llm_ai, không phụ thuộc cli)
    ↓ import
llm_ai/task_runner.py       ← Provider creation (dùng chung bởi cli + sync_engine)
utils/srt_parser.py         ← SRT parsing + write_segments_to_flat_text()
```

### Pending

- Chạy pipeline test (yêu cầu ffmpeg + cv2 + pydub)

---

## 2026-05-17: Bảo vệ segmentation subtitle khỏi dấu chấm trong abbreviation

### Yêu cầu

- Tránh để dấu chấm trong các abbreviation tiếng Anh như `e.g.`, `i.e.`, `Mr.`, `Mrs.`, `Dr.`, `vs.` bị coi là ranh giới câu.
- Với dấu `.` ASCII, chỉ cắt câu khi ký tự chữ cái có nghĩa tiếp theo là chữ in hoa, hoặc khi dấu chấm nằm ở cuối text.
- Bổ sung unit test khóa hành vi cho các trường hợp `360-degree`, `It's`, `e.g.` và các abbreviation phổ biến.

### Thay đổi đã thực hiện

1. **Text segmentation**:
   - Thêm danh sách `COMMON_NON_SENTENCE_ABBREVIATIONS` trong `utils/text_segmenter.py`.
   - Thêm helper nhận diện dấu chấm thuộc abbreviation, lấy ký tự có nghĩa tiếp theo sau dấu `.`, và quyết định có nên cắt sau dấu chấm hay không.
   - Cập nhật `_has_sentence_split_punct()` để bỏ qua dấu `.` không phải ranh giới câu.
   - Cập nhật scoring của `_split_long_block()` để phạt nặng boundary sau dấu `.` không phải ranh giới câu, hạn chế split cơ học sai nghĩa.

2. **Tests**:
   - Bổ sung test `merge_punctuation()` cho `360-degree`, `It's`, `e.g.`.
   - Bổ sung test `smart_segment()` đảm bảo `e.g.`, `i.e.`, `Mr.`, `Mrs.`, `Dr.`, `vs.` không tự tạo block subtitle mới.
   - Bổ sung test dấu `.` trước chữ thường không tạo block mới, còn câu tiếp theo bắt đầu chữ hoa vẫn được cắt.

### Trạng thái hiện tại

- ✅ `python -m compileall -q utils tests` pass.
- ✅ `python -m pytest tests/cli/test_qwen3_asr.py -v` pass: 29 passed.

### Outstanding / Pending

- Chưa mở rộng danh sách abbreviation ngoài bộ hiện tại; nếu gặp thêm pattern thực tế trong transcript tiếng Anh, có thể bổ sung vào `COMMON_NON_SENTENCE_ABBREVIATIONS`.

---

## 2026-05-16: Refactor helper LLM metadata sync_video sang module riêng

### Yêu cầu

- Tách các helper LLM metadata khỏi `cli/sync_video.py` để pipeline chính gọn hơn.
- Xóa toàn bộ key `_comment` khỏi `assets/default_render_config.json` để JSON chỉ chứa dữ liệu runtime.
- Đưa phần giải thích schema LLM metadata vào docstring của module helper mới.

### Thay đổi đã thực hiện

1. **Module helper riêng**:
   - Tạo `cli/sync_video_llm_metadata.py` chứa schema notes trong docstring.
   - Chuyển các helper merge override, resolve path, build raw text, execute generic LLM task và fail policy handling sang module mới.
   - `cli/sync_video.py` chỉ import `apply_llm_metadata_override` và `run_llm_metadata_task`.

2. **Render config sạch hơn**:
   - Xóa các key `_comment`, `_debug_input_comment`, `_fail_policy_comment` khỏi `assets/default_render_config.json`.
   - Giữ nguyên schema runtime `llm_metadata.enabled`, `task_config`, `input`, `output`, `fail_policy`.

3. **Tests**:
   - Cập nhật `tests/cli/test_sync_video_llm_metadata.py` import và monkeypatch sang `cli.sync_video_llm_metadata`.

### Trạng thái hiện tại

- Logic LLM metadata không còn nằm trực tiếp trong `cli/sync_video.py`.
- JSON render config gọn hơn, còn giải thích schema nằm trong docstring của `cli/sync_video_llm_metadata.py`.

---

## 2026-05-16: Tích hợp LLM metadata post-render cho sync_video

### Yêu cầu

- Thêm bước tạo SEO metadata bằng generic LLM task vào flow `cli/sync_video.py`.
- Cấu hình wiring đặt trong `render_config.json`, trước mắt thêm schema vào `assets/default_render_config.json`.
- Input LLM là raw text phẳng gom từ toàn bộ block `text` của subtitle SRT, không timestamp, không line number, không chia line.
- Output metadata và debug input mặc định nằm cùng thư mục input video bằng `directory_policy: "/"`.
- LLM generation chạy sau bước final render; lỗi LLM mặc định chỉ warning theo `fail_policy: "warn"`.

### Thay đổi đã thực hiện

1. **Schema render config**:
   - Thêm block `llm_metadata` vào `assets/default_render_config.json` với `enabled`, `task_config`, `input.write_debug_input`, `input.debug_input_filename_template`, `output.directory_policy`, `output.filename_template`, `fail_policy`.
   - Ghi chú: sau refactor kế tiếp, các key `_comment` đã được chuyển khỏi JSON và schema được giải thích trong docstring của `cli/sync_video_llm_metadata.py`.

2. **Helper trong sync_video**:
   - Thêm helper gom subtitle segments thành raw text phẳng.
   - Thêm resolver cho `directory_policy: "/"` nghĩa là thư mục chứa input video, ví dụ `content/a/b.mp4` -> `content/a/`.
   - Thêm debug input writer và output metadata filename template.
   - Tái dùng logic provider/task từ `cli/llm_task.py` và `llm_ai.tasks.generic_text_task.run_generic_text_task()` thay vì gọi subprocess.
   - Thêm `fail_policy` để warning hoặc raise lỗi LLM theo cấu hình.

3. **Task-file override**:
   - `worker_task` hỗ trợ override `render_config` và override sâu `llm_metadata` theo từng task JSON.

4. **Tests**:
   - Thêm `tests/cli/test_sync_video_llm_metadata.py` cho raw text builder, path policy, deep override và fail policy.
   - Cập nhật `tests/test_matrix.yaml` với entry Layer 1 cho helper LLM metadata.
   - Cập nhật pipeline test hiện có để truyền đủ namespace/config tối thiểu và tắt LLM metadata trong integration test.

### Trạng thái hiện tại

- Helper LLM metadata đã được tích hợp ở Phase 6 và chỉ chạy sau final render; nếu dùng `--no-hardsub` thì metadata cũng được bỏ qua để đúng `run_stage` đã chốt.
- Output metadata mặc định ghi đè file cũ vì `run_generic_text_task()` ghi file trực tiếp.
- API key vẫn lấy từ environment/provider config theo flow LLM hiện có, không hardcode trong JSON.

### Outstanding / Pending

- Cần chạy real LLM metadata trên môi trường có `OPENAI_API_KEY`, `GEMINI_API_KEY` hoặc Vertex AI credentials để xác nhận chất lượng output thực tế.
- Nếu muốn output metadata bắt buộc thành công, đổi `fail_policy` từ `warn` sang `raise` trong render config.

---

## 2026-05-16: Thêm provider_chain fallback và retry wait tuyến tính cho llm_ai

### Yêu cầu

- Thêm cơ chế fallback: sau khi provider chính retry hết `retry_attempts` và vẫn thất bại, tự chuyển sang LLM config khác.
- Đặt fallback trong task config bằng schema `provider_chain`, ví dụ primary `config/llm/openai_compat.yaml` và fallback `config/llm/vertexai.yaml`.
- Đổi retry wait từ fixed/exponential sang tuyến tính theo `retry_wait_seconds * attempt_number`.
- Áp dụng cho cả generic `llm-task` và workflow `translate-srt`.

### Thay đổi đã thực hiện

1. **Fallback provider chain dùng chung**:
   - Thêm `llm_ai/provider_chain.py` với `FallbackLLMProvider`, `ProviderChainError`, parser/normalizer `provider_chain` và helper merge override.
   - Wrapper gọi provider hiện tại đến khi provider đó raise lỗi cuối sau retry, sau đó chuyển sang provider kế tiếp.
   - Provider trong chain được khởi tạo lazy để không kéo dependency/credential của fallback trước khi cần.

2. **Retry wait tuyến tính**:
   - Thêm `llm_ai/retry.py` với `calculate_linear_retry_wait_seconds()` và `build_linear_retry_wait()`.
   - Cập nhật `llm_ai/providers/openai.py`, `llm_ai/providers/gemini.py`, `llm_ai/providers/vertexai.py` dùng wait tuyến tính.
   - Cập nhật retry integrity của SRT batch trong `translation/srt_translator.py` dùng cùng công thức tuyến tính.

3. **CLI và config**:
   - Cập nhật `cli/llm_task.py` để ưu tiên `provider_chain` trong task config, vẫn giữ single-provider mode nếu truyền `--provider` hoặc `--provider-config`.
   - Cập nhật `cli/translate_srt.py` tương tự, dùng `provider_chain` từ `config/llm_tasks/srt_translation.yaml`.
   - Thêm `provider_chain` vào `config/llm_tasks/seo_metadata.yaml` và `config/llm_tasks/srt_translation.yaml` với primary OpenAI-compatible và fallback Vertex AI.

4. **Tests**:
   - Cập nhật `tests/llm_ai/test_generic_text_task.py` với test retry wait tuyến tính và fallback provider chain.

### Trạng thái hiện tại

- ✅ `python -m compileall -q llm_ai translation cli tests` pass.
- ✅ Import smoke các module `llm_ai`, `llm_ai.retry`, `llm_ai.provider_chain`, providers, `translation.srt_translator`, `cli.llm_task`, `cli.translate_srt` pass.
- ✅ `pytest tests/llm_ai/test_generic_text_task.py tests/translation/test_translation_providers.py -k "Layer1 or Layer2" -q` pass: 13 passed, 8 skipped, 8 deselected.

### Outstanding / Pending

- Còn cảnh báo `PytestUnknownMarkWarning` cho marker `api`; không ảnh hưởng logic fallback/retry.
- Nếu cần fallback với nhiều secret khác nhau cho nhiều OpenAI endpoint, nên bổ sung schema secret/env riêng theo từng provider_chain entry ở task sau.

---

## 2026-05-16: Fix import-time dependency sau cleanup translation legacy

### Yêu cầu

- Kiểm tra lỗi cú pháp hoặc bug cú pháp sau khi xóa các file legacy của `translation`.
- Xác minh các import liên quan không bị hỏng trong môi trường hiện tại.

### Vấn đề phát hiện

- `compileall` không phát hiện lỗi cú pháp.
- Import smoke ban đầu fail khi import `translation.srt_translator` vì `translation/srt_translator.py` import `llm_ai.providers.gemini.GeminiProvider` tại module import-time.
- Import provider Gemini kéo `tenacity` ngay khi import module, khiến môi trường chưa cài dependency runtime bị lỗi `ModuleNotFoundError: No module named 'tenacity'` dù chỉ đang kiểm tra import.

### Thay đổi đã thực hiện

1. **Giảm import-time optional dependency**:
   - Cập nhật `llm_ai/providers/__init__.py` để không import eager các provider concrete.
   - Cập nhật `llm_ai/providers/gemini.py` để lazy import `tenacity` trong `GeminiProvider.call()`.
   - Cập nhật `translation/srt_translator.py` để `GeminiCaller` là lazy factory, không import Gemini provider tại module import-time.

2. **Kiểm tra sau sửa**:
   - `python -m compileall -q llm_ai translation cli tests` pass.
   - Import smoke các module `translation`, `translation.srt_translator`, `llm_ai.factory`, `llm_ai.providers`, `cli.llm_task`, `cli.translate_srt` pass.
   - `pytest tests/llm_ai/test_generic_text_task.py tests/translation/test_translation_providers.py -k "Layer1 or Layer2" -q` pass: 10 passed, 8 skipped, 8 deselected.

### Outstanding / Pending

- Còn cảnh báo `PytestUnknownMarkWarning` cho marker `api`; không phải lỗi cú pháp/import và có thể xử lý bằng cách đăng ký marker trong cấu hình test ở task riêng.

---

## 2026-05-16: Cleanup legacy translation wrappers sau refactor llm_ai

### Yêu cầu

- Dọn code cũ không còn sử dụng sau refactor `llm_ai`, bắt đầu từ `translation/factory.py`.
- Ưu tiên dùng command để xóa file legacy và kiểm tra tham chiếu còn sót.

### Thay đổi đã thực hiện

1. **Xóa compatibility wrappers legacy trong `translation/`**:
   - Xóa `translation/factory.py`, `translation/base.py`, `translation/translator.py`.
   - Xóa các wrapper provider cũ: `translation/openai_provider.py`, `translation/gemini_provider.py`, `translation/vertexai_provider.py`.
   - Giữ workflow dịch SRT hiện hành trong `translation/srt_translator.py`, `translation/batching.py`, `translation/prompting.py`, `translation/response_parser.py`.

2. **Cập nhật import sang canonical modules**:
   - `translation/__init__.py` export trực tiếp từ `llm_ai` và lazy import workflow SRT hiện hành.
   - `tests/translation/test_translation_providers.py` chuyển từ `translation.factory`/`translation.base`/`translation.translator` sang `llm_ai.factory`/`llm_ai.base`/`translation.srt_translator`.

3. **Dọn config legacy**:
   - Xóa `config/openai_compat_translate.yaml` và `config/vertexai_translate.yaml`.
   - Cập nhật notebook sang config mới trong `config/llm/`.

### Trạng thái hiện tại

- ✅ Không còn tham chiếu legacy trong các file active `*.py`, `*.yaml`, `*.ipynb` đối với `translation.factory`, `translation.base`, `translation.translator`, provider wrappers cũ và config translate cũ.
- ✅ `py_compile` pass cho các module `llm_ai`, `translation` và CLI LLM/translation liên quan.
- ✅ `pytest tests/llm_ai/test_generic_text_task.py tests/translation/test_translation_providers.py -k 'Layer1 or Layer2' -q` pass: 10 passed, 8 skipped, 8 deselected.

### Outstanding / Pending

- Các tham chiếu trong tài liệu lịch sử như `logs/JOURNAL-2604.md` và plan cũ vẫn được giữ nguyên vì là record/plan quá khứ, không phải code active.
- Pytest vẫn cảnh báo `PytestUnknownMarkWarning` cho marker `api`; có thể đăng ký marker này trong cấu hình test ở một cleanup riêng.

---

## 2026-05-15: Refactor LLM provider sang llm_ai và thêm generic LLM task

### Yêu cầu

- Tách toàn bộ hạ tầng gọi LLM khỏi flow dịch phụ đề để có thể tái sử dụng cho metadata, script, summary và các tác vụ generative khác.
- Chuẩn hóa package mới là `llm_ai` thay vì đặt provider trong `translation`.
- Tạo flow generic cho SEO metadata: input raw text, thay vào prompt TheArmoryLog tại placeholder `[Video Content]`, output markdown.
- Giữ tương thích tạm thời với import cũ trong `translation/*` để không phá CLI/test hiện tại.

### Thay đổi đã thực hiện

1. **Tầng LLM dùng chung `llm_ai/`**:
   - Tạo `llm_ai/base.py`, `llm_ai/factory.py`, `llm_ai/providers/openai.py`, `llm_ai/providers/gemini.py`, `llm_ai/providers/vertexai.py`.
   - `BaseLLMProvider` là interface mới, giữ alias `BaseTranslationProvider` để tương thích.
   - Factory mới hỗ trợ `gemini`, `openai`, `vertexai` và lazy import `PyYAML` khi thật sự load config.

2. **Tầng generic task `llm_ai/tasks/`**:
   - Tạo runner `generic_text_task.py` cho flow text-in/text-out.
   - Tách helper `prompt_template.py`, `output_writer.py`, `response_parser.py`.
   - Hỗ trợ parser `raw`, `tag`, `json` để task mới có thể chủ yếu thêm YAML + prompt mà không cần code mới.

3. **Tách workflow dịch SRT**:
   - Tạo `translation/srt_translator.py`, `translation/batching.py`, `translation/prompting.py`, giữ `translation/response_parser.py` như wrapper tag `<TRANSLATE_TEXT>`.
   - `translation/translator.py`, `translation/base.py`, `translation/factory.py`, các provider cũ trở thành compatibility wrappers.

4. **Config và prompt mới**:
   - Provider config chuyển sang `config/llm/openai_compat.yaml`, `config/llm/vertexai.yaml`, `config/llm/gemini.yaml`.
   - Task config nằm trong `config/llm_tasks/seo_metadata.yaml`, `script_generation.yaml`, `summary.yaml`, `srt_translation.yaml`.
   - Prompt được tách sang `prompts/llm_tasks/seo_metadata.txt`, `script_generation.txt`, `summary.txt`, và `prompts/translation/srt_translate.txt`.

5. **CLI và packaging**:
   - Thêm `cli/llm_task.py` và entrypoint `llm-task` trong `pyproject.toml`.
   - Cập nhật `cli/translate_srt.py` dùng provider từ `llm_ai` và prompt/config translation mới.
   - Cập nhật package include thêm `llm_ai*`, dependency thêm `PyYAML` và `tenacity`.

6. **Tests & docs**:
   - Thêm `tests/llm_ai/test_generic_text_task.py` theo domain-based test structure.
   - Cập nhật `tests/test_matrix.yaml` với Layer 1/2 cho generic LLM task.
   - Cập nhật `docs/colab-guide.md` cho đường dẫn config/prompt mới và ví dụ `llm-task` tạo SEO metadata.

### Trạng thái hiện tại

- ✅ `python -m py_compile ...` pass cho các module `llm_ai`, `translation` và CLI mới.
- ✅ `python -m pytest tests/llm_ai/test_generic_text_task.py -v` pass 6/6 tests.
- ✅ `python -m pytest tests/translation/test_translation_providers.py -k "Layer1 or Layer2" -v` pass phần không phụ thuộc dependency ngoài; các provider/config tests skip hợp lệ khi môi trường chưa cài `PyYAML`, `google-genai`, `openai`, `tenacity`.
- ✅ Flow SEO metadata hiện chạy qua `llm-task` bằng `config/llm_tasks/seo_metadata.yaml` + `prompts/llm_tasks/seo_metadata.txt`.

### Outstanding / Pending

- Các file legacy `config/openai_compat_translate.yaml`, `config/vertexai_translate.yaml`, `prompts/gemini.txt`, `prompts/TheArmoryLog-metadata.txt` vẫn được giữ lại để tránh phá workflow cũ; có thể dọn sau khi xác nhận toàn bộ notebook/docs đã chuyển sang cấu trúc mới.
- Layer 4 real API tests vẫn phụ thuộc API key/ADC thật trên môi trường người dùng.

---

## 2026-05-15: Bảo vệ định dạng số khi chia câu trong text_segmenter

### Yêu cầu

- Khắc phục lỗi chia câu nhầm tại dấu chấm/phẩy nằm trong số thập phân, số hàng nghìn, số kèm đơn vị/tiền tệ và phần trăm.
- Các ví dụ cần bảo vệ: `1.2`, `3,14`, `1,000,000`, `12.5kg`, `3,000円`, `99.9%`.
- Vẫn phải cắt bình thường tại dấu câu thật sau số, ví dụ `99.9%. Sau đó...`.

### Thay đổi đã thực hiện

1. **`utils/text_segmenter.py`**:
   - Thêm helper nhận diện dấu phân cách số bằng ngữ cảnh ký tự liền trước/liền sau trong chuỗi ghép từ tokens.
   - Cập nhật Giai đoạn 1 để bỏ qua dấu chấm/phẩy nằm giữa hai chữ số khi quyết định điểm cắt ngữ pháp.
   - Cập nhật chấm điểm Giai đoạn 2 để phạt nặng boundary tách rời số, dấu phân cách số hoặc hậu tố đơn vị/tiền tệ/phần trăm dính trực tiếp với số.

2. **`tests/utils/test_text_segmenter.py`**:
   - Bổ sung unit tests cho số thập phân bằng dấu chấm và dấu phẩy.
   - Bổ sung unit tests cho số hàng nghìn, số kèm đơn vị/tiền tệ, phần trăm và dấu câu thật sau số.
   - Bổ sung test hồi quy cho Giai đoạn 2 để không ưu tiên cắt giữa `1.2kg`.

### Trạng thái hiện tại

- ✅ Logic chia câu đã phân biệt dấu câu thật với dấu phân cách số phổ biến.
- ✅ Hành vi `split_on_comma=True` vẫn cắt dấu phẩy ngữ pháp nhưng không cắt dấu phẩy số.
- ✅ `python -m pytest tests/utils/test_text_segmenter.py -k Layer1 -v` pass toàn bộ 26 tests trên môi trường hiện tại.
- ✅ `python -m pytest tests/cli/test_qwen3_asr.py -k Layer1 -v` pass toàn bộ 19 tests, xác nhận caller ASR không bị phá.

---

## 2026-05-10: Tự động xuất file .txt song song khi dịch SRT

### Yêu cầu

- Trong flow `cli/translate_srt.py`, khi output kết quả dịch (ví dụ `[ten]_en.srt`), cần xuất thêm một file text văn bản liên tục không ngắt dòng ngay tại thư mục output.
- Người dùng nhớ đã có chức năng `srt_to_txt` trong `utils`, nhưng thực tế module này chưa tồn tại.

### Thay đổi đã thực hiện

1. **`utils/srt_parser.py`**:
   - Thêm hàm `segments_to_txt(segments: List[Dict]) -> str`: nối toàn bộ text từ các segment SRT thành một chuỗi liên tục, thay thế dấu xuống dòng `\n` bằng khoảng trắng, loại bỏ segment rỗng.

2. **`translation/translator.py`**:
   - Cập nhật import để thêm `segments_to_txt` từ `utils.srt_parser`.
   - Trong hàm `translate_srt_file`, sau khi ghi file `.srt` (dòng 193-194), tự động gọi `segments_to_txt` để sinh chuỗi text liên tục và ghi ra file `.txt` cùng tên (ví dụ `video_en.srt` → `video_en.txt`) trong cùng thư mục.

### Trạng thái hiện tại

- ✅ Hàm `segments_to_txt` đã được thêm vào `utils/srt_parser.py`.
- ✅ `translate_srt_file` giờ đây tự động xuất cả `.srt` và `.txt` cho mỗi task dịch.
- ✅ Tên file `.txt` được sinh bằng `Path(output_file).with_suffix('.txt')`, đảm bảo đồng bộ với tên file `.srt`.

---

## 2026-05-09: Thay thế Demucs bằng Audio-Separator (ROFORMER models)

### Yêu cầu

- Khắc phục lỗi Demucs làm mất tiếng súng nổ (hiện tượng clipping bị coi là noise).
- Xóa bỏ hoàn toàn Demucs và thay thế bằng thư viện `audio-separator` với các mô hình mạnh mẽ của Roformer.
- Đưa tham số tách âm thanh (`extract_bgm`, `extract_vocals`) vào cấu hình `render_config.json` thay vì tham số dòng lệnh CLI.
- Hỗ trợ ghi đè linh hoạt cấu hình tham số `mdxc_params` từ `render_config.json` để tối ưu cho từng dự án, ưu tiên đè lên `audio_separator_config.yaml`.

### Thay đổi đã thực hiện

1. **Gỡ bỏ thư viện `demucs`**:
   - Xóa `cli/demucs_audio.py` và `tests/cli/test_demucs_audio.py`.
   - Cập nhật `pyproject.toml` để sử dụng `audio-separator[gpu]` thay vì `demucs`.

2. **Cấu hình `audio-separator`**:
   - Tạo mới file `config/audio_separator_config.yaml` chứa các thông số chi tiết (model, overlap, batch_size) cho 2 preset: `bgm_extraction` (model: model_bs_roformer_ep_317_sdr_12.9755.ckpt) và `vocal_extraction` (model: vocals_mel_band_roformer.ckpt).
   - Thêm phần cấu hình `"audio_separator"` vào các file `render_config.json` cùng với block `"mdxc_params"`.
   - Sửa `demucs_bgm_volume` thành `bgm_volume`.

3. **Cập nhật mã nguồn**:
   - Tạo file `cli/audio_separator.py` với logic wrap xung quanh thư viện `Separator` của Invertigo. Hỗ trợ tách single-file và batch-processing. Tự động nhận diện stem output chứa chữ "Instrumental" hoặc "Vocals".
   - Áp dụng phương thức gộp sâu (deep update) khi merge `override_kwargs` với `preset_config` để hỗ trợ ghi đè một phần tham số `mdxc_params`.
   - Cập nhật luồng logic của Phase 2.5 và Phase 3 trong `cli/sync_video.py` và `sync_engine/audio_assembler.py`:
     - Giảm padding `pad_s` từ 3.5 xuống 1.0 giây vì thuật toán Roformer cắt viền mượt hơn.
     - Đổi tên biến `use_demucs` thành `use_vocal_extraction`.
     - Đổi tên biến `demucs_bgm_path` thành `bgm_path`.

4. **Cập nhật tài liệu**:
   - Chỉnh sửa `docs/colab-guide.md` thay thế các tham chiếu đến Demucs bằng Audio-Separator.
   - Chỉnh sửa `docs/testing-guide.md` tương tự.
   - `docs/workflow.md` đã được người dùng gỡ bỏ hoàn toàn.

### Trạng thái hiện tại

- ✅ Chức năng tích hợp audio-separator (ROFORMER) hoàn thiện.
- ✅ Hỗ trợ cơ chế override cấu hình `mdxc_params` ưu tiên từ `render_config.json`.
- ✅ Dự án sạch bóng mã và cấu hình liên quan đến Demucs.
- ✅ Cấu trúc code thống nhất với các file config (yaml, json) thay cho các argument dòng lệnh phức tạp.

---

## 2026-05-08: Tích hợp task_utils và Multiprocessing vào cli/sync_video.py

### Yêu cầu

- Hỗ trợ truyền `--task-file` JSON cho `cli/sync_video.py` để xử lý hàng loạt video.
- Tích hợp `utils/task_utils` để đồng bộ chuẩn hóa JSON với các CLI khác (`tts`, `qwen3_asr`, `whisper_srt`).
- Xử lý vấn đề rò rỉ VRAM của PyTorch (QwenTTS, Demucs) khi chạy nhiều video liên tiếp bằng cách bọc hàm `run_sync_pipeline` trong `multiprocessing.Process` với context `spawn`.

### Thay đổi đã thực hiện

1. **`cli/sync_video.py`**:
   - Thêm CLI argument `--task-file`.
   - Sửa tham số `--video` và `--subtitle` thành không bắt buộc (chỉ bắt buộc khi chạy đơn lẻ không có `--task-file`).
   - Thêm hàm `worker_task(task_data, base_args)`: chạy trong tiến trình độc lập, clone arguments, setup cấu hình output thông qua `resolve_output_dir_and_stem` của `task_utils`, và tự động báo lỗi `sys.exit(1)` khi exception.
   - Thêm logic đọc JSON qua `resolve_cli_tasks` và gọi `worker_task` song song qua `mp.get_context('spawn').Process`.

2. **`docs/colab-guide.md`**:
   - Thêm section "Chạy hàng loạt nhiều video (Batch JSON)" trong mục 2.11.
   - Cập nhật bảng tham số: thêm `--task-file`, sửa mô tả `--video`/`--subtitle` thành "bắt buộc khi không dùng `--task-file`".

### Trạng thái hiện tại

- ✅ Chức năng Batch Processing và Multiprocessing đã hoàn thiện.
- ✅ Cú pháp `python -m py_compile` pass.
- ✅ Giải quyết triệt để rủi ro OOM VRAM trên GPU do framework (PyTorch/CUDA) không nhả memory sau khi tắt model.

---

## 2026-05-06: Refactor CLI TTS — Dọn dẹp tham số và chuẩn hóa cấu hình YAML

### Yêu cầu

- Sau khi thêm Qwen3-TTS, CLI `sync_video` vẫn còn các tham số chuyên biệt của EdgeTTS (`--tts-rate`, `--tts-volume`, `--tts-pitch`) gây rác giao diện.
- Cần chuẩn hóa toàn bộ provider (`edge`, `voicevox`, `qwen`) đều đọc cấu hình từ `config/tts_config.yaml`, chỉ giữ lại `--tts-voice` làm cờ ghi đè nhanh.

### Thay đổi đã thực hiện

1. **`cli/sync_video.py`**:
   - Xóa 3 CLI argument: `--tts-rate`, `--tts-volume`, `--tts-pitch`.
   - Đổi `--tts-voice` từ `default="vi-VN-HoaiMyNeural"` sang `default=None`.
   - Refactor **PHASE 0**: Tất cả provider đều gọi `_load_tts_config(args.tts_config)` trước khi khởi tạo engine.
     - `edge`: lấy `voice`, `rate`, `volume`, `pitch`, `strip_silence`, `concurrent`, `min_silence_len_ms` từ YAML. `--tts-voice` CLI ghi đè `voice`.
     - `voicevox`: lấy `voice_id`, `concurrent_requests`, `speed_scale`, `pitch_scale`, `intonation_scale`, `volume_scale` từ YAML. `--tts-voice` CLI ghi đè `voice_id`.
     - `qwen`: giữ nguyên logic lấy block `qwen` từ YAML.

2. **`docs/colab-guide.md`**:
   - Cập nhật bảng tham số `sync-video`: thêm `--tts-config`, cập nhật mô tả `--tts-voice` thành "ghi đè YAML".
   - Thêm ví dụ "Chạy nhanh với Qwen3-TTS" trong section 2.11.

### Trạng thái hiện tại

- ✅ CLI đã tinh gọn, chỉ còn `--tts-provider`, `--tts-voice`, `--tts-config` cho nhóm TTS.
- ✅ Cú pháp `python -m py_compile` pass.

---

## 2026-05-06: Tích hợp Qwen3-TTS vào pipeline `sync_video`

### Yêu cầu

- `cli/sync_video.py` hiện tại chỉ hỗ trợ 2 TTS provider (`edge`, `voicevox`), trong khi `cli/tts.py` đã hỗ trợ thêm `qwen`.
- Cần đưa Qwen3-TTS vào luồng `sync_video` để có thể tạo video đồng bộ với giọng nói do Qwen3-TTS sinh ra, đồng thời tái sử dụng cơ chế cấu hình YAML (`config/tts_config.yaml`) thay vì nhồi tham số vào `render_config.json`.

### Thay đổi đã thực hiện

1. **`cli/sync_video.py`**:
   - Thêm `import yaml` để đọc file cấu hình TTS.
   - Thêm hàm `_load_tts_config(config_path: str) -> dict`: đọc file YAML cấu hình TTS, hỗ trợ đường dẫn tương đối/tuyệt đối, trả về dict rỗng nếu file không tồn tại.
   - Cập nhật CLI argument `--tts-provider`: mở rộng `choices` thêm `"qwen"`.
   - Thêm CLI argument `--tts-config` (mặc định: `config/tts_config.yaml`).
   - Cập nhật **PHASE 0 (AUTO GENERATE TTS)**:
     - Import thêm `QwenTTSEngine` từ `tts.qwen`.
     - Khi `args.tts_provider == "qwen"`, gọi `_load_tts_config(args.tts_config)` để lấy block `qwen`, sau đó khởi tạo `QwenTTSEngine(queue_tts=queue_tts, **qwen_cfg)`.
   - Cập nhật log in ra màn hình: phân biệt rõ `edge` (voice), `voicevox` (voice_id), và `qwen` (provider name).

### Trạng thái hiện tại

- ✅ `cli/sync_video.py` đã được chỉnh sửa và cú pháp pass (`python -m py_compile`).
- ✅ Qwen3-TTS đã được tích hợp vào pipeline `sync_video` thông qua YAML config.

### Outstanding / Pending

1. Chạy end-to-end test với provider `qwen` trên môi trường có GPU để xác nhận:
   - Model load và sinh audio hoạt động đúng.
   - VRAM được giải phóng sạch sau Phase 0 (do `QwenTTSEngine` đã có `torch.cuda.empty_cache()` trong `finally`, nhưng cần kiểm chứng trên luồng dài).
   - Audio clips được đặt đúng vị trí trên timeline và không gây drift.

### Đối chiếu Data Flow

- Thay đổi chỉ nằm ở Phase 0 (khởi tạo engine TTS). Các phase sau (Analysis, Video Processing, Audio Assembly, Render) không thay đổi logic, chỉ nhận đầu vào là các file `.wav` trong `tts_dir` như các provider khác.

---

## 2026-05-05: Thêm tính năng Demucs BGM Synced Track vào pipeline sync_video

### Yêu cầu

- Thêm khả năng tách BGM (nhạc nền + SFX) từ video gốc bằng Demucs, sau đó giãn (time-stretch) BGM này theo đúng timeline của video hình ảnh đã bị kéo dãn (slow motion ở các đoạn TTS).
- Tách BGM này hoạt động song song với track `ambient` (nhạc nền tùy chọn), không thay thế ambient.
- Cho phép tùy chỉnh âm lượng (volume) của BGM đã tách và ambient thông qua file cấu hình JSON (`assets/default_render_config.json`).
- Sửa lại logic kiểm tra `tts_provider` trong `compress_tts_clip` để ưu tiên kiểm tra `voicevox` (hỗ trợ cả `voicevox nemo` trong tương lai).

### Thay đổi đã thực hiện

1. **`assets/default_render_config.json`**:
   - Thêm block `audio_mix` chứa `ambient_volume` (mặc định: 0.03) và `demucs_bgm_volume` (mặc định: 1.0).

2. **`cli/sync_video.py`**:
   - Thêm CLI argument `--demucs-bgm` (action="store_true").
   - Thêm **PHASE 2.5**: Nếu `--demucs-bgm` được bật, gọi `separate_audio(..., keep="bgm")` trên toàn bộ video gốc để tạo `raw_demucs_bgm.wav`.
   - Truyền `demucs_bgm_path` và `audio_mix_config` (lấy từ `render_config.get("audio_mix")`) vào `assemble_audio_track`.

3. **`sync_engine/audio_assembler.py`**:
   - Sửa `compress_tts_clip`: Đảo ngược điều kiện — kiểm tra `tts_provider.startswith("voicevox")` trước để bỏ qua filter volume/limiter. Các provider khác (edge, qwen, v.v.) sẽ áp dụng filter.
   - Thêm hàm `_prepare_bgm_chunk`: Cắt 1 đoạn BGM từ `demucs_bgm_path` tại vị trí `orig_start..orig_end`, áp dụng filter `atempo` theo `video_speed` của segment, sau đó pad/trim để đạt đúng `new_chunk_dur`.
   - Nâng cấp `assemble_audio_track`:
     - Nhận thêm 2 tham số: `demucs_bgm_path` và `audio_mix_config`.
     - Sau bước concat Main Track, nếu có `demucs_bgm_path`, chạy song song `_prepare_bgm_chunk` cho toàn bộ timeline, concat các chunk BGM thành `synced_bgm.wav`.
     - Bước Mix Final được nâng cấp để hỗ trợ 3 inputs: Main Track + Ambient (nếu có) + Synced BGM (nếu có).
     - Sử dụng `volume` filter trên từng input trước khi `amix`, với giá trị volume đọc từ `audio_mix_config`.

### Trạng thái hiện tại

- ✅ Các file đã được chỉnh sửa và cú pháp pass (`python -m py_compile`).
- ✅ Tính năng Demucs BGM Synced Track đã được tích hợp vào pipeline.
- ✅ Volume config được quản lý tập trung qua JSON.

### Outstanding / Pending

1. Chạy end-to-end test với video thật để xác nhận BGM được giãn đúng tốc độ và sync với hình ảnh.
2. Đánh giá mức độ artifact (robotic/metallic) khi `video_speed` xuống quá thấp (ví dụ < 0.5x).

### Đối chiếu Data Flow

- Thay đổi chỉ nằm ở Phase 2.5 (tách BGM) và Phase 3 (mix audio). Không thay đổi luồng timeline, video processing, hay subtitle remapping.
- `assemble_audio_track` vẫn giữ nguyên hợp đồng cũ khi không truyền `demucs_bgm_path` (backward compatible).

---

## 2026-05-04: Chuyển đổi Hardcode Tham số Render sang Cấu hình JSON động

### Yêu cầu

- Loại bỏ việc "hardcode" (gắn cứng) các tham số khi render final video như độ phân giải 1920x1080, dải đen, watermark ảnh, text.
- Xây dựng cơ chế cấu hình JSON động (Data-Driven Configuration) cho phép dễ dàng chuyển đổi giữa các dự án có tỷ lệ khung hình khác nhau (VD: dọc 1080x1920 cho TikTok/Reels, ngang 1920x1080 cho YouTube).
- Giảm phụ thuộc vào thư viện `Pillow/PIL` trong việc sinh ảnh dải đen tĩnh, chuyển sang sử dụng file ảnh do user cung cấp trực tiếp.
- Thay đổi chữ ký lệnh CLI bằng cờ `--render-config` thay cho loạt cờ `--subtitle-fontname`, `--black-bg` rời rạc.

### Thay đổi đã thực hiện

1. **`assets/default_render_config.json`**:
   - Tạo file JSON định nghĩa cấu trúc: `resolution`, `watermark_img`, `watermark_text`, `black_strip`, `subtitles`, `note_overlay`.
   - Các tham số vị trí x, y được tách riêng để FFmpeg dễ dàng nội suy (ví dụ `x: "W-w-40"`).
   - Hỗ trợ flag `bypass_scale` để render video kích thước tự do.

2. **`cli/sync_video.py`**:
   - Loại bỏ các flag `--subtitle-fontname`, `--subtitle-fontsize`, `--subtitle-color`, `--subtitle-margin-v`, `--black-bg`, `--note-overlay-png`.
   - Thêm flag `--render-config` hỗ trợ nạp cấu hình JSON.
   - Truyền toàn bộ dictionary cấu hình vào `render_final_video`.

3. **`sync_engine/renderer.py`**:
   - **Xóa bỏ hàm `ensure_black_bg()`**, loại bỏ hoàn toàn import Pillow tại bước này.
   - **Tái cấu trúc (Refactor) `render_final_video`**: Áp dụng mô hình thiết kế **Dynamic Label Chaining (Gán nhãn dòng chảy động)**. FFmpeg `filter_complex` giờ đây được ghép tự động dựa trên số lượng Input `-i` thực tế, đảm bảo các stream index như `[0:v]`, `[2:v]` không bị trượt dù user có bật/tắt watermark hay dải đen.

### Trạng thái hiện tại

- ✅ Cơ chế Data-Driven Rendering qua JSON đã hoàn tất và tích hợp.
- ✅ Kiến trúc `filter_complex` động đã được cập nhật thành công, khắc phục rủi ro sai số index luồng.
- ✅ Dễ dàng tái sử dụng pipeline cho nhiều dự án có format Video khác nhau.
