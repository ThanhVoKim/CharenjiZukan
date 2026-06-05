# Project Journal

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

## 2026-06-04: Bugfix — miệng mở trễ ~1s (bug gộp event _merge_short_silence)

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
| File | Mô tả |
|------|-------|
| [`sync_engine/tuber_mouth_events.py`](../sync_engine/tuber_mouth_events.py) | Phân tích RMS amplitude TTS → mouthEvents [{frame, state}] per segment |
| [`sync_engine/tuber_prerender.py`](../sync_engine/tuber_prerender.py) | Port mouthWarp.ts (affine 2-triangle), pre-render body×mouth → PNG |

### File sửa
| File | Thay đổi |
|------|----------|
| [`sync_engine/tuber_config.py`](../sync_engine/tuber_config.py) | +mouth (mode, silenceDb, minSilenceMs, cadenceMs, mouthStates) +prerender config |
| [`sync_engine/tuber_manifest.py`](../sync_engine/tuber_manifest.py) | +compute_character_box, +mouthEvents build, +compWidth/compHeight/compOffset, +prerenderManifest trong run_manifest |
| [`sync_engine/tuber_overlay.py`](../sync_engine/tuber_overlay.py) | +composite offset_x/offset_y, +prerender path (_build_prerender_frame_list, use_prerender param), +mouthEvents map lookup |
| [`assets/tuber_overlay_config.json`](../assets/tuber_overlay_config.json) | Config mới với mode=amplitude + prerender section |
| [`docs/tuber-overlay-guide.md`](../docs/tuber-overlay-guide.md) | Update architecture, mouth config, prerender docs |

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
