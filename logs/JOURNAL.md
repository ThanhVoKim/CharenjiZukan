# Project Journal

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
