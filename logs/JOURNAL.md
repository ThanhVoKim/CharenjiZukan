# Project Journal

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
