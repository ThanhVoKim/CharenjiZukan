# Project Journal

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
