# Hướng dẫn tham số OpenAI-compatible LLM

## 1. Mục đích

Tài liệu này mô tả cách cấu hình các endpoint OpenAI-compatible trong [`config/llm`](config/llm), đặc biệt khi mỗi [`base_url`](config/llm/openai_compat.yaml:3) có mức hỗ trợ tính năng khác nhau.

Provider hiện tại là [`OpenAICompatibleProvider`](llm_ai/providers/openai.py:7), đang dùng Chat Completions tại [`chat.completions.create()`](llm_ai/providers/openai.py:64). Vì vậy mọi mở rộng phải giữ tương thích với đường mặc định này.

## 2. Nguyên tắc chung

1. Mỗi endpoint nên có một profile config riêng.
2. Chat Completions cơ bản là baseline mặc định.
3. Tất cả tham số nâng cao phải đi qua capability flag.
4. Không gửi tham số mà endpoint chưa xác nhận hỗ trợ.
5. Fail-fast với custom exception rõ nguyên nhân khi config yêu cầu tính năng nhưng capability flag không cho phép.
6. Không hardcode API key trong config.
7. Capability probe chỉ xác nhận khả năng endpoint, không thay thế đánh giá chất lượng dịch.
8. Production telemetry chỉ ghi metadata từ request thật, không tạo request phụ để test cache.
9. Capability report cần có lịch sử theo profile và timestamp.
10. Stateful features chỉ dùng khi endpoint hỗ trợ Responses API thật.

## 3. Nhóm tham số được hỗ trợ trong scope hiện tại

| Nhóm                    | Tham số                                  | Mức tương thích                             |
| ----------------------- | ---------------------------------------- | ------------------------------------------- |
| Chat Completions cơ bản | model, messages, temperature, max tokens | Cao                                         |
| Reasoning               | reasoning effort                         | Phụ thuộc model và endpoint                 |
| Output length/detail    | verbosity                                | Phụ thuộc model và endpoint                 |
| Output reliability      | structured output                        | Có fallback prompt-based JSON               |
| Cache routing           | prompt cache key                         | Phụ thuộc endpoint                          |
| Stateful continuation   | previous response state                  | Responses API only                          |
| Context management      | compaction                               | Responses API only hoặc project-level local |

## 4. Tham số cơ bản

### 4.1. model

`model` là tên model mà endpoint nhận. Với OpenAI-compatible gateway, tên model có thể không trùng OpenAI official.

Khuyến nghị:

- Mỗi profile config nên cố định một model mặc định.
- Nếu cần override từ CLI, ghi rõ trong log provider name.
- Capability probe nên cho phép override model bằng env để tránh sửa config nhiều lần.

### 4.2. messages

Trong Chat Completions, provider gửi messages gồm system prompt nếu có và user message. Logic hiện nằm trong [`OpenAICompatibleProvider.call()`](llm_ai/providers/openai.py:45).

Quy tắc:

- Giữ messages đơn giản cho baseline.
- Không đưa previous response state vào messages.
- Không trộn state server-side với prompt history client-side nếu chưa có thiết kế session rõ ràng.

### 4.3. temperature

`temperature` kiểm soát độ ngẫu nhiên. Dịch phụ đề thường cần ổn định, nhưng vẫn cần tự nhiên.

Gợi ý:

| Workflow              | Gợi ý               |
| --------------------- | ------------------- |
| Dịch SRT              | 0.3 đến 1 tùy model |
| SEO metadata          | 0.7 đến 1           |
| Summary               | 0.2 đến 0.8         |
| Structured extraction | 0 đến 0.4           |

### 4.4. max tokens

Trong Chat Completions hiện provider gửi [`max_tokens`](llm_ai/providers/openai.py:68). Với Responses API, một số endpoint dùng tên tương đương như max_output_tokens.

Quy tắc:

- Giữ `max_tokens` trong config để backward-compatible.
- Request builder chịu trách nhiệm map sang tên tham số phù hợp với API mode.
- Với structured output, max tokens phải đủ lớn để không cắt JSON giữa chừng.

## 5. Reasoning effort

Reasoning effort điều chỉnh lượng suy luận model dùng trước khi trả lời.

Giá trị thường gặp:

| Giá trị | Khi dùng                                                                |
| ------- | ----------------------------------------------------------------------- |
| none    | Endpoint hoặc model cho phép tắt reasoning; dùng cho task cực đơn giản. |
| low     | Extraction, routing, format cleanup, batch dịch đơn giản.               |
| medium  | Dịch có ngữ cảnh, summary, metadata, phân tích vừa.                     |
| high    | Phân tích cốt truyện, nhân vật, consistency validation.                 |
| xhigh   | Chỉ dùng khi eval chứng minh đáng chi phí và latency.                   |

Mapping đề xuất:

| API mode         | Mapping                               |
| ---------------- | ------------------------------------- |
| Chat Completions | reasoning_effort nếu endpoint hỗ trợ. |
| Responses API    | reasoning.effort nếu endpoint hỗ trợ. |

Capability flag:

```yaml
capability_flags:
  supports_reasoning_effort: false
request_options:
  reasoning_effort: medium
```

Quy tắc:

- Nếu supports_reasoning_effort là false thì không gửi tham số.
- Nếu endpoint reject tham số, cập nhật profile về false.
- Không dùng high hoặc xhigh làm mặc định cho dịch batch lớn nếu chưa benchmark.

## 6. Verbosity

Verbosity kiểm soát độ dài và mức chi tiết của output.

Giá trị thường gặp:

| Giá trị | Khi dùng                                                     |
| ------- | ------------------------------------------------------------ |
| low     | Câu trả lời ngắn, dịch phụ đề cần gọn, output ít giải thích. |
| medium  | Mặc định cân bằng.                                           |
| high    | Summary dài, phân tích, hướng dẫn chi tiết.                  |

Mapping đề xuất:

| API mode         | Mapping                             |
| ---------------- | ----------------------------------- |
| Chat Completions | verbosity nếu endpoint hỗ trợ.      |
| Responses API    | text.verbosity nếu endpoint hỗ trợ. |

Capability flag:

```yaml
capability_flags:
  supports_verbosity: false
request_options:
  verbosity: medium
```

Quy tắc:

- Dịch SRT nên dùng low hoặc medium.
- Summary và script generation có thể dùng medium hoặc high.
- Capability probe chỉ xác nhận endpoint chấp nhận tham số, không đảm bảo tham số có hiệu lực rõ ràng.

## 7. Structured output

Structured output giúp model trả dữ liệu theo cấu trúc ổn định để parser xử lý.

### 7.1. Prompt-based JSON

Prompt-based JSON là chế độ tương thích nhất. Prompt yêu cầu model trả JSON, sau đó client parse bằng [`parse_json_response()`](llm_ai/tasks/response_parser.py:39).

Ưu điểm:

- Dùng được với hầu hết endpoint.
- Không cần endpoint hỗ trợ schema API.
- Phù hợp trích xuất character bible, glossary, metadata.

Nhược điểm:

- Model vẫn có thể trả JSON lỗi.
- Cần retry hoặc repair nếu parse fail.

Capability flag:

```yaml
capability_flags:
  structured_output:
    supports_prompt_json: true
```

### 7.2. API-enforced schema

API-enforced schema gửi schema vào request để endpoint ràng buộc output.

Ưu điểm:

- Output ổn định hơn nếu endpoint enforce thật.
- Giảm lỗi parser.

Nhược điểm:

- Không phải endpoint OpenAI-compatible nào cũng hỗ trợ.
- Có gateway nhận tham số nhưng không enforce.
- Schema quá phức tạp có thể làm request fail.

Capability flags:

```yaml
capability_flags:
  structured_output:
    supports_chat_response_format: false
    supports_responses_text_format: false
request_options:
  structured_output:
    mode: none
    schema_name: null
    schema: null
```

Quy tắc:

- Luôn validate output phía client.
- Với task dịch SRT, tag format hiện tại vẫn có thể phù hợp hơn JSON nếu batch nhỏ.
- Với translation memory, glossary, nhân vật và metadata thì JSON nên được ưu tiên.

## 8. Prompt cache key và telemetry cache

Prompt cache key giúp endpoint route các request có prefix giống nhau tới cache phù hợp nếu endpoint hỗ trợ.

Capability flag:

```yaml
capability_flags:
  supports_prompt_cache_key: false
request_options:
  prompt_cache_key: null
```

Khuyến nghị tạo key:

```text
llm-task:srt_translation:model:gpt-5.5:prompt:v1:target:en
```

Quy tắc:

- Không đưa nội dung batch động vào cache key.
- Không đưa secret hoặc dữ liệu nhạy cảm vào cache key.
- Không giả định endpoint thật sự cache nếu probe chỉ báo accepted.
- Hiệu quả cache cần đo qua metadata, usage stats hoặc billing nếu provider có cung cấp.

### 8.1. Capability cache probe

Capability cache probe dùng request nhỏ có kiểm soát để xác nhận:

- Endpoint có chấp nhận prompt cache key không.
- Endpoint có trả usage token không.
- Endpoint có trả cached token hoặc cache-related field không.
- Endpoint có trả cache header như x-cache-status không.
- Khi lặp lại cùng prefix/cache key, report có dấu hiệu hit hoặc vẫn unknown.

Probe này thuộc nhóm test opt-in, nên chạy bằng tag llm_capability_probe hoặc external_api. Probe có thể tạo request thật và có thể tốn chi phí.

### 8.2. Production telemetry

Production telemetry không test cache bằng request phụ. Nó chỉ ghi metadata từ request thật của pipeline.

Config đề xuất:

```yaml
telemetry:
  enabled: false
  capture_usage: true
  capture_cache_headers: true
  capture_raw_headers: false
  log_level: summary
  output_path: logs/llm_telemetry.jsonl
```

Dữ liệu nên capture nếu provider trả về:

| Nhóm              | Ví dụ                                                   | Ghi chú                               |
| ----------------- | ------------------------------------------------------- | ------------------------------------- |
| Request identity  | profile_name, base_url_hash, model, api_mode, task_name | Dùng để tổng hợp theo endpoint.       |
| Cache metadata    | prompt_cache_key, cache_status, cached_tokens           | Dùng để đo cache hit thật.            |
| Usage             | input_tokens, output_tokens, total_tokens               | Dùng để đo chi phí.                   |
| Latency           | latency_ms, retry_count                                 | Dùng để so sánh profile.              |
| Provider metadata | request_id, resolved_model, system_fingerprint          | Dùng để debug hoặc đối chiếu support. |

Quy tắc bảo mật:

- Không log API key hoặc Authorization header.
- Không log full prompt/output mặc định.
- Sanitize hoặc hash base_url nếu endpoint nhạy cảm.
- Thiếu telemetry không được làm fail request thành công.

## 9. Responses API mode

Responses API là mode optional. Không nên bật mặc định cho mọi OpenAI-compatible endpoint.

Capability flag:

```yaml
api_mode: chat_completions
capability_flags:
  supports_responses_api: false
```

Chỉ đặt `api_mode` thành responses khi:

- Endpoint đã probe pass Responses API basic.
- Model hỗ trợ schema request tương ứng.
- Caller đã xử lý output format của Responses API.
- Không dùng provider fallback giữa stateful session.

## 10. Previous response state

Previous response state dùng response id trước đó để tiếp tục ngữ cảnh ở request sau.

Capability flag:

```yaml
capability_flags:
  supports_previous_response_state: false
stateful_options:
  store: false
  use_previous_response_id: false
```

Khi nên dùng:

- Một session dịch theo chương hoặc arc.
- Cùng một endpoint và model.
- Có nhu cầu continuity gần.
- Endpoint hỗ trợ Responses API và store state.

Khi không nên dùng:

- Provider chain có fallback.
- Dịch toàn bộ video 72 giờ trong một state duy nhất.
- Cần inspect hoặc sửa memory thủ công.
- Cần chuyển session giữa nhiều base_url.

Quy tắc:

- State server-side không thay thế glossary hoặc character bible local.
- Nếu previous response state fail, workflow phải có cách resume bằng local translation memory.
- Không lưu response id như nguồn sự thật duy nhất.

## 11. Compaction

Compaction có hai nghĩa cần tách rõ.

### 11.1. API compaction

API compaction nén context/state của Responses API để tiếp tục session.

Capability flag:

```yaml
capability_flags:
  supports_compaction: false
stateful_options:
  compact_threshold: null
```

Khi nên dùng:

- Responses API session dài.
- Cần giảm context cũ nhưng vẫn tiếp tục workflow.
- Endpoint đã probe pass compact endpoint hoặc context management.

Không nên dùng nếu:

- Endpoint chỉ hỗ trợ Chat Completions.
- Workflow cần provider fallback.
- Cần memory có thể đọc và sửa thủ công.

### 11.2. Project-level compaction

Project-level compaction là hướng nên ưu tiên cho dịch video dài. Đây là memory local do dự án tạo và quản lý.

Nên lưu:

- Character bible.
- Glossary.
- Relationship map.
- Style guide.
- Plot summary.
- Pronoun và honorific rules.
- Translation decisions.
- Unresolved ambiguities.

Đối với video 30 đến 72 giờ, project-level compaction quan trọng hơn API compaction vì nó portable giữa các provider và có thể review thủ công.

## 12. Fail-fast khi capability mâu thuẫn

Nếu request_options hoặc task yêu cầu tính năng mà capability_flags chưa bật, provider phải fail-fast trước khi gọi API thật. Mục tiêu là báo đúng cờ YAML cần chỉnh, thay vì để API trả lỗi mơ hồ.

Exception đề xuất:

| Exception                    | Khi dùng                                                                  |
| ---------------------------- | ------------------------------------------------------------------------- |
| CapabilityNotEnabledError    | Tính năng được yêu cầu nhưng capability flag tương ứng là false.          |
| CapabilityModeError          | Tính năng yêu cầu api_mode khác với profile hiện tại.                     |
| CapabilityRejectedError      | Capability flag đã bật nhưng endpoint thật reject request.                |
| CapabilityProbeRequiredError | Tính năng cần probe xác nhận nhưng profile chưa có probe metadata hợp lệ. |

Message lỗi nên có:

- profile_name.
- feature được yêu cầu.
- capability flag cần bật hoặc cần giữ false.
- api_mode hiện tại.
- gợi ý fallback, ví dụ đổi structured_output.mode sang prompt_json.

Ví dụ:

```text
CapabilityNotEnabledError: structured_output.api_schema was requested for profile openrouter_gpt55, but capability_flags.structured_output.supports_chat_response_format is false. Enable this flag only after probe verifies support, or switch request_options.structured_output.mode to prompt_json.
```

## 13. Capability flags chuẩn đề xuất

```yaml
capability_flags:
  supports_chat_completions: true
  supports_responses_api: false
  supports_reasoning_effort: false
  supports_verbosity: false
  supports_prompt_cache_key: false
  supports_previous_response_state: false
  supports_compaction: false
  structured_output:
    supports_prompt_json: true
    supports_chat_response_format: false
    supports_responses_text_format: false
```

Mặc định an toàn:

- supports_chat_completions: true.
- supports_prompt_json: true.
- Các flag còn lại: false cho đến khi probe xác nhận.

## 14. Capability test tags

Để dễ chạy test capability riêng, cần bổ sung các tag sau vào taxonomy trong [`docs/testing-guide.md`](docs/testing-guide.md:321):

| Tag                  | Ý nghĩa                                          |
| -------------------- | ------------------------------------------------ |
| llm_capability       | Tất cả test capability flags.                    |
| openai_compat        | Nhóm test riêng cho OpenAI-compatible provider.  |
| llm_capability_probe | Test gọi endpoint thật để probe capability.      |
| external_api         | Test có gọi dịch vụ ngoài và có thể tốn chi phí. |

Cách chạy đề xuất:

```bash
python run_colab_tests.py --tags llm_capability
python run_colab_tests.py --tags openai_compat
python run_colab_tests.py --tags llm_capability_probe
python run_colab_tests.py --tags external_api
```

Lưu ý: [`run_colab_tests.py`](run_colab_tests.py:328) lọc tag động theo entries trong [`tests/test_matrix.yaml`](tests/test_matrix.yaml:15), nhưng tài liệu testing vẫn phải cập nhật để tag mới không vi phạm strict tagging.

## 15. Real endpoint probe

Real endpoint probe nên được thiết kế opt-in.

Env bắt buộc:

| Env                            | Ý nghĩa                           |
| ------------------------------ | --------------------------------- |
| OPENAI_COMPAT_PROFILE          | Đường dẫn profile YAML cần probe. |
| OPENAI_API_KEY                 | API key.                          |
| OPENAI_COMPAT_PROBE_ALLOW_COST | Phải bằng 1 để test gọi API thật. |

Env tùy chọn:

| Env                         | Ý nghĩa                   |
| --------------------------- | ------------------------- |
| OPENAI_COMPAT_PROBE_MODEL   | Override model khi probe. |
| OPENAI_COMPAT_PROBE_TIMEOUT | Override timeout.         |

Probe output nên phân biệt:

| Trạng thái  | Ý nghĩa                                        |
| ----------- | ---------------------------------------------- |
| unsupported | Endpoint reject.                               |
| accepted    | Endpoint nhận request.                         |
| verified    | Có assertion chứng minh tính năng hoạt động.   |
| skipped     | Thiếu env hoặc prerequisite.                   |
| error       | Auth, quota, network hoặc lỗi không phân loại. |

## 16. Versioned capability report

Real endpoint probe nên ghi report có lịch sử, không lưu đè một file duy nhất.

Đường dẫn đề xuất:

```text
tests/test_reports/openai_compat_capabilities/<profile_name>/<YYYYMMDD-HHMMSSZ>.json
tests/test_reports/openai_compat_capabilities/<profile_name>/latest.json
```

Trạng thái probe:

| Trạng thái  | Ý nghĩa                                                            |
| ----------- | ------------------------------------------------------------------ |
| unsupported | Endpoint reject endpoint hoặc tham số.                             |
| accepted    | Endpoint nhận request nhưng chưa chứng minh chức năng có hiệu lực. |
| verified    | Có assertion chứng minh tính năng hoạt động đúng.                  |
| skipped     | Thiếu env hoặc prerequisite.                                       |
| error       | Auth, quota, network hoặc lỗi không phân loại.                     |

Report nên có:

- probe_schema_version.
- timestamp_utc.
- profile_name.
- base_url_hash hoặc sanitized URL.
- model thực tế.
- sdk_version nếu lấy được.
- kết quả từng capability.
- telemetry_summary về usage/cache headers/cached tokens.
- errors_sanitized.

Không đưa API key, Authorization header, full prompt nhạy cảm hoặc full response không cần thiết vào report.

## 17. Gợi ý profile mẫu tối giản

```yaml
provider: openai
profile_name: openai_compat_safe_default
base_url: https://api.example.com/v1
model: gpt-5.5
api_mode: chat_completions

temperature: 1
max_tokens: 8192
request_timeout: 300
retry_attempts: 3
retry_wait_seconds: 5

capability_flags:
  supports_chat_completions: true
  supports_responses_api: false
  supports_reasoning_effort: false
  supports_verbosity: false
  supports_prompt_cache_key: false
  supports_previous_response_state: false
  supports_compaction: false
  structured_output:
    supports_prompt_json: true
    supports_chat_response_format: false
    supports_responses_text_format: false

request_options:
  reasoning_effort: null
  verbosity: null
  prompt_cache_key: null
  structured_output:
    mode: none
    schema_name: null
    schema: null

stateful_options:
  store: false
  use_previous_response_id: false
  compact_threshold: null

telemetry:
  enabled: false
  capture_usage: true
  capture_cache_headers: true
  capture_raw_headers: false
  log_level: summary
  output_path: logs/llm_telemetry.jsonl
```

## 18. Checklist khi thêm endpoint mới

- Tạo profile riêng trong [`config/llm`](config/llm).
- Không hardcode API key.
- Chạy baseline Chat Completions probe.
- Probe từng capability nâng cao.
- Cập nhật capability flags theo kết quả probe.
- Kiểm tra config mâu thuẫn có raise custom capability exception rõ ràng.
- Chạy unit và mock tests với tag llm_capability.
- Chỉ chạy real probe khi chấp nhận chi phí.
- Lưu report theo profile_name và timestamp, đồng thời cập nhật latest.json.
- Nếu bật prompt_cache_key, bật telemetry phù hợp để đo cache hit/usage thật trong production.
- Ghi chú endpoint nào chỉ accepted và endpoint nào verified.
- Không bật previous response state nếu workflow dùng provider chain.
- Không dùng API compaction thay thế project-level translation memory.
