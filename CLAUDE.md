# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## MANDATORY: Web search & URL fetch must use the `web-haiku` subagent

To keep web research fast and cheap, you MUST NOT call the `WebSearch` or `WebFetch` tools directly. Instead, ALWAYS delegate any task that needs a web search or URL fetch to the `web-haiku` subagent (it runs on the Haiku model) via the Agent tool.

- Need to search the web for information → spawn `web-haiku`.
- Need to fetch/read/summarize a URL → spawn `web-haiku`.
- This applies even for a single quick lookup. The only exception is authenticated/private URLs that `WebFetch` cannot access (use the appropriate MCP tool or `gh` for those).

Pass the full query or URL plus what you need extracted, and act on the results the subagent returns.

## Overview

This is a **video dubbing pipeline** that redubs Chinese (or other source-language) videos into another language. The primary runtime environment is **Google Colab** with GPU.

### Colab workflow (end-to-end)

1. **Setup** — Mount Google Drive, copy unprocessed video folders to local Colab storage, install dependencies (`uv`, `rubberband-cli`, `audio-separator`, etc.), clone this repo, and install the package.
2. **ASR / Transcription** — Run `qwen3-asr-srt` (in a dedicated `.venv-qwen3asr` virtualenv that inherits Colab's PyTorch) to generate SRT files from the source audio.
3. **Translation** — Run `translate-srt` to translate the SRT into the target language via an LLM provider (Gemini / VertexAI / OpenAI).
4. **Sync & Dub** — Run `sync-video` with a TTS provider (Qwen, EdgeTTS, Voicevox) to generate dubbed audio, time-stretch the video to match, and burn in hardcoded subtitles.
5. **Upload** — Output folders (`sync_output/`) are written back to Google Drive; the Drive sync step skips any folder that already has a `sync_output/` directory.

Input videos and task JSON files live on Google Drive under `MyDrive/CharenjiZukan/<batch>/`. Each batch folder contains one sub-folder per video episode.

## Architecture

This is a **video dubbing & subtitle pipeline** tool: it takes a Chinese (or other source) video + SRT subtitle, generates TTS audio, then time-stretches the video to fit the dubbed audio, and burns subtitles into the final output.

### Pipeline (sync-video main flow)

`cli/sync_video.py::run_sync_pipeline` orchestrates 5 phases:

| Phase | Module                              | What it does                                                                       |
| ----- | ----------------------------------- | ---------------------------------------------------------------------------------- |
| 0     | `tts/`                              | Generate TTS `.wav` clips from subtitle lines                                      |
| 1     | `sync_engine/analyzer.py`           | Classify timeline blocks (tts / mute / gap), compute per-segment video/audio speed |
| 2     | `sync_engine/video_processor.py`    | FFmpeg: stretch video chunks in parallel, concat                                   |
| 3     | `sync_engine/audio_assembler.py`    | Mix TTS clips + original audio + ambient + BGM                                     |
| 4     | `sync_engine/timestamp_remapper.py` | Recalculate SRT/ASS timestamps for the stretched timeline                          |
| 5     | `sync_engine/renderer.py`           | FFmpeg: hardsub + watermark + note_overlay → final MP4                             |

Data flows through `sync_engine/models.py` via two dataclasses:

- `SubBlock` — one subtitle/mute/gap entry with TTS clip path + duration
- `TimelineSegment` — one video chunk with original and new time bounds + speeds

### Key modules

**`cli/`** — Entry points registered as `[project.scripts]` in `pyproject.toml`. Each is self-contained with `argparse`. `sync_video.py` is the master pipeline; others handle standalone operations (translate, TTS only, audio separation, OCR, etc.).

**`sync_engine/`** — Core pipeline logic. `analyzer.py` is the algorithmic heart: it classifies blocks, computes the speed factor per block so TTS audio fits within the original subtitle slot, and builds the `TimelineSegment` list.

**`tts/`** — TTS engine abstraction. `tts/base.py` defines the interface; concrete engines: `edgetts.py` (Microsoft Edge TTS), `voicevox.py` / `voicevox_nemo.py` (Voicevox HTTP API, local), `qwen.py` (GPU model). All engines receive a `queue_tts` list and return stats `{ok, err}`.

**`llm_ai/`** — LLM provider abstraction. `base.py` defines `BaseLLMProvider` with `call(message) -> str`. `factory.py` instantiates providers (gemini / openai / vertexai). `provider_chain.py` implements `FallbackLLMProvider` which tries providers in order until one succeeds. Task configs (YAML) in `config/llm_tasks/` control model, prompts, and provider_chain.

**`translation/`** — SRT translation logic using `llm_ai/`. `srt_translator.py` batches subtitle lines, calls the LLM provider, and parses responses.

**`video_subtitle_extractor/`** — OCR-based subtitle extraction from video frames.

### Configuration

- `config/tts_config.yaml` — Default TTS settings (voice, speed, concurrency) per provider.
- `assets/default_render_config.json` — Render settings: resolution, subtitle style, watermark, audio mix volumes, `audio_policies` (bgm/mute/ambient), `note_overlay`, `image_overlay`, `forced_alignment_subtitle`, `llm_metadata`.
- `config/llm_tasks/*.yaml` — Per-task LLM configs (model, system_prompt, provider_chain).
- `config/llm/*.yaml` — LLM provider secrets/API keys.

### Task file mode

All CLI tools that handle multiple videos support `--task-file tasks.json` (a JSON array of task objects). Each task object maps to the same CLI arguments. `sync_video` runs each task in a fresh `multiprocessing.spawn` subprocess to fully reclaim GPU memory between videos.

### External dependencies

- **FFmpeg + ffprobe** must be on PATH (video/audio processing).
- **HEVC NVENC GPU** is used by default for video encoding (`hevc_nvenc -preset p4 -tune hq -cq 28`); pass `--no-gpu` only for compatibility (the flag is legacy — GPU is always attempted first).
- **Voicevox / Voicevox Nemo** requires a local HTTP server (`127.0.0.1:50021` or `:50121`).
- Qwen TTS requires a CUDA GPU and the `qwen-tts` optional dependency group.
