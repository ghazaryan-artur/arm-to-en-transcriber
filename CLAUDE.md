# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Live Armenian → English transcriber: captures mic + system audio on macOS, transcribes Armenian, translates to English, prints to the terminal labelled by speaker. Status: early skeleton — most modules are stubs.

## Hard constraint: local-only

Transcription, translation, and speaker ID must run entirely on-device. Never add cloud speech/translation/LLM APIs or runtime network calls. `cli.py` sets `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` / `HF_DATASETS_OFFLINE=1` before any model import — keep that ordering (env vars must be set before `transformers`/`huggingface_hub` are imported). The only code allowed to hit the network is `scripts/download_models.py` (one-time weight download into `models/`). Model weights and audio files (`*.wav`, etc.) are gitignored and must never be committed.

## Commands

Environment: Python 3.11 venv, Apple Silicon (ASR uses `mlx-whisper`). Run everything from this directory with the venv activated.

```bash
pip install -r requirements.txt pytest       # pytest is not in requirements.txt
python -m pytest                              # all tests
python -m pytest tests/test_output.py::test_render   # single test
python -m armtranscribe --list-devices
python -m armtranscribe --mic <id|name> --system <id|name> [--asr ...] [--mt ...] [--debug]
python scripts/download_models.py             # one-time weight download into models/
python scripts/bench_asr.py / bench_mt.py     # model bake-offs (WER/CER via jiwer, latency)
```

No linter/formatter is configured.

System audio comes from **BlackHole 2ch** via a macOS Multi-Output Device (speakers/headphones + BlackHole 2ch in Audio MIDI Setup). Claude can't hear or produce live audio here, so verify audio code with recorded WAVs or synthetic arrays.

## Architecture

```
mic InputStream ──┐                                   ┌ source="mic"    → "Me"
                  ├► resample to 16 kHz mono ► VAD ───┤
BlackHole stream ─┘   (soxr, consumer thread)          └ source="system" → SpeakerTracker → "Speaker N"
                                     │
                     Segment queue → single ASR worker → MT worker → output.render
```

- **Concurrency**: plain threads + `queue.Queue`, no asyncio. PortAudio callbacks only `indata.copy()` into a queue — never block, resample, or run models in a callback. One ASR worker (GPU is single-use); segments processed in end-time order.
- **Data types** (`models.py`): `Segment` (source, t_start/t_end in seconds since session start, float32 mono audio @ `SAMPLE_RATE`=16000) flows into the workers; `Result` (speaker, hy, en, latency) flows to `output.render`.
- **Swappable backends** via `typing.Protocol` so benchmarks can compare models: `asr.Transcriber.transcribe(audio) -> str`, `translate.Translator.translate(hy) -> str`, `speakers.SpeakerTracker.assign(audio) -> str`. New model backends implement these; `--asr` / `--mt` select them.
- **Default models**: ASR `mlx-community/whisper-large-v3-mlx` (Whisper with `language="hy"`, `temperature=0`, `condition_on_previous_text=False`, filter hallucinations / high `no_speech_prob`); MT NLLB-200 distilled 600M converted to CTranslate2 int8 at `models/nllb-600m-ct2` (`hye_Armn → eng_Latn`); speaker embeddings SpeechBrain ECAPA (`spkrec-ecapa-voxceleb`) + online cosine clustering.
- **Latency target**: p95 segment-end → printed < 5 s. Main knobs are in the VAD (≈600 ms silence to end a segment, force-cut at ≈8 s, drop < 0.4 s).
- Heavy imports (`sounddevice`, `torch`, models) are done lazily inside functions so `--list-devices` and tests stay fast.

## Repo conventions

- `chats/` holds exported Claude/Cowork session transcripts (one `.md` per session) — part of the project deliverable.
- README sections marked `_TODO_` (setup, model choices with benchmark numbers, architecture, limitations) are to be filled in as phases complete.
