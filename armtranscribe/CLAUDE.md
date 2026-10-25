# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the program in `armtranscribe/` (the repo root is the workspace above it).

Live Armenian → English transcriber: captures mic + system audio on macOS, transcribes Armenian, translates to English, prints to the terminal labelled by speaker. Status: Phase 4 done — live mic + system audio → HY → EN works (`python -m app --system BlackHole`, mic labelled `Me`, system audio `Speaker N` via `speakers.OnlineSpeakerTracker`, one VAD segmenter per stream); file mode uses the same streaming VAD segmenter (`--speakers` labels speakers). Now: testing in real calls (recordings in `../recordings/calls/`, findings in docs/TECH_DEBT.md) — fixes driven by the calls (segmentation, continued lines, names, short replies) come before the rest of Phase 5 (latency & hardening).

## Hard constraint: local-only

Transcription, translation, and speaker ID must run entirely on-device. Never add cloud speech/translation/LLM APIs or runtime network calls. `cli.py` sets `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` / `HF_DATASETS_OFFLINE=1` before any model import — keep that ordering (env vars must be set before `transformers`/`huggingface_hub` are imported). The only code allowed to hit the network is `scripts/download_models.py` (one-time weight download into `models/`). Model weights and audio files (`*.wav`, etc.) are gitignored and must never be committed.

## Commands

Environment: Python 3.11 venv in `armtranscribe/.venv` (gitignored), Apple Silicon (default ASR runs on the CPU via `onnx-asr`; Whisper alternatives use `mlx-whisper`). Run everything from this directory with the venv activated.

```bash
./setup.command                               # creates .venv, installs requirements.txt, downloads default models (also by double-click)
source .venv/bin/activate
pip install pytest                            # dev only: pytest is not in requirements.txt
python -m pytest                              # all tests
python -m pytest tests/test_output.py::test_render   # single test
python -m app --list-devices
python -m app [--mic <id|name>] [--system <id|name>] [--asr ...] [--mt ...] [--debug] [--save-audio DIR]   # live; no --mic = system default input, no --system = mic only
python -m app.file <audio> [--asr ...] [--mt ...] [--debug]   # offline: file -> HY -> EN
python scripts/download_models.py [--bench | --list | <names>]        # one-time download/convert into models/
python scripts/bench_asr.py <manifest.tsv> [--models ...] [--save hyp.tsv] [--vad]   # WER/CER (jiwer) + latency; --vad: cut long clips like live
python scripts/bench_mt.py <manifest.tsv> [--asr-hyps hyp.tsv]               # chrF++/BLEU (sacrebleu) + latency
python scripts/tune_speakers.py <conv.wav> [--thresholds ...] [--timeline]  # speaker clustering vs threshold
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
- **Lines** (`pipeline._Line`): a segment that follows the same source + speaker within `MERGE_GAP_S` (1.5 s between segments ≈ a 2 s pause) extends the previous line (max 3 segments / 30 s); `CT2Translator.translate_continuation` translates the previous segment + the new one with the previous segment's English forced as the output prefix (only one segment of context, so MT time stays bounded) and returns (previous English as it now reads, new English) (falls back to translating the new part alone when NLLB stops early — it tends to drop the second of two sentences). `output.Terminal` redraws the line in place (ANSI) if nothing was printed since, else prints it again marked `↻`; all other terminal output goes through `output.notice` so a redraw never erases it.
- **Glossary** (`glossary.py`, wired in `cli.with_glossary`): `NamedTranscriber` capitalises known names (`COMMON_NAMES` anywhere; `JAN_ONLY` word-like names and any capitalised non-kin word only before `ջան`) and drops a leading dialogue dash; `GlossaryTranslator` translates sentence by sentence (NLLB drops the 2nd of two sentences), writes English loanwords in English (`terms.py`, loose spelling classes) and names in Latin (`Gor-ի`), answers one-word replies from `SHORT_REPLIES`, fixes terms (`drachma` → `dram`). NLLB decoding: `no_repeat_ngram_size=4`, output ≤ 1.5× source + 10 tokens (repetition loops).
- **Echo filter** (`echo.py`): the pipeline keeps the last 40 s of the system stream (`StreamTail`); a mic segment whose normalised cross-correlation with it (±0.3 s) is ≥ 0.3 is dropped before ASR — leaked call audio scores 0.37–0.59 on the recorded calls, own speech ≤ 0.16.
- **Swappable backends** via `typing.Protocol` so benchmarks can compare models: `asr.Transcriber.transcribe(audio) -> str`, `translate.Translator.translate(hy) -> str`, `speakers.SpeakerTracker.assign(audio) -> str`. New model backends implement these; `--asr` / `--mt` select them.
- **Default models** (chosen by bake-offs, numbers in README): ASR `fastconformer-hy-onnx` = NVIDIA `stt_hy_fastconformer_hybrid_large_pc` as ONNX int8 (`Vana-Labs/stt-fastconformer-armenian`), run by `onnx-asr` on the CPU provider only (`asr.NemoONNX`; writes punctuation + capitalisation). Alternative `whisper-turbo-hy-mlx` = `Chillarmo/whisper-large-v3-turbo-armenian` converted HF → MLX fp16 (Whisper with `language="hy"`, `temperature=0`, `condition_on_previous_text=False`, filter hallucinations / high `no_speech_prob`); MT `nllb-1.3b-ct2` = NLLB-200 distilled 1.3B in CTranslate2 int8 (`hye_Armn → eng_Latn`, beam 2, 6 threads); speaker embeddings SpeechBrain ECAPA (`spkrec-ecapa-voxceleb`) + online cosine clustering.
- `--asr` / `--mt` take a model name under `models/` (see `download_models.py --list`) or a path; `asr.load_transcriber` picks the backend from the dir's `config.json` (NeMo ONNX, MLX Whisper or MMS CTC), `translate.CT2Translator` handles NLLB and Marian dirs.
- Benchmark manifests are header-less TSV `file<TAB>hy<TAB>en` with paths relative to the manifest; the audio itself is never committed (it lives in the gitignored `recordings/` next to `armtranscribe/`).
- **Latency target**: p95 segment-end → printed < 5 s. Main knobs are in the VAD (≈1 s silence to end a segment, force-cut at ≈12 s in the longest pause of the last 3 s, drop < 0.4 s). Longer segments cost time-to-first-word but give MT whole sentences; tuned on the first real call (see docs/TECH_DEBT.md).
- Heavy imports (`sounddevice`, `torch`, models) are done lazily inside functions so `--list-devices` and tests stay fast.

## Repo conventions

- `README.md` here is the program's README (setup incl. audio devices, usage, models with benchmark numbers, architecture, limitations); the repo-root README is only an overview of the repository. Sections marked `_TODO_` (setup, model choices with benchmark numbers, architecture, limitations) are to be filled in as phases complete.
