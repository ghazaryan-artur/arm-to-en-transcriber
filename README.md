# Live Armenian → English Transcriber

A Python program for macOS that listens to a video call — your microphone and the call audio — transcribes spoken Armenian, translates it to English and prints every line live in the terminal with the time and the speaker (`Me` for the microphone, `Speaker 1`, `Speaker 2` … for the people on the call side). All models run locally; no cloud speech or translation APIs. Built for the Mathrix AI Coding Challenge (brief: English within 5 s of the end of a phrase, local models only, Python).

> **The application is in [`armtranscribe/`](armtranscribe/).** How to install and use it, the models and their benchmark numbers, the architecture and the known limitations: **[armtranscribe/README.md](armtranscribe/README.md)**.
>
> In short: `brew install python@3.11 blackhole-2ch` → clone → double-click `armtranscribe/setup.command` → set up a Multi-Output Device once → double-click `armtranscribe/run.command`.

## Result in numbers

| | |
|---|---|
| ASR | NVIDIA FastConformer (Armenian), chosen from 13 local models: CER 2.4–2.9% on read news and the author's voice, ~17% on a conversational podcast; 0.04–0.1 s per segment on the CPU |
| Translation | NLLB-200 1.3B (CTranslate2 int8): chrF++ 66.0 on FLEURS from the reference Armenian, 70.7 end to end (speech → English) on the author's colloquial phrases; plus a glossary for names, English loanwords and one-word replies |
| Speakers | ECAPA-TDNN embeddings + online clustering; stable labels on the test podcast and on real calls with 2 people on the call side |
| Latency | current code, a recorded call replayed in real time on the Mac: p50 2.05 s, p95 3.23 s (limit 5 s); live calls before the last fixes p50 1.4–2.2 s, one p95 of 6.2 s under CPU load, which led to the fallback-model switch |

Per-model numbers below; the reasoning and rejected alternatives are in the [app README](armtranscribe/README.md#models).

## Benchmark results per model

Measured on a MacBook Air M5, 16 GB, after one warm-up call per model. Text is lowercased and stripped of punctuation before WER/CER; `և` / `եւ` / `ու` count as the same word. Test sets:

- **FLEURS**: 20 clips from the `hy_am` **test** split (Eastern Armenian read news), 4.9–10 s each, 10 male / 10 female speakers; the n-way parallel `en_us` sentences are the English references.
- **Own phrases**: 20 short colloquial lines (Yerevan forms, English tech loanwords, names, numbers, short replies) read by the author into the MacBook microphone through the live app (`--save-audio`).
- **Own monologue**: 11 longer conversational sentences, same setup; held out from any tuning.
- **Podcast**: 10 × ~2 min (22 min) of a two-person Armenian tech podcast, cut with the live VAD segmenter. The references are YouTube's *automatic* captions (Google's ASR), which write English terms in Latin script and numbers as digits, so these scores are "distance from YouTube": useful for ranking models, inflated as absolute error rates.

**ASR**: CER / WER per test set (lower is better), 13 local models:

| Model | FLEURS | Own phrases | Own monologue | Podcast | s / segment | Punctuation |
|---|---|---|---|---|---|---|
| **NVIDIA FastConformer hy** (ONNX int8, CPU) | **2.9 / 7.0** | **2.5 / 11.3** | **2.4 / 8.3** | **17.9 / 37.6** | **0.04** | yes + caps |
| SeamlessM4T v2 large (Meta) | 4.2 / 11.2 | 3.6 / 11.3 | 3.1 / 9.0 | 28.9 / 46.7 | 0.7 | yes |
| w2v-BERT 2.0 CTC (`ErikMkrtchyan/w2v-bert-2.0-armenian-CV20.0`) | 3.5 / 11.5 | 7.7 / 32.7 | 3.6 / 17.2 | 19.8 / 56.0 | 0.2 | no |
| Whisper large-v2 FT (`arampacha/whisper-large-hy-2`) | 3.3 / 9.4 | 12.1 / 36.9 | 4.1 / 17.9 | 21.5 / 46.9 | slow¹ | no |
| Whisper large-v3 FT (`BuzzASR/armenian`) | 5.0 / 12.2 | 10.5 / 30.4 | 6.3 / 17.9 | 20.7 / 43.9 | 0.9 | partly |
| Whisper large-v3-turbo FT (`Chillarmo`, previous default, MLX) | 3.8 / 13.3 | 9.7 / 36.3 | 3.8 / 17.2 | 31.8 / 61.2 | 0.6 | yes |
| Whisper large-v3-turbo FT (`ditord`) | 3.8 / 13.3 | 9.5 / 35.1 | 3.8 / 16.6 | 35.4 / 64.4 | 0.7 | yes |
| Meta MMS-1B-all, `hye` adapter (CTC) | 4.5 / 21.3 | 15.0 / 55.4 | 7.3 / 36.6 | 24.0 / 64.0 | 0.3 | no |
| Whisper medium FT (`ArthurYeghinyan/whisper-hy-am-asr-v2`) | 5.1 / 18.9 | 11.8 / 45.2 | 4.9 / 22.1 | 52.4 / 85.5 | 3–6 | yes |
| Whisper large-v3 (base) | 9.6 / 40.2 | 17.6 / 53.0 | 10.5 / 40.7 | 37.8 / 70.3 | 2.9 | yes |
| Whisper large-v3-turbo (base) | 10.5 / 45.1 | 41.6 / 106.0 | 12.1 / 44.8 | 66.0 / 111.4 | 0.8 | yes |
| Gemma 4 E2B (4-bit MLX, prompted) | 14.3 / 38.8 | 20.6 / 51.8 | 19.4 / 45.5 | 35.7 / 67.4 | 0.6 | yes |
| Gemma 4 E4B (4-bit MLX, prompted) | 24.2 / 51.4 | 22.5 / 48.2 | 26.0 / 43.4 | 38.3 / 74.6 | 1.1 | yes |

Seconds per segment are indicative only: backends differ (ONNX on CPU, MLX on the GPU, PyTorch on MPS). ¹ The arampacha run was disturbed by the laptop sleeping; as full large-v2 in PyTorch it is several times slower than the turbo models.

**Translation** (`scripts/bench_mt.py`; chrF++ / BLEU against the English references, gold Armenian input, FLEURS):

| Model | chrF++ | BLEU | s / sentence |
|---|---|---|---|
| **NLLB-200 distilled 1.3B** | **66.0** | **45.6** | 1.11 |
| NLLB-200 distilled 600M | 59.6 | 39.0 | 0.55 |
| Helsinki-NLP opus-mt-hy-en | 44.6 | 22.4 | 0.10 |

NLLB beam size 1/2/4 gives chrF++ 65.0/66.0/66.0 (1.3B), so beam 2 is used.

**End to end** (ASR → NLLB-1.3B, chrF++):

| Input to NLLB | FLEURS | Own phrases | Own monologue |
|---|---|---|---|
| Gold Armenian (ceiling) | 66.0 | 72.8 | 66.5 |
| **NVIDIA FastConformer** | 60.4 | **70.7** | **60.6** |
| Whisper turbo FT (previous default) | 60.6 | 58.9 | 56.5 |

**Numbers** (`tools/numbers_mt/`, 30 sentences with numbers, NLLB-1.3B with the full glossary): number right in the English 27/30 with number words as the ASR writes them (`Քսաներեք։` → "It's twenty.", 2018 → 1818, 2026 → 2600), 29/30 with numbers above ten converted to digits (`app/numbers.py`; the remaining one, `Տասներկուսին։` → "The 12th.", is ambiguous).

**Tuning on recorded calls** (speakers, echo filter, VAD): defaults confirmed by sweeps over all saved sessions; numbers in the [app README](armtranscribe/README.md#benchmark).

**Speaker ID** (SpeechBrain ECAPA-TDNN, `scripts/tune_speakers.py`, 3 podcast clips, 9 min, 2 people): same voice 0.6–0.9 cosine similarity to its centroid, the other voice 0.0–0.35; thresholds 0.35–0.55 give identical labels (default 0.45), exactly 2 speakers per clip. Only one embedding model was tested; pyannote / Resemblyzer were rejected without a run (see the app README).

Harness and per-clip hypotheses of the ASR run: [`tools/asr_bakeoff/`](tools/asr_bakeoff/) (`results/<model>.json`). Why each model was chosen or rejected: [app README → Models](armtranscribe/README.md#models).

## Repository layout

| Path | What it is |
|---|---|
| [`armtranscribe/`](armtranscribe/) | **The application**: package `app/`, `setup.command` / `run.command`, `requirements.txt`, unit tests, model download and benchmark scripts. Self-contained: runs without anything else in this repository. |
| [`docs/PLAN.md`](docs/PLAN.md) | The implementation plan the work followed, phase by phase (0 setup → 6 README and rehearsal), with the parameters and risks decided up front. |
| [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) | Problems we know about and could fix but deliberately leave within this assignment (unplugged headsets, a new speaker's first short replies …), each with the reason and the fix. |
| [`tools/`](tools/) | Development tools around the app (not needed to run it), see below. |
| [`chats/`](chats/) | The AI sessions of this project, one Markdown file per session; the file name's prefix names the tool: `cowork - …` from Claude Cowork, `claude - …` from Claude Code (the user's messages and Claude's replies). |
| [`CLAUDE.md`](CLAUDE.md), [`armtranscribe/CLAUDE.md`](armtranscribe/CLAUDE.md) | Project instructions for the AI assistant: layout, the local-only rule, commands, architecture and conventions. Kept up to date as the project went on; also a compact technical overview for a human reader. |
| [`.claude/settings.json`](.claude/settings.json) | Assistant hooks: blocks network / cloud-API imports in `armtranscribe/app/` (local-only rule) and runs the tests after every Python edit. |

### `tools/`

| Tool | Purpose |
|---|---|
| `record_test.py` | Records a few seconds from the mic and BlackHole: checks the audio routing on this machine. |
| `fetch_fleurs.py` | Fetches the 20 FLEURS `hy_am` test clips with Armenian and English references (benchmark set 1). |
| `build_live_test.py` | Turns `--save-audio` sessions of the author reading a script into the "own phrases" / "own monologue" test sets. |
| `make_podcast_set.py` | Cuts 22 min of an Armenian podcast into clips with YouTube-caption references (conversational test set). |
| `replay_live.py` | Replays a recording through the live pipeline in real time and reports latency p50 / p95; used instead of new test calls. |
| `bench_mt_threads.py` | NLLB speed vs CPU thread count, run with Zoom open. |
| `numbers_mt/` | Numbers in translation: 30 sentences with numbers, NLLB with and without the digit conversion (`eval.py`, results in `last_run.txt`). |
| `export_session.py` | Exports the current Cowork chat (its `.jsonl` transcript) into a Markdown file in `chats/` (`claude - …` / `cowork - …`): the user's messages and Claude's replies, without thinking, tool calls or notes between them. |
| `asr_bakeoff/` | Harness and per-clip results of the 13-model ASR comparison (Whisper fine-tunes, MMS, SeamlessM4T, w2v-BERT, Gemma, FastConformer); source of the ASR table in the app README. |

## How the work was done

1. **Plan first** ([`docs/PLAN.md`](docs/PLAN.md)): a thin end-to-end path (file → Armenian → English) on day one, then live mic, system audio, speakers, latency.
2. **Models by measurement**, not by reputation: an ASR bake-off on four test sets (read news, the author's own phrases and monologue, a two-person podcast) and an MT comparison with chrF++ / BLEU. The default ASR changed halfway (Whisper fine-tune → NVIDIA FastConformer) because the numbers said so.
3. **Real calls as tests**: recorded test calls with friends (Zoom, up to 3 people) drove the segmentation, line merging, echo filter, names and loanwords fixes; each finding with its before / after numbers is in the commit messages and in the chats. After that, changes were checked on the saved recordings with `tools/replay_live.py` rather than new calls.
4. **Unit tests** with synthetic audio and fake models (89 tests) for the VAD, echo filter, clustering, glossary, pipeline and output.

All of the work was done with Claude, in two tools:
- **Claude Cowork** (the desktop app, linked to the repository on the MacBook): planning, research, model choice, analysis of the test calls and recordings, most of the code, the READMEs.
- **Claude Code** (in the terminal, in this repository; [`CLAUDE.md`](CLAUDE.md) and the [`.claude/`](.claude/) hooks are its project setup): coding sessions run directly on the Mac, where the models and audio devices are.

The sessions of both are in [`chats/`](chats/); each file name starts with the tool it comes from: `cowork - <date>_<title>.md` for Claude Cowork, `claude - <date>_<title>.md` for Claude Code.

## Not in the repository

- **Model weights**: downloaded by `armtranscribe/setup.command` into `armtranscribe/models/` (gitignored).
- **Audio**: test recordings, call recordings, FLEURS clips and the podcast stay in a local, gitignored `recordings/` folder (they contain other people's voices). So the benchmarks can be re-run only where that audio exists; FLEURS can be re-fetched with `tools/fetch_fleurs.py`.
- The virtual environment, the assignment brief and third-party transcripts.

**Network use.** The transcriber never uses the network at runtime. Only the one-time model download in `armtranscribe/` and two development tools (`tools/fetch_fleurs.py`, `tools/asr_bakeoff/download.py`, benchmark data and candidate models from Hugging Face) do.
