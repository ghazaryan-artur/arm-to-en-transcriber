# Live Armenian → English Transcriber

Listens to the microphone and system audio, transcribes spoken Armenian, translates it to English and prints it live in the terminal, labelled by speaker. All models run locally.

**Local-only rule:** transcription, translation and speaker identification run entirely on this machine. No cloud speech or translation APIs are used. Model weights are downloaded once during setup (`setup.command` → `scripts/download_models.py`, the only code that uses the network); at runtime the app runs in offline mode (`HF_HUB_OFFLINE=1`) and works with Wi-Fi turned off.

> **Status: work in progress.** Done: live microphone + system audio → Armenian → English, mic lines labelled `Me`, system-audio lines by speaker (`Speaker 1`, `Speaker 2` …), file mode. Not yet: latency hardening (Phase 5).

## Setup

Requirements: a Mac with Apple Silicon, [Homebrew](https://brew.sh), ~4 GB of disk for the app (packages 1.6 GB, models 1.4 GB) plus ~5.5 GB of temporary download.

### 1. Install the program

```bash
brew install python@3.11
git clone <this repository> && cd <repository>/armtranscribe
./setup.command                  # or double-click armtranscribe/setup.command in Finder
```

`setup.command` creates `.venv/` in this folder, installs `requirements.txt` and downloads the default models into `models/`:
the ASR model (NVIDIA FastConformer, 130 MB) and NLLB-200 1.3B, which is downloaded as a ~5.5 GB checkpoint and converted locally to 1.3 GB. The checkpoint stays in `~/.cache/huggingface/` and can be deleted afterwards. The script can be re-run at any time: it reuses the venv and skips models that are already there. At the end it prints the audio devices it sees.

If the repository was downloaded as a ZIP, macOS may refuse to open `setup.command` on double-click: right-click it → Open, or run it from Terminal.

### 2. Microphone permission

The first run of `python -m app` asks for microphone access for the app you run it from (Terminal, iTerm, VS Code). If it was denied: System Settings → Privacy & Security → Microphone → enable that app, then restart it. Without permission the mic records silence.

### 3. System audio (the other side of a call)

macOS cannot record its own output directly, so the call audio is routed through a virtual device, [BlackHole](https://github.com/ExistentialAudio/BlackHole), while you keep hearing it:

1. `brew install blackhole-2ch`, then restart the Mac (or `sudo killall coreaudiod`).
2. Open **Audio MIDI Setup** → `+` → **Create Multi-Output Device**. Tick your headphones (or speakers) and **BlackHole 2ch**, choose the headphones as **Primary Device** (the clock source) and enable **Drift Correction** for BlackHole.
3. Make the Multi-Output Device the output: System Settings → Sound → Output, or the speaker setting inside Zoom / Meet / Telegram.
4. Volume keys don't work on a Multi-Output Device: set the volume on the headphones in Audio MIDI Setup or in the call app.
5. Use headphones, at a normal call volume. With speakers the microphone also picks up the other side, and their words appear twice (echo handling is out of scope). Very loud headphones leak into a headset mic too: in our test at ~100 % volume the call came through the headset mic only ~4 dB below the user's own voice and was transcribed again as `Me`.

### 4. Check

```bash
source .venv/bin/activate
python -m app --list-devices        # the mic and "BlackHole 2ch" should both be listed
python -m app --system BlackHole    # say something in Armenian ("Me" line), play an Armenian video ("Speaker 1" line)
```

To check that nothing goes over the network, turn Wi-Fi off and run it again.

## Usage

Run from this folder with the venv active (`source .venv/bin/activate`):

```bash
python -m app --list-devices
python -m app                          # live: default microphone only, lines labelled "Me"
python -m app --system BlackHole       # live: microphone + system audio ("Speaker 1", "Speaker 2" …), the normal call setup
python -m app --system BlackHole --spk-threshold 0.4   # lower = fewer speakers (merges), higher = more (splits)
python -m app --mic <id|name> --debug  # pick the mic; --debug prints per-stage timings and latency p50/p95 on exit
python -m app --save-audio DIR         # also save the session: full stream, segments, segments.tsv (file, hy, en)
python -m app.file recording.wav       # offline: transcribe + translate a file (add --speakers to label speakers)
python scripts/tune_speakers.py conv.wav --timeline     # speaker labels for several thresholds, to tune on a recording
```
`--asr` / `--mt` select another model from `models/` (`python scripts/download_models.py --list`; fetch more with `python scripts/download_models.py <name>`). Stop with Ctrl+C: segments already captured are still finished.

## Models

| Stage | Default | Runtime |
|---|---|---|
| ASR | [`nvidia/stt_hy_fastconformer_hybrid_large_pc`](https://huggingface.co/nvidia/stt_hy_fastconformer_hybrid_large_pc) (FastConformer RNNT, 115M params, writes punctuation and capitalisation), ONNX int8 export by [`Vana-Labs/stt-fastconformer-armenian`](https://huggingface.co/Vana-Labs/stt-fastconformer-armenian) | `onnx-asr`, CPU |
| Translation | [`facebook/nllb-200-distilled-1.3B`](https://huggingface.co/facebook/nllb-200-distilled-1.3B), CTranslate2 int8, `hye_Armn → eng_Latn`, beam 2 | `ctranslate2`, CPU |
| Speaker ID (system audio) | [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (ECAPA-TDNN, 192-d embeddings) + online cosine clustering | `speechbrain` / PyTorch, CPU |

The previous default ASR, [`Chillarmo/whisper-large-v3-turbo-armenian`](https://huggingface.co/Chillarmo/whisper-large-v3-turbo-armenian) converted to MLX, is still available as `--asr whisper-turbo-hy-mlx`.

### Benchmark

Measured on a MacBook Air M5, 16 GB, after one warm-up call per model. Before WER/CER, text is lowercased and stripped of punctuation, and the word "and" is unified: formal `և` / `եւ` and colloquial `ու` count as the same word. Four test sets:

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

NVIDIA's FastConformer is best on every set, by a wide margin on the speech that matters (own voice CER 2.5% vs 9.7% for the previous default, podcast 17.9% vs 31.8%), is ~15× faster and runs on the CPU, leaving the GPU free. It is a transducer, so it barely inserts text (0–4% insertions vs 8–12% for the Whisper fine-tunes, which loop on conversational audio), and it capitalises names. It tends to write literary forms (`Այս` for colloquial `Էս`, `Երեկ` for `Էրեկ`), which costs WER against colloquial references but helps translation. onnxruntime is pinned to the CPU provider: its default CoreML partitioning is ~4× slower for this graph.

Not benchmarked: `infinitejoy/wav2vec2-large-xls-r-300m-armenian` (its published vocabulary doesn't match the model; the author's own eval log is garbage, WER 69%), Qwen3-ASR (no Armenian), `CAST-ML/whisper-large-v3-mwa-hy` (Western Armenian), `alphaedge-ai` trimmed Whisper (vocabulary-trimmed base model, no Armenian training), Meta Omnilingual ASR (not distributed on Hugging Face).

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

**Live latency** (2 min of the podcast replayed in real time through the live pipeline): segment end → printed p50 1.3 s, p95 2.1 s; ASR 0.04–0.1 s and MT 0.6–1.2 s per segment. With the previous Whisper default: p50 2.1 s, p95 10.7 s, the tail caused by Whisper and NLLB repetition loops.

**Speaker ID** (`scripts/tune_speakers.py`, 3 podcast clips, 9 min, 2 people): the same voice scores 0.6–0.9 against its centroid, the other voice 0.0–0.35; thresholds 0.35–0.55 give identical labels, so the default 0.45 sits in the middle. Each clip ends with exactly 2 speakers; the same person's centroids from different clips agree at 0.89–0.99.

### Rejected alternatives
- **Whisper `task="translate"`**: drops the Armenian transcript and translates worse than a dedicated MT model.
- **SeamlessM4T v2 as ASR**: close to the default on the author's voice, but worse on the podcast, ~15× slower, and CC-BY-NC. Its direct speech → English translation is not evaluated yet.
- **Gemma 4 E2B/E4B**: mixes in Georgian, Cyrillic and Sinhala characters when transcribing Armenian.
- **MADLAD-400 3B**: possibly better MT, but ~3× the size/latency of NLLB-1.3B, which already uses most of the MT budget.
- **Local LLM for MT** (Ollama Gemma/Qwen): slow and weak on Armenian at laptop sizes.
- **pyannote diarization pipeline / diart** for speakers: built for offline or heavier streaming diarization, adds latency, and the pyannote weights are gated behind a Hugging Face token. One embedding per VAD segment is enough here, since the VAD already cuts at pauses. **Resemblyzer**: lighter but weaker embeddings.

## Architecture
_TODO_

## Known limitations
- **Spontaneous, colloquial speech is much harder than read speech.** The ASR models were trained on read Common Voice / FLEURS sentences; Yerevan colloquial forms (`ա` for `է`, `էդ`, `էրկու`) and English tech loanwords (`ինթելիջենս`, `ռոբոտիքս`) are often misspelled, and NLLB then translates the misspelling ("Dune scenario" → "snow scenario"). With the FastConformer most podcast lines now keep their meaning, but CER against YouTube's captions is still ~18%.
- **Proper names and English terms.** NLLB may still translate a name that is also a common word, and English loanwords written in Armenian script are often mistranslated (`բեթրիփեք` "battery pack" → "tripectin bacteria").
- **ASR errors cascade into MT.** One wrong letter can change the meaning (`ՄԱԿ-ի` "UN's" → `մակի` → "mouse expert").
- **Licenses.** NVIDIA FastConformer: CC-BY-4.0 (attribution to NVIDIA). NLLB-200: CC-BY-NC-4.0 (non-commercial). SpeechBrain ECAPA: Apache-2.0.
- **Echo.** The microphone is not echo-cancelled: with speakers, or very loud headphones, the other side is transcribed twice (`System` and `Me`). Use headphones at a normal volume.
- **Overlapping speech.** When both sides talk at once, the mic segment also contains the other side, and the `Me` line mixes both.
- **Speaker ID works per segment.** A segment in which the speaker changes (fast turn-taking, no pause) gets one label. Segments under 1 s are too short for a reliable voice embedding and keep the previous label; a new speaker is only created from a segment of ≥ 3 s, so a new person's short first replies are attributed to someone already known (this keeps laughs and cross-talk fragments from creating phantom speakers).
- **Speaker numbers are per session** and depend on who speaks first; there is no enrolment of known voices. Similar voices (same gender, similar pitch) can merge at a low threshold; noisy audio or music can split one person into two at a high one (`--spk-threshold`).
- _TODO: Western Armenian._
