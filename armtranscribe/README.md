# Live Armenian → English Transcriber

Listens to a call (your microphone and the system audio, i.e. the other side), transcribes spoken Armenian, translates it to English and prints each line live in the terminal with the time and the speaker. Everything from the microphone is `Me`; people on the call side are told apart as `Speaker 1`, `Speaker 2` … and keep their label for the whole session.

```
[00:01:12] Speaker 1
HY: Բարև, ինչպե՞ս ես։
EN: Hi, how are you?

[00:01:15] Me
HY: Լավ եմ, շնորհակալություն։
EN: I'm fine, thank you.
```

**Local only.** Transcription, translation and speaker identification run on this Mac; no cloud speech or translation APIs. The models are downloaded once by `setup.command` (the only step that uses the network). At runtime the app sets Hugging Face offline mode (`HF_HUB_OFFLINE=1`) and works with Wi-Fi off.

Requirements: a Mac with Apple Silicon (developed and tested on a MacBook Air M5, 16 GB, macOS), [Homebrew](https://brew.sh), headphones, ~5 GB of disk for the app (packages 1.6 GB, models 2.1 GB) plus ~8 GB of temporary download.

## Quick start

```bash
brew install python@3.11 blackhole-2ch          # then restart the Mac once (BlackHole is an audio driver)
git clone https://github.com/ghazaryan-artur/arm-to-en-transcriber.git
open arm-to-en-transcriber/armtranscribe/setup.command     # or double-click it in Finder
```

1. `setup.command` installs everything into `armtranscribe/.venv` and downloads the models (first run 15–30 min, depends on the connection).
2. Once, in **Audio MIDI Setup**, create a Multi-Output Device = headphones + BlackHole 2ch, and pick it as the speaker in the call app ([step 3](#3-system-audio-the-other-side-of-the-call)).
3. Double-click **`armtranscribe/run.command`**. When it prints `Ready, listening`, talk. Ctrl+C stops.

The rest of this section explains each step in detail.

## Setup

### 1. Install the program

```bash
brew install python@3.11
git clone https://github.com/ghazaryan-artur/arm-to-en-transcriber.git
cd arm-to-en-transcriber/armtranscribe
./setup.command                  # or double-click armtranscribe/setup.command in Finder
```

`setup.command`:
- creates `.venv/` in this folder with Python 3.11 and installs `requirements.txt` (versions pinned to the tested set, ~1.6 GB);
- downloads the models into `models/` (skipped if already there):

  | Model | Role | On disk |
  |---|---|---|
  | `fastconformer-hy-onnx` | Armenian ASR | 130 MB |
  | `nllb-1.3b-ct2` | Armenian → English translation (main) | 1.3 GB |
  | `nllb-600m-ct2` | faster translation, used only while the app falls behind | 0.6 GB |
  | `ecapa-voxceleb` | speaker embeddings | 85 MB |

  The NLLB models are downloaded as original checkpoints (~5.5 GB + 2.5 GB) and converted locally to CTranslate2 int8. The checkpoints stay in `~/.cache/huggingface/` and can be deleted afterwards;
- prints the audio devices it sees and warns if BlackHole is missing.

It can be re-run any time (after a failed download, for example): it reuses the venv and skips models that are already there.

If the repository was downloaded as a ZIP, macOS may refuse to open `setup.command` / `run.command` on double-click: right-click → Open once, or run it from Terminal.

### 2. Microphone permission

On the first start macOS asks for microphone access for Terminal (or iTerm / VS Code, whichever runs the app). If it was denied: System Settings → Privacy & Security → Microphone → enable that app, then restart it. Without the permission the mic records silence and no `Me` lines appear.

### 3. System audio (the other side of the call)

macOS can't record its own output, so the call audio is sent through a virtual device, [BlackHole](https://github.com/ExistentialAudio/BlackHole), while you keep hearing it in your headphones:

1. `brew install blackhole-2ch`, then restart the Mac (or `sudo killall coreaudiod`).
2. Open **Audio MIDI Setup** (Spotlight: "Audio MIDI") → `+` bottom left → **Create Multi-Output Device**. Tick your **headphones** and **BlackHole 2ch**, set the headphones as **Primary Device** (the clock) and tick **Drift Correction** for BlackHole.
3. Send the call to it. Either in the call app:
   - **Zoom**: Settings → Audio → **Speaker** = *Multi-Output Device*, **Microphone** = your headset / MacBook mic;
   - **Google Meet** (Chrome): ⋮ → Settings → Audio → **Speakers** = *Multi-Output Device*;
   - **Telegram**: Settings → Speakers and Camera → **Output device** = *Multi-Output Device*;

   or for every app at once: System Settings → Sound → Output → *Multi-Output Device*.
4. Volume keys don't work on a Multi-Output Device: set the volume in the call app or on the headphones in Audio MIDI Setup.
5. Use **headphones at a normal call volume**. With speakers the mic hears the other side again; the echo filter drops most of that (see [Known limitations](#known-limitations)), but not when you talk at the same time.

The Multi-Output Device contains specific headphones. If you take them off, plug in others, or macOS switches the output to newly connected headphones by itself, BlackHole receives nothing: the app warns after 15 s of digital silence on the system stream. Re-select the Multi-Output Device (or make a second one with the MacBook speakers + BlackHole and switch between the two in the menu-bar sound menu).

### 4. Check

```bash
source .venv/bin/activate
python -m app --list-devices        # your mic and "BlackHole 2ch" should both be listed
python -m app --system BlackHole    # say something in Armenian -> a "Me" line;
                                    # play an Armenian YouTube video (output = Multi-Output) -> "Speaker 1" lines
```

To confirm that nothing goes over the network, turn Wi-Fi off and run it again.

## Usage

Double-click `run.command`, or run from this folder with the venv active (`source .venv/bin/activate`):

```bash
./run.command                              # = python -m app --system BlackHole: default mic + call audio, the normal setup
./run.command --mic "External Microphone" --debug    # run.command passes any option on

python -m app --list-devices               # input devices with ids
python -m app                              # microphone only, every line "Me"
python -m app --system BlackHole           # microphone + system audio ("Speaker 1", "Speaker 2" …)
python -m app --mic <id|name>              # another mic (default: the macOS default input); pick the same one as in the call app
python -m app --system BlackHole --debug   # + per-line timings, speaker similarity; latency and stage p50/p95 on exit
python -m app --system BlackHole --spk-threshold 0.4  # lower = fewer speakers (merges), higher = more (splits); default 0.45
python -m app --save-audio DIR             # also save the session: both streams, every segment, segments.tsv (file, hy, en)
python -m app.file recording.wav [--speakers]         # offline: transcribe + translate a file
```

| Option | Default | Meaning |
|---|---|---|
| `--mic` | macOS default input | microphone, by id or (part of the) name |
| `--system` | none (mic only) | loopback device for the call audio, normally `BlackHole` |
| `--asr` | `fastconformer-hy-onnx` | ASR model in `models/` or a path |
| `--mt` | `nllb-1.3b-ct2` | translation model |
| `--mt-fast` | `nllb-600m-ct2` | translation model while the app falls behind; `none` = only warn |
| `--speaker-model` | `ecapa-voxceleb` | speaker embedding model |
| `--spk-threshold` | `0.45` | cosine similarity needed to join a known speaker |
| `--debug` | off | timings per line and a summary on exit |
| `--save-audio DIR` | off | keep the session audio for later evaluation |

Other models: `python scripts/download_models.py --list` and `python scripts/download_models.py <name>` (for example `whisper-turbo-hy-mlx`, the previous ASR default, then `--asr whisper-turbo-hy-mlx`).

Stop with **Ctrl+C**: segments already captured are still finished; a second Ctrl+C quits at once.

### What you see

- **One block per line**: `[time since start] speaker`, `HY:` and `EN:`. Lines appear after the speaker pauses (~1 s of silence) or after at most ~12 s of continuous speech.
- **Lines that continue.** People pause mid-sentence. A segment that follows the same speaker within ~2 s of silence is translated together with the previous one and that block is updated in place (or printed again marked `↻` if another line came in between). The English already shown stays, only the continuation is added.
- **`Speaker ?`**: call audio before the first speaker is established (that needs ≥ 3 s of one voice, so laughs and short fragments don't create phantom speakers).
- **Warnings** go to stderr: no audio from a device (unplugged headset, silent BlackHole), the app falling behind and switching to the fast translation model, and back.

### Names, English words, short replies

The ASR writes names in lowercase and NLLB translates them as words (`աղաս` → "you little prick", `գոռի` → "the old man's"). About 330 Armenian first names (and Russian ones common in Armenia) are recognised with their case endings (`Գոռի`, `Աննային`) and passed to the translator in Latin (`Gor-ի` → "Gor's"). Names that are usually ordinary words in speech (`Արա` "hey / do!", `Սիրուն` "pretty", `Արև` "sun" …) count as names only before `ջան` (`Արա ջան` → "Ara"); so does any other capitalised word before `ջան` (rare names), except kin words (`Մամ ջան`). `Անուշ` is a name except in `անուշ եղի / լինի` ("enjoy your meal"); `Արա,` without `ջան` is translated as "Hey,".

English words said in Armenian (`չատ ջիպիթի`, `վեյբ կոդինգ`, `սկիլերը`, `դեդլայնից` …, ~80 IT / startup terms in `app/terms.py`, matched loosely because the ASR mixes up similar consonants) are passed to the translator in English (`skill-երը` → "skills"). Each sentence of a segment is translated on its own: given two, NLLB often drops the second. One-word replies come from a fixed table (`Ապրես` → "Thanks, well done!", `Այո` → "Yes.").

**Numbers** above ten are passed to the translator as digits with their ending (`Տասներկուսին` → `12-ին`, `երկու հազար տասնութ թվականին` → `2018 թվականին`): as words NLLB got 3 of 30 test sentences wrong (`Քսաներեք։` → "It's twenty.", 2018 → 1818, 2026 → 2600), as digits none of those. A sentence that is only a number is printed as digits (`23.`). Numbers up to ten stay words (`մի երկու բան` "a couple of things"). **Colloquial `էս / էդ / էն`** (this / that) and their compounds (`էսօր`, `էդքան`, `էնտեղ`) are passed as literary `այս / այդ / այն`: NLLB read `Էս` as a name ("Ace Vegas code") and `էդ` often as "your"; on the 10 test-script lines that contain them chrF++ 68.3 → 73.1.

## Before a call (checklist)

- [ ] Headphones on; the call app's **speaker** = Multi-Output Device, its **microphone** = the mic you give the app (or the macOS default).
- [ ] `python -m app --list-devices` shows the mic and BlackHole 2ch.
- [ ] Laptop on the charger, heavy apps closed (NLLB uses 6 CPU threads; the call app needs the rest).
- [ ] Start the app (`./run.command`) before the call and wait for `Ready, listening`: loading and warming up the models takes a little while.
- [ ] Ask the first person on the call side to say a full sentence or two: the first speaker label is created from ≥ 3 s of speech.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| No `Speaker` lines, warning about silence on the system stream | The call doesn't go to BlackHole: the output (call app speaker or System Settings → Sound) isn't the Multi-Output Device, or macOS switched to newly connected headphones. |
| You hear nothing, but `Speaker` lines appear | The headphones in the Multi-Output Device are off / disconnected: BlackHole still gets the audio. Re-check the device in Audio MIDI Setup. |
| No `Me` lines | Microphone permission (setup step 2), or the app listens to another mic than you speak into: `--list-devices`, then `--mic`. |
| The other side also appears as `Me` | Their voice reaches your mic (speakers, very loud headphones). The echo filter drops what is a clear copy of the call; lower the volume / use headphones. |
| `Device not found` | The error lists the input devices; pass an id or a part of the name to `--mic` / `--system`. |
| One person split into two speakers, or two merged | `--spk-threshold` higher splits more, lower merges more (0.35–0.55 gave the same labels on the test podcast). |
| "Falling behind" warnings | CPU busy (video / screen share in the call app). The app translates with the faster model until it has caught up. |
| `setup.command` fails | Read the error above the "Setup failed" line; fix (network, disk space, `brew install python@3.11`) and run it again: finished steps are skipped. |

## How it works

```
 Microphone ──► InputStream ─┐                                      ┌─ mic    ──► echo filter ──► "Me"
                             ├─► resample to 16 kHz ─► VAD segmenter┤
 BlackHole 2ch ► InputStream ┘   mono (soxr)          (one per stream)└─ system ──► speaker tracker ─► "Speaker N"
                                                                │
                                         segment queue (source, start, end, audio)
                                                                │
                                ASR worker (Armenian text, names) ──► MT worker (English) ──► terminal
                                                                       (NLLB-1.3B, or 600M while behind)
```

- **Capture** (`audio.py`): one PortAudio input stream per device. The callbacks only copy the audio into a queue; resampling to 16 kHz mono (`soxr`) happens in a consumer thread. If a device won't open at its default rate, 48 / 44.1 / 16 kHz are tried; a device that delivers nothing for 3 s is reported.
- **Segmentation** (`vad.py`): Silero VAD on 32 ms frames, one segmenter per stream. A segment starts at speech (with 0.25 s of pre-roll) and ends after **1 s of silence**; continuous speech is force-cut at **12 s**, in the longest pause of the last 3 s, so words and sentences are kept whole. Segments under 0.4 s are dropped. These values come from the first real call, where shorter pauses split sentences and the translation lost their meaning.
- **Echo filter** (`echo.py`): the last 40 s of the system stream are kept; a mic segment whose normalised cross-correlation with it is ≥ 0.3 is call audio leaking into the mic and is dropped before ASR (leaked audio 0.37–0.59 on recorded calls, own speech ≤ 0.16).
- **ASR** (`asr.py`): one worker thread, segments in end-time order. NVIDIA FastConformer through `onnx-asr` on the CPU, with punctuation and capitalisation. `glossary.NamedTranscriber` fixes name capitalisation and drops dialogue dashes and single-letter output.
- **Speaker ID** (`speakers.py`), system stream only: a 192-d ECAPA embedding per segment, compared by cosine similarity with each known speaker's running-mean centroid. ≥ 0.45 → that speaker; otherwise a new `Speaker N`, but only from a segment ≥ 3 s. Segments < 1 s keep the previous label.
- **Lines** (`pipeline.py`): a segment from the same source and speaker within ~2 s of silence continues the previous line (max 3 segments / 30 s). It is translated with only the segment before as context, with that segment's English forced as the prefix, so the English already shown doesn't change and MT time doesn't grow with the line.
- **Translation** (`translate.py`, `glossary.py`, `terms.py`): NLLB-200 1.3B in CTranslate2 int8, 6 threads, beam 2, `no_repeat_ngram_size=4` and an output length cap (against repetition loops). Sentence by sentence, with names and English loanwords pre-written in Latin script, numbers above ten as digits, colloquial `էս/էդ/էն` as literary forms, one-word replies from a table.
- **Backpressure** (`pipeline.py`): the lag is measured when MT starts; at ≥ 2.5 s behind, the app warns and translates with NLLB-600M (≈ 2× faster) until the lag is ≤ 1 s.
- **Output** (`output.py`): blocks with time, speaker, HY, EN; a continued line is redrawn in place when nothing else was printed after it.
- **Startup** (`cli.py`): sets offline mode before any model import, loads every model and runs a warm-up call through each and prints `Ready` only then. Ctrl+C finishes the queued segments; a second one quits.

Threads and `queue.Queue`, no asyncio. Each stage sits behind a small interface (`Transcriber.transcribe(audio) -> str`, `Translator.translate(hy) -> str`, `SpeakerTracker.assign(audio) -> str`), which is how the benchmarks swap models and the tests use fakes.

| File | What it does |
|---|---|
| `app/cli.py` | command line, model loading / warm-up, wiring, exit summary |
| `app/audio.py` | device listing and lookup, capture, resampling, silent-device check |
| `app/vad.py` | streaming Silero VAD segmenter |
| `app/echo.py` | drops mic segments that are the call audio |
| `app/asr.py` | ASR backends: NeMo ONNX (default), MLX Whisper, MMS CTC |
| `app/speakers.py` | ECAPA embeddings + online clustering |
| `app/translate.py` | CTranslate2 translator (NLLB, Marian) incl. continuation |
| `app/glossary.py`, `app/terms.py`, `app/numbers.py` | names, English loanwords, numbers → digits, colloquial forms, short replies, sentence split |
| `app/pipeline.py` | worker threads, line merging, backpressure, latency |
| `app/output.py` | terminal rendering |
| `app/file.py` | offline mode: a file through the same segmenter and models |
| `app/models.py` | `Segment` / `Result` dataclasses |
| `scripts/` | `download_models.py` (setup), `bench_asr.py`, `bench_mt.py`, `tune_speakers.py` |
| `tests/` | 89 unit tests (VAD, echo, speakers, glossary, pipeline, output …) with synthetic audio and fake models |

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

**Live latency**: see [Latency](#latency).

**Speaker ID** (`scripts/tune_speakers.py`, 3 podcast clips, 9 min, 2 people): the same voice scores 0.6–0.9 against its centroid, the other voice 0.0–0.35; thresholds 0.35–0.55 give identical labels, so the default 0.45 sits in the middle. Each clip ends with exactly 2 speakers; the same person's centroids from different clips agree at 0.89–0.99.

### Rejected alternatives
- **Whisper `task="translate"`**: drops the Armenian transcript and translates worse than a dedicated MT model.
- **SeamlessM4T v2 as ASR**: close to the default on the author's voice, but worse on the podcast, ~15× slower, and CC-BY-NC. Its direct speech → English translation is not evaluated yet.
- **Gemma 4 E2B/E4B**: mixes in Georgian, Cyrillic and Sinhala characters when transcribing Armenian.
- **MADLAD-400 3B**: possibly better MT, but ~3× the size/latency of NLLB-1.3B, which already uses most of the MT budget.
- **Local LLM for MT** (Ollama Gemma/Qwen): slow and weak on Armenian at laptop sizes.
- **pyannote diarization pipeline / diart** for speakers: built for offline or heavier streaming diarization, adds latency, and the pyannote weights are gated behind a Hugging Face token. One embedding per VAD segment is enough here, since the VAD already cuts at pauses. **Resemblyzer**: lighter but weaker embeddings.

## Latency

The brief allows at most 5 s from the end of a spoken phrase to its English line. Where the time goes, per line, on the MacBook Air M5:

| Stage | Time |
|---|---|
| silence that ends the segment | 1.0 s (minus the 0.2 s pad kept after speech) |
| ASR (FastConformer, CPU) | 0.04–0.1 s |
| speaker embedding (system audio) | 0.05–0.08 s |
| translation (NLLB-1.3B, 6 threads) | 0.5–1.2 s per segment; NLLB-600M about half |

Measured on real calls (`--debug`, latency from the segment end, i.e. 0.2 s after the last speech frame; the app now measures from the last speech frame, so add ~0.2 s):

| Session | p50 | p95 |
|---|---|---|
| Calls 2–3 (2 people on the call side, 4 min) | 1.4 s | 2.1 s |
| Call 6 (Zoom, 2 people on the call side, 2 min) | 2.2 s | 6.2 s |
| **Call 6 replayed after Phase 5** (`tools/replay_live.py` on the Mac, from the end of the phrase, Zoom not running) | **2.05 s** | **3.23 s** |

The call 6 tail came from ~30 s in which every stage ran 2–4× slower (CPU contention with Zoom; the thread count was ruled out with `tools/bench_mt_threads.py`). That is what the backpressure switch to NLLB-600M is for; it was added after that call (the replay above is the current code, measured as the brief defines latency), is covered by unit tests and a replay, and has not yet been exercised in a real overload. Earlier replays of the podcast (`tools/replay_live.py`, then with 0.6 s of silence) gave p50 1.3 s / p95 2.1 s, against p95 10.7 s with the previous Whisper ASR, whose repetition loops caused the tail.

A long monologue is the other cost: a line is printed when the speaker pauses, so the first English of a 12 s sentence appears ~13 s after it started (still within 5 s of the end of that phrase).

## Known limitations
- **Spontaneous, colloquial speech is much harder than read speech.** The ASR models were trained on read Common Voice / FLEURS sentences; Yerevan colloquial forms (`ա` for `է`, `էդ`, `էրկու`) and English tech loanwords outside `terms.py` are often misspelled, and NLLB then translates the misspelling ("Dune scenario" → "snow scenario"). Most podcast lines keep their meaning, but CER against YouTube's captions is still ~17%.
- **ASR errors cascade into MT.** One wrong letter can change the meaning (`ՄԱԿ-ի` "UN's" → `մակի` → "mouse expert"); the user gets no confidence signal.
- **Numbers the ASR mishears** can't be fixed by the digit conversion (`հիսուներկու` 52 heard as `մինիսուն երկու` → "two and a half"); a lone number with an ending is ambiguous (`Տասներկուսին` "at twelve" → "The 12th.").
- **Proper names and English terms** outside the built-in lists are still translated as words; English words in Armenian script are often mistranslated (`բեթրիփեք` "battery pack" → "tripectin bacteria").
- **Western Armenian is not supported.** All benchmarked data and the chosen ASR are Eastern Armenian; Western Armenian pronunciation, vocabulary and (classical) spelling will raise the error rate a lot. `CAST-ML/whisper-large-v3-mwa-hy` (Western Armenian) was found but not tested.
- **Echo.** The microphone is not echo-cancelled. A mic segment that is just the call leaking in is dropped (`echo.py`); when you talk while the other side is heard in your mic, their words can still get mixed into your `Me` line.
- **Overlapping speech.** When both sides talk at once, the `Me` line can contain fragments of the other side; two people talking at once on the call side get one label.
- **Speaker ID works per segment.** A segment in which the speaker changes (fast turn-taking without a pause) gets one label; short answers inside the asker's segment go to the asker. Segments under 1 s keep the previous label; a new speaker is created only from ≥ 3 s, so a new person's first short replies go to someone already known.
- **Speaker numbers are per session** and depend on who speaks first; no enrolment of known voices (out of scope). Tuned on one podcast (2 male voices) and checked on 6 short calls with up to 2 people on the call side, by ear: no labelled reference, no DER.
- **Fixed audio routing.** The Multi-Output Device names specific headphones and doesn't follow output changes; when a headset is unplugged, the stream isn't reopened (the app warns; restart it).
- **The microphone is always captured**: there is no system-audio-only mode.
- **Lines print in the order segments end**, so a long remote line can appear after a later short `Me` line. Mic and BlackHole are separate clocks; fine for a call-length session.
- **Memory**: with the fallback both NLLB models are loaded (~0.6 GB more); peak RAM on an 8 GB Mac is not measured (the app was tested on 16 GB).
- **Hugging Face revisions are not pinned** in `scripts/download_models.py`; if an upstream repository changes, the download can differ from the tested models.
- **Licenses.** NVIDIA FastConformer: CC-BY-4.0 (attribution to NVIDIA). NLLB-200: CC-BY-NC-4.0, non-commercial only. SpeechBrain ECAPA: Apache-2.0. Silero VAD: MIT.

## What I would improve with more time
- **System audio without BlackHole**: Core Audio process taps (macOS 14.4+) through a small Swift helper; no Multi-Output Device, any output works and follows device changes.
- **ASR fine-tuning on conversational Armenian**: `--save-audio` already collects segments + transcripts from real sessions; correct them by hand and fine-tune FastConformer on colloquial speech, loanwords and numbers.
- **Turn-taking**: detect a speaker change inside a segment from sliding 1.5 s embeddings and split before ASR; a pending state for a new voice instead of the hard 3 s rule. Needs a speaker-labelled test set (DER) first.
- **Translation with dialogue context** (the question before a short answer), a larger MT model if a faster runtime is available (CTranslate2 has no Metal backend on the Mac).
- **Reliability**: reopen a device that disappears, a `--no-mic` mode, pinned model revisions, an opt-in end-to-end test with the real models.

## Tests

```bash
source .venv/bin/activate && pip install pytest
python -m pytest              # 89 tests, synthetic audio and fake models, no model download needed
```
