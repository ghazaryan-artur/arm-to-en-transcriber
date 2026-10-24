# Live Armenian → English Transcriber

Listens to the microphone and system audio, transcribes spoken Armenian, translates it to English and prints it live in the terminal, labelled by speaker. All models run locally.

**Local-only rule:** transcription, translation and speaker identification run entirely on this machine. No cloud speech or translation APIs are used. Model weights are downloaded once during setup; at runtime the app runs in offline mode (`HF_HUB_OFFLINE=1`) and works with Wi-Fi turned off.

> Work in progress.

## Setup
_TODO_

## Usage
```bash
python scripts/download_models.py               # once: default ASR + MT into models/ (~5 GB download)
python -m armtranscribe.file recording.wav      # offline: transcribe + translate a file
python -m armtranscribe --list-devices
python -m armtranscribe                         # live: default microphone, lines labelled "Me"
python -m armtranscribe --mic <id|name> --debug  # pick the mic; --debug prints per-stage timings
python -m armtranscribe --save-audio DIR         # also save the session: full stream, segments, segments.tsv (file, hy, en)
```
`--asr` / `--mt` select another model from `models/` (`scripts/download_models.py --list`).

## Models

| Stage | Default | Runtime |
|---|---|---|
| ASR | [`nvidia/stt_hy_fastconformer_hybrid_large_pc`](https://huggingface.co/nvidia/stt_hy_fastconformer_hybrid_large_pc) (FastConformer RNNT, 115M params, writes punctuation and capitalisation), ONNX int8 export by [`Vana-Labs/stt-fastconformer-armenian`](https://huggingface.co/Vana-Labs/stt-fastconformer-armenian) | `onnx-asr`, CPU |
| Translation | [`facebook/nllb-200-distilled-1.3B`](https://huggingface.co/facebook/nllb-200-distilled-1.3B), CTranslate2 int8, `hye_Armn → eng_Latn`, beam 2 | `ctranslate2`, CPU |

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

### Rejected alternatives
- **Whisper `task="translate"`**: drops the Armenian transcript and translates worse than a dedicated MT model.
- **SeamlessM4T v2 as ASR**: close to the default on the author's voice, but worse on the podcast, ~15× slower, and CC-BY-NC. Its direct speech → English translation is not evaluated yet.
- **Gemma 4 E2B/E4B**: mixes in Georgian, Cyrillic and Sinhala characters when transcribing Armenian.
- **MADLAD-400 3B**: possibly better MT, but ~3× the size/latency of NLLB-1.3B, which already uses most of the MT budget.
- **Local LLM for MT** (Ollama Gemma/Qwen): slow and weak on Armenian at laptop sizes.

## Architecture
_TODO_

## Known limitations
- **Spontaneous, colloquial speech is much harder than read speech.** The ASR models were trained on read Common Voice / FLEURS sentences; Yerevan colloquial forms (`ա` for `է`, `էդ`, `էրկու`) and English tech loanwords (`ինթելիջենս`, `ռոբոտիքս`) are often misspelled, and NLLB then translates the misspelling ("Dune scenario" → "snow scenario"). With the FastConformer most podcast lines now keep their meaning, but CER against YouTube's captions is still ~18%.
- **Proper names and English terms.** NLLB may still translate a name that is also a common word, and English loanwords written in Armenian script are often mistranslated (`բեթրիփեք` "battery pack" → "tripectin bacteria").
- **ASR errors cascade into MT.** One wrong letter can change the meaning (`ՄԱԿ-ի` "UN's" → `մակի` → "mouse expert").
- - **Licenses.** NVIDIA FastConformer: CC-BY-4.0 (attribution to NVIDIA). NLLB-200: CC-BY-NC-4.0 (non-commercial).
- _TODO: echo without headphones, overlapping speech, speaker ID on short segments, Western Armenian._
