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
python -m armtranscribe --mic <id> --system <id>   # live (in progress)
```
`--asr` / `--mt` select another model from `models/` (`scripts/download_models.py --list`).

## Models

| Stage | Default | Runtime |
|---|---|---|
| ASR | [`Chillarmo/whisper-large-v3-turbo-armenian`](https://huggingface.co/Chillarmo/whisper-large-v3-turbo-armenian), converted to MLX fp16 | `mlx-whisper`, Apple GPU |
| Translation | [`facebook/nllb-200-distilled-1.3B`](https://huggingface.co/facebook/nllb-200-distilled-1.3B), CTranslate2 int8, `hye_Armn → eng_Latn`, beam 2 | `ctranslate2`, CPU |

### Benchmark

Test set: 20 clips from the FLEURS `hy_am` **test** split (Eastern Armenian read speech), 4.9–10 s each (163 s total), 20 distinct sentences, 10 male / 10 female speakers. FLEURS is n-way parallel, so the matching `en_us` sentences are the English references. Measured on a MacBook Air M5, 16 GB, after one warm-up call per model. Before WER/CER, text is lowercased and stripped of punctuation, and the word "and" is unified: formal `և` and colloquial `ու` (the more common one in speech) count as the same word.

**ASR** (`scripts/bench_asr.py`):

| Model | WER | CER | s / clip (mean) | s / clip (p90) | Punctuation |
|---|---|---|---|---|---|
| **Whisper large-v3-turbo, Armenian fine-tune** (Chillarmo) | **13.3%** | **3.8%** | 0.79 | 0.94 | yes |
| Meta MMS-1B-all, `hye` adapter (CTC) | 21.3% | 4.5% | 0.37 | 0.43 | no |
| Whisper large-v3 | 40.2% | 9.6% | 2.73 | 3.43 | yes |
| Whisper large-v3-turbo | 45.1% | 10.4% | 0.78 | 0.92 | yes |

The fine-tune was trained on Common Voice 20 Armenian, not on FLEURS, so this is out-of-domain for it. Base Whisper misspells Armenian heavily (WER ~40%) and large-v3 misses the ≤ 1.5 s latency budget for an ~8 s segment. MMS is fast with a good CER but outputs no punctuation or casing, which costs translation quality (below).

**Translation** (`scripts/bench_mt.py`; chrF++ / BLEU against the English references):

| Model | chrF++ (gold Armenian) | BLEU | chrF++ (after fine-tuned ASR) | s / sentence |
|---|---|---|---|---|
| **NLLB-200 distilled 1.3B** | **66.0** | **45.6** | **62.0** | 1.11 |
| NLLB-200 distilled 600M | 59.6 | 39.0 | 56.8 | 0.55 |
| Helsinki-NLP opus-mt-hy-en | 44.6 | 22.4 | – | 0.10 |

Full cascade chrF++ with NLLB-1.3B by ASR model: fine-tuned Whisper 62.0, MMS 53.9, Whisper large-v3 45.1. NLLB beam size 1/2/4 gives chrF++ 65.0/66.0/66.0 (1.3B), so beam 2 is used. Estimated latency per ~8 s segment: 0.6 s end-of-speech silence + 0.8 s ASR + 1.1 s MT ≈ 2.5 s (target p95 < 5 s).

**Own voice** (7 conversational sentences read by the author, 48 s total): fine-tuned Whisper CER 3.0% / WER 18.1% (0.63 s per clip), MMS CER 5.3% / WER 22.9%, Whisper large-v3 CER 9.8% / WER 32.5%, large-v3-turbo CER 35.5% / WER 63.9%. Cascade with NLLB-1.3B: chrF++ 70.9.

**Spontaneous speech** (5 min of a two-person Armenian tech podcast, no reference): 58 segments, mean 4.9 s; ASR 0.64 s + MT 0.65 s per segment on average, worst case 3.0 s. Quality drops clearly compared with read speech — see Known limitations.

### Rejected alternatives
- **Whisper `task="translate"`**: drops the Armenian transcript and translates worse than a dedicated MT model.
- **SeamlessM4T v2**: large, non-commercial license, slower on a laptop.
- **MADLAD-400 3B**: possibly better MT, but ~3× the size/latency of NLLB-1.3B, which already uses most of the MT budget.
- **Local LLM for MT** (Ollama Gemma/Qwen): slow and weak on Armenian at laptop sizes.

## Architecture
_TODO_

## Known limitations
- **Spontaneous, colloquial speech is much harder than read speech.** The ASR fine-tune was trained on read Common Voice sentences; Yerevan colloquial forms (`ա` for `է`, `էդ`, `էրկու`) and English tech loanwords (`ինթելիջենս`, `ռոբոտիքս`) are often misspelled, and NLLB then translates the misspelling ("Dune scenario" → "snow scenario"). On a podcast roughly half the lines keep their meaning.
- **Proper names.** The ASR writes names in lowercase, so names that are also common words get translated literally: `Լուսինե` (a name) → `լուսին` "moon".
- **ASR errors cascade into MT.** One wrong letter can change the meaning (`ՄԱԿ-ի` "UN's" → `մակի` → "mouse expert").
- _TODO: echo without headphones, overlapping speech, speaker ID on short segments, Western Armenian, NLLB non-commercial license._
