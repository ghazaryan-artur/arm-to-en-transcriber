# Tech debt & open issues

Running list of what was left aside and known weaknesses. Each item: what, why it matters, idea for a fix.
Add new items under the phase where they appeared; move fixed ones to "Resolved" with the commit.

---

## Phase 1 — offline skeleton + model bake-off

### Left aside

- **Podcast has no reference transcript.** Spontaneous-speech quality ("about half the lines keep their meaning") is judged by eye on 5 min only — no WER/CER, no chrF.
  → Hand-transcribe 1–2 min of `recordings/podcast/excerpt_10-15min.wav` (and the English for it) and add to a manifest.
- **Common Voice `hy-AM` not used.** Test set is FLEURS (read news sentences) + 7 own sentences only.
  → If access to CV is easy, add 20 clips; CV is also what the chosen fine-tune was trained on, so use the **test** split only and treat it as in-domain.
- **ASR candidates not benchmarked:** `arampacha/whisper-large-hy-2` (large-v2 FT on CV11 + FLEURS train), `CAST-ML/whisper-large-v3-mwa-hy` (large-v3 FT, 2026), `ditord/whisper-large-v3-turbo-armenian`, wav2vec2 XLS-R Armenian models, Omnilingual ASR, SeamlessM4T v2.
  → `CAST-ML` is the most interesting (full large-v3, recent) — but large-v3 was 2.7 s/clip, over budget. Worth one run if conversational quality stays the main problem.
- **MT candidates not benchmarked:** MADLAD-400 3B, local LLM (Ollama). Rejected on latency without numbers.
- **MMS is unusable as-is**: no punctuation/casing → MT quality drops (cascade chrF++ 53.9 vs 62.0). Would need punctuation restoration to be a fast fallback.
- **Converted fine-tune has no `alignment_heads`** → Whisper word timestamps unavailable. Not needed now; needed if we ever split segments by words.

### Weaknesses

- **Test set is small** (20 + 7 sentences). Differences of a few chrF/WER points are noise: on own voice NLLB-600M beat 1.3B (75.8 vs 70.6), opposite of FLEURS.
  → Grow to 50+ sentences, ideally conversational; only then fine-tune choices like 600M vs 1.3B or beam size.
- **Spontaneous / colloquial speech is weak.** ASR fine-tune is trained on read speech; Yerevan colloquial forms (`ա` for `է`, `էդ`, `էրկու`, `սաղ`) and English tech loanwords (`ինթելիջենս`, `ռոբոտիքս`, `ակտուատոր`) get misspelled, MT then translates the misspelling ("Dune scenario" → "snow scenario").
  → Cheap: Whisper `initial_prompt` with a colloquial sample; MT with the previous segment as context (Phase 5 optional). Expensive: fine-tune on conversational Armenian.
- **Proper names come out lowercase** → names that are common words are translated literally (`Լուսինե` → "moon", `me_04`).
  → Try `initial_prompt` with capitalised names; or keep base Whisper casing for names (it capitalises, but misspells more).
- **ASR errors cascade into MT** (`ՄԱԿ-ի` "UN's" → `մակի` → "mouse expert"). Inherent to the cascade; no confidence signal is passed to the user.
  → Optionally mark low-confidence lines (Whisper `avg_logprob`) in the output.
- **Hallucination filter is a guess.** `asr.HALLUCINATIONS` lists typical phrases, not verified on real silence / music / YouTube outros with *this* fine-tune; `no_speech_threshold=0.6` is untuned. Only 3 s of synthetic noise was tested.
  → Record silence, room noise, music, keyboard; collect what the model emits; tune threshold.
- **Cosmetic ASR output**: the fine-tune sometimes adds a leading `- ` (dialogue dash) and trailing `…`; NLLB copies them.
  → Strip leading dashes in `MLXWhisper.transcribe`.
- **File-mode segmentation ≠ live segmentation.** `file.py` uses Silero `get_speech_timestamps(max_speech_duration_s=8)`, which drops a remainder < 0.4 s after a forced cut; the live VAD (Phase 2) will cut at the lowest-energy frame. Benchmarks of file mode won't exactly match live behaviour.
  → Once `vad.py` exists, make file mode feed the file through the same streaming segmenter.
- **First call is slow** (ASR 1.6 s vs 0.8 s steady state). → Warm-up at startup (Phase 5).
- **MT latency is 3–4× the plan's estimate** (NLLB-1.3B ≈ 1.1 s/sentence, 600M ≈ 0.55 s; plan assumed 0.1–0.3 s). CTranslate2 has no Metal backend on Mac → CPU only. Not yet measured while ASR runs concurrently in the live pipeline.
  → Measure in Phase 2/5; keep `--mt nllb-600m-ct2` as the fallback if p95 slips.
- **Memory**: loading many CTranslate2 models in one process got OOM-killed (beam/threads sweep). `bench_mt.py` loads 3 models sequentially in one process — works now, but fragile. Live RAM footprint (ASR 1.6 GB + MT 1.3 GB + ECAPA + torch) not measured.
  → Run each model in a subprocess in `bench_mt.py`, or `del` + `gc.collect()` between models; measure RSS in `--debug`.
- **`own/me_02` reference unverified**: both ASR models hear `ուզում են`, reference says `ուզում եմ` — may be pronunciation, may be reference error.

### Reproducibility

- **Dependencies are unpinned** (`requirements.txt` has no versions; we run transformers 5.x, ctranslate2 4.8, mlx-whisper 0.4.3). A fresh install may break the CT2 converter or MLX loading.
  → Pin versions (`pip freeze` subset) before the final README check.
- **HF model revisions are unpinned** in `download_models.py`; upstream repos can change. Current: Chillarmo `44df5fa4c1`, NLLB-1.3B `7be3e24664`, MMS `3d33597edb`, mlx turbo `a4aaeec063`.
  → Pass `revision=` to `snapshot_download`.
- **Source checkpoints stay in the HF cache** (`~/.cache/huggingface`, several GB after `--bench`); nothing cleans them up.
  → Mention in README; optional `--clean-cache`.
- **Benchmark data is not in the repo.** The prep scripts are (`tools/fetch_fleurs.py`, `make_podcast_set.py`, `build_live_test.py`), but the audio isn't: FLEURS can be re-fetched, the podcast needs its mp3 + YouTube transcript, the own recordings can't be shared. A fresh clone can reproduce the FLEURS numbers only.

### Licensing

- **NLLB-200 and MMS are CC-BY-NC-4.0** (non-commercial). Chillarmo fine-tune: Apache-2.0, trained on CC0 data. Fine for this project; must be stated in README limitations.

### Tests

- HF → MLX key mapping in `download_models.py` has no unit test (verified once by hand: 586/586 tensors match `mlx-community/whisper-large-v3-turbo` in names and shapes).
- No integration test of `app.file` (needs models; could be an opt-in test marked `slow`).

---

## Phase 2 — live mic end-to-end

### Weaknesses

- **ASR switched to NVIDIA FastConformer (bake-off of 13 models, numbers in README).** This removes the Whisper repetition loops and most hallucinations (insertions 0–4%), cuts own-voice CER 9.7% → 2.5% and podcast CER 31.8% → 17.9%, and live latency p95 10.7 s → 2.1 s. Still open: NLLB can loop on its own (`no_repeat_ngram_size` / length cap not added yet); English terms in Armenian script are still mistranslated (glossary); the ONNX export is a third-party one (Vana-Labs, checked against NeMo by its author) and the HF revision isn't pinned; bake-off harness lives in `tools/asr_bakeoff/`, not in the app (Gemma / Seamless / HF Whisper backends were not added to `asr.py`).

- **Repetition loops blow the latency budget.** 2 min of the podcast replayed in real time through the live pipeline: p50 2.1 s, p95 10.7 s. All of the tail is 2 of 20 segments where a model looped: NLLB `let's say, let's say, …` up to `max_decoding_length=256` (MT 8.5 s), Whisper `մի մի մի …` / `բանջական բանջական …`. Without these two, p95 ≈ 4 s.
  → MT: `no_repeat_ngram_size` (3–4) and `max_decoding_length` proportional to source length (≈ 2× tokens + 10). ASR: collapse a phrase repeated ≥ 3 times; or drop segments with high compression ratio (mlx-whisper computes it, we ignore it at `temperature=0`). Re-run `bench_mt.py` to check chrF doesn't drop.
- **Sentences split at hesitation pauses.** Live test, phrase 13 read with ~0.7 s pauses at commas → 4 segments, each translated alone into nonsense ("And even the good ones", "I didn't see it that way"). `min_silence_ms=600` is shorter than natural pauses.
  → First: raise to 800–1000 ms and measure on `recordings/live_test` (+0.2–0.4 s to every line; p95 was ≈ 4 s without the repetition loops, so still < 5 s for one source — re-check after Phase 3, when two sources queue for one ASR).
  → If that's not enough (pauses > 1 s, 8 s force-cuts): **revise the last line when the sentence continues** (user idea). Print the draft as now; if the next segment of the same source continues the sentence — the previous segment was force-cut, or the gap is < ~1.5 s (a trailing `…` is NOT a reliable signal: the fine-tune also puts it on complete short phrases — `Իհարկե…`, `Լավ, համաձայն…` in the live test) — re-run ASR on the merged audio, translate the whole sentence and replace the previous block(s). Cap at ~20 s / 3 segments. Terminal: redraw the last block while one source prints; with interleaved sources print `↳ corrected: …`. Keeps the fast first draft; costs one extra ASR+MT (~1–2 s) only on merges.
- **Live test set, part 1** (`recordings/live_test/part1`, 20 read colloquial lines, user's voice, MacBook mic). Live output: ASR WER 38.1%, CER 11.1%; end-to-end EN chrF++ 54.7 (BLEU 33.3). NLLB-1.3B on the *reference* Armenian: chrF++ 72.8 (600M: 69.4) → ~18 points are lost to ASR errors. Typical ASR errors are one-letter misspellings of colloquial words that flip the meaning: `էրեկ`→`երեք` ("three" for "yesterday"), `Գինը`→`Կինը` ("woman" for "price"), `ձեռք`→`ձերբ` ("arrest"), names lowercased (`նարեկը` → "narc", `արարատի` → "creator").
- **Live test set, part 2** (`recordings/live_test/part2`, 11-line read monologue, held out). Live: ASR WER 17.2%, CER 3.8% (much better than part 1: longer, more literary sentences, and the user's reading improved); end-to-end chrF++ 56.9; NLLB-1.3B on reference HY 66.5. Errors again in colloquial words and terms: `հայերենի` → `հայրենի` ("at home" for "on Armenian"), `դատասեթը` → "case file", `Մարիամի` → `մարյա մի` (name lost), `Առավոտը` → `Արոտը` ("the problem started"). Line 29 split before the unfamiliar `լարջ լանգվիջ` (hesitation pause). Earlier takes of the monologue are kept in their session dirs (extra material for fine-tuning); one of them (`20261023-223912`) has another ASR loop (`համար համար …`).
- **Podcast set with YouTube-caption references** (`recordings/podcast/yt_set`, `tools/make_podcast_set.py`: 10 × ~2 min, 22 min of native conversational speech, references = YouTube auto-captions, i.e. Google's ASR). `bench_asr.py --vad`: turbo-hy WER 61.2% / CER 31.8% (S 39.9, D 13.2, I 8.0); MMS 64.0% / **24.0%** (I only 2.6%); base turbo 111% / 66%. Absolute numbers are inflated — the reference is itself ASR and writes English terms in Latin and numbers as digits — but the ranking is informative: on native speech MMS beats our Whisper on CER, mostly because Whisper inserts text (loops / hallucinations). On the user's voice our model is clearly best (part 1 CER 9.7% vs MMS 15.0%, part 2 3.8% vs 7.3%).
- **NLLB misreads colloquial forms and pads short replies.** On correct input: `Էս` ("this") read as a name ("Ace Vegas code", "Ace Fitch"); `Հա` → "Yes, you are", `Իհարկե` → "Of course you did".
  → Colloquial → literary normalisation before MT (`էս/էդ/էն`→`այս/այդ/այն`, `ա`→`է`, `էրեկ`→`երեկ`, `ախպեր`→`եղբայր`, `սաղ`→`ամբողջ` …): reference HY 72.8 → 77.8 chrF++, live HY 55.1 → 56.3. Word list was made looking at part 1. **Held-out check (part 2 monologue): 66.5 → 65.5 on reference HY, live 56.9 → 54.3** — no line got worse in meaning (one got better: "config last night"), the drop is rephrasing noise on 11 lines. So the +5 on part 1 came from fixing `Էս`-as-a-name, which part 2 doesn't contain. Keep only the targeted `էս/էդ/էն` → `այս/այդ/այն` fix (it removes a catastrophic error type); the broad list is unproven.
  → Short replies: a small fixed table for one-word replies (`Հա`, `Չէ`, `Իհարկե`, `Լավ` …) instead of MT.
- **Short replies are dropped.** `Հա` (≈ 0.3 s of speech, Silero max prob 0.92) fell under `min_segment_s=0.4` and never reached ASR. In conversation "yes / no / ok" matter.
  → Lower to ~0.25 s, but for segments < 0.4 s require a high VAD peak (≥ 0.9) to keep noise blips out; check on saved sessions that it doesn't add hallucinations.
- **Force-cut is cruder than Silero's offline split.** Ours cuts at the lowest-energy frame in the last 1 s once 8 s is reached; `get_speech_timestamps` cuts at the last pause ≥ 98 ms. On the podcast ours gives fewer, longer segments (49 vs 58, 4 vs 1 at the 8 s cap); a cut mid-word may be what triggers some Whisper loops.
  → Prefer the last frame below `neg_threshold` within the window, fall back to lowest energy.
- **Remainder after a force-cut can be dropped** if it holds < 0.4 s of speech before the next pause (the end of a word lost). Rare; the cut window is 1 s.
- **Latency is measured on the audio clock** (`perf_counter` since stream start vs sample count). PortAudio buffering and clock drift are not accounted for; fine for p50/p95 at this scale.
- **Hallucination on short non-speech in the live room.** First `--save-audio` smoke test (nobody speaking): a 0.96 s segment passed the VAD and Whisper produced a repeating line (`այդ դինական այդ դինական …` → "It's the same thing" ×3). Same failure mode as the repetition loops above; `no_speech_prob` didn't catch it.
  → Repetition / compression-ratio filter; consider a higher speech threshold or longer min segment for short isolated blips; collect such segments from saved sessions.
- **Mic is chosen once at startup** (system default, or `--mic`). Following the call app's mic is Phase 5 (see PLAN). Live mic tested only for startup / silence / Ctrl+C here; speaking test is manual.

---

## Resolved
_(none yet)_
