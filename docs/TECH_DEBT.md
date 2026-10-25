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

## Phase 3 — system audio

### Weaknesses

- **Echo into the mic → every remote line twice (`System` + `Me`).** → **Fixed** by `echo.py` (see "Call 5"). Original finding: Live test, podcast through the Multi-Output Device, user silent. With MacBook speakers in the Multi-Output: echo 53 ms behind, normalised xcorr 0.31. With wired headphones at ~100 % volume: still picked up by the headset mic at −34 dBFS vs the user's voice at −30 dBFS, transcribed almost word for word as `Me`. Out of scope per the brief, but confusing in the interview.
  → First: headphones at a normal call volume (README setup step 5), re-check with 30 s of podcast and a silent user.
  → Phase 5 option: drop a `Me` line when a `System` segment overlapping it in time (±1.5 s) has near-identical text (char similarity > ~0.6), optionally also when the mic segment's voice doesn't match the user's ECAPA profile (Phase 4 embedder). Hold `Me` lines until the system stream has been transcribed up to the same time (≤ ~1 s extra latency; p95 is 2.65 s now). Never drop on voice alone: with overlapping speech the embedding is mixed and real user speech would be lost; the brief says everything from the mic is `Me`.
- **Overlapping speech.** User reading over the podcast (headphones): the user's lines are recognised but with pieces of the podcast mixed in (`Վերջեր, ինչ ասեմ, այսօր…`). Two people talking at once on one mic is not separable here.
- **Force-cut splits words.** A 9 s FLEURS sentence through BlackHole was cut inside `հիմնականում` → `հիմնական` | `կանում`. Same root cause as the Phase 2 force-cut item (cut at the lowest-energy frame, not at a pause).
- **Two streams, two clocks.** Each Segmenter counts its own samples since its stream started; BlackHole and the mic are separate CoreAudio clocks. Lines from the two sources at the same moment can be printed in either order (seen: `System` 00:01:09 before `Me` 00:01:08). Fine for a 10–20 min session; drift would matter only for hour-long ones.
- **Silent BlackHole is not reported.** If the Multi-Output Device isn't the output (it was not, at the first test), the system stream is digital silence and the app just shows no `System` lines.
  → Done (Phase 4): the pipeline warns once after 15 s of digital silence on the system stream. Happened again in the first Phase 4 live test: output was the headphones, not the Multi-Output Device.
- **Routing is a fixed Multi-Output Device and doesn't follow the output.** Found in the Phase 4 live test: with the headphones switched off, the Multi-Output (headphones + BlackHole) sent sound only to BlackHole — the app worked, the user heard nothing; without headphones the user needs a different Multi-Output (speakers + BlackHole). Plugging headphones in also makes macOS switch the output to them directly, bypassing the Multi-Output → BlackHole goes silent (now at least warned about after 15 s). Won't fix in this project.
  → Workaround: two Multi-Output Devices (headphones + BlackHole, speakers + BlackHole), pick one in the menu-bar sound menu before the call.
  → Proper fix: capture system audio with Core Audio process taps (macOS 14.4+) via a small Swift helper piping PCM to the app — no BlackHole / Multi-Output, any output device works, switching is automatic. Needs the "System Audio Recording" permission; ~3–4 h incl. debugging on the Mac.
- **No system-audio-only mode.** The mic is always captured. Listening to a call on speakers with the mic muted in Meet, the app still records the mic, hears the other side from the speakers and prints it a second time as `Me`. Won't fix now.
  → `--no-mic` flag (~10 min), or the duplicate filter above.

---

## Phase 4 — speaker ID on system audio

Tuned on 3 clips of the Rearrange #326 podcast (`recordings/podcast/excerpt_10-15min.wav`, `yt_set/yt_11m11s.wav`, `yt_set/yt_45m10s.wav`; 9 min, 2 people) with `scripts/tune_speakers.py`. Same voice vs its centroid 0.6–0.9, other voice 0.0–0.35; thresholds 0.35–0.55 give identical labels, 0.65 splits one person into up to 6. Defaults: threshold 0.45, new speaker only from a segment ≥ 3 s. Result: exactly 2 speakers in each clip. Cross-clip check: centroids of the same person from different clips agree at 0.89–0.99.

### Weaknesses

- **Small tuning set, no ground truth.** One podcast, two male voices, labels checked by ear only (no speaker-annotated reference, no DER).
  → Label 5–10 min by hand (start/end/speaker), compute DER; add a recording with 3 people and a female voice.
- **Similar voices are hard to tell apart by ear, the model separates them.** In `yt_45m10s` 01:19–01:42 the user first heard the same person as before; the model gave a new speaker (similarity 0.01–0.27 to the first voice, 0.89 to the second podcast participant from the other clips). Checked by ear against `excerpt_10-15min` 00:00–00:23 → the model was right. Takeaway: the threshold isn't the weak spot here; checking the labels is.
- **Phantom speakers from short fragments.** A 2.8 s laugh / cross-talk fragment (`yt_45m10s` 00:55) matched nobody and became `Speaker 2`, so the real second person got number 3. Fixed by `min_new_s=3.0`. Cost: a new person's first replies < 3 s are attributed to the nearest known speaker.
  → A pending-speaker state: open a candidate on a short unknown segment and confirm it only when a second segment matches; label it with the nearest speaker until then.
- **One label per segment.** A segment with a speaker change scores ~0.6 against both centroids (`excerpt` 03:13: 0.62 / 0.59; `yt_11m11s` 02:02: 0.57 / 0.65) and goes entirely to one of them.
  → Embed 1.5 s sub-windows, detect the change point, split the segment before ASR; or a shorter `max_segment_s` for the system stream.
- **First live test was invalid.** Two men in a video through BlackHole: everything went to `Speaker 1` (0.65–0.68) and the ASR text was gibberish — broken routing (headphones off inside the Multi-Output). On Mac `spk` = 0.05–0.08 s per segment, negligible.
- **One-word answers in fast turn-taking go to the asker.** Podcast 53:00–53:20 (host asks, guest answers `Չորս, չորս` and `բավական`), analysed offline (`recordings/podcast/cut_52m50s.wav`, Silero frames + 1 s ECAPA windows). The answers are not lost — they sit inside the host's segments:
  - from 53:01.9 to 53:24 there is no pause ≥ 600 ms at all; the longest is 448 ms (just before `Չորս`), the guest answers `բավական` with < 150 ms gap. Pauses inside the host's own sentences are 200–400 ms, so no `min_silence_ms` separates the turns without chopping every sentence.
  - the stream is therefore force-cut every 8 s; one cut (53:17.0) fell inside `բավական`: the first half stayed in the host's segment, the second started the next one, where ASR heard `Լավ`.
  - 1 s windows do see the guest for ~0.5–0.75 s (guest 0.37–0.42 vs host 0.07–0.21), but the margin is small and a one-window flip also happens on the host's own pauses.
  → Possible: sliding-window change detection (cut when ≥ 3 consecutive windows prefer another known speaker by > 0.15) before ASR; fragile on 0.5 s words, needs a labelled test set first. Force-cut at a pause instead of the quietest frame (Phase 2 item) would at least keep words whole. A real fix is a diarization model trained on turn-taking/overlap. Not doing it in this project.
- **Live speaker ID confirmed (Phase 4).** Second live run with working routing (podcast, host + guest): guest `Speaker 1` at 0.72–0.84, host `Speaker 2` at 0.76–0.82, cross 0.02–0.30; checked against the YouTube captions, the disputed line at 53:30 (`Կարան տարբեր ձեւը ընկալեն…`) is the host, as labelled; the earlier phantom `Speaker 3` came from rewinding the video (scrubbing fragments of both voices in one segment). `--debug` now also prints segments dropped for empty ASR output.
- **Not tested on call audio.** ECAPA is trained on VoxCeleb (interview audio); Zoom / Meet / Telegram compress and denoise the voice, which can lower same-speaker similarity.
  → The rehearsal call (Phase 6): 10+ min with 2 people on the other side, check label stability; re-tune with `tune_speakers.py` on a `--save-audio` recording of it.
- **Embedding latency on the Mac not measured.** ~0.5 s per segment on the cloud CPU used for tuning; runs in the ASR thread, so it adds to every system line.
  → Check `spk` in `--debug` on the MacBook; if > 0.2 s, embed only the first 3 s of long segments or move it to its own thread.
- **Speaker numbers live for one session**, depend on who speaks first; no enrolment of known voices. Centroids keep growing (running mean over the whole session) — fine for a call, slow to adapt if the audio changes mid-session.
- **Short segments < 1 s** are not embedded and keep the previous label — wrong after a quick turn change ("Հա", "Լավ" from the other person).

---

## First real call (2026-10-25, before Phase 5)

`recordings/calls/20261025-145539`: real call, one person on the other side, headphones, 65 s. Latency 0.7–2.2 s per line, no echo duplicates, the remote voice stayed `Speaker 1` (0.50–0.85). Main problem: **meaning was lost at segment cuts**, not in ASR: `փաստաթ` | `Տերը, որ ձեզ ուղարկենք` → "The Lord sends you" (word split by the 8 s force-cut at the quietest frame), `ես էսօր բարելավել եմ` | `Ձայնային ճանաչման…` → "I got better today" (sentence split), `համակարգի օպտիմալացման` split at a ~0.7 s hesitation → `Դիմման եւ` → "Apply and".

**Fixed** (segmenter defaults): `min_silence_ms` 600 → 1000, `max_segment_s` 8 → 12, force-cut in the longest pause (frames below `neg_threshold`) of the last 3 s, else after the least speech-like frame (was: quietest frame of the last 1 s). On the call all three errors above become whole sentences ("…and we're now preparing the documents to send to you", "I improved the speech recognition function today"). Podcast `yt_set` (22 min, YouTube-caption refs, FastConformer): WER/CER 37.4 / 17.8 % → 35.2 / 16.7 %; segments 235 → 151, mean 5.3 → 8.4 s, segments at the cap 105 → 41. Whole-clip ASR (no cuts) is 34.0 / 16.5 %, so the cuts now cost little ASR accuracy; the bigger gain is MT getting whole sentences (not measured: no English reference for the podcast).
Cost: a long monologue now shows its first line up to ~13 s after it started (was ~9 s), and every line ends 0.4 s later (longer silence wait). Segment-end → print latency is unchanged apart from MT on longer input.

### Still open after the call
- **Hesitation pauses ≥ 1 s still split sentences** (`համակարգի օպտիմալ եւ` | `…ենթակառուցվածքը…`). → revise-the-last-line idea in Phase 2 above, or MT with the previous segment as context.
- **Lone numbers / short answers**: `Քսաներեք։` → "It's twenty" (`Քսաներեք տարեկան եմ` → "I am twenty-three years old" is right). → MT with the previous line (the question) as context, or a numbers/short-reply table.
- Leading `֊ ` dash in ASR output (FastConformer too, not only Whisper) → strip it.
- Lines print in segment-end order, so a long remote line can appear after a later short `Me` line (`00:00:07 Me` before `00:00:04 Speaker 1`).

## Calls 2 and 3 (2026-10-25)

`recordings/calls/20261025-153131` (2.5 min, two people on the other side, each speaking in turn) and `…-153533` (1.5 min, frequent interruptions). Latency p50 1.4 s / p95 2.1 s; no echo (mic at −55…−60 dBFS while the call plays, xcorr < 0.25); speakers stable even with interruptions (Gor = `Speaker 1`, Aghas = `Speaker 2` throughout call 3). Call 2 had one phantom `Speaker 2` (a 3.5 s line of `Speaker 1`, similarity 0.39 — below the 0.45 threshold).

**Fixed:**
- **Fragments at thinking pauses (> 1 s)** were translated alone ("And we hope that our product." / "The world will know and we will get positive."). A segment that follows the same speaker within 1.5 s (between segments, ≈ 2 s of silence) now extends the previous line, max 3 segments / 30 s → "…and we hope that our product will be recognized and received positively in the world". Plain joining made NLLB **drop the second sentence** in 4 of 8 two-segment joins (`Պատասխանեմ … սեւ հաց էր։ Դու ինչեր հարցնում…` → only the first), so the join is translated with the English already shown forced as the prefix, and the new part is translated alone if the continuation comes out empty. Re-run on calls 2–3 (NLLB in the VM): nothing dropped.
- **NLLB repetition loop on a joined line** ("we're going to put up, …" × 28): `no_repeat_ngram_size=4`, output ≤ 1.5 × source + 10 tokens. Live test part 1: identical output and chrF++ (72.2) with and without the guard.
- **Names**: `աղաս` → "you little prick" / "pray", `գոռի` → "the old man's". Capitalising alone gives "Alas", "Gore"; Latin with the case ending kept (`Gor-ի`, `Aghas-ին`) gives "Gor's question", "Aghas". ~330 built-in names; names that are usually words (`Արա`, `Սիրուն`, `Արև` …) only before `ջան` (Artur: that's how they're used), plus any capitalised non-kin word before `ջան`; `Անուշ` except in `անուշ եղի / լինի`; `Արա,` → "Hey,"; `ռ` = `ր` in names (ASR wrote `Գոր` for `Գոռ` → "Gore"). Call 4 (`20261025-170639`) was invalid — BlackHole silent, the remote side came in through the mic as `Me`. A `--names` flag and a NER model were considered and rejected (Artur). Names outside the list without `ջան` are still translated as words. The built-in list has both `Աղաս` and `Աղասի`, so `Աղասին` is read as Aghasi + ն.
- **One-word replies** from a table (`Ապրես` "You live" → "Thanks, well done!", `Այո` "Yes, you did" → "Yes."), also as a sentence inside a segment (`Ապրես։ Աղաս դու ասա։` lost the `Ապրես`). `drachmas` → `drams`. Leading `֊ ` dash dropped.

### Still open
- **Numbers**: ASR `հիսուներկու` → `մինիսուն երկու` → "two and a half"; `Տասներկուսին` ("at twelve") → "It's twelve".
- **Colloquial sentences still mistranslated** now and then (`Կտրենք խոսքդ…` → "Let's cut to the chase" and the list of breads dropped; `ինչ ա արել` → "…to know").
- **Overlap**: when both sides speak at once, short `Me` fragments get odd text (`Ես եմ հարով լավ`).
- **Line rewrite in the terminal** assumes nothing else wrote to the terminal and the window wasn't resized since the line was printed; otherwise the old block may be partly left on screen (cosmetic).
- `segments.tsv` from `--save-audio`: for a segment that continued a line, `en` is the translation of the whole line.

## Call 5 (2026-10-25, `recordings/calls/20261025-171728`)

New mic (`External Microphone`, a wired headset) leaked the call into the mic: remote speech at −48 dBFS in the mic vs the user's −28, < 5 ms behind the system stream. Every remote line was printed twice (`Speaker 1` and `Me`), and the `Me` copies broke the line merging (lines interleaved, `↻` everywhere).
**Fixed**: `echo.py` — a mic segment whose normalised cross-correlation with the system stream (best lag within ±0.3 s, each window normalised by its own energy) is ≥ 0.3 is dropped before ASR. On all five calls: leaked segments 0.37–0.59 (call 5 ×6, and `Ես եմ հարով լավ` in call 3, which was also echo), own speech 0.01–0.16, including talking over the other side. A segment where the user speaks *and* the leak is heard still passes and can mix both.

**Next run (Zoom, podcast played into the call):** a 3-segment line (26.7 s of audio) took MT 4.6 s → latency 6.7 s, over the 5 s target: the continuation re-translated the whole line so far. Now only the segment before is the context (its English forced as prefix), so MT time no longer grows with the line; the 3-segment line from call 2 still reads as one sentence ("…and we hope that our product will be recognized by many in the world and we will get a positive response").

## Call 6 (2026-10-25, Zoom, Artur + Gor + Aghas, 2 min)

Echo filter dropped 2 leaked mic segments (xcorr 0.45–0.47); Gor and Aghas kept their labels through the call (0.63–0.82); p50 2.2 s.
- **Phantom first speaker**: a 1.2 s `Որ` on the system stream founded `Speaker 1` (the first speaker skipped the ≥ 3 s rule), so Gor became 2 and Aghas 3. **Fixed**: the first speaker needs ≥ 3 s too, shorter segments before it are `Speaker ?`.
- **Single-letter ASR output** (`Հ.` → "C. What is it?"): **fixed**, outputs with < 2 letters are dropped.
- **MT slowed 3–4× for ~30 s** (MT 4.2 / 5.9 / 4.0 s, also `Դրական։` 1.6 s; ASR and speaker ID 2–3× slower at the same time) → p95 6.2 s, latency up to 9.2 s. CPU contention with Zoom is the likely cause; NLLB runs with 8 threads on a 4P+4E MacBook Air. → measure with `tools/bench_mt_threads.py` while a Zoom call is open, pick the fastest thread count.
- **English tech words in Armenian script**: `չատ ջիպիթի`, `Ջեմինի`, `վեյբ կոդինգ`, `գրոկ`, `սկիլերը` → "chat jeep", "WebCat", "the grok", … (also `Սիրիս էյ` Series A → "My dear fellow" in the previous run). → a term list like the names (Armenian spelling → English, case endings kept), if wanted.
- Overlap: a `Me` segment with Artur talking while Gor's voice leaked in passed the echo filter (mixed text) — expected limit.

---

## Resolved
- **NLLB repetition loops** (Phase 2) → `no_repeat_ngram_size=4` + length cap (see "Calls 2 and 3").
- **Force-cut splits words / sentences** (Phase 2, Phase 3 items) → pause-based cut, 1 s silence, 12 s cap (see "First real call").
