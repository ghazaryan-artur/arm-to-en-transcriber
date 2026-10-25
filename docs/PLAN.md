# Live Armenian → English Transcriber — Implementation Plan

Target machine: MacBook Air (macOS). Budget: ~20 h. Strategy: **thin end-to-end first** — get one path (audio → Armenian text → English → terminal) working on day 1, then widen it (second source, speakers, latency, hardening).

---

## 1. Target architecture

```
 Mic (InputStream) ─────┐                         ┌─ source="mic"    → label "Me"
                        ├─► per-source ring buffer ─► VAD segmenter ─┤
 BlackHole (InputStream)┘   (resample → 16 kHz mono)                  └─ source="system" → SpeakerTracker → "Speaker N"
                                                                │
                                                     Segment queue (source, t_start, t_end, audio)
                                                                │
                                              ASR worker (Armenian text)  →  MT worker (English)
                                                                │
                                                   Printer: [hh:mm:ss] Speaker / HY / EN
```

- Threads + `queue.Queue`, no asyncio needed. Audio callbacks only copy data into queues (never block).
- One ASR worker (GPU/ANE is single-use anyway), segments processed in end-time order.
- Every component behind a tiny interface so models can be swapped for benchmarking:
  - `Transcriber.transcribe(audio: np.ndarray) -> str`
  - `Translator.translate(hy: str) -> str`
  - `SpeakerTracker.assign(audio: np.ndarray) -> str`

### Repo layout
```
app/
  cli.py          # argparse: --list-devices, --mic, --system, --asr, --mt, --debug
  audio.py        # device listing, capture, resampling
  vad.py          # Silero VAD → Segment objects
  asr.py          # Transcriber backends (mlx-whisper, faster-whisper, MMS/Omnilingual…)
  translate.py    # Translator backends (NLLB via CTranslate2, …)
  speakers.py     # embedding model + online clustering
  pipeline.py     # threads, queues, wiring
  output.py       # terminal rendering (rich)
  models.py       # Segment / Result dataclasses
scripts/
  bench_asr.py    # WER + latency on Armenian test clips
  bench_mt.py     # translation quality/latency on sample sentences
  record_test.py  # record a few clips from each device for debugging
tests/            # unit tests for VAD segmentation, clustering, formatting
README.md
chats/            # exported Cowork sessions (.md, one per session)
.gitignore        # models/, *.wav, .env, venv
```

---

## 2. Phase 0 — Environment & audio setup (≈2 h)

1. **Check the machine**: chip (M1/M2/M3…) and RAM (`sysctl hw.memsize`). With 8 GB RAM plan for Whisper turbo / quantized models; with 16 GB large-v3 + NLLB-1.3B fit.
2. **Tools**: `brew install python@3.11 portaudio ffmpeg blackhole-2ch` (reboot or restart CoreAudio after BlackHole install).
3. **Audio MIDI Setup**:
   - Create **Multi-Output Device** = Built-in speakers/headphones (master clock) + BlackHole 2ch; tick "Drift correction" on BlackHole.
   - Set it as system output (or as the speaker device inside Zoom/Meet/Telegram).
   - Note: volume keys don't work on a Multi-Output Device — set volume on the sub-device.
4. **Permissions**: System Settings → Privacy & Security → Microphone → allow Terminal/iTerm/VS Code.
5. **Python env**: `python3.11 -m venv .venv && pip install sounddevice numpy soxr silero-vad torch rich`.
6. **Sanity script** (`scripts/record_test.py`): print `sounddevice.query_devices()`, record 5 s from mic and 5 s from BlackHole while a YouTube video plays, save WAVs, listen back. ✅ Done when both WAVs contain the expected audio.
7. `git init`, `.gitignore`, first commit. Chats are exported once, at the end (Phase 6).

---

## 3. Phase 1 — Offline skeleton + model bake-off (≈4 h)

Goal: `python -m app.file sample.wav` prints HY + EN. No live audio yet — this isolates model problems from audio problems.

1. **Test data**: 15–20 Eastern Armenian clips with reference text — FLEURS `hy_am` test split / Common Voice `hy-AM` + 3–4 clips of your own voice and a two-person podcast/interview from YouTube (local only, never committed).
2. **ASR candidates** (all local):
   | Candidate | Runtime on Mac | Why consider | Watch out for |
   |---|---|---|---|
   | Whisper large-v3 | `mlx-whisper` (Apple GPU) | Best general Whisper for low-resource langs, punctuation | Slower; hallucinations on silence |
   | Whisper large-v3-turbo | `mlx-whisper` | ~4–8× faster | Pruned decoder → often worse on low-resource languages |
   | Armenian fine-tuned Whisper (HF community) | convert to MLX / CT2 | May beat base on Armenian | Check training data & eval honestly |
   | Meta MMS-1B-all (`hye` adapter) / Omnilingual ASR CTC | transformers / PyTorch MPS | Strong low-resource coverage, fast CTC | No punctuation/casing → hurts MT |
   | SeamlessM4T v2 | PyTorch | ASR + direct speech translation | Big, non-commercial license, slower |
3. **`bench_asr.py`**: for each model → WER/CER (`jiwer`) and seconds per clip. Pick by CER + latency for a ~5 s clip (target ≤ 1.5 s). Write numbers into README.
   - Whisper settings: `language="hy"`, `task="transcribe"`, `temperature=0`, `condition_on_previous_text=False`, drop output if `no_speech_prob` high or text is a known hallucination phrase.
4. **MT candidates**:
   | Candidate | Notes |
   |---|---|
   | **NLLB-200 distilled 600M / 1.3B** via CTranslate2 int8 (`hye_Armn → eng_Latn`) | Default pick: good Armenian, ~100–300 ms/sentence on CPU |
   | MADLAD-400 3B (CT2) | Possibly better quality, slower |
   | Helsinki-NLP opus-mt-hy-en | Tiny & fast, weaker quality |
   | Local LLM (Ollama: Gemma/Qwen) | Fluent English but slow, may hallucinate; Armenian weak in small models |
   | Whisper `task="translate"` | Rejected: loses the Armenian text, weaker quality |
   - Convert: `ct2-transformers-converter --model facebook/nllb-200-distilled-600M --output_dir models/nllb-600m-ct2 --quantization int8`.
5. Wire `Transcriber` + `Translator` into the file CLI and print in the final format. ✅ **Thin E2E #1 done**: WAV → HY → EN → terminal.

---

## 4. Phase 2 — Live mic end-to-end (≈3 h)

1. `audio.py`: `sd.InputStream(device=mic, samplerate=native, channels=1, callback=...)`; callback puts `indata.copy()` into a queue. Resample to 16 kHz with `soxr` in the consumer thread (don't trust PortAudio to resample).
2. `vad.py`: Silero VAD on 32 ms frames (512 samples @16 kHz).
   - Start segment when speech prob > 0.5; keep 200–300 ms pre-roll.
   - End segment after **~600 ms silence** (main latency knob).
   - **Force-cut at ~8 s** of continuous speech (cut at the lowest-energy frame in the last second) so long monologues still meet the 5 s target.
   - Drop segments < 0.4 s.
3. `pipeline.py`: segment → ASR → MT → `output.py`. Label every mic segment `Me`. Timestamp = segment start relative to session start.
4. `cli.py`: `--list-devices`, `--mic <id|name>`, `--system <id|name>` (wired up in Phase 3), `--debug` (prints per-stage timings).
   - Mic choice: `--mic` if given, else the system default input, picked once at startup. Following the call app's mic is Phase 5 (step 5).
5. ✅ **Thin E2E #2 done**: speak Armenian into the mic → lines appear live.

---

## 5. Phase 3 — Second source (system audio) (≈2 h)

1. Open a second `InputStream` on BlackHole with its own VAD segmenter → same job queue, `source="system"`, temporary label `System`.
2. Test: play an Armenian YouTube interview through the Multi-Output Device while speaking into the mic. Both streams appear, correctly labelled, interleaved by time.
3. Note: if you don't wear headphones the mic picks up the speakers → duplicates. Echo handling is out of scope; use headphones in the interview and mention this in README.
4. ✅ **Done (Oct 24)**: `--system BlackHole` opens a second capture with its own Segmenter. Live test with the default output switched to the Multi-Output Device, a podcast playing and the user reading over it and alone: both streams labelled, latency p50 1.4 s / p95 2.65 s, 0 input overflows. Echo into the headset mic at full headphone volume → duplicates (later fixed by the echo filter); a mic-side echo filter is a Phase 5 option.

---

## 6. Phase 4 — Speaker identification on system audio (≈3 h)

1. **Embedding model** (local): SpeechBrain `spkrec-ecapa-voxceleb` (ECAPA-TDNN, 192-d) — small, fast on CPU. Alternatives to mention: pyannote/WeSpeaker ResNet34, Resemblyzer (lighter, weaker). Full pyannote diarization pipeline / diart rejected as primary: built for offline or heavier streaming, adds latency and HF-token gated weights.
2. **Online clustering** (`speakers.py`):
   - Embed each system segment (skip/hold if < 1 s — too short for a reliable embedding; assign to last speaker or nearest with low-confidence flag).
   - Cosine similarity to each speaker centroid; if best ≥ threshold (start ~0.55, tune) → that speaker, update centroid with running mean; else → new `Speaker N`.
   - Guard against speaker explosion: require ≥ 1.5 s for creating a new speaker; optional merge of centroids that converge (> 0.75).
3. **Tuning**: record a 2–3 person Armenian conversation (YouTube podcast) → run offline, compare labels by ear, tune threshold. Add unit tests for clustering with synthetic embeddings.
4. Known limitation: a segment containing a speaker change gets one label. Mitigation if time: shorter max segment for the system stream, or sub-window embeddings (1.5 s windows) to detect a change point and split.

---

## 7. Phase 5 — Latency & reliability (≈3 h)

1. **Measure**: log `t_segment_end → t_printed` per segment in `--debug`; print p50/p95 on exit. Target p95 < 5 s.
   Typical budget: silence detection 0.6 s + ASR 0.7–1.5 s + speaker 0.1 s + MT 0.2 s ≈ 2–3 s.
2. **Backpressure**: if the queue grows (both sides talk at once), log a warning; optionally switch to the faster ASR model or drop overlap rather than fall further behind.
3. **Warm-up**: run each model once on silence at startup so the first real line isn't slow; print "Ready" only after.
4. **Robustness**: graceful Ctrl+C (close streams, flush queue), device-not-found errors with the device list, sample-rate mismatches, empty/hallucinated ASR output filtered.
5. **Follow the call app's mic** (only when `--mic` is not given; until now the mic is the system default, picked once at startup):
   - Find which input device the call app (Zoom, Chrome/Safari for Meet, Teams, Telegram, FaceTime) is recording from via CoreAudio (`kAudioHardwarePropertyProcessObjectList` → per-process input devices, macOS 14.2+; probed and working).
   - **At startup**: use the call app's device if one is recording, else the system default; print which one and why.
   - **During the session**: poll every ~2 s. On change: flush the current segment, reopen the mic stream on the new device, keep the session clock continuous, print `Mic → <device> (used by <app>)`. Covers both "transcriber started before the call" and "call switched mic mid-session".
   - Open question: whether Zoom/Meet keep the mic open while muted (if not, keep the last device instead of falling back to the default).
   - ✅ Done when: started before the call, the app moves to the call's mic once the call starts; switching the mic in the call app mid-session switches capture without a restart.
6. **Optional quality boosts** (only if core is stable): translate with the previous segment as context, Whisper `initial_prompt` with Armenian punctuation to encourage punctuated output.

---

## 8. Phase 6 — README, chats, rehearsal (≈3 h)

1. **README**: setup (brew, BlackHole, Multi-Output, permissions, venv, model download/convert), run commands, model choices with your benchmark numbers and rejected alternatives, architecture diagram, latency numbers, known limitations (echo, overlapping speech, short-utterance speaker errors, Western Armenian, accents/noise, NLLB non-commercial license), future work.
2. **Chats**: at the very end, export every Cowork session in full and unedited, each as its own `.md`, into `chats/` in the workspace root (not into `armtranscribe/`); keep the originals available, the interviewers open them.
3. **Rehearsal**: a real call (Zoom/Meet/Telegram) with 2 people on the other side, speaking Armenian, with the app running for 10+ minutes. Check labels stay stable, no memory growth, latency holds.
4. Final check: fresh clone → follow README from scratch → works.

---

## 9. Time budget

| Phase | Hours |
|---|---|
| 0 Setup & audio devices | 2 |
| 1 Offline skeleton + bake-off | 4 |
| 2 Live mic E2E | 3 |
| 3 System audio | 2 |
| 4 Speaker ID | 3 |
| 5 Latency & hardening | 3 |
| 6 README, chats, rehearsal | 3 |
| **Total** | **20** |

## 10. Main risks

| Risk | Mitigation |
|---|---|
| Armenian ASR quality poor | Bake-off on real data in Phase 1; be honest about CER in README |
| Whisper hallucinations on silence/noise | VAD gating, `no_speech_prob`, phrase blacklist, no previous-text conditioning |
| Latency > 5 s on Air (thermal throttling) | Turbo/quantized fallback flag, force-cut long segments, warm-up |
| Speakers merge/split | Min durations, threshold tuning on recorded conversation |
| Audio routing breaks on interview day | Setup checklist in README, test with the actual call app beforehand |

## 11. Interview-day checklist
- [ ] Multi-Output Device selected in the call app; headphones on
- [ ] `--list-devices` → correct IDs for mic and BlackHole
- [ ] Models already downloaded (offline run works with Wi-Fi off)
- [ ] Laptop on charger (avoids throttling), other heavy apps closed
- [ ] Chat history open in Cowork
