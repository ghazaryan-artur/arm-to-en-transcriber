# Known issues

Problems we know about, understand, and could fix, but deliberately don't within this test assignment (~20 h budget, "make the core reliable rather than add features"). Each entry: what happens, why it is left, and how it would be fixed. The user-facing summary is in [armtranscribe/README.md → Known limitations](armtranscribe/README.md#known-limitations).

## Audio devices and routing

- **Unplugged headset / device disappears.** The app warns after 3 s without audio ("back" when it returns), but the stream is not reopened; if macOS gives the device a new id, the app needs a restart.
  *Why left:* rare during one call, and every fix needs testing by hand with real hardware. *Fix:* on a stall, close the stream and reopen the device by name every few seconds (~1 h).
- **The Multi-Output Device doesn't follow output changes.** It names specific headphones: headphones off → you hear nothing (BlackHole still works); new headphones plugged in → macOS switches the output to them, BlackHole gets silence (warning after 15 s).
  *Why left:* a macOS routing limitation, solved by a pre-call checklist. *Fix:* capture the call with Core Audio process taps (macOS 14.4+) through a small Swift helper, no BlackHole / Multi-Output at all (~3–4 h).
- **The microphone is chosen once at startup.** Switching the mic in the call app later doesn't switch it in the transcriber.
  *Fix:* follow the system default input, or reopen on change.
- **No system-audio-only mode.** The mic is always captured; listening on speakers with your mic muted in the call still records the room. *Why left:* the brief needs both sources. *Fix:* a `--no-mic` flag (~10 min).
- **Echo while both talk.** Call audio leaking into the mic is dropped only when the mic segment is a clear copy of it; if you speak at the same time, the other side's words can mix into your `Me` line. Seen in the rehearsal: Gor's first sentence also appeared at the end of a `Me` line that continued Artur's. *Why left:* echo handling is out of scope per the brief. *Fix:* a real acoustic echo canceller (WebRTC AEC3) on the mic stream.
- **Two clocks.** Mic and BlackHole are separate CoreAudio clocks; lines from both at the same second can print in either order, and drift would matter only in hour-long sessions.

## Segmentation and output

- **Very short replies (< 0.4 s, a quick `Հա`) are dropped** by the VAD's minimum segment length. *Why left:* lowering it lets noise blips through; needs a noise test set to tune safely. *Fix:* min 0.25 s, but only with a high VAD peak (≥ 0.9).
- **Lines appear in the order segments end**, so a long remote line can be printed after a later short `Me` line.
- **A long monologue** shows its first line only at the next pause or at the 12 s cap (up to ~13 s after it started; still < 5 s after the end of that phrase).
- **In-place redraw of a continued line** assumes the terminal wasn't resized since the line was printed; otherwise parts of the old block can stay on screen (cosmetic).

## Speakers

- **Similar voices and excited speech split one person** (rehearsal, Zoom, Gor + Aghas: 4 speakers instead of 2). On the recorded calls the two voices score up to 0.5–0.6 against each other, and one person laughing or excited can drop to ~0.3–0.4 against their own centroid, so no single threshold separates them cleanly. The grey zone (0.35–0.45 → nearest speaker, only for segments < 6 s) removes the phantom seen in an earlier call; applied to all segments it merged a new person into a known one (Meet test: an 11 s first line at 0.39; another Meet test put both people on the call side into `Speaker 1`). Whether the current rule fixes those sessions and the rehearsal's Speaker 3 / 4 can't be checked: only one of them was saved, and it comes out right. *Why left:* better separation needs a stronger embedding model or diarization fine-tuned on call audio and a labelled test set. *Fix:* record sessions with `--save-audio`, label them, tune; enrol each voice from its first long segment with several embeddings instead of one centroid.
- **A new person's first short replies** (< 3 s) go to an already known speaker, or to `Speaker ?` before the first one exists. *Why left:* the 3 s rule is what keeps laughs and cross-talk from creating phantom speakers; the alternative needs a labelled test set. *Fix:* a pending candidate opened by a short unknown segment, confirmed by the next matching one.
- **Speaker change inside one segment** (fast turn-taking without a pause): the whole segment gets one label, so short answers inside the asker's segment are attributed to the asker. *Fix:* sliding 1.5 s embeddings to find the change point and split before ASR, or a diarization model trained on turn-taking.
- **Segments under 1 s** are too short for a reliable embedding and keep the previous label.
- **No labelled evaluation.** Tuned on one podcast (2 male voices) and checked by ear on recorded calls with up to 2 people on the call side; no speaker-annotated reference, no DER. *Fix:* label 10 min by hand incl. a female voice and 3 people.

## Recognition and translation

- **Colloquial speech and English loanwords** outside the built-in lists are misspelled by the ASR and then mistranslated; one wrong letter can change the meaning. *Fix:* fine-tune the ASR on conversational Armenian (the `--save-audio` sessions are a start).
- **Numbers the ASR mishears** (`հիսուներկու` → `մինիսուն երկու`) can't be repaired by the digit conversion. A lone number with a case ending is ambiguous (`Տասներկուսին` → "The 12th." where "At twelve." was meant).
- **Names outside the built-in list** (without `ջան`) are still translated as words.
- **Overlapping speech** on one stream can't be separated.
- **Noise read as an abbreviation** (`Հովհ.` → "How can we do this?") still passes the single-letter filter.
- **Western Armenian** is not supported or tested.

## Performance and resources

- **The fallback-model switch is untested under a real overload** (unit tests and a replay only), and with both NLLB models loaded RAM grows by ~0.6 GB; not measured on an 8 GB Mac.
- **CPU contention with the call app** (video, screen share) can slow every stage 2–4× for a while; the fallback model limits it but doesn't remove it.

## Reproducibility

- **Hugging Face revisions are not pinned** in `scripts/download_models.py`; an upstream change could change the downloaded model. *Fix:* `revision=` per model.
- **The NLLB source checkpoints (~8 GB)** stay in `~/.cache/huggingface/` after conversion; nothing cleans them up.
- **Benchmarks need local audio** (own voice, calls, podcast) that isn't committed; only the FLEURS numbers can be reproduced from a fresh clone.
- **No end-to-end test with the real models**: the unit tests use synthetic audio and fake models.
