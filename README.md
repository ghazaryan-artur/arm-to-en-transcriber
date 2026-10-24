# Live Armenian → English Transcriber

A Python program that listens to a call (your microphone + the system audio), transcribes spoken Armenian, translates it to English and prints each line live in the terminal, labelled by speaker, with only local models.

**The program, how to install and use it, the model choices and their benchmark numbers: [armtranscribe/README.md](armtranscribe/README.md).**

## Repository layout

| Path | What it is |
|---|---|
| [`armtranscribe/`](armtranscribe/) | The transcriber: the `app` package, tests, setup, benchmark scripts. Self-contained: it runs without anything else in this repository. |
| [`docs/PLAN.md`](docs/PLAN.md) | The implementation plan the work followed, phase by phase. |
| [`docs/TECH_DEBT.md`](docs/TECH_DEBT.md) | Known weaknesses and open issues found along the way, with measurements and fix ideas. |
| [`tools/`](tools/) | Development tools around the app: audio routing check (`record_test.py`), test-set preparation (`fetch_fleurs.py`, `make_podcast_set.py`, `build_live_test.py`), real-time latency replay (`replay_live.py`), and the 13-model ASR bake-off with its per-clip results (`asr_bakeoff/`), which the ASR table in the app README comes from. |
| [`chats/`](chats/) | The AI chat sessions of this project, one `.md` each. |

Not committed: test audio (own recordings, FLEURS clips, a podcast), model weights and the virtual environment. The benchmark numbers can therefore be re-run only where that audio is available (FLEURS can be re-fetched with `tools/fetch_fleurs.py`).

**Network use.** The transcriber itself never uses the network at runtime. Besides the one-time model download in `armtranscribe/`, only the development tools do: `tools/fetch_fleurs.py` and `tools/asr_bakeoff/download.py` fetch benchmark data and candidate models from Hugging Face.
