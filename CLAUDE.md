# CLAUDE.md

Workspace and git repository of the Armenian → English transcriber. **`armtranscribe/` is the program**; `docs/`, `tools/` and `chats/` around it are the development record and are committed too (the assignment grades the working method as well as the result). Test audio, model weights, `.venv` and the assignment brief are gitignored.

## Layout

```
armenian-transcriber/      repo root: README.md (overview: layout, how the work was done), .gitignore
  armtranscribe/           the program: package app/, tests/, scripts/, requirements, setup.command, README.md (setup, usage, models),
                           models/ and .venv/ (both gitignored; .venv is shared with tools/)
  docs/PLAN.md             design plan — source of truth for phases, parameters, tuning targets;
                           read the relevant phase before implementing a module
  docs/TECH_DEBT.md        known weaknesses / open issues per phase
  chats/                   AI chat sessions, exported once at the very end (one .md each, full and unedited)
  tools/                   development tools, not part of the program: record_test.py (does my mic / BlackHole routing work);
                           benchmark data prep: fetch_fleurs.py, make_podcast_set.py (podcast clips + YouTube-caption references);
                           build_live_test.py part1|part2 (recordings/live_test sessions -> part1/part2 test sets);
                           replay_live.py <wav> [sec] (replay audio through the live Pipeline in real time: latency p50/p95);
                           asr_bakeoff/ (13-model ASR comparison harness + results, see its README)
  recordings/              (gitignored) test audio: record_test output, benchmark clips (pass to armtranscribe/scripts/bench_*.py by path)
                           live_test/: script.tsv (31 reference lines) + --save-audio sessions -> part1/ (20 phrases), part2/ (11-line monologue, held out)
                           podcast/: Rearrange #326 mp3, youtube_auto_hy.txt (YouTube auto captions), yt_set/ (22 min, 10 clips)
                           fleurs_hy/: 20 FLEURS test clips
  .claude/                 Claude Code hooks (local-only import guard, pytest after .py edits)
```

- Run git commands from this directory (the repo root).
- Don't copy plan, tools, recordings or `.claude/` into `armtranscribe/`. Keep `armtranscribe/` self-contained: no references to files outside it.
- Never commit `recordings/`, weights, the `*.docx` brief or third-party transcripts; the `.gitignore` covers them, check `git status` before committing.
- `tools/` scripts import the package from `armtranscribe/` via `sys.path`.

## Commands (from this directory)

```bash
armtranscribe/setup.command               # armtranscribe/.venv + requirements + default models
source armtranscribe/.venv/bin/activate
pip install pytest                        # dev only
(cd armtranscribe && python -m pytest)
python tools/record_test.py [--mic ...] [--system BlackHole] [-s 5]   # writes recordings/mic.wav, system.wav
```

What belongs where: tests of the code and its functionality (pytest, benchmarks of the app's models) are part of the project and live in `armtranscribe/`; checks of this machine's hardware and audio routing live in `tools/`.

App commands (running the transcriber, model download, benchmarks) are in `armtranscribe/CLAUDE.md`.

@armtranscribe/CLAUDE.md
