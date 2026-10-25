# Skill recommendations for armenian-transcriber

Exported from the Claude Code chat transcript by `tools/export_session.py`: the user's messages and Claude's replies as shown in the chat. Claude's thinking, tool calls (commands, file reads and edits) with their results are left out; Claude's messages between tool calls are kept, as Claude Code shows them.

## User · 2026-10-22 14:29 UTC

which skills do you recommend I use for this project?

## Claude · 2026-10-22 14:30 UTC

Your project is a local-only Python CLI (live Armenian→English transcription with MLX Whisper, CTranslate2, and SpeechBrain), and it's still at the skeleton stage. These are the skills I'd use, most useful first:

**Worth using now**
- **`/init`**: creates a `CLAUDE.md`. This matters most for you because it can record rules I'd otherwise have to rediscover each session: no cloud APIs, `HF_HUB_OFFLINE=1` at runtime, Apple Silicon/MLX, and where models live (`models/`). After that, I'll treat the local-only rule as a hard rule in every session.
- **`/run`**: starts the app and checks a change actually works, not just that tests pass. Live audio is limited here because I can't hear your mic, but it still catches startup errors, `--list-devices` output, and runs on the benchmark scripts.
- **`/code-review`** and **`/simplify`**: use these after each milestone in `docs/PLAN.md`. Concurrency bugs in the audio→VAD→ASR→translate pipeline, like queue backpressure, threads that never shut down, or sample-rate mismatches, are the kind of thing `/code-review` is good at finding.

**Useful a bit later**
- **`update-config`** (you just ask, e.g. "run pytest after every edit"): sets up hooks. Two that fit this project: run `pytest` after edits, and flag any new import of networking libraries like `requests` or `openai` to protect the local-only rule.
- **`dataviz`**: when you fill in the README's "Models / benchmark results" section from `bench_asr.py` and `bench_mt.py` (WER, latency, real-time factor), this makes the charts clean and consistent.
- **`/security-review`**: run it once before calling the project done, mainly to check that nothing sends audio or text off the machine.
- **`/fewer-permission-prompts`**: after a few sessions, it allowlists the commands you keep approving, such as `pytest` and `python -m armtranscribe`.
- **`skill-creator`**: if you keep repeating a workflow, like "record a test clip, run the pipeline, compare against a reference transcript," you can turn it into a project skill.

**You can skip:** `claude-api` (it conflicts with the local-only design), the docx/pptx/xlsx/pdf skills, `loop`/`schedule`, and the Gmail/Drive tools.

My suggestion: run `/init` first, so later sessions start with the right constraints.

## User · 2026-10-22 14:32 UTC

install what needed now

## Claude · 2026-10-22 14:33 UTC

I created [CLAUDE.md](CLAUDE.md) at the repo root. I haven't run any of the commands in it.

It covers:
- **Local-only rule:** no cloud APIs and no runtime network calls, and the offline env vars in `cli.py` must be set before any model import. Only `scripts/download_models.py` may use the network.
- **Commands:** venv setup, running the tests (including a single test), the CLI flags, and the scripts. It notes that `pytest` isn't in `requirements.txt` and that no linter is configured.
- **Architecture:**
  - Threads and queues carry audio from the mic and BlackHole through VAD, ASR, translation and output.
  - PortAudio callbacks only copy data.
  - ASR, translation and speaker ID sit behind swappable `Protocol` interfaces.
  - The default models and latency targets come from `docs/PLAN.md`, which it points to as the source of truth.

I found an OpenAI Codex config at `~/.codex/config.toml` and didn't read it. To import its MCP servers, commands, skills and instructions, reply `/import` and I'll list what's importable. Then apply it with `/import --yes=<digest>`, using the digest from that scan.

The other items I suggested earlier (hooks, `/run`, `/code-review`) need nothing installed. I can set up the pytest and no-network hooks whenever you want them.

## User · 2026-10-22 14:33 UTC

/model sonnet

## User · 2026-10-22 14:35 UTC

so the skills that worth using now is already used?

## Claude · 2026-10-22 14:35 UTC

Partly. Of the "now" group, only `/init` is done.

- **`/init`:** done. [CLAUDE.md](CLAUDE.md) is written.
- **`/run`, `/code-review` and `/simplify`:** there's nothing to run or review yet. Most modules are empty stubs. They become useful once the pipeline has real code, so use them after each phase in `docs/PLAN.md`.
- **`update-config` hooks:** not set up. I suggested a pytest-after-edit hook and a guard against network imports. I haven't added either, so say so if you want them.
- **`/import`:** not run. It would pull in your Codex config, and only if you want that.

To finish setting things up now, I can add those two hooks to `.claude/settings.json`. Otherwise you can start on Phase 0 of the plan.

## User · 2026-10-22 14:35 UTC

add

## Claude · 2026-10-22 14:36 UTC

pytest isn't installed in the venv yet, so I'll add it first. The hook needs it and CLAUDE.md already lists it as a manual step.

The three parse errors come from my test harness: `echo` expanded `\n` into real newlines. I'll re-run those cases with `printf`, and also check that the pytest hook fails correctly.

All pipe-tests behave as expected. Now I'll write the settings file with `jq`, so the shell quoting is escaped correctly.

Now I'll check that the guard fires in the live session. I'll try a deliberately bad edit, which should be blocked.

The guard fired live and blocked the edit. That also shows the session picked up the new settings file, so the pytest hook is loaded too. I'll confirm nothing was written.

Both hooks are in [.claude/settings.json](.claude/settings.json), and the project settings are live in this session.

- **No-network guard:** this runs before any Write or Edit to a `.py` file under `armtranscribe/`. It blocks imports of `requests`, `httpx`, `aiohttp`, `urllib3`, `urllib.request`, `http.client`, `websockets`, `openai`, `anthropic`, `boto3`, `google.cloud`, `azure` and `deepl`. `scripts/` is exempt, so `download_models.py` can still use the network. I tried a real edit adding `import requests` to `vad.py`, and the hook blocked it. The file is unchanged.
- **pytest after edits:** after any Write or Edit to a `.py` file, it runs `pytest -q -x` in the venv. It stays quiet when tests pass. When they fail, it feeds the last 20 lines of output back to me. I tested the failure path with a deliberately failing test. I didn't watch this hook fire in the live session.

I also installed `pytest` into `.venv`, since the hook needs it. The two existing tests pass.

- **Possible false positive:** the guard checks only the new text of an edit. If I edit a line that sits next to an existing banned import, it can block the edit even though no import was added. It would be rare in practice.
- **Commit status:** the settings file and `CLAUDE.md` are not committed. Tell me if you want me to commit them.
- **Review:** you can review or disable the hooks any time with `/hooks`.

## User · 2026-10-22 14:37 UTC

в итоге у меня есть все скилы которые лучше использовать сначала?

## Claude · 2026-10-22 14:37 UTC

Да, всё, что стоило использовать в начале, уже сделано или доступно.

**Сделано:**
- **`/init`:** создан [CLAUDE.md](CLAUDE.md).
- **`update-config`:** добавлены два хука в [.claude/settings.json](.claude/settings.json), запрет сетевых импортов в `armtranscribe/` и запуск pytest после правок `.py`. Заодно в `.venv` поставлен `pytest`.

**Уже доступны, ставить ничего не нужно:** `/run`, `/code-review`, `/simplify`, `/security-review`, `/fewer-permission-prompts`, `dataviz`, `skill-creator`. Это встроенные скиллы, и они подключаются без установки. Пока в проекте почти нет кода, поэтому применять их рано:
- `/code-review` и `/simplify` запускайте после первых рабочих модулей (VAD, ASR, пайплайн).
- `/run` пригодится, когда `--mic` и `--system` начнут работать.
- `/security-review` нужен перед завершением проекта.
- `dataviz` пригодится для графиков бенчмарков в README.

**Не сделано и по желанию:** `/import`, перенос настроек из `~/.codex/config.toml`. Он нужен, только если вы пользуетесь Codex и хотите перенести оттуда MCP-серверы или инструкции.

Дальше можно начинать Phase 0 из `docs/PLAN.md`. Это проверка железа, BlackHole и аудио-маршрутизации, затем `scripts/record_test.py`. Начнём?

## User · 2026-10-25 18:44 UTC

запусти export_session.py и ничего не коммить
