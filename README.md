# Live Armenian → English Transcriber

Listens to the microphone and system audio, transcribes spoken Armenian, translates it to English and prints it live in the terminal, labelled by speaker. All models run locally.

**Local-only rule:** transcription, translation and speaker identification run entirely on this machine. No cloud speech or translation APIs are used. Model weights are downloaded once during setup; at runtime the app runs in offline mode (`HF_HUB_OFFLINE=1`) and works with Wi-Fi turned off.

> Work in progress — see `docs/PLAN.md`.

## Setup
_TODO_

## Usage
```bash
python -m armtranscribe --list-devices
python -m armtranscribe --mic <id> --system <id>
```

## Models
_TODO: choices, alternatives, benchmark results_

## Architecture
_TODO_

## Known limitations
_TODO_
