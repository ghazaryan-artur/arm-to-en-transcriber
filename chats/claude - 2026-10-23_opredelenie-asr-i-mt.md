# Определение ASR и MT

Exported from the Claude Code chat transcript by `tools/export_session.py`: the user's messages and Claude's replies as shown in the chat. Claude's thinking, tool calls (commands, file reads and edits) with their results are left out; Claude's messages between tool calls are kept, as Claude Code shows them.

## User · 2026-10-23 14:00 UTC

напомни что такое ASR и MT

## Claude · 2026-10-23 14:01 UTC

**ASR** (Automatic Speech Recognition) — автоматическое распознавание речи, то есть перевод звука в текст. В проекте это первый шаг: аудиосегмент превращается в армянский текст (HY). Отвечает за него интерфейс `asr.Transcriber.transcribe(audio) -> str`. Модель по умолчанию — `whisper-turbo-hy-mlx`: Whisper large-v3-turbo, дообученный на армянском и сконвертированный в MLX для Apple Silicon.

**MT** (Machine Translation) — машинный перевод одного текста в другой. В проекте это второй шаг: армянский текст переводится на английский (EN). Отвечает за него `translate.Translator.translate(hy) -> str`. Модель по умолчанию — `nllb-1.3b-ct2`: NLLB-200 1.3B в CTranslate2 int8, направление `hye_Armn → eng_Latn`.

Вместе это выглядит так: **аудио → ASR → армянский текст → MT → английский текст**. Флаги `--asr` и `--mt` выбирают модели для каждого шага, а качество измеряют скрипты `bench_asr.py` (WER/CER) и `bench_mt.py` (chrF++/BLEU).

## User · 2026-10-25 18:45 UTC

запусти export_session.py и ничего не коммить
