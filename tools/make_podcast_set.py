"""Cut a podcast into ~2 min clips with references from a YouTube transcript.

    python tools/make_podcast_set.py                     # Rearrange #326 -> recordings/podcast/yt_set/
    python tools/make_podcast_set.py --group 3 --every 4 --skip 90

The transcript is YouTube's "Show transcript" text: paragraphs starting with [mm:ss] or [h:mm:ss].
Each clip spans `--group` consecutive paragraphs (paragraph edges cut words, so longer clips keep
that noise small); every `--every`-th group is taken, after skipping the intro. References are
cleaned of [music]-style tags and "Ըըը" fillers. Writes WAVs plus manifest.tsv (file<TAB>hy<TAB>)
for armtranscribe/scripts/bench_asr.py --vad.

The references are YouTube's automatic captions (Google's ASR), not human transcripts: scores are
"distance from YouTube", fine for comparing models, not an absolute error rate. Google also writes
English terms in Latin script and numbers as digits, which our models don't.
"""
import argparse
import re
import sys
from pathlib import Path

import soundfile as sf

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE / "armtranscribe"))
from app.audio import load_audio  # noqa: E402

PODCAST = WORKSPACE / "recordings" / "podcast"
SR = 16_000


def paragraphs(path: Path) -> list[tuple[float, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\[(?:(\d+):)?(\d+):(\d+)\]\s*(.*)", line)
        if m:
            h, mnt, sec, text = m.groups()
            out.append((int(h or 0) * 3600 + int(mnt) * 60 + int(sec), text))
    return out


def clean(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", " ", text)                   # [երաժշտություն] etc.
    text = re.sub(r"(?<!\w)[ԸըԷէԱա]{2,}(?!\w)", " ", text)     # Ըըը, էէ fillers
    return " ".join(text.split())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--transcript", type=Path, default=PODCAST / "youtube_auto_hy.txt")
    p.add_argument("--audio", type=Path, default=PODCAST / "audio-podcast-example.mp3")
    p.add_argument("--out", type=Path, default=PODCAST / "yt_set")
    p.add_argument("--group", type=int, default=3, help="paragraphs per clip")
    p.add_argument("--every", type=int, default=4, help="take every N-th group")
    p.add_argument("--skip", type=float, default=90, help="seconds of intro to skip")
    args = p.parse_args()

    audio = load_audio(str(args.audio))
    paras = paragraphs(args.transcript)
    ends = [start for start, _ in paras[1:]] + [len(audio) / SR]
    spans = [(s, e, t) for (s, t), e in zip(paras, ends) if s >= args.skip]
    groups = [spans[i:i + args.group] for i in range(0, len(spans) - args.group + 1, args.group)][::args.every]

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0.0
    with open(args.out / "manifest.tsv", "w", encoding="utf-8") as m:
        for g in groups:
            start, end = g[0][0], g[-1][1]
            name = f"yt_{int(start) // 60:02d}m{int(start) % 60:02d}s.wav"
            sf.write(args.out / name, audio[int(start * SR):int(end * SR)], SR, subtype="PCM_16")
            m.write(f"{name}\t{clean(' '.join(t for _, _, t in g))}\t\n")
            total += end - start
    print(f"{len(groups)} clips, {total / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
