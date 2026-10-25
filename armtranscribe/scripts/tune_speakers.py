"""Tune speaker clustering on a recorded conversation: same audio, several thresholds.

    python scripts/tune_speakers.py ../recordings/podcast/excerpt_10-15min.wav
    python scripts/tune_speakers.py conv.wav --thresholds 0.45 0.55 0.65 --timeline

The file is cut by the live VAD segmenter, every segment is embedded once (ECAPA), then the
online clustering is replayed for each threshold. Prints, per threshold, how many speakers it
found and how many segments each got; --timeline adds one row per segment (start, length,
label per threshold) to check against the recording by ear.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cli import DEFAULT_SPK, resolve_model  # noqa: E402  (first: sets offline mode)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="audio file with 2-3 people talking")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.35, 0.45, 0.55, 0.65])
    p.add_argument("--speaker-model", default=DEFAULT_SPK)
    p.add_argument("--min-new", type=float, default=3.0, help="seconds of audio needed to create a new speaker")
    p.add_argument("--timeline", action="store_true", help="print the label of every segment")
    args = p.parse_args()

    from app.audio import load_audio
    from app.file import split_speech
    from app.models import SAMPLE_RATE
    from app.output import fmt_time
    from app.speakers import OnlineSpeakerTracker, ecapa_embedder

    segments = split_speech(load_audio(args.path), source="system")
    print(f"{len(segments)} segments, {sum(s.t_end - s.t_start for s in segments):.0f} s of speech")
    embed = ecapa_embedder(resolve_model(args.speaker_model))
    t = time.perf_counter()
    trackers = [OnlineSpeakerTracker(embed, threshold=th, min_new_s=args.min_new) for th in args.thresholds]
    min_embed = trackers[0].min_embed
    embeddings = [embed(s.audio) if len(s.audio) >= min_embed else None for s in segments]
    n_emb = sum(e is not None for e in embeddings)
    print(f"embedded {n_emb} in {time.perf_counter() - t:.1f} s ({(time.perf_counter() - t) / max(1, n_emb):.3f} s each)\n")

    table = []  # per segment: one Decision per threshold
    for seg, e in zip(segments, embeddings):
        row = []
        for tr in trackers:
            if e is None:  # short: not embedded, keeps the previous label
                tr.assign(seg.audio)
                row.append(tr.last)
            else:
                row.append(tr.assign_embedding(e, len(seg.audio)))
        table.append(row)

    for j, (th, tr) in enumerate(zip(args.thresholds, trackers)):
        counts: dict[str, int] = {}
        hows: dict[str, int] = {}
        for row in table:
            counts[row[j].label] = counts.get(row[j].label, 0) + 1
            hows[row[j].how] = hows.get(row[j].how, 0) + 1
        per = ", ".join(f"{k.split()[-1]}:{v}" for k, v in sorted(counts.items()))
        print(f"threshold {th:.2f}: {len(tr.speakers)} speakers at the end, labels used [{per}]  "
              f"({', '.join(f'{k} {v}' for k, v in sorted(hows.items()))})")

    if args.timeline:
        print("\n start     len   " + "".join(f"{th:>7.2f}" for th in args.thresholds) + "   sims @" +
              f"{args.thresholds[len(args.thresholds) // 2]:.2f}")
        mid = len(args.thresholds) // 2
        for seg, row in zip(segments, table):
            labels = "".join(f"{d.label.split()[-1]:>7s}" for d in row)
            print(f" {fmt_time(seg.t_start)} {seg.t_end - seg.t_start:4.1f}s {labels}   {row[mid]}")
    print(f"\n(segment lengths: < {min_embed / SAMPLE_RATE:.1f} s keep the previous label without embedding)")


if __name__ == "__main__":
    main()
