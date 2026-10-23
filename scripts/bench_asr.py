"""Compare ASR models on Armenian clips: WER/CER and latency.

    python scripts/bench_asr.py <manifest.tsv> [--models whisper-large-v3-mlx mms-1b-all ...] [--save hyp.tsv]

manifest.tsv: <wav path relative to the manifest><TAB><Armenian reference>[<TAB>...], no header.
Text is normalised before scoring (lowercase, no punctuation, եւ -> և). Each model is warmed
up on the first clip; latency is wall time per clip on this machine.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from armtranscribe.cli import resolve_model  # noqa: E402  (also sets offline mode)
from armtranscribe.asr import load_transcriber, normalize  # noqa: E402
from armtranscribe.audio import load_audio  # noqa: E402
from armtranscribe.models import SAMPLE_RATE  # noqa: E402

DEFAULT_MODELS = ["whisper-large-v3-mlx", "whisper-large-v3-turbo-mlx", "whisper-turbo-hy-mlx", "mms-1b-all"]


def read_manifest(path: Path) -> list[tuple[Path, list[str]]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            file, *texts = line.split("\t")
            rows.append((path.parent / file, texts))
    return rows


def bench(model: str, clips: list[tuple[str, np.ndarray, str]]) -> dict:
    import jiwer

    t = time.perf_counter()
    asr = load_transcriber(resolve_model(model))
    asr.transcribe(clips[0][1])  # warm-up: first call compiles kernels / loads weights
    load_s = time.perf_counter() - t

    hyps, secs = [], []
    for _, audio, _ in clips:
        t = time.perf_counter()
        hyps.append(asr.transcribe(audio))
        secs.append(time.perf_counter() - t)
    refs = [normalize(ref) for _, _, ref in clips]
    norm = [normalize(h) or "<empty>" for h in hyps]
    audio_s = sum(len(a) for _, a, _ in clips) / SAMPLE_RATE
    return {
        "model": model, "hyps": hyps, "load_s": load_s,
        "wer": jiwer.wer(refs, norm), "cer": jiwer.cer(refs, norm),
        "mean_s": float(np.mean(secs)), "p90_s": float(np.percentile(secs, 90)), "rtf": sum(secs) / audio_s,
        "punct": sum(any(c in h for c in ",։:՝?") for h in hyps) / len(hyps),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", type=Path)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="model dirs or names in models/")
    p.add_argument("--save", type=Path, help="write per-clip hypotheses as TSV")
    args = p.parse_args()

    rows = read_manifest(args.manifest)
    clips = [(f.name, load_audio(str(f)), texts[0]) for f, texts in rows]
    durs = [len(a) / SAMPLE_RATE for _, a, _ in clips]
    print(f"{len(clips)} clips, {sum(durs):.0f} s audio, {min(durs):.1f}-{max(durs):.1f} s each\n")

    results = []
    for model in args.models:
        print(f"running {model} ...", file=sys.stderr)
        results.append(bench(model, clips))

    print("| Model | WER | CER | s/clip (mean) | s/clip (p90) | RTF | punctuated | load+warm-up s |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['model']} | {r['wer']:.1%} | {r['cer']:.1%} | {r['mean_s']:.2f} | {r['p90_s']:.2f} "
              f"| {r['rtf']:.2f} | {r['punct']:.0%} | {r['load_s']:.1f} |")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write("file\tref\t" + "\t".join(r["model"] for r in results) + "\n")
            for i, (name, _, ref) in enumerate(clips):
                f.write("\t".join([name, ref] + [r["hyps"][i] for r in results]) + "\n")


if __name__ == "__main__":
    main()
