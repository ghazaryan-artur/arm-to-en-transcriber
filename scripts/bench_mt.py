"""Compare translation models on Armenian sentences: chrF++ / BLEU against English references, latency.

    python scripts/bench_mt.py <manifest.tsv> [--models nllb-600m-ct2 nllb-1.3b-ct2 ...]
    python scripts/bench_mt.py <manifest.tsv> --asr-hyps hyp.tsv --asr-column whisper-large-v3-mlx

manifest.tsv: <file><TAB><Armenian reference><TAB><English reference>, no header (FLEURS is
n-way parallel, so tools/fetch_fleurs.py fills the English column). With --asr-hyps the input
is ASR output saved by bench_asr.py --save, i.e. the full speech -> English cascade is scored.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from armtranscribe.cli import resolve_model  # noqa: E402  (also sets offline mode)
from armtranscribe.translate import load_translator  # noqa: E402

DEFAULT_MODELS = ["nllb-600m-ct2", "nllb-1.3b-ct2", "opus-mt-hy-en-ct2"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", type=Path)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="model dirs or names in models/")
    p.add_argument("--asr-hyps", type=Path, help="TSV from bench_asr.py --save to translate instead of references")
    p.add_argument("--asr-column", help="which ASR model's column to use from --asr-hyps")
    p.add_argument("--show", action="store_true", help="print every translation")
    args = p.parse_args()

    import sacrebleu

    rows = [line.split("\t") for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    sources = {r[0]: r[1] for r in rows}
    refs = {r[0]: r[2] for r in rows}
    if args.asr_hyps:
        with open(args.asr_hyps, encoding="utf-8") as f:
            hyps = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
        column = args.asr_column or list(hyps[0])[2]
        sources = {h["file"]: h[column] for h in hyps}
        print(f"input: ASR output of {column}")
    files = [f for f in sources if refs.get(f)]
    print(f"{len(files)} sentences\n")

    results = []
    for model in args.models:
        print(f"running {model} ...", file=sys.stderr)
        t = time.perf_counter()
        mt = load_translator(resolve_model(model))
        mt.translate(sources[files[0]])  # warm-up
        load_s = time.perf_counter() - t
        outs, secs = [], []
        for f in files:
            t = time.perf_counter()
            outs.append(mt.translate(sources[f]))
            secs.append(time.perf_counter() - t)
        ref_list = [[refs[f] for f in files]]
        results.append({
            "model": model, "outs": outs, "load_s": load_s,
            "chrf": sacrebleu.corpus_chrf(outs, ref_list, word_order=2).score,
            "bleu": sacrebleu.corpus_bleu(outs, ref_list).score,
            "mean_s": float(np.mean(secs)), "p90_s": float(np.percentile(secs, 90)),
        })

    print("| Model | chrF++ | BLEU | s/sentence (mean) | s/sentence (p90) | load+warm-up s |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['model']} | {r['chrf']:.1f} | {r['bleu']:.1f} | {r['mean_s']:.2f} | {r['p90_s']:.2f} "
              f"| {r['load_s']:.1f} |")

    if args.show:
        for i, f in enumerate(files):
            print(f"\nHY:  {sources[f]}\nREF: {refs[f]}")
            for r in results:
                print(f"{r['model']}: {r['outs'][i]}")


if __name__ == "__main__":
    main()
