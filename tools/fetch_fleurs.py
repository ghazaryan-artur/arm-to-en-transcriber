"""Fetch a small Armenian ASR benchmark set from FLEURS (hy_am, test split).

    python tools/fetch_fleurs.py                  # 20 clips -> recordings/fleurs_hy/
    python tools/fetch_fleurs.py -n 30 --max-dur 12

Picks distinct sentences of 4-10 s (live segments are force-cut at ~8 s), half male / half
female voices, deterministically. Writes the WAVs plus manifest.tsv (file<TAB>hy<TAB>en) for
armtranscribe/scripts/bench_asr.py and bench_mt.py; FLEURS is n-way parallel, so the English reference
is the en_us sentence with the same id. Downloads ~520 MB into the Hugging Face cache once.
"""
import argparse
import csv
import tarfile
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
REPO = "google/fleurs"
SR = 16_000


def pick(rows: list[list[str]], n: int, min_dur: float, max_dur: float) -> list[list[str]]:
    # FLEURS tsv: id, file_name, raw_transcription, transcription, phonemes, num_samples, gender
    seen, pools = set(), {"MALE": [], "FEMALE": []}
    for r in sorted(rows, key=lambda r: r[1]):
        if r[0] in seen or not min_dur <= int(r[5]) / SR <= max_dur or r[6] not in pools:
            continue
        seen.add(r[0])
        pools[r[6]].append(r)
    out = []
    for gender, pool in pools.items():
        k = n // 2 + (n % 2 if gender == "MALE" else 0)
        step = max(1, len(pool) // k)
        out += pool[::step][:k]
    return sorted(out, key=lambda r: r[1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", type=int, default=20)
    p.add_argument("--min-dur", type=float, default=4.0)
    p.add_argument("--max-dur", type=float, default=10.0)
    p.add_argument("--out-dir", type=Path, default=WORKSPACE / "recordings" / "fleurs_hy")
    p.add_argument("--download-dir", type=Path, help="reuse/keep the raw download here instead of the HF cache")
    args = p.parse_args()

    from huggingface_hub import hf_hub_download

    def fetch(name: str, lang: str = "hy_am") -> str:
        return hf_hub_download(REPO, f"data/{lang}/{name}", repo_type="dataset", local_dir=args.download_dir)

    def read_tsv(path: str) -> list[list[str]]:
        with open(path, encoding="utf-8") as f:
            return list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))

    english = {r[0]: r[2] for split in ("test", "dev", "train") for r in read_tsv(fetch(f"{split}.tsv", "en_us"))}
    chosen = pick(read_tsv(fetch("test.tsv")), args.n, args.min_dur, args.max_dur)
    missing = [r[1] for r in chosen if r[0] not in english]
    if missing:
        print(f"warning: no English reference for {len(missing)} clip(s)")
    wanted = {f"test/{r[1]}": r for r in chosen}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fetch("audio/test.tar.gz")) as tar:
        for member in tar:
            if member.name in wanted:
                member.name = Path(member.name).name
                tar.extract(member, args.out_dir, filter="data")

    with open(args.out_dir / "manifest.tsv", "w", encoding="utf-8") as f:
        for r in chosen:
            f.write(f"{r[1]}\t{r[2]}\t{english.get(r[0], '')}\n")
    total = sum(int(r[5]) for r in chosen) / SR
    print(f"{len(chosen)} clips, {total:.0f} s total -> {args.out_dir}/manifest.tsv")


if __name__ == "__main__":
    main()
