"""Numbers in translation: NLLB with and without app/numbers.digitize on numbers.tsv.

    python tools/numbers_mt/eval.py [--mt nllb-1.3b-ct2] [--limit-s 150]

numbers.tsv: hy <TAB> English reference <TAB> expected numbers (';' between numbers, each a regex
of accepted forms) <TAB> source (call = said in a recorded call, script = live-test script, own =
written for this test). A line counts as right when every expected number is in the English.
Translations are cached in cache_<mt>.json, so a run cut by --limit-s continues where it stopped.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent.parent / "armtranscribe"
sys.path.insert(0, str(APP))

from app import glossary  # noqa: E402
from app.numbers import digitize  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mt", default="nllb-1.3b-ct2")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit-s", type=float, default=1e9, help="stop translating after this many seconds")
    args = p.parse_args()

    rows = [l.rstrip("\n").split("\t") for l in open(HERE / "numbers.tsv", encoding="utf-8") if l.strip()]
    cache_path = HERE / f"cache_{args.mt}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    names = glossary.Names()
    mt = None
    t0 = time.perf_counter()
    variants = {"words": lambda h: h, "digits": digitize}  # before / after app/numbers.py
    for hy, *_ in rows:
        for v, f in variants.items():
            key = f"{v}\t{hy}"
            if key in cache:
                continue
            if time.perf_counter() - t0 > args.limit_s:
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
                print(f"stopped: {len(cache)}/{2 * len(rows)} done, run again to continue")
                return
            if mt is None:
                from app.translate import CT2Translator
                mt = CT2Translator(str(APP / "models" / args.mt), threads=args.threads)
            glossary.digitize = f  # the "words" run switches the conversion off inside the glossary
            cache[key] = glossary.GlossaryTranslator(mt, names).translate(hy)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1))

    score = {v: 0 for v in variants}
    for hy, ref, expected, src in rows:
        marks = []
        for v in variants:
            en = cache[f"{v}\t{hy}"]
            ok = all(re.search(rf"(?<![\w,])(?:{e})(?!\w|,\d)", en, re.I) for e in expected.split(";"))
            score[v] += ok
            marks.append(f"{'+' if ok else '-'} {v:6s} {en}")
        print(f"[{src}] {hy}\n  ref    {ref}\n  " + "\n  ".join(marks))
    print("\n" + ", ".join(f"{v}: {n}/{len(rows)} numbers right" for v, n in score.items()))


if __name__ == "__main__":
    main()
