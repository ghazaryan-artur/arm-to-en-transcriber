"""MT speed vs CPU threads, to run while the call app is open (it competes for the CPU).

    python tools/bench_mt_threads.py [--threads 2 4 6 8] [--mt nllb-1.3b-ct2]

Translates the same 8 call-like sentences with each thread count and prints mean / max seconds.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "armtranscribe"))
import app.cli as cli  # noqa: E402  (first: sets offline mode)
from app.translate import CT2Translator  # noqa: E402

SENTENCES = [
    "Ես էլ եմ համաձայն Գոռի հետ ու գիտեմ, որ էդ ծրագիրը ցույց կտա իրեն դրական կողմով։",
    "Ինձ թվում ա, որ դա շատ հետաքրքիր որոշում ա եւ հուսով եմ շատ լավ կաշխատի։",
    "Որովհետեւ մենք շատ ժամանակ ենք տրամադրել էդ ծրագրի վրա։",
    "Դրական։",
    "Հիմա քննարկենք, օրինակ՝ ինչ ես մտածում, որ լավ կողմը ցույց կտա։",
    "Մենք ներկայումս աշխատում ենք մեր ծրագրի վրա, որտեղ ավելացնում ենք նոր մեթոդներ։",
    "Չէ, բայց կոնկրետ պատճառ։",
    "Պատասխանեմ Գոռի հարցին՝ սեւ հաց էր։",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threads", nargs="+", type=int, default=[2, 4, 6, 8])
    p.add_argument("--mt", default=cli.DEFAULT_MT)
    args = p.parse_args()
    for n in args.threads:
        mt = CT2Translator(cli.resolve_model(args.mt), threads=n)
        mt.translate(SENTENCES[0])  # warm-up
        secs = []
        for s in SENTENCES:
            t = time.perf_counter()
            mt.translate(s)
            secs.append(time.perf_counter() - t)
        print(f"threads {n}: mean {sum(secs) / len(secs):.2f} s, max {max(secs):.2f} s", flush=True)
        del mt


if __name__ == "__main__":
    main()
