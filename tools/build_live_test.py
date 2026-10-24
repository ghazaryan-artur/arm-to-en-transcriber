"""Build a live-test benchmark set from recorded --save-audio sessions.

    python tools/build_live_test.py part1      # 20 colloquial phrases -> recordings/live_test/part1/
    python tools/build_live_test.py part2      # 11-line monologue (held out) -> recordings/live_test/part2/

The lines were read from recordings/live_test/script.tsv (id<TAB>hy<TAB>en) in takes of a few lines
each; TAKES maps every line to the segments of the session it was recorded in. Writes one clip per
line (a line split by the VAD is joined with 0.3 s gaps), manifest.tsv (clips/<id>.wav<TAB>hy<TAB>en)
for armtranscribe/scripts/bench_*.py, and live.tsv (id<TAB>hy<TAB>en as the live run printed them).
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

LIVE_TEST = Path(__file__).resolve().parent.parent / "recordings" / "live_test"
SR = 16_000
GAP = np.zeros(int(0.3 * SR), np.float32)

# part -> session -> [(script id, segment numbers)]; latest take wins (c1, g1 were re-recorded at the end)
TAKES = {
    "part1": {
        "20261023-183409": [("a1", [1]), ("a2", [2]), ("a3", [3]), ("a4", [4]), ("b1", [5])],
        "20261023-183827": [("b2", [1]), ("b3", [2]), ("c2", [4]), ("c3", [5])],
        "20261023-184629": [("d1", [1]), ("d2", [2]), ("e1", [3, 4, 5, 6])],
        "20261023-184905": [("e2", [1, 2]), ("f2", [3])],
        "20261023-185723": [("f3", [1]), ("f4", [2])],
        "20261023-190032": [("g2", [1]), ("c1", [2]), ("g1", [3])],
    },
    "part2": {
        "20261023-223602": [("h1", [1]), ("h2", [2]), ("h3", [3])],
        "20261023-223945": [("h4", [1]), ("h5", [2]), ("h6", [3])],
        "20261023-224231": [("h7", [1]), ("h8", [2]), ("h9", [3, 4])],
        "20261023-224412": [("h10", [1]), ("h11", [2])],
    },
}
# lines the live VAD dropped, cut from the session's full stream: id -> (session, start s, end s)
DROPPED = {
    "part1": {"f1": ("20261023-184905", 17.45, 18.35)},  # Հա: below min_segment_s
    "part2": {},
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("part", choices=TAKES)
    args = p.parse_args()
    out = LIVE_TEST / args.part

    script = {}
    for line in (LIVE_TEST / "script.tsv").read_text(encoding="utf-8").splitlines():
        sid, hy, en = line.split("\t")
        script[sid] = (hy, en)

    clips, live = {}, {}
    for sess, takes in TAKES[args.part].items():
        rows = [l.split("\t") for l in (LIVE_TEST / sess / "segments.tsv").read_text(encoding="utf-8").splitlines()]
        for sid, nums in takes:
            parts = []
            for n in nums:
                parts += [sf.read(LIVE_TEST / sess / f"{n:04d}_mic.wav", dtype="float32")[0], GAP]
            clips[sid] = np.concatenate(parts[:-1])
            live[sid] = (" ".join(rows[n - 1][1] for n in nums), " ".join(rows[n - 1][2] for n in nums))
    for sid, (sess, t0, t1) in DROPPED[args.part].items():
        stream = sf.read(LIVE_TEST / sess / "mic.wav", dtype="float32")[0]
        clips[sid] = stream[int(t0 * SR):int(t1 * SR)]
        live[sid] = ("", "")

    (out / "clips").mkdir(parents=True, exist_ok=True)
    ids = [sid for sid in script if sid in clips]  # script order
    with open(out / "manifest.tsv", "w", encoding="utf-8") as m, open(out / "live.tsv", "w", encoding="utf-8") as lv:
        for sid in ids:
            sf.write(out / "clips" / f"{sid}.wav", clips[sid], SR, subtype="PCM_16")
            m.write(f"clips/{sid}.wav\t{script[sid][0]}\t{script[sid][1]}\n")
            lv.write(f"{sid}\t{live[sid][0]}\t{live[sid][1]}\n")
    print(len(ids), "lines:", " ".join(ids))


if __name__ == "__main__":
    main()
