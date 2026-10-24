"""Score results/*.json against data.pkl references -> markdown tables (stdout) + scores.json."""
import json
import pickle
import sys
from pathlib import Path

import jiwer
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "armtranscribe"))
from app.asr import normalize  # noqa: E402

SETS = ["fleurs", "own_phrases", "own_monologue", "podcast"]
TITLES = {"fleurs": "FLEURS (news, read)", "own_phrases": "Your phrases (part 1)",
          "own_monologue": "Your monologue (part 2)", "podcast": "Podcast vs YouTube captions"}

data = pickle.load(open(HERE / "data.pkl", "rb"))
rows = []
for path in sorted((HERE / "results").glob("*.json")):
    r = json.load(open(path))
    row = {"label": r["label"], "backend": r["backend"], "load_s": r["load_s"]}
    for name in SETS:
        if name not in r["sets"]:
            continue
        s = r["sets"][name]
        refs = [normalize(ref) for _, _, ref in data[name]]
        hyps = [normalize(h) or "<empty>" for h in s["hyps"]]
        w = jiwer.process_words(refs, hyps)
        n = w.hits + w.substitutions + w.deletions
        secs, durs = np.array(s["secs"]), np.array(s["durs"])
        row[name] = {"wer": w.wer, "cer": jiwer.cer(refs, hyps), "S": w.substitutions / n, "D": w.deletions / n,
                     "I": w.insertions / n, "mean_s": secs.mean(), "p90_s": np.percentile(secs, 90),
                     "rtf": secs.sum() / durs.sum(), "empty": sum(not h.strip() for h in s["hyps"]) / len(s["hyps"])}
    rows.append(row)

for name in SETS:
    have = [r for r in rows if name in r]
    print(f"\n### {TITLES[name]}\n")
    print("| Model | WER | CER | subst | del | ins | s/segment (mean) | s/segment (p90) | RTF |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(have, key=lambda r: r[name]["cer"]):
        x = r[name]
        print(f"| {r['label']} | {x['wer']:.1%} | {x['cer']:.1%} | {x['S']:.1%} | {x['D']:.1%} | {x['I']:.1%} "
              f"| {x['mean_s']:.2f} | {x['p90_s']:.2f} | {x['rtf']:.2f} |")
json.dump(rows, open(HERE / "scores.json", "w"), indent=1, default=float)
