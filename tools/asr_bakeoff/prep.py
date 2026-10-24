"""Freeze the bake-off inputs: {set: [(clip, [16 kHz pieces], reference)]} -> data.pkl.

Podcast clips are pre-cut with the live VAD segmenter so every model sees identical segments.
"""
import pickle
import sys
from pathlib import Path

W = Path(__file__).resolve().parent.parent.parent  # workspace
sys.path.insert(0, str(W / "armtranscribe"))
from app.audio import load_audio  # noqa: E402
from app.file import split_speech  # noqa: E402

SETS = {
    "fleurs": (W / "recordings/fleurs_hy/manifest.tsv", False),
    "own_phrases": (W / "recordings/live_test/part1/manifest.tsv", False),
    "own_monologue": (W / "recordings/live_test/part2/manifest.tsv", False),
    "podcast": (W / "recordings/podcast/yt_set/manifest.tsv", True),
}
data = {}
for name, (manifest, vad) in SETS.items():
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        file, ref = line.split("\t")[:2]
        audio = load_audio(str(manifest.parent / file))
        pieces = [s.audio for s in split_speech(audio)] if vad else [audio]
        rows.append((file, pieces or [audio], ref))
    data[name] = rows
    n = sum(len(p) for _, p, _ in rows)
    print(f"{name}: {len(rows)} clips, {n} pieces, {sum(len(a) for _, ps, _ in rows for a in ps) / 16000 / 60:.1f} min of speech")
pickle.dump(data, open(Path(__file__).resolve().parent / "data.pkl", "wb"))
