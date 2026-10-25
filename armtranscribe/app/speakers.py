"""Speaker embeddings + online clustering for the system-audio side.

Each system segment is embedded with SpeechBrain ECAPA-TDNN (192-d) and compared by cosine
similarity with the centroid of every speaker heard so far: close enough -> that speaker,
otherwise a new `Speaker N`. Runs on the CPU, ~0.05-0.1 s per segment.
"""
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from app.models import SAMPLE_RATE

UNKNOWN = "Speaker ?"  # only before the first speaker exists and the segment is too short to embed


class SpeakerTracker(Protocol):
    def assign(self, audio: np.ndarray) -> str: ...


def ecapa_embedder(model_dir: str) -> Callable[[np.ndarray], np.ndarray]:
    """SpeechBrain `spkrec-ecapa-voxceleb` loaded from a local dir (scripts/download_models.py ecapa-voxceleb)."""
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    # hyperparams.yaml points its checkpoints at the HF repo (`pretrained_path`); redirect to the local dir
    model = EncoderClassifier.from_hparams(source=model_dir, savedir=model_dir, run_opts={"device": "cpu"},
                                           overrides={"pretrained_path": model_dir})

    def embed(audio: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model.encode_batch(torch.from_numpy(np.ascontiguousarray(audio, np.float32))[None]).reshape(-1).numpy()

    return embed


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64).reshape(-1)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


@dataclass(eq=False)  # identity, not array comparison (list.remove)
class _Speaker:
    id: int
    total: np.ndarray       # sum of unit embeddings; its direction is the centroid
    n: int = 1

    @property
    def centroid(self) -> np.ndarray:
        return _unit(self.total)


@dataclass
class Decision:
    """How the last segment was labelled, for --debug and tuning."""
    label: str
    how: str                # match | new | first | nearest (too short for a new speaker) | short (not embedded)
    sims: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        sims = " ".join(f"{k.split()[-1]}:{v:.2f}" for k, v in self.sims.items())
        return f"{self.how}" + (f" {sims}" if sims else "")


class OnlineSpeakerTracker:
    """Online cosine clustering of segment embeddings into `Speaker 1`, `Speaker 2`, ...

    - segment < `min_embed_s`: too short for a reliable embedding, gets the previous label
    - best similarity >= `threshold`: that speaker; the centroid moves towards the segment
    - below threshold: a new speaker, but only from >= `min_new_s` of audio (shorter -> nearest)
    - two centroids that drift above `merge_threshold` are merged into the older speaker
    Numbers are never reused, so a label printed once keeps meaning the same person.

    Defaults tuned on 3 podcast clips (2 people, 9 min, scripts/tune_speakers.py): the same voice
    scores 0.6-0.9 against its centroid, the other one 0.0-0.35; thresholds 0.35-0.55 give the same
    labels, 0.65 splits one person into several. A 2.8 s laugh/overlap fragment that matched nobody
    created a phantom speaker at min_new_s=1.5-2.5; 3.0 removes it.
    """

    def __init__(self, embed: Callable[[np.ndarray], np.ndarray], threshold: float = 0.45,
                 min_embed_s: float = 1.0, min_new_s: float = 3.0, merge_threshold: float = 0.75,
                 sample_rate: int = SAMPLE_RATE):
        self.embed = embed
        self.threshold, self.merge_threshold = threshold, merge_threshold
        self.min_embed, self.min_new = int(min_embed_s * sample_rate), int(min_new_s * sample_rate)
        self.speakers: list[_Speaker] = []
        self._next_id = 1
        self._last: str | None = None
        self.last: Decision | None = None

    def assign(self, audio: np.ndarray) -> str:
        self.last = self._decide(audio)
        self._last = self.last.label
        return self.last.label

    def assign_embedding(self, e: np.ndarray, n_samples: int) -> Decision:
        """Clustering step on a precomputed embedding (tuning script, tests)."""
        self.last = self._cluster(_unit(e), n_samples)
        self._last = self.last.label
        return self.last

    def _decide(self, audio: np.ndarray) -> Decision:
        if len(audio) < self.min_embed:
            return Decision(self._last or UNKNOWN, "short")
        return self._cluster(_unit(self.embed(audio)), len(audio))

    def _cluster(self, e: np.ndarray, n_samples: int) -> Decision:
        if not self.speakers:
            return Decision(self._new(e), "first")
        sims = [float(s.centroid @ e) for s in self.speakers]
        named = {self._label(s): round(x, 3) for s, x in zip(self.speakers, sims)}
        i = int(np.argmax(sims))
        if sims[i] >= self.threshold:
            best = self.speakers[i]
            best.total, best.n = best.total + e, best.n + 1
            label = self._merge_into(best)
            return Decision(label, "match", named)
        if n_samples >= self.min_new:
            return Decision(self._new(e), "new", named)
        # too short to found a speaker: nearest one, centroid left alone (low confidence)
        return Decision(self._label(self.speakers[i]), "nearest", named)

    def _new(self, e: np.ndarray) -> str:
        s = _Speaker(self._next_id, e.copy())
        self._next_id += 1
        self.speakers.append(s)
        return self._label(s)

    def _merge_into(self, moved: _Speaker) -> str:
        """After `moved`'s centroid changed: merge it with any speaker it now overlaps. Returns its label."""
        while True:
            close = [o for o in self.speakers
                     if o is not moved and float(o.centroid @ moved.centroid) > self.merge_threshold]
            if not close:
                return self._label(moved)
            other = close[0]
            keep, gone = (other, moved) if other.id < moved.id else (moved, other)
            keep.total, keep.n = keep.total + gone.total, keep.n + gone.n
            self.speakers.remove(gone)
            moved = keep

    @staticmethod
    def _label(s: _Speaker) -> str:
        return f"Speaker {s.id}"
