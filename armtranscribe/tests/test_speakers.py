import numpy as np

from app.speakers import UNKNOWN, OnlineSpeakerTracker

SR = 16000
rng = np.random.default_rng(0)
A, B, C = (rng.standard_normal(192) for _ in range(3))  # random 192-d directions are ~orthogonal


def voice(base, noise=0.3):
    """An embedding of speaker `base`: same direction plus a little per-segment noise."""
    return base + noise * np.linalg.norm(base) / np.sqrt(192) * rng.standard_normal(192)


def make(embed, **kw):
    """Tracker with the limits the cases below are written for (the defaults are tuned on real audio)."""
    return OnlineSpeakerTracker(embed, **{"threshold": 0.55, "min_new_s": 1.5, "new_margin": 0.0, **kw})


def audio(seconds):
    return np.zeros(int(seconds * SR), np.float32)


class Script:
    """Fake embedder: returns the queued embeddings in order, counts calls."""

    def __init__(self, *embeddings):
        self.queue, self.calls = list(embeddings), 0

    def __call__(self, a):
        self.calls += 1
        return self.queue.pop(0)


def labels(tracker, items):
    return [tracker.assign(audio(s)) for s in items]


def test_same_voice_keeps_label_and_new_voice_gets_next_number():
    t = make(Script(voice(A), voice(A), voice(B), voice(A), voice(B), voice(C)))
    assert labels(t, [2, 2, 2, 2, 2, 2]) == ["Speaker 1", "Speaker 1", "Speaker 2", "Speaker 1", "Speaker 2",
                                             "Speaker 3"]
    assert t.last.how == "new" and set(t.last.sims) == {"Speaker 1", "Speaker 2"}


def test_too_short_to_embed_gets_previous_label_without_embedding():
    embed = Script(voice(A), voice(B))
    t = make(embed)
    assert t.assign(audio(0.5)) == UNKNOWN          # nobody yet
    assert labels(t, [2, 0.8, 2, 0.6]) == ["Speaker 1", "Speaker 1", "Speaker 2", "Speaker 2"]
    assert embed.calls == 2 and t.last.how == "short"


def test_short_unfamiliar_segment_does_not_found_a_speaker():
    t = make(Script(voice(A), voice(B), voice(B)))
    assert labels(t, [2, 1.2, 1.6]) == ["Speaker 1", "Speaker 1", "Speaker 2"]
    assert [s.n for s in t.speakers] == [1, 1]      # the 1.2 s "nearest" guess did not move speaker 1


def test_threshold_controls_splitting():
    near = A + 0.9 * B                              # similarity to A ~0.74
    strict = make(Script(A, near), threshold=0.8, merge_threshold=0.95)
    loose = make(Script(A, near), threshold=0.6)
    assert labels(strict, [2, 2]) == ["Speaker 1", "Speaker 2"]
    assert labels(loose, [2, 2]) == ["Speaker 1", "Speaker 1"]


def test_converging_centroids_merge_into_older_speaker_and_numbers_are_not_reused():
    e1, e2, e3 = np.eye(192)[:3]
    a, x = e1, 0.5 * e1 + np.sqrt(0.75) * e2        # similarity 0.5: below threshold -> two speakers
    between = a + x                                 # 0.87 to both: joins one and pulls it towards the other
    t = make(Script(a, x, between, a, e3), threshold=0.6, merge_threshold=0.7)
    out = labels(t, [2] * 5)
    assert out[:2] == ["Speaker 1", "Speaker 2"]
    assert out[2] == "Speaker 1"                    # centroids now 0.71 apart -> merged into the older one
    assert out[3:] == ["Speaker 1", "Speaker 3"]    # number 2 is not handed out again
    assert sorted(s.id for s in t.speakers) == [1, 3] and t.speakers[0].n == 4


def test_centroid_follows_running_mean():
    t = make(lambda a: None)
    for e in (A, A + 0.2 * B, A - 0.2 * B):
        t.assign_embedding(e, 2 * SR)
    (s,) = t.speakers
    assert s.n == 3 and float(s.centroid @ (A / np.linalg.norm(A))) > 0.99


def test_defaults_need_3s_to_create_a_speaker():
    t = OnlineSpeakerTracker(Script(voice(A), voice(B), voice(B)))
    assert labels(t, [4, 2.8, 3.2]) == ["Speaker 1", "Speaker 1", "Speaker 2"]


def test_first_speaker_also_needs_min_new_seconds():
    t = OnlineSpeakerTracker(lambda a: np.ones(4), min_new_s=3.0)
    assert t.assign(np.zeros(int(1.2 * 16000), np.float32)) == UNKNOWN
    assert t.last.how == "short" and not t.speakers
    assert t.assign(np.zeros(3 * 16000, np.float32)) == "Speaker 1"


def test_grey_zone_goes_to_the_nearest_speaker_without_moving_it():
    e1, e2 = np.eye(4)[0], np.eye(4)[1]
    known_excited = 0.4 * e1 + np.sqrt(1 - 0.16) * e2   # similarity 0.40: between 0.35 and 0.45
    new_voice = 0.3 * e1 + np.sqrt(1 - 0.09) * e2       # similarity 0.30: below 0.35
    t = OnlineSpeakerTracker(Script(e1, known_excited, new_voice), threshold=0.45)  # default margin 0.10
    long = np.zeros(4 * 16000, np.float32)
    assert t.assign(long) == "Speaker 1"
    assert t.assign(long) == "Speaker 1" and t.last.how == "unsure"
    assert np.allclose(t.speakers[0].centroid, e1)      # centroid not moved by the unsure segment
    assert t.assign(long) == "Speaker 2"
