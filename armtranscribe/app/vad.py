"""Streaming voice-activity segmenter: 16 kHz audio chunks in, speech Segments out."""
from collections import deque
from typing import Callable

import numpy as np

from app.models import SAMPLE_RATE, Segment

FRAME = 512  # 32 ms; the only frame size Silero accepts at 16 kHz
FRAME_S = FRAME / SAMPLE_RATE


def silero_prob() -> Callable[[np.ndarray], float]:
    """Speech probability of one FRAME; the model keeps state across calls, so one per stream."""
    import torch
    from silero_vad import load_silero_vad

    model = load_silero_vad()

    def prob(frame: np.ndarray) -> float:
        with torch.no_grad():
            return model(torch.from_numpy(frame), SAMPLE_RATE).item()

    return prob


def _frames(ms: float) -> int:
    return max(1, round(ms / 1000 / FRAME_S))


class Segmenter:
    """Cuts a live stream into speech segments.

    A segment starts when speech prob >= `threshold` (with `pre_roll_ms` of audio before it) and
    ends after `min_silence_ms` below `threshold - 0.15`, keeping `pad_ms` of the trailing silence.
    Continuous speech is force-cut at `max_segment_s`, in the longest pause (run of frames below
    `threshold - 0.15`) within the last `cut_window_s`, or at the frame with the lowest speech
    probability if there is no pause, so that words stay whole; the rest carries over into the
    next segment. Segments with less than `min_segment_s` of speech are dropped.
    """

    def __init__(self, source: str, speech_prob: Callable[[np.ndarray], float] | None = None,
                 threshold: float = 0.5, min_silence_ms: float = 1000, max_segment_s: float = 12.0,
                 min_segment_s: float = 0.4, pre_roll_ms: float = 250, pad_ms: float = 200,
                 cut_window_s: float = 3.0):
        self.source = source
        self.prob = speech_prob or silero_prob()
        self.threshold, self.neg_threshold = threshold, threshold - 0.15
        self.min_silence = _frames(min_silence_ms)
        self.max_frames = _frames(max_segment_s * 1000)
        self.min_speech = _frames(min_segment_s * 1000)
        self.pad = _frames(pad_ms)
        self.cut_window = _frames(cut_window_s * 1000)
        self.pre_roll: deque[np.ndarray] = deque(maxlen=_frames(pre_roll_ms))
        self._rest = np.zeros(0, np.float32)  # samples short of a full frame
        self._n = 0                           # frames consumed so far: the stream clock
        self._seg: list[np.ndarray] | None = None
        self._probs: list[float] = []         # speech prob of each frame in _seg
        self._pre_probs: deque[float] = deque(maxlen=self.pre_roll.maxlen)
        self._seg_start = 0                   # frame index of _seg[0]
        self._speech_start = 0                # first and last speech frame, relative to _seg
        self._last_speech = 0
        self._silence = 0                     # trailing frames below neg_threshold

    def feed(self, audio: np.ndarray) -> list[Segment]:
        """Consume float32 mono samples @ SAMPLE_RATE; return the segments completed by them."""
        audio = np.concatenate([self._rest, np.asarray(audio, np.float32)])
        n = len(audio) // FRAME * FRAME
        self._rest = audio[n:]
        out = []
        for i in range(0, n, FRAME):
            seg = self._step(audio[i:i + FRAME])
            if seg:
                out.append(seg)
        return out

    def flush(self) -> list[Segment]:
        """End of stream: emit the segment in progress, if any."""
        if self._seg is None:
            return []
        seg = self._close()
        return [seg] if seg else []

    def _step(self, frame: np.ndarray) -> Segment | None:
        p = self.prob(frame)
        self._n += 1
        if self._seg is None:
            if p < self.threshold:
                self.pre_roll.append(frame)
                self._pre_probs.append(p)
                return None
            self._seg = [*self.pre_roll, frame]
            self._probs = [*self._pre_probs, p]
            self._seg_start = self._n - len(self._seg)
            self._speech_start = self._last_speech = len(self._seg) - 1
            self._silence = 0
            self.pre_roll.clear()
            self._pre_probs.clear()
            return None

        self._seg.append(frame)
        self._probs.append(p)
        if p >= self.threshold:
            self._last_speech, self._silence = len(self._seg) - 1, 0
        elif p < self.neg_threshold:
            self._silence += 1
        # between the thresholds: neither speech nor silence, the segment just goes on

        if self._silence >= self.min_silence:
            frames, probs = self._seg, self._probs
            seg = self._close()
            self.pre_roll.extend(frames[-self.pre_roll.maxlen:])  # trailing silence is the next pre-roll
            self._pre_probs.extend(probs[-self.pre_roll.maxlen:])
            return seg
        if len(self._seg) >= self.max_frames:
            return self._force_cut()
        return None

    def _cut_point(self) -> int:
        """Index in _seg to cut at: the middle of the longest pause in the cut window (the latest
        one on ties), else just after the frame with the lowest speech probability."""
        start = max(self._speech_start + 1, len(self._seg) - self.cut_window)
        probs = self._probs[start:]
        best, run = (0, 0), 0  # (length, end) of the longest run below neg_threshold
        for i, p in enumerate(probs):
            run = run + 1 if p < self.neg_threshold else 0
            if run and run >= best[0]:
                best = (run, i + 1)
        if best[0]:
            length, end = best
            return start + end - (length + 1) // 2
        return start + int(len(probs) - 1 - np.argmin(probs[::-1])) + 1

    def _force_cut(self) -> Segment | None:
        cut = self._cut_point()
        rest, rest_probs = self._seg[cut:], self._probs[cut:]
        last_speech, silence = self._last_speech - cut, min(self._silence, len(rest))
        seg = self._emit(cut)
        # the carried-over audio is a new segment that is already in speech
        self._seg, self._probs, self._seg_start = rest, rest_probs, self._n - len(rest)
        self._speech_start, self._last_speech, self._silence = 0, max(last_speech, 0), silence
        return seg

    def _close(self) -> Segment | None:
        """End the segment `pad` frames after its last speech."""
        seg = self._emit(min(len(self._seg), self._last_speech + 1 + self.pad))
        self._seg = None
        return seg

    def _emit(self, end: int) -> Segment | None:
        """Segment of _seg[:end], or None if it holds too little speech."""
        speech = min(self._last_speech + 1, end) - self._speech_start
        if speech < self.min_speech:
            return None
        t_start = self._seg_start * FRAME_S
        return Segment(self.source, t_start, t_start + end * FRAME_S, np.concatenate(self._seg[:end]))
