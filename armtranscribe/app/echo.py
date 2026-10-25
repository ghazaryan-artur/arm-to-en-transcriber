"""Echo filter: drops microphone segments that are only the call audio leaking into the mic.

Some headsets leak the playback into their own microphone (call 5: the remote side 20 dB under
the user's voice, < 5 ms behind the system stream), so every remote line was printed twice, also
as `Me`. A leaked copy is a filtered, delayed copy of the system audio: its normalised
cross-correlation with the system stream is 0.37-0.59 on the recorded calls, while the user's own
speech scores <= 0.16, also while both sides talk at once.
"""
import threading

import numpy as np

from app.models import SAMPLE_RATE

ECHO_XCORR = 0.3     # >= this: the mic segment is a copy of the system audio
MAX_LAG_S = 0.3      # the two streams' clocks and the leak path are within this of each other


def xcorr(a: np.ndarray, b: np.ndarray) -> float:
    """Max normalised cross-correlation of `a` with any same-length window of the longer `b`
    (each window normalised by its own energy, so a longer `b` doesn't dilute the score)."""
    m = len(a)
    if not m or len(b) < m:
        return 0.0
    n = 1 << (len(a) + len(b) - 1).bit_length()
    c = np.fft.irfft(np.conj(np.fft.rfft(a, n)) * np.fft.rfft(b, n), n)[:len(b) - m + 1]  # c[k] = a . b[k:k+m]
    e = np.concatenate([[0.0], np.cumsum(b.astype(np.float64) ** 2)])
    win = np.sqrt(np.maximum(e[m:] - e[:-m], 0) + 1e-9 * m)
    return float(np.max(np.abs(c) / (win * (np.linalg.norm(a) + 1e-9))))


class StreamTail:
    """The last `seconds` of a 16 kHz stream, addressable by stream time (s since stream start)."""

    def __init__(self, seconds: float = 40.0):
        self._buf = np.zeros(int(seconds * SAMPLE_RATE), np.float32)
        self._n = 0  # samples appended so far
        self._lock = threading.Lock()
        self._grew = threading.Condition(self._lock)

    @property
    def end_s(self) -> float:
        return self._n / SAMPLE_RATE

    def append(self, block: np.ndarray) -> None:
        block = np.asarray(block, np.float32)[-len(self._buf):]
        with self._lock:
            i = self._n % len(self._buf)
            first = min(len(block), len(self._buf) - i)
            self._buf[i:i + first] = block[:first]
            self._buf[:len(block) - first] = block[first:]
            self._n += len(block)
            self._grew.notify_all()

    def get(self, t0: float, t1: float, wait_s: float = 0.0) -> np.ndarray:
        """Samples between t0 and t1 that are still held (waits up to `wait_s` for t1 to arrive)."""
        a, b = max(0, int(t0 * SAMPLE_RATE)), int(t1 * SAMPLE_RATE)
        with self._lock:
            self._grew.wait_for(lambda: self._n >= b, timeout=wait_s)
            a, b = max(a, self._n - len(self._buf)), min(b, self._n)
            if b <= a:
                return np.zeros(0, np.float32)
            idx = np.arange(a, b) % len(self._buf)
            return self._buf[idx].copy()


def is_echo(mic: np.ndarray, t_start: float, t_end: float, system: StreamTail) -> tuple[bool, float]:
    """(mic segment is a leaked copy of the system audio, its cross-correlation score)."""
    ref = system.get(t_start - MAX_LAG_S, t_end + MAX_LAG_S, wait_s=MAX_LAG_S + 0.5)
    score = xcorr(np.asarray(mic, np.float32), ref)
    return score >= ECHO_XCORR, score
