"""Shared data types."""
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000


@dataclass
class Segment:
    source: str          # "mic" or "system"
    t_start: float       # seconds since session start
    t_end: float
    audio: np.ndarray    # float32 mono @ SAMPLE_RATE


@dataclass
class Result:
    t_start: float
    speaker: str
    hy: str
    en: str
    latency: float       # seconds from segment end to print
    audio_s: float = 0.0 # segment duration and per-stage times, for --debug
    asr_s: float = 0.0
    mt_s: float = 0.0
