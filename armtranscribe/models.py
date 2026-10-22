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
