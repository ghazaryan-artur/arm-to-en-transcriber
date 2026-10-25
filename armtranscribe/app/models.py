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
    speech_end: float | None = None  # end of the last speech frame (t_end minus trailing silence)

    @property
    def phrase_end(self) -> float:
        """When the speaker stopped: latency is measured from here (the brief: "from the end of a
        spoken phrase"), so the trailing pad and the silence wait count against it."""
        return self.t_end if self.speech_end is None else self.speech_end


@dataclass
class Result:
    t_start: float
    speaker: str
    hy: str
    en: str
    latency: float       # seconds from the end of the spoken phrase to print
    audio_s: float = 0.0 # segment duration and per-stage times, for --debug
    asr_s: float = 0.0
    mt_s: float = 0.0
    spk_s: float = 0.0   # speaker embedding + clustering (system audio only)
    spk_info: str = "" # how the speaker was chosen, similarities to each centroid
    line: int = 0        # id of the printed line; a continuation keeps the id of the line it extends
    merged: int = 1      # segments in this line: > 1 = the previous line(s), re-translated with this one
    mt_fast: bool = False  # translated by the fast fallback model (the pipeline was behind)
