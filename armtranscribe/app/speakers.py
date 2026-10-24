"""Speaker embeddings + online clustering for the system-audio side."""
from typing import Protocol

import numpy as np


class SpeakerTracker(Protocol):
    def assign(self, audio: np.ndarray) -> str: ...
