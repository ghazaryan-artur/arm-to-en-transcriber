"""Armenian speech recognition backends."""
from typing import Protocol

import numpy as np


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...
