"""Replay an audio file through the live Pipeline in real time (stand-in for the mic) and report latency.

    python tools/replay_live.py <audio> [seconds] [--asr name] [--mt name]

Feeds the first `seconds` (default 120) at wall-clock pace, prints every line with per-stage timings
and ends with latency p50/p95 (segment end -> printed), as `python -m app --debug` would.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "armtranscribe"))
import app.cli as cli  # noqa: E402  (first: sets offline mode)

import numpy as np  # noqa: E402

from app.models import SAMPLE_RATE  # noqa: E402


class RealtimeWav:
    """Capture-like source (start / stop / read) that releases audio as fast as a live device would."""

    def __init__(self, audio: np.ndarray, block: int = 480):  # 30 ms blocks
        self.audio, self.block, self.pos, self.t0 = audio, block, 0, 0.0

    def start(self) -> None:
        self.t0 = time.perf_counter()

    def stop(self) -> None:
        self.audio = self.audio[:self.pos]

    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        if self.pos >= len(self.audio):
            time.sleep(timeout)
            return None
        due = int((time.perf_counter() - self.t0) * SAMPLE_RATE)
        if due - self.pos < self.block:
            time.sleep(self.block / SAMPLE_RATE)
            return np.zeros(0, np.float32)
        out = self.audio[self.pos:due]
        self.pos = due
        return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="audio file")
    p.add_argument("seconds", nargs="?", type=float, default=120, help="replay only the first N seconds")
    cli.add_model_args(p)
    args = p.parse_args()

    from app.asr import load_transcriber
    from app.audio import load_audio
    from app.output import percentile_summary, render, render_timings
    from app.pipeline import Pipeline
    from app.translate import load_translator
    from app.vad import Segmenter

    audio = load_audio(args.path)[:int(args.seconds * SAMPLE_RATE)]
    asr = load_transcriber(cli.resolve_model(args.asr))
    mt = load_translator(cli.resolve_model(args.mt))
    asr.transcribe(np.zeros(SAMPLE_RATE, np.float32))  # warm-up, as the live CLI does
    mt.translate("Բարև")

    latencies = []

    def emit(r):
        latencies.append(r.latency)
        print(render(r) + render_timings(r), flush=True)

    src = RealtimeWav(audio)
    pipeline = Pipeline(asr, mt, emit)
    pipeline.add_source(src, Segmenter("mic"))
    pipeline.start()
    while src.pos < len(src.audio):
        time.sleep(0.2)
    time.sleep(1.0)  # let the trailing silence close the last segment
    pipeline.stop()
    print(f"ASR={args.asr} MT={args.mt}:", percentile_summary(latencies))


if __name__ == "__main__":
    main()
