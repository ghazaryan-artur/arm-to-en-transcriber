"""Offline mode: transcribe and translate an audio file, printed in the live output format.

    python -m armtranscribe.file recording.wav [--asr ...] [--mt ...] [--debug]

Long files are split into speech segments with Silero VAD using the live segmenter's limits.
"""
import argparse
import time

from armtranscribe.cli import add_model_args, resolve_model  # first: sets offline mode

import numpy as np  # noqa: E402

from armtranscribe.models import SAMPLE_RATE, Result, Segment  # noqa: E402
from armtranscribe.output import render  # noqa: E402

MIN_SILENCE_MS = 600
MAX_SEGMENT_S = 8.0
MIN_SEGMENT_S = 0.4
SPEECH_PAD_MS = 200


def split_speech(audio: np.ndarray, source: str = "file") -> list[Segment]:
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    stamps = get_speech_timestamps(
        torch.from_numpy(audio), load_silero_vad(), sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=MIN_SILENCE_MS, max_speech_duration_s=MAX_SEGMENT_S,
        min_speech_duration_ms=int(MIN_SEGMENT_S * 1000), speech_pad_ms=SPEECH_PAD_MS,
    )
    return [Segment(source, s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE, audio[s["start"]:s["end"]])
            for s in stamps]


def main() -> None:
    p = argparse.ArgumentParser(prog="armtranscribe.file", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="audio file (wav/flac/...)")
    add_model_args(p)
    p.add_argument("--debug", action="store_true", help="print per-stage timings")
    args = p.parse_args()

    from armtranscribe.asr import load_transcriber
    from armtranscribe.audio import load_audio
    from armtranscribe.translate import load_translator

    asr = load_transcriber(resolve_model(args.asr))
    mt = load_translator(resolve_model(args.mt))
    segments = split_speech(load_audio(args.path))
    if not segments:
        print("no speech found")
        return
    for seg in segments:
        t0 = time.perf_counter()
        hy = asr.transcribe(seg.audio)
        t1 = time.perf_counter()
        if not hy:
            continue
        en = mt.translate(hy)
        t2 = time.perf_counter()
        print(render(Result(seg.t_start, "Speaker", hy, en, latency=t2 - t0)), flush=True)
        if args.debug:
            print(f"  [{seg.t_end - seg.t_start:.1f} s audio: asr {t1 - t0:.2f} s, mt {t2 - t1:.2f} s]\n")


if __name__ == "__main__":
    main()
