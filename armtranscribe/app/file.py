"""Offline mode: transcribe and translate an audio file, printed in the live output format.

    python -m app.file recording.wav [--asr ...] [--mt ...] [--speakers [--spk-threshold 0.45]] [--debug]

Long files are cut into speech segments by the live VAD segmenter, fed as if streamed,
so the segmentation matches the live mode.
"""
import argparse
import time

from app.cli import add_model_args, add_speaker_args, load_speaker_tracker, resolve_model  # first: offline mode

import numpy as np  # noqa: E402

from app.models import Result, Segment  # noqa: E402
from app.output import render, render_timings  # noqa: E402

CHUNK = 1600  # 100 ms, roughly what the live capture delivers per read


def split_speech(audio: np.ndarray, source: str = "file") -> list[Segment]:
    from app.vad import Segmenter

    seg = Segmenter(source)
    segments = []
    for i in range(0, len(audio), CHUNK):
        segments += seg.feed(audio[i:i + CHUNK])
    return segments + seg.flush()


def main() -> None:
    p = argparse.ArgumentParser(prog="app.file", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="audio file (wav/flac/...)")
    add_model_args(p)
    add_speaker_args(p)
    p.add_argument("--speakers", action="store_true",
                   help="label speakers as for system audio (Speaker N) instead of one 'Speaker'")
    p.add_argument("--debug", action="store_true", help="print per-stage timings")
    args = p.parse_args()

    from app.asr import load_transcriber
    from app.audio import load_audio
    from app.translate import load_translator

    asr = load_transcriber(resolve_model(args.asr))
    mt = load_translator(resolve_model(args.mt))
    tracker = load_speaker_tracker(args) if args.speakers else None
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
        speaker, spk_s, spk_info = "Speaker", 0.0, ""
        if tracker:
            speaker = tracker.assign(seg.audio)
            spk_s, spk_info = time.perf_counter() - t1, str(tracker.last)
        t1s = time.perf_counter()
        en = mt.translate(hy)
        t2 = time.perf_counter()
        r = Result(seg.t_start, speaker, hy, en, latency=t2 - t0, audio_s=seg.t_end - seg.t_start,
                   asr_s=t1 - t0, mt_s=t2 - t1s, spk_s=spk_s, spk_info=spk_info)
        print(render(r) + (render_timings(r) if args.debug else ""), flush=True)


if __name__ == "__main__":
    main()
