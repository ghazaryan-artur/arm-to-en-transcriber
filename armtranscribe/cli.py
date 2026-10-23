"""Command-line entry point."""
import argparse
import os
import sys
import time
from pathlib import Path

# Local-only rule: models are downloaded once by scripts/download_models.py,
# at runtime nothing may go over the network (no cloud ASR/MT, no hub calls).
for _var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ.setdefault(_var, "1")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_ASR = "whisper-turbo-hy-mlx"
DEFAULT_MT = "nllb-1.3b-ct2"


def resolve_model(spec: str) -> str:
    """A model dir path, or the name of one under models/ (see scripts/download_models.py --list)."""
    path = Path(spec) if Path(spec).exists() else MODELS_DIR / spec
    if not path.is_dir():
        raise SystemExit(f"model not found: {spec} (run scripts/download_models.py {Path(spec).name})")
    return str(path)


def add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--asr", default=DEFAULT_ASR, help=f"ASR model name in models/ or path (default {DEFAULT_ASR})")
    p.add_argument("--mt", default=DEFAULT_MT, help=f"translation model name in models/ or path (default {DEFAULT_MT})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="armtranscribe", description="Live Armenian -> English transcription")
    p.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    p.add_argument("--mic", help="microphone device id or name")
    p.add_argument("--system", help="system-audio (loopback) device id or name, e.g. BlackHole")
    add_model_args(p)
    p.add_argument("--debug", action="store_true", help="print per-stage timings")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.list_devices:
        from armtranscribe.audio import list_devices
        list_devices()
        return

    import numpy as np

    from armtranscribe.asr import load_transcriber
    from armtranscribe.audio import Capture, default_input, resolve_device
    from armtranscribe.models import SAMPLE_RATE
    from armtranscribe.output import percentile_summary, render, render_timings
    from armtranscribe.pipeline import Pipeline
    from armtranscribe.translate import load_translator
    from armtranscribe.vad import Segmenter

    try:
        mic = resolve_device(args.mic) if args.mic else default_input()
    except ValueError as e:
        raise SystemExit(str(e))
    if args.system:
        print("--system is not supported yet (Phase 3), ignored", file=sys.stderr)

    print("Loading models...", flush=True)
    asr = load_transcriber(resolve_model(args.asr))
    mt = load_translator(resolve_model(args.mt))
    asr.transcribe(np.zeros(SAMPLE_RATE, np.float32))  # warm-up: the first call loads weights
    mt.translate("Բարև")

    latencies = []

    def emit(r):
        latencies.append(r.latency)
        print(render(r) + (render_timings(r) if args.debug else ""), flush=True)

    capture = Capture(mic)
    pipeline = Pipeline(asr, mt, emit)
    pipeline.add_source(capture, Segmenter("mic"))
    print(f"Mic: {capture.name}" + ("" if args.mic else " (system default)"))
    print("Listening, Ctrl+C to stop\n", flush=True)
    pipeline.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping, finishing queued segments (Ctrl+C again to quit now)...", flush=True)
        pipeline.stop()
    if args.debug:
        print(percentile_summary(latencies) + f", input overflows {capture.overflows}")
