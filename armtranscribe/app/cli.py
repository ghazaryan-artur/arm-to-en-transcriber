"""Command-line entry point."""
import argparse
import os
import time
from pathlib import Path

# Local-only rule: models are downloaded once by scripts/download_models.py,
# at runtime nothing may go over the network (no cloud ASR/MT, no hub calls).
for _var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ.setdefault(_var, "1")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_ASR = "fastconformer-hy-onnx"
DEFAULT_MT = "nllb-1.3b-ct2"
DEFAULT_SPK = "ecapa-voxceleb"


def resolve_model(spec: str) -> str:
    """A model dir path, or the name of one under models/ (see scripts/download_models.py --list)."""
    path = Path(spec) if Path(spec).exists() else MODELS_DIR / spec
    if not path.is_dir():
        raise SystemExit(f"model not found: {spec} (run scripts/download_models.py {Path(spec).name})")
    return str(path)


def add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--asr", default=DEFAULT_ASR, help=f"ASR model name in models/ or path (default {DEFAULT_ASR})")
    p.add_argument("--mt", default=DEFAULT_MT, help=f"translation model name in models/ or path (default {DEFAULT_MT})")


def add_speaker_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--speaker-model", default=DEFAULT_SPK,
                   help=f"speaker embedding model name in models/ or path (default {DEFAULT_SPK})")
    p.add_argument("--spk-threshold", type=float, default=0.45,
                   help="cosine similarity to join an existing speaker; lower = fewer speakers (default 0.45)")


def load_speaker_tracker(args):
    """ECAPA embedder + online clustering, warmed up (the first call is slow)."""
    import numpy as np

    from app.models import SAMPLE_RATE
    from app.speakers import OnlineSpeakerTracker, ecapa_embedder

    embed = ecapa_embedder(resolve_model(args.speaker_model))
    embed(np.zeros(2 * SAMPLE_RATE, np.float32))  # warm-up straight on the embedder: creates no speaker
    return OnlineSpeakerTracker(embed, threshold=args.spk_threshold)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="app", description="Live Armenian -> English transcription")
    p.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    p.add_argument("--mic", help="microphone device id or name")
    p.add_argument("--system", help="system-audio (loopback) device id or name, e.g. BlackHole")
    add_model_args(p)
    add_speaker_args(p)
    p.add_argument("--debug", action="store_true", help="print per-stage timings")
    p.add_argument("--save-audio", metavar="DIR",
                   help="save the session (streams, segments, segments.tsv) into a new subdir of DIR")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.list_devices:
        from app.audio import list_devices
        list_devices()
        return

    import numpy as np

    from app.asr import load_transcriber
    from app.audio import Capture, Recorder, default_input, resolve_device
    from app.models import SAMPLE_RATE
    from app.output import percentile_summary, render, render_timings
    from app.pipeline import Pipeline
    from app.translate import load_translator
    from app.vad import Segmenter

    try:
        mic = resolve_device(args.mic) if args.mic else default_input()
        system = resolve_device(args.system) if args.system else None
    except ValueError as e:
        raise SystemExit(str(e))
    if system == mic:
        raise SystemExit("--system must be a different device than the microphone (see --list-devices)")

    print("Loading models...", flush=True)
    asr = load_transcriber(resolve_model(args.asr))
    mt = load_translator(resolve_model(args.mt))
    asr.transcribe(np.zeros(SAMPLE_RATE, np.float32))  # warm-up: the first call loads weights
    mt.translate("Բարև")
    speakers = load_speaker_tracker(args) if system is not None else None

    latencies = []

    def emit(r):
        latencies.append(r.latency)
        print(render(r) + (render_timings(r) if args.debug else ""), flush=True)

    captures = {"mic": Capture(mic)}
    if system is not None:
        captures["system"] = Capture(system)
    recorder = Recorder(args.save_audio) if args.save_audio else None
    pipeline = Pipeline(asr, mt, emit, recorder, speakers, debug=args.debug)
    for source, capture in captures.items():
        pipeline.add_source(capture, Segmenter(source))  # one VAD per stream: Silero keeps state
    print(f"Mic: {captures['mic'].name}" + ("" if args.mic else " (system default)"))
    if "system" in captures:
        print(f"System audio: {captures['system'].name}"
              + f" (speakers: threshold {args.spk_threshold})")
    if recorder:
        print(f"Saving to {recorder.dir}")
    print("Listening, Ctrl+C to stop\n", flush=True)
    pipeline.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping, finishing queued segments (Ctrl+C again to quit now)...", flush=True)
        pipeline.stop()
    if args.debug:
        overflows = ", ".join(f"{s} {c.overflows}" for s, c in captures.items())
        print(percentile_summary(latencies) + f", input overflows: {overflows}")
