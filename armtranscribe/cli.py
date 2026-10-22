"""Command-line entry point."""
import argparse
import os

# Local-only rule: models are downloaded once by scripts/download_models.py,
# at runtime nothing may go over the network (no cloud ASR/MT, no hub calls).
for _var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ.setdefault(_var, "1")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="armtranscribe", description="Live Armenian -> English transcription")
    p.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    p.add_argument("--mic", help="microphone device id or name")
    p.add_argument("--system", help="system-audio (loopback) device id or name, e.g. BlackHole")
    p.add_argument("--asr", default="mlx-community/whisper-large-v3-mlx", help="ASR model")
    p.add_argument("--mt", default="models/nllb-600m-ct2", help="translation model dir")
    p.add_argument("--debug", action="store_true", help="print per-stage timings")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.list_devices:
        from armtranscribe.audio import list_devices
        list_devices()
        return
    raise SystemExit("Pipeline not implemented yet")
