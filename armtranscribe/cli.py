"""Command-line entry point."""
import argparse
import os
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
    raise SystemExit("Live pipeline not implemented yet; for files use: python -m armtranscribe.file <wav>")
