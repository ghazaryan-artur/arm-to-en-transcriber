"""Audio device listing and capture (resampled to 16 kHz mono)."""
import numpy as np

from armtranscribe.models import SAMPLE_RATE


def to_mono_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    """(frames,) or (frames, channels) at `rate` -> float32 mono @ SAMPLE_RATE."""
    import soxr

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = soxr.resample(audio, rate, SAMPLE_RATE).astype(np.float32)
    return audio


def load_audio(path: str) -> np.ndarray:
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    return to_mono_16k(audio, rate)


def list_devices() -> None:
    import sounddevice as sd
    print(sd.query_devices())


def resolve_device(spec: str, kind: str = "input") -> int:
    """Map a device id or a case-insensitive name substring to a sounddevice index."""
    import sounddevice as sd

    channels = "max_input_channels" if kind == "input" else "max_output_channels"
    devices = sd.query_devices()
    if spec.isdigit():
        idx = int(spec)
        if not 0 <= idx < len(devices) or devices[idx][channels] == 0:
            raise ValueError(f"device {idx} is not an {kind} device (see --list-devices)")
        return idx
    matches = [i for i, d in enumerate(devices) if spec.lower() in d["name"].lower() and d[channels] > 0]
    if not matches:
        raise ValueError(f"no {kind} device matches {spec!r} (see --list-devices)")
    if len(matches) > 1:
        names = ", ".join(f"{i}:{devices[i]['name']}" for i in matches)
        raise ValueError(f"{spec!r} is ambiguous: {names}")
    return matches[0]
