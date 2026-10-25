"""Audio device listing and capture (resampled to 16 kHz mono)."""
import queue

import numpy as np

from app.models import SAMPLE_RATE


class Capture:
    """One input device. The PortAudio callback only queues raw blocks; read() resamples them.

    Opened at the device's native rate (PortAudio's own resampling is not trusted) and mixed to
    mono; read() is meant to be called from a single consumer thread.
    """

    FALLBACK_RATES = (48000, 44100, 16000)

    def __init__(self, device: int):
        import sounddevice as sd
        import soxr

        info = sd.query_devices(device)
        self.name = info["name"]
        self.overflows = 0
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        channels = min(info["max_input_channels"], 2)
        # the device's default rate, else common ones (some devices report a rate they won't open at)
        rates = dict.fromkeys([int(info["default_samplerate"]), *self.FALLBACK_RATES])
        errors = []
        for rate in rates:
            try:
                self._stream = sd.InputStream(device=device, samplerate=rate, dtype="float32",
                                              channels=channels, callback=self._callback)
                break
            except sd.PortAudioError as e:
                errors.append(f"{rate} Hz: {e}")
        else:
            raise ValueError(f"cannot open {self.name!r}: " + "; ".join(errors))
        self.rate = rate
        self._resampler = soxr.ResampleStream(self.rate, SAMPLE_RATE, 1, dtype="float32")

    def _callback(self, indata, frames, time, status) -> None:
        if status.input_overflow:
            self.overflows += 1
        self._q.put(indata.copy())

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()

    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        """Next block as float32 mono @ SAMPLE_RATE (may be empty), or None if nothing arrived."""
        try:
            block = self._q.get(timeout=timeout)
        except queue.Empty:
            return None
        return self._resampler.resample_chunk(block.mean(axis=1))


class Recorder:
    """Saves a session for later evaluation and fine-tuning, into a new timestamped dir:

    <source>.wav         the whole 16 kHz stream of each source (to re-run the VAD with other settings)
    NNNN_<source>.wav    each transcribed segment
    segments.tsv         `file<TAB>hy<TAB>en` per segment: a benchmark manifest once hy/en are corrected
                         (en is the translation of the whole printed line when the segment continued one)
    """

    def __init__(self, root: str):
        from datetime import datetime
        from pathlib import Path

        self.dir = Path(root) / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True)
        self._streams = {}
        self._n = 0

    def write_stream(self, source: str, block: np.ndarray) -> None:
        import soundfile as sf

        if source not in self._streams:
            self._streams[source] = sf.SoundFile(self.dir / f"{source}.wav", "w", SAMPLE_RATE, 1, "PCM_16")
        self._streams[source].write(block)

    def write_segment(self, seg, hy: str, en: str) -> None:
        import soundfile as sf

        self._n += 1
        name = f"{self._n:04d}_{seg.source}.wav"
        sf.write(self.dir / name, seg.audio, SAMPLE_RATE, subtype="PCM_16")
        clean = lambda s: " ".join(s.split())  # no tabs/newlines inside a TSV field
        with open(self.dir / "segments.tsv", "a", encoding="utf-8") as f:
            f.write(f"{name}\t{clean(hy)}\t{clean(en)}\n")

    def close(self) -> None:
        for s in self._streams.values():
            s.close()


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


def input_devices() -> str:
    """Input devices as `  id: name`, for error messages."""
    import sounddevice as sd

    return "\n".join(f"  {i}: {d['name']}" for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0)


def default_input() -> int:
    """The system default input device (System Settings → Sound → Input)."""
    import sounddevice as sd

    try:
        return sd.query_devices(kind="input")["index"]
    except sd.PortAudioError as e:
        raise ValueError("no default input device (see --list-devices)") from e


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
