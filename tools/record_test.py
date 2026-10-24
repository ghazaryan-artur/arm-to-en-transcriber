"""Record a few seconds from input devices at once to check audio routing.

    python tools/record_test.py                          # default mic only
    python tools/record_test.py --system BlackHole       # mic + system audio
    python tools/record_test.py --mic 0 --system 2 -s 8

Play a YouTube video while it records, then listen to the WAVs in the workspace's recordings/.
Both streams run concurrently, like in the real pipeline.
"""
import argparse
import queue
import sys
import time
import wave
from contextlib import ExitStack
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE / "armtranscribe"))
from app.audio import default_input, list_devices, resolve_device  # noqa: E402

SILENT_DBFS = -60.0


def dbfs(x: float) -> float:
    return 20 * np.log10(x) if x > 0 else float("-inf")


def save_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(audio.shape[1])
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())


def record(devices: dict[str, int], seconds: float, out_dir: Path) -> bool:
    import sounddevice as sd

    out_dir.mkdir(parents=True, exist_ok=True)
    queues = {label: queue.Queue() for label in devices}
    rates = {}

    def make_callback(q: queue.Queue):
        def callback(indata, frames, time_info, status):
            if status:
                print(f"  stream status: {status}", file=sys.stderr)
            q.put(indata.copy())  # never block in a PortAudio callback
        return callback

    with ExitStack() as stack:
        for label, idx in devices.items():
            info = sd.query_devices(idx)
            rates[label] = int(info["default_samplerate"])
            stack.enter_context(sd.InputStream(
                device=idx,
                channels=min(2, info["max_input_channels"]),
                samplerate=rates[label],
                callback=make_callback(queues[label]),
            ))
        print(f"Recording {seconds:g} s — make some noise / play audio now...")
        time.sleep(seconds)

    ok = True
    for label, idx in devices.items():
        chunks = []
        while not queues[label].empty():
            chunks.append(queues[label].get())
        if not chunks:
            print(f"[{label}] device {idx}: no audio received")
            ok = False
            continue
        audio = np.concatenate(chunks)
        path = out_dir / f"{label}.wav"
        save_wav(path, audio, rates[label])
        peak = dbfs(float(np.abs(audio).max()))
        rms = dbfs(float(np.sqrt(np.mean(audio ** 2))))
        verdict = "OK" if peak > SILENT_DBFS else "SILENT — check routing / permissions"
        ok &= peak > SILENT_DBFS
        print(f"[{label}] device {idx} @ {rates[label]} Hz, {audio.shape[1]} ch: "
              f"peak {peak:.1f} dBFS, rms {rms:.1f} dBFS -> {verdict}\n  saved {path}")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mic", help="microphone device id or name (default: system default input)")
    p.add_argument("--system", help="loopback device id or name, e.g. BlackHole")
    p.add_argument("-s", "--seconds", type=float, default=5.0)
    p.add_argument("--out-dir", type=Path, default=WORKSPACE / "recordings")
    args = p.parse_args()

    list_devices()
    print()

    devices = {}
    try:
        devices["mic"] = resolve_device(args.mic) if args.mic else default_input()
        if args.system:
            devices["system"] = resolve_device(args.system)
    except ValueError as e:
        raise SystemExit(f"error: {e}")

    raise SystemExit(0 if record(devices, args.seconds, args.out_dir) else 1)


if __name__ == "__main__":
    main()
