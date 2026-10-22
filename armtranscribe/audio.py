"""Audio device listing and capture (resampled to 16 kHz mono)."""


def list_devices() -> None:
    import sounddevice as sd
    print(sd.query_devices())
