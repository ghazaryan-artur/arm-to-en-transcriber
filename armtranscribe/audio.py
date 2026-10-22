"""Audio device listing and capture (resampled to 16 kHz mono)."""


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
