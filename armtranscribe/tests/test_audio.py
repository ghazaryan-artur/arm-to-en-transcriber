import sys
import types

import pytest

from app.audio import resolve_device

DEVICES = [
    {"name": "MacBook Air Microphone", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "MacBook Air Speakers", "max_input_channels": 0, "max_output_channels": 2},
    {"name": "BlackHole 2ch", "max_input_channels": 2, "max_output_channels": 2},
]


@pytest.fixture(autouse=True)
def fake_sounddevice(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(query_devices=lambda: DEVICES))


def test_resolve_by_id():
    assert resolve_device("2") == 2


def test_resolve_by_name_case_insensitive():
    assert resolve_device("blackhole") == 2


def test_resolve_respects_kind():
    assert resolve_device("MacBook Air", kind="output") == 1
    assert resolve_device("MacBook Air", kind="input") == 0


def test_resolve_rejects_wrong_kind_id():
    with pytest.raises(ValueError):
        resolve_device("1")  # speakers: no input channels


def test_resolve_unknown_and_ambiguous():
    with pytest.raises(ValueError):
        resolve_device("zoom")
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_device("ac")  # "MacBook..." and "BlackHole..." both match


class FakeSD:
    """Stands in for `sounddevice`: one input device that only opens at `ok_rates`."""

    class PortAudioError(Exception):
        pass

    def __init__(self, ok_rates, default=44100):
        self.ok_rates, self.default, self.opened = ok_rates, default, []
        sd = self

        class InputStream:
            def __init__(self, device, samplerate, **kw):
                sd.opened.append(samplerate)
                if samplerate not in sd.ok_rates:
                    raise sd.PortAudioError(f"Invalid sample rate {samplerate}")

        self.InputStream = InputStream

    def query_devices(self, device=None, kind=None):
        devices = [{"name": "Speakers", "max_input_channels": 0, "default_samplerate": 48000},
                   {"name": "USB Mic", "max_input_channels": 1, "default_samplerate": self.default}]
        return devices if device is None else devices[device]


def test_capture_falls_back_to_a_rate_the_device_accepts(monkeypatch):
    import sys

    from app.audio import Capture

    sd = FakeSD(ok_rates={48000})
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    c = Capture(1)
    assert c.rate == 48000 and sd.opened == [44100, 48000]


def test_capture_that_wont_open_says_why(monkeypatch):
    import sys

    import pytest

    from app.audio import Capture, input_devices

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSD(ok_rates=set()))
    with pytest.raises(ValueError, match="cannot open 'USB Mic'.*44100 Hz"):
        Capture(1)
    assert input_devices() == "  1: USB Mic"
