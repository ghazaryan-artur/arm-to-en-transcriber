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
