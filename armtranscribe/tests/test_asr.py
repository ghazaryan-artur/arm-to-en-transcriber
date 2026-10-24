import numpy as np

from app.asr import MLXWhisper, normalize


def test_normalize_strips_punctuation_and_case():
    assert normalize("Բարև, աշխարհ։") == "բարև աշխարհ"
    assert normalize("  Hello,   World! ") == "hello world"


def test_normalize_unifies_yev_ligature():
    assert normalize("Եւ") == normalize("և") == "և"


def test_normalize_colloquial_and():
    assert normalize("Արամն ու Լուսինեն") == normalize("Արամն և Լուսինեն")
    assert normalize("Ու հետո") == "և հետո"
    assert normalize("ուզում ենք") == "ուզում ենք"  # ու inside a word is untouched


def fake_whisper(segments):
    w = MLXWhisper.__new__(MLXWhisper)  # skip loading mlx_whisper
    w.path, w.no_speech_threshold = "unused", 0.6
    w._transcribe = lambda audio, **kw: {"segments": segments}
    return w


def test_whisper_drops_no_speech_and_hallucinations():
    w = fake_whisper([
        {"text": " Բարև ձեզ։", "no_speech_prob": 0.1},
        {"text": " noise", "no_speech_prob": 0.9},
        {"text": " Շնորհակալություն դիտելու համար։", "no_speech_prob": 0.2},
        {"text": " Ինչպե՞ս եք", "no_speech_prob": 0.3},
    ])
    assert w.transcribe(np.zeros(16000, np.float32)) == "Բարև ձեզ։ Ինչպե՞ս եք"


def test_whisper_all_filtered_is_empty():
    w = fake_whisper([{"text": " Thank you.", "no_speech_prob": 0.4}])
    assert w.transcribe(np.zeros(16000, np.float32)) == ""


def test_load_transcriber_picks_nemo_onnx_by_config(tmp_path, monkeypatch):
    import json
    import sys
    import types

    from app.asr import NemoONNX, load_transcriber

    calls = {}

    class FakeModel:
        def recognize(self, audio, sample_rate):
            calls["recognize"] = (audio.dtype, sample_rate)
            return " Բարև ձեզ։ "

    def load_model(model_type, path, quantization, providers):
        calls["load"] = (model_type, path, quantization, providers)
        return FakeModel()

    monkeypatch.setitem(sys.modules, "onnx_asr", types.SimpleNamespace(load_model=load_model))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "nemo-conformer-rnnt"}))
    asr = load_transcriber(str(tmp_path))
    assert isinstance(asr, NemoONNX)
    assert calls["load"] == ("nemo-conformer-rnnt", str(tmp_path), "int8", ["CPUExecutionProvider"])
    assert asr.transcribe(np.zeros(16000, np.float64)) == "Բարև ձեզ։"
    assert calls["recognize"] == (np.float32, 16000)
