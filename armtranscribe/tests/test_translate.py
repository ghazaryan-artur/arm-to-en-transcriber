import types

import pytest

from app.cli import resolve_model
from app.translate import CT2Translator


class FakeSP:
    def encode(self, text, out_type=str):
        return ["▁" + w for w in text.split()]

    def decode(self, pieces):
        return "".join(pieces).replace("▁", " ").strip()


def fake_translator(src_prefix, tgt_prefix, output):
    t = CT2Translator.__new__(CT2Translator)  # skip loading ctranslate2
    t.src_sp = t.tgt_sp = FakeSP()
    t.src_prefix, t.tgt_prefix, t.beam_size = src_prefix, tgt_prefix, 4
    t.calls = []

    def translate_batch(batch, target_prefix=None, **kw):
        t.calls.append((batch, target_prefix))
        return [types.SimpleNamespace(hypotheses=[output])]

    t.model = types.SimpleNamespace(translate_batch=translate_batch)
    return t


def test_nllb_language_tokens():
    t = fake_translator(["hye_Armn"], ["eng_Latn"], ["eng_Latn", "▁Hello", "▁there"])
    assert t.translate("Բարև ձեզ") == "Hello there"
    assert t.calls == [([["hye_Armn", "▁Բարև", "▁ձեզ", "</s>"]], [["eng_Latn"]])]


def test_marian_no_prefix():
    t = fake_translator([], [], ["▁Hi"])
    assert t.translate("Բարև") == "Hi"
    assert t.calls == [([["▁Բարև", "</s>"]], None)]


def test_empty_input_skips_model():
    t = fake_translator(["hye_Armn"], ["eng_Latn"], [])
    assert t.translate("   ") == ""
    assert t.calls == []


def test_resolve_model(tmp_path, monkeypatch):
    monkeypatch.setattr("app.cli.MODELS_DIR", tmp_path)
    (tmp_path / "nllb").mkdir()
    assert resolve_model("nllb") == str(tmp_path / "nllb")
    assert resolve_model(str(tmp_path / "nllb")) == str(tmp_path / "nllb")
    with pytest.raises(SystemExit, match="download_models"):
        resolve_model("missing")


def scripted_translator(outputs):
    """NLLB-like fake: translate_batch returns the next scripted output after the forced prefix."""
    t = fake_translator(["hye_Armn"], ["eng_Latn"], [])
    t.calls, outputs = [], list(outputs)

    def translate_batch(batch, target_prefix=None, **kw):
        t.calls.append((batch, target_prefix, kw))
        return [types.SimpleNamespace(hypotheses=[target_prefix[0] + outputs.pop(0)])]

    t.model = types.SimpleNamespace(translate_batch=translate_batch)
    return t


def test_continuation_keeps_shown_part_and_drops_its_full_stop():
    t = scripted_translator([["▁will", "▁be", "▁known."]])
    assert t.translate_continuation("ա բ։", "Our product.", "գ դ") == "Our product will be known."
    (batch, prefix, kw), = t.calls
    assert batch == [["hye_Armn", "▁ա", "▁բ։", "▁գ", "▁դ", "</s>"]]
    assert prefix == [["eng_Latn", "▁Our", "▁product"]]
    assert kw["no_repeat_ngram_size"] == 4 and kw["max_decoding_length"] == int(1.5 * 6) + 10 + 2


def test_continuation_falls_back_when_model_stops_early():
    t = scripted_translator([["."], ["▁Hello"]])  # the model ends right after the forced part
    assert t.translate_continuation("ա։", "Yes.", "բարև ձեզ") == "Yes. Hello"
    assert t.calls[1][0] == [["hye_Armn", "▁բարև", "▁ձեզ", "</s>"]]


def test_decoding_is_guarded_against_loops():
    t = fake_translator(["hye_Armn"], ["eng_Latn"], ["eng_Latn", "▁Hi"])
    seen = {}
    inner = t.model.translate_batch
    t.model.translate_batch = lambda batch, **kw: seen.update(kw) or inner(batch, **kw)
    t.translate("Բարև ձեզ")
    assert seen["no_repeat_ngram_size"] == 4 and seen["max_decoding_length"] == int(1.5 * 4) + 10
