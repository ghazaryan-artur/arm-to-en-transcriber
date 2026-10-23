import types

import pytest

from armtranscribe.cli import resolve_model
from armtranscribe.translate import CT2Translator


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
    monkeypatch.setattr("armtranscribe.cli.MODELS_DIR", tmp_path)
    (tmp_path / "nllb").mkdir()
    assert resolve_model("nllb") == str(tmp_path / "nllb")
    assert resolve_model(str(tmp_path / "nllb")) == str(tmp_path / "nllb")
    with pytest.raises(SystemExit, match="download_models"):
        resolve_model("missing")
