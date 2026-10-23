"""Armenian -> English translation backends."""
from pathlib import Path
from typing import Protocol


class Translator(Protocol):
    def translate(self, hy: str) -> str: ...


class CT2Translator:
    """CTranslate2 seq2seq model with SentencePiece tokenizers: NLLB-200 or Marian (opus-mt-hy-en)."""

    def __init__(self, path: str, beam_size: int = 2, threads: int = 8):
        import ctranslate2
        import sentencepiece as spm

        p = Path(path)
        self.model = ctranslate2.Translator(str(p), device="cpu", compute_type="int8", intra_threads=threads)
        self.beam_size = beam_size
        if (p / "sentencepiece.bpe.model").exists():  # NLLB: one shared vocab + language tokens
            self.src_sp = self.tgt_sp = spm.SentencePieceProcessor(model_file=str(p / "sentencepiece.bpe.model"))
            self.src_prefix, self.tgt_prefix = ["hye_Armn"], ["eng_Latn"]
        else:  # Marian: language pair fixed by the model
            self.src_sp = spm.SentencePieceProcessor(model_file=str(p / "source.spm"))
            self.tgt_sp = spm.SentencePieceProcessor(model_file=str(p / "target.spm"))
            self.src_prefix, self.tgt_prefix = [], []

    def translate(self, hy: str) -> str:
        if not hy.strip():
            return ""
        source = self.src_prefix + self.src_sp.encode(hy, out_type=str) + ["</s>"]
        r = self.model.translate_batch(
            [source], target_prefix=[self.tgt_prefix] if self.tgt_prefix else None,
            beam_size=self.beam_size, max_decoding_length=256,
        )
        return self.tgt_sp.decode(r[0].hypotheses[0][len(self.tgt_prefix):])


def load_translator(path: str) -> Translator:
    return CT2Translator(path)
