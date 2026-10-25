"""Armenian -> English translation backends."""
import re
from pathlib import Path
from typing import Protocol


class Translator(Protocol):
    def translate(self, hy: str) -> str: ...


class CT2Translator:
    """CTranslate2 seq2seq model with SentencePiece tokenizers: NLLB-200 or Marian (opus-mt-hy-en).

    Decoding is guarded against NLLB's repetition loops ("we're going to put up, we're going to
    put up, ..." up to the length limit, 8 s of MT): no 4-gram may repeat, and the output is capped
    at 1.5x the source length + 10 tokens.
    """

    def __init__(self, path: str, beam_size: int = 2, threads: int = 6):
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

    def _decode(self, hy: str, forced: list[str]) -> list[str]:
        """Target tokens after the language token, the first ones forced to `forced`."""
        source = self.src_prefix + self.src_sp.encode(hy, out_type=str) + ["</s>"]
        prefix = self.tgt_prefix + forced
        r = self.model.translate_batch(
            [source], target_prefix=[prefix] if prefix else None,
            beam_size=self.beam_size, no_repeat_ngram_size=4,
            max_decoding_length=min(256, int(1.5 * len(source)) + 10 + len(forced)),
        )
        return r[0].hypotheses[0][len(self.tgt_prefix):]

    def translate(self, hy: str) -> str:
        if not hy.strip():
            return ""
        return self.tgt_sp.decode(self._decode(hy, []))

    def translate_continuation(self, prev_hy: str, prev_en: str, hy: str) -> tuple[str, str]:
        """English for `hy`, which follows `prev_hy`, already shown translated as `prev_en`.

        Returns (`prev_en` as it should now read, English for `hy`). `hy` usually continues a
        sentence cut at a pause, so `prev_hy hy` is translated as one text with the output forced
        to start with `prev_en` minus its final full stop: the part already read stays, the new
        part is translated in context. NLLB tends to drop the second of two sentences; if the
        continuation has far fewer letters than `hy`, `hy` is translated on its own instead.
        """
        head = prev_en.rstrip().rstrip(".։").rstrip()
        if not hy.strip():
            return prev_en, ""
        if not head:
            return "", self.translate(f"{prev_hy} {hy}")
        forced = self.tgt_sp.encode(head, out_type=str)
        added = self.tgt_sp.decode(self._decode(f"{prev_hy} {hy}", forced)[len(forced):])
        if len(re.findall(r"\w", added)) < 0.3 * len(re.findall(r"\w", hy)):  # ~ as many letters in EN as in HY
            return prev_en.rstrip(), self.translate(hy)
        return head, added


def load_translator(path: str) -> Translator:
    return CT2Translator(path)
