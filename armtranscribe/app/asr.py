"""Armenian speech recognition backends."""
import json
import re
from pathlib import Path
from typing import Protocol

import numpy as np

from app.models import SAMPLE_RATE

# Stock phrases Whisper emits on silence, music or noise (compared after normalize()).
HALLUCINATIONS = {
    "շնորհակալություն", "շնորհակալություն դիտելու համար", "շնորհակալ եմ դիտելու համար",
    "բաժանորդագրվեք մեր ալիքին", "բաժանորդագրվեք", "thank you", "thanks for watching",
    "субтитры сделал dimatorzok", "продолжение следует",
}


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, unify the և ligature; used for hallucination filtering and WER.

    The word "and" is written և (formal) or ու (colloquial, more common in speech); both count as և.
    """
    words = re.sub(r"[^\w\s]", " ", text.lower().replace("եւ", "և")).split()
    return " ".join("և" if w == "ու" else w for w in words)


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...


class MLXWhisper:
    """Whisper on the Apple GPU via mlx-whisper; `path` is a converted model dir in models/."""

    def __init__(self, path: str, no_speech_threshold: float = 0.6):
        import mlx_whisper

        self._transcribe = mlx_whisper.transcribe
        self.path = path
        self.no_speech_threshold = no_speech_threshold

    def transcribe(self, audio: np.ndarray) -> str:
        r = self._transcribe(
            audio.astype(np.float32), path_or_hf_repo=self.path, language="hy", task="transcribe",
            temperature=0.0, condition_on_previous_text=False, verbose=None,
        )
        texts = [s["text"].strip() for s in r["segments"] if s["no_speech_prob"] < self.no_speech_threshold]
        return " ".join(t for t in texts if t and normalize(t) not in HALLUCINATIONS)


class MMSCTC:
    """Meta MMS-1B-all with the Armenian adapter (CTC: fast, but no punctuation or casing)."""

    def __init__(self, path: str, lang: str = "hye"):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(path, target_lang=lang)
        self.model = Wav2Vec2ForCTC.from_pretrained(path, target_lang=lang, ignore_mismatched_sizes=True)
        self.model.to(self.device).eval()

    def transcribe(self, audio: np.ndarray) -> str:
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
        return self.processor.decode(logits[0].argmax(dim=-1)).strip()


class NemoONNX:
    """NVIDIA NeMo FastConformer exported to ONNX, run by onnx-asr on the CPU (int8).

    The Armenian model writes punctuation and capitalisation itself and, being a transducer,
    doesn't loop or hallucinate on silence the way Whisper does.
    """

    def __init__(self, path: str, model_type: str):
        import onnx_asr

        # CPU only: ~4x faster than onnxruntime's default CoreML partitioning for this graph (0.04 vs
        # 0.15 s per segment), loads in 0.4 s, and rules out any non-local execution provider
        self.model = onnx_asr.load_model(model_type, path, quantization="int8", providers=["CPUExecutionProvider"])

    def transcribe(self, audio: np.ndarray) -> str:
        return self.model.recognize(audio.astype(np.float32), sample_rate=SAMPLE_RATE).strip()


def load_transcriber(path: str) -> Transcriber:
    """Pick the backend from the model dir's config.json."""
    config = json.loads((Path(path) / "config.json").read_text())
    if config.get("model_type", "").startswith("nemo-"):
        return NemoONNX(path, config["model_type"])
    if "n_mels" in config:
        return MLXWhisper(path)
    if config.get("model_type") == "wav2vec2":
        return MMSCTC(path)
    raise ValueError(f"{path}: unrecognised ASR model")
