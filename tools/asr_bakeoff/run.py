"""Run one ASR candidate over data.pkl -> results/<label>.json (hyps per clip, seconds per piece).

    python run.py <label> <backend> <model>
backends: mlx (mlx_whisper dir), hfw (HF Whisper repo), ctc (HF CTC repo), mms (app MMS dir),
          seamless (HF repo), gemma (mlx-vlm repo), onnx (onnx-asr NeMo dir/repo)
"""
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
APP = HERE.parent.parent / "armtranscribe"
SR = 16000


def hf_device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def make(backend: str, model: str):
    if backend == "mlx":  # exactly the app's live ASR (incl. its hallucination filter)
        sys.path.insert(0, str(APP))
        from app.asr import MLXWhisper
        return MLXWhisper(model).transcribe
    if backend == "hfw":
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        dev = hf_device()
        proc = WhisperProcessor.from_pretrained(model)
        m = WhisperForConditionalGeneration.from_pretrained(model, dtype=torch.float16).to(dev).eval()
        # BuzzASR bakes language/task into its generation config and recommends these settings
        kw = ({"num_beams": 1, "no_repeat_ngram_size": 3, "repetition_penalty": 1.2} if "BuzzASR" in model
              else {"num_beams": 1, "language": "hy", "task": "transcribe"})
        if not hasattr(m.generation_config, "lang_to_id"):  # pre-2023 generation config (arampacha)
            tok = proc.tokenizer
            prompt = [tok.convert_tokens_to_ids(t) for t in
                      ("<|startoftranscript|>", "<|hy|>", "<|transcribe|>", "<|notimestamps|>")]
            kw = {"num_beams": 1, "decoder_input_ids": torch.tensor([prompt], device=dev)}
        def f(a):
            feats = proc(a, sampling_rate=SR, return_tensors="pt").input_features.to(dev, torch.float16)
            with torch.inference_mode():
                ids = m.generate(feats, max_new_tokens=220, **kw)
            return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return f
    if backend == "ctc":
        import torch
        from transformers import AutoModelForCTC, AutoProcessor
        dev = hf_device()
        proc = AutoProcessor.from_pretrained(model)
        m = AutoModelForCTC.from_pretrained(model).to(dev).eval()
        # greedy CTC by hand: the blank is the model's pad_token_id, which some repos' tokenizers
        # disagree with (infinitejoy: config blank 47 = "։" in vocab, tokenizer pad = 49)
        blank = m.config.pad_token_id
        vocab = {i: t for t, i in proc.tokenizer.get_vocab().items()}
        delim = proc.tokenizer.word_delimiter_token
        def f(a):
            inputs = proc(a, sampling_rate=SR, return_tensors="pt").to(dev)
            with torch.inference_mode():
                ids = m(**inputs).logits.argmax(-1)[0].tolist()
            out = [vocab.get(i, "") for k, i in enumerate(ids) if i != blank and (k == 0 or i != ids[k - 1])]
            text = "".join(" " if t == delim else t for t in out if not (t.startswith("[") or t.startswith("<")))
            return " ".join(text.split())
        return f
    if backend == "mms":
        sys.path.insert(0, str(APP))
        from app.asr import MMSCTC
        return MMSCTC(model).transcribe
    if backend == "seamless":
        import torch
        from transformers import (SeamlessM4TFeatureExtractor, SeamlessM4TProcessor, SeamlessM4TTokenizer,
                                  SeamlessM4Tv2ForSpeechToText)
        dev = hf_device()
        # explicit slow (sentencepiece) tokenizer: AutoProcessor's fast conversion fails on this repo
        proc = SeamlessM4TProcessor(SeamlessM4TFeatureExtractor.from_pretrained(model),
                                    SeamlessM4TTokenizer.from_pretrained(model))
        m = SeamlessM4Tv2ForSpeechToText.from_pretrained(model, dtype=torch.float16).to(dev).eval()
        def f(a):
            inputs = proc(audio=a, sampling_rate=SR, return_tensors="pt").to(dev)
            inputs["input_features"] = inputs["input_features"].half()
            with torch.inference_mode():
                ids = m.generate(**inputs, tgt_lang="hye", num_beams=1, max_new_tokens=220)
            return proc.decode(ids[0], skip_special_tokens=True).strip()
        return f
    if backend == "gemma":
        import soundfile as sf
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        m, proc = load(model)
        prompt = apply_chat_template(
            proc, m.config,
            "Transcribe this audio verbatim. The speech is in Eastern Armenian; write it in Armenian script. "
            "Output only the transcription.", num_audios=1)
        tmp = tempfile.mkdtemp()
        def f(a):
            path = os.path.join(tmp, "piece.wav")
            sf.write(path, a, SR)
            out = generate(m, proc, prompt, audio=[path], max_tokens=220, temperature=0.0, verbose=False)
            return (out.text if hasattr(out, "text") else str(out)).strip()
        return f
    if backend == "onnx":
        import onnx_asr
        from huggingface_hub.constants import HF_HUB_CACHE
        path = model if Path(model).exists() else \
            next(Path(HF_HUB_CACHE).glob(f"models--{model.replace('/', '--')}/snapshots/*"))
        m = onnx_asr.load_model("nemo-conformer-rnnt", path, quantization="int8")
        return lambda a: m.recognize(a, sample_rate=SR).strip()
    raise SystemExit(f"unknown backend {backend}")


def main() -> None:
    label, backend, model = sys.argv[1:4]
    data = pickle.load(open(HERE / "data.pkl", "rb"))
    t = time.perf_counter()
    f = make(backend, model)
    f(data["fleurs"][0][1][0])  # warm-up
    load_s = time.perf_counter() - t
    out = {"label": label, "backend": backend, "model": model, "load_s": load_s, "sets": {}}
    for name, clips in data.items():
        hyps, secs, durs = [], [], []
        for _, pieces, _ in clips:
            texts = []
            for a in pieces:
                t = time.perf_counter()
                texts.append(f(a.astype(np.float32)))
                secs.append(time.perf_counter() - t)
                durs.append(len(a) / SR)
            hyps.append(" ".join(x for x in texts if x))
        out["sets"][name] = {"hyps": hyps, "secs": secs, "durs": durs}
        print(f"{label} {name}: {sum(secs):.0f} s for {sum(durs):.0f} s audio", file=sys.stderr, flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "results" / f"{label}.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
