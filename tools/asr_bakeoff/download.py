"""Fetch the ASR bake-off candidates into the HF cache (benchmark only, outside armtranscribe/)."""
import time
from huggingface_hub import snapshot_download

CFG = ["*.json", "*.txt", "*.model", "*.jinja", "merges.txt", "vocab.json"]
REPOS = {
    "Vana-Labs/stt-fastconformer-armenian": ["*.onnx", "*.txt", "*.json"],
    "infinitejoy/wav2vec2-large-xls-r-300m-armenian": CFG + ["pytorch_model.bin"],
    "ErikMkrtchyan/w2v-bert-2.0-armenian-CV20.0": CFG + ["model.safetensors"],
    "BuzzASR/armenian": CFG + ["model.safetensors"],
    "ditord/whisper-large-v3-turbo-armenian": CFG + ["model.safetensors"],
    "ArthurYeghinyan/whisper-hy-am-asr-v2": CFG + ["model.safetensors"],
    "arampacha/whisper-large-hy-2": CFG + ["pytorch_model.bin"],
    "mlx-community/gemma-4-E2B-it-4bit": CFG + ["*.safetensors"],
    "mlx-community/gemma-4-E4B-it-4bit": CFG + ["*.safetensors"],
    "facebook/seamless-m4t-v2-large": CFG + ["*.safetensors"],
}
for repo, pats in REPOS.items():
    t = time.time()
    path = snapshot_download(repo, allow_patterns=pats)
    print(f"{repo}: {time.time() - t:.0f} s -> {path}", flush=True)
