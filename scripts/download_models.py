"""One-time setup: download model weights into models/ (the only step that uses the network).

    python scripts/download_models.py                 # default ASR + MT
    python scripts/download_models.py --bench         # every benchmark candidate (~20 GB download)
    python scripts/download_models.py nllb-1.3b-ct2   # specific models (see --list)

Source checkpoints go to the Hugging Face cache; converted models land in models/<name>/.
After this the app runs fully offline.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# name -> (how, HF repo, extra)
MODELS = {
    # ASR
    "whisper-large-v3-mlx": ("snapshot", "mlx-community/whisper-large-v3-mlx", None),
    "whisper-large-v3-turbo-mlx": ("snapshot", "mlx-community/whisper-large-v3-turbo", None),
    "whisper-turbo-hy-mlx": ("whisper-to-mlx", "Chillarmo/whisper-large-v3-turbo-armenian", None),
    "mms-1b-all": ("snapshot", "facebook/mms-1b-all", [
        "config.json", "preprocessor_config.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "model.safetensors", "adapter.hye.safetensors",
    ]),
    # MT
    "nllb-600m-ct2": ("ct2", "facebook/nllb-200-distilled-600M", ["sentencepiece.bpe.model"]),
    "nllb-1.3b-ct2": ("ct2", "facebook/nllb-200-distilled-1.3B", ["sentencepiece.bpe.model"]),
    "opus-mt-hy-en-ct2": ("ct2", "Helsinki-NLP/opus-mt-hy-en", ["source.spm", "target.spm"]),
}
DEFAULT = ["whisper-turbo-hy-mlx", "nllb-1.3b-ct2"]

# HF transformers Whisper parameter names -> mlx_whisper (OpenAI-style) names
HF_TO_MLX = [
    ("model.", ""), ("encoder.layers.", "encoder.blocks."), ("decoder.layers.", "decoder.blocks."),
    ("self_attn_layer_norm", "attn_ln"), ("encoder_attn_layer_norm", "cross_attn_ln"),
    ("self_attn.", "attn."), ("encoder_attn.", "cross_attn."),
    ("q_proj", "query"), ("k_proj", "key"), ("v_proj", "value"), ("out_proj", "out"),
    ("fc1", "mlp1"), ("fc2", "mlp2"), ("final_layer_norm", "mlp_ln"),
    ("encoder.layer_norm", "encoder.ln_post"), ("decoder.layer_norm", "decoder.ln"),
    ("embed_tokens", "token_embedding"), ("decoder.embed_positions.weight", "decoder.positional_embedding"),
]


def whisper_hf_to_mlx(src: Path, out: Path) -> None:
    import mlx.core as mx
    from safetensors.numpy import load_file

    cfg = json.loads((src / "config.json").read_text())
    dims = {
        "n_mels": cfg["num_mel_bins"], "n_audio_ctx": cfg["max_source_positions"],
        "n_audio_state": cfg["d_model"], "n_audio_head": cfg["encoder_attention_heads"],
        "n_audio_layer": cfg["encoder_layers"], "n_vocab": cfg["vocab_size"],
        "n_text_ctx": cfg["max_target_positions"], "n_text_state": cfg["d_model"],
        "n_text_head": cfg["decoder_attention_heads"], "n_text_layer": cfg["decoder_layers"],
    }
    weights = {}
    for f in sorted(src.glob("*.safetensors")):
        for key, w in load_file(str(f)).items():
            # encoder positions are fixed sinusoids in mlx_whisper; output projection is tied
            if key in ("model.encoder.embed_positions.weight", "proj_out.weight"):
                continue
            for old, new in HF_TO_MLX:
                key = key.replace(old, new)
            if "conv" in key and w.ndim == 3:
                w = w.transpose(0, 2, 1)  # torch Conv1d (out, in, k) -> MLX (out, k, in)
            weights[key] = mx.array(w).astype(mx.float16)
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "weights.safetensors"), weights)
    (out / "config.json").write_text(json.dumps({"model_type": "whisper", **dims}, indent=2))


def ct2_convert(src: Path, out: Path, copy_files: list[str]) -> None:
    from ctranslate2.converters import TransformersConverter

    TransformersConverter(str(src), copy_files=copy_files).convert(str(out), quantization="int8", force=True)


def fetch(name: str) -> None:
    from huggingface_hub import snapshot_download

    how, repo, extra = MODELS[name]
    out = MODELS_DIR / name
    if out.exists():
        print(f"{name}: already in {out}, skipping")
        return
    print(f"{name}: downloading {repo}")
    if how == "snapshot":
        snapshot_download(repo, local_dir=out, allow_patterns=extra)
        return
    src = Path(snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.spm"],
                                 ignore_patterns=["*.msgpack", "*.h5", "*.onnx", "*.pth"]))
    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{name}: converting -> {out}")
    if how == "whisper-to-mlx":
        whisper_hf_to_mlx(src, tmp)
    else:
        ct2_convert(src, tmp, extra)
    tmp.rename(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help="models to fetch (default: %s)" % ", ".join(DEFAULT))
    p.add_argument("--bench", action="store_true", help="fetch every benchmark candidate")
    p.add_argument("--list", action="store_true", help="list known models and exit")
    args = p.parse_args()

    if args.list:
        for name, (_, repo, _) in MODELS.items():
            mark = "*" if name in DEFAULT else " "
            print(f"{mark} {name:28s} {repo}")
        return
    names = list(MODELS) if args.bench else args.names or DEFAULT
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        sys.exit(f"unknown model(s): {', '.join(unknown)} (see --list)")
    for name in names:
        fetch(name)


if __name__ == "__main__":
    main()
