# ASR bake-off (development tool, not part of armtranscribe/)

13 local ASR models on 4 test sets; results summarised in the repository README ("Benchmark").

    ../../armtranscribe/.venv/bin/python prep.py                 # freeze inputs -> data.pkl (podcast pre-cut with the live VAD)
    ../../armtranscribe/.venv/bin/python download.py             # candidate weights -> HF cache (network; benchmark only)
    python3.11 -m venv venv2 && venv2/bin/pip install mlx-vlm "onnx-asr[cpu,hub]" soundfile numpy
    ./queue.sh local.txt; ./queue.sh new.txt       # one process per model -> results/<label>.json
    ../../armtranscribe/.venv/bin/python score.py                # WER/CER, S/D/I, speed per set -> markdown tables

venv2 is separate so mlx-vlm (Gemma) can't change the mlx version mlx-whisper uses. Seamless needs
`protobuf` in the main venv. results/ holds the hypotheses of the run reported in the README.
Excluded: infinitejoy XLS-R (broken published vocab), Qwen3-ASR (no Armenian), CAST-ML (Western
Armenian), alphaedge (trimmed base Whisper).
