#!/usr/bin/env bash
# Sample invocation of src/evaluate.py: runs the Higgs Audio v3 baseline (quantization
# not yet supported for this model -- see models/higgs_model.py).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

python "$REPO_ROOT/src/evaluate.py" \
    --model higgs \
    --model-name multimodalart/higgs-audio-v3-tts-4b-transformers \
    --quant-type none \
    --prompts-file "$REPO_ROOT/src/prompts.txt" \
    --num-samples 50 \
    --output-dir "$REPO_ROOT/outputs" \
    --device xpu \
    --max-new-tokens 1200 \
    --temperature 1.0 \
    --top-p 0.95 \
    --seed 0

# evaluate.py leaves WER/CER blank; fill them in from the audio just generated.
"$SCRIPT_DIR/transcribe.sh" "$REPO_ROOT/outputs/evaluation/higgs-audio-v3-tts-4b-transformers"
