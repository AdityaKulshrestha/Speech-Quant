#!/usr/bin/env bash
# Sample invocation of src/evaluate.py for the Orpheus baseline (default, non-quantized) model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

python "$REPO_ROOT/src/evaluate.py" \
    --model orpheus \
    --model-name canopylabs/orpheus-3b-0.1-ft \
    --quant-type none \
    --prompts-file "$REPO_ROOT/src/prompts.txt" \
    --num-samples 5 \
    --output-dir "$REPO_ROOT/src/outputs/orpheus_baseline" \
    --voice tara \
    --device xpu \
    --max-new-tokens 1200 \
    --temperature 0.6 \
    --top-p 0.95 \
    --repetition-penalty 1.1 \
    --seed 0
