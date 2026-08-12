#!/usr/bin/env bash
# Sample invocation of src/evaluate.py: runs the Orpheus baseline and an
# int4 (torchao) quantized variant, then compares them (FDP / D(t)).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

python "$REPO_ROOT/src/evaluate.py" \
    --model orpheus \
    --model-name canopylabs/orpheus-3b-0.1-ft \
    --quant-type gptq-4bit \
    --prompts-file "$REPO_ROOT/src/prompts.txt" \
    --num-samples 50 \
    --output-dir "$REPO_ROOT/outputs" \
    --voice tara \
    --device xpu \
    --max-new-tokens 1200 \
    --temperature 0.6 \
    --top-p 0.95 \
    --repetition-penalty 1.1 \
    --seed 0
