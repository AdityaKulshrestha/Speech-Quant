#!/usr/bin/env bash
# Sample invocation of src/evaluate.py: runs the OuteTTS baseline and a
# gptq-4bit quantized variant, then compares them (FDP / D(t) / KL / prob-diff).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

python "$REPO_ROOT/src/evaluate.py" \
    --model outetts \
    --model-name OuteAI/Llama-OuteTTS-1.0-1B \
    --quant-type gptq-4bit \
    --prompts-file "$REPO_ROOT/src/prompts.txt" \
    --num-samples 50 \
    --output-dir "$REPO_ROOT/outputs" \
    --device xpu \
    --max-new-tokens 1200 \
    --temperature 0.4 \
    --top-p 0.9 \
    --repetition-penalty 1.1 \
    --seed 0
