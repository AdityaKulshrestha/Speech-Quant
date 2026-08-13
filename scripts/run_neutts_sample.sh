#!/usr/bin/env bash
# Sample invocation of src/evaluate.py: runs the NeuTTS baseline and a
# gptq-4bit quantized variant, then compares them (FDP / D(t) / KL / prob-diff).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

python "$REPO_ROOT/src/evaluate.py" \
    --model neutts \
    --model-name neuphonic/neutts-nano \
    --quant-type gptq-4bit \
    --prompts-file "$REPO_ROOT/src/prompts.txt" \
    --num-samples 50 \
    --output-dir "$REPO_ROOT/outputs" \
    --voice emily \
    --device xpu \
    --max-new-tokens 1200 \
    --temperature 1.0 \
    --seed 0
