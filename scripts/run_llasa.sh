#!/usr/bin/env bash
# Sample invocation of src/evaluate.py: runs the Llasa-1B baseline and a
# gptq-4bit quantized variant, then compares them (FDP / D(t) / KL / prob-diff).
# Requires the dedicated .venv-llasa environment — see models/llasa_model.py's
# docstring for setup (xcodec2 needs a separate venv from the main .venv).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

"$REPO_ROOT/.venv-llasa/bin/python3" "$REPO_ROOT/src/evaluate.py" \
    --model llasa \
    --model-name HKUSTAudio/Llasa-1B \
    --quant-type gptq-4bit \
    --prompts-file "$REPO_ROOT/src/prompts.txt" \
    --num-samples 50 \
    --output-dir "$REPO_ROOT/outputs" \
    --device xpu \
    --max-new-tokens 1200 \
    --temperature 0.8 \
    --top-p 1.0 \
    --seed 0
