#!/usr/bin/env bash
# Qwen3 TTS baseline vs gptq-4bit evaluation.
# Speakers: Ethan, Serena, Cherry, Tom, Emily, Anna, Ashley, Eric
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

python "$REPO_ROOT/src/evaluate.py" \
    --model qwen \
    --model-name Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --quant-type gptq-4bit \
    --prompts-file "$REPO_ROOT/src/prompts.txt" \
    --num-samples 50 \
    --output-dir "$REPO_ROOT/outputs" \
    --voice Ethan \
    --device xpu \
    --max-new-tokens 2048 \
    --temperature 0.9 \
    --seed 0
