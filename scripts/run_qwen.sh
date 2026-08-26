#!/usr/bin/env bash
# Qwen3 TTS baseline vs every rtn/gptq/awq/sq quant flavour at 4bit and 8bit.
# Speakers: Ethan, Serena, Cherry, Tom, Emily, Anna, Ashley, Eric
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# All rtn/gptq/awq/sq x 4bit/8bit combinations (see src/quants/config.py).
QUANT_TYPES="rtn-4bit,rtn-8bit,gptq-4bit,gptq-8bit,awq-4bit,awq-8bit,sq-4bit,sq-8bit"

python "$REPO_ROOT/src/evaluate.py" \
    --model qwen \
    --model-name Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --quant-type "$QUANT_TYPES" \
    --prompts-file "$REPO_ROOT/src/prompts.txt" \
    --num-samples 50 \
    --output-dir "$REPO_ROOT/outputs" \
    --voice Ethan \
    --device xpu \
    --max-new-tokens 2048 \
    --temperature 0.9 \
    --seed 0
