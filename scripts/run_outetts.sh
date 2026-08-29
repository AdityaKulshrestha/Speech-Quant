#!/usr/bin/env bash
# Sample invocation of src/evaluate.py: runs the OuteTTS baseline against every
# rtn/gptq/awq/sq quant flavour at 4bit and 8bit, then compares them (FDP / D(t) / KL / prob-diff).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"
export UV_PROJECT_ENVIRONMENT=.venv-outetts
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

# All rtn/gptq/awq/sq x 4bit/8bit combinations (see src/quants/config.py).
QUANT_TYPES="rtn-4bit,rtn-8bit,gptq-4bit,gptq-8bit,awq-4bit,awq-8bit,sq-4bit,sq-8bit"

python "$REPO_ROOT/src/evaluate.py" \
    --model outetts \
    --model-name OuteAI/Llama-OuteTTS-1.0-1B \
    --quant-type "$QUANT_TYPES" \
    --prompts-file "$REPO_ROOT/src/prompts.txt" \
    --num-samples 50 \
    --output-dir "$REPO_ROOT/outputs" \
    --device xpu \
    --max-new-tokens 1200 \
    --temperature 0.4 \
    --top-p 0.9 \
    --repetition-penalty 1.1 \
    --seed 0

# evaluate.py leaves WER/CER blank; fill them in from the audio just generated.
"$SCRIPT_DIR/transcribe.sh" "$REPO_ROOT/outputs/evaluation/Llama-OuteTTS-1.0-1B"
