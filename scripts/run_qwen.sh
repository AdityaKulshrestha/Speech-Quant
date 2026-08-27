#!/usr/bin/env bash
# Sample invocation of src/evaluate.py: runs the supported Qwen baseline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"

# Qwen's specialized talker does not yet support the shared quantization path.
QUANT_TYPES="none"

"$REPO_ROOT/.venv-qwen/bin/python" "$REPO_ROOT/src/evaluate.py" \
    --model qwen \
    --model-name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --quant-type "$QUANT_TYPES" \
    --prompts-file "$REPO_ROOT/src/prompts.txt" \
    --num-samples 50 \
    --output-dir "$REPO_ROOT/outputs" \
    --device xpu \
    --max-new-tokens 2048 \
    --temperature 0.9 \
    --seed 0

# evaluate.py leaves WER/CER blank; fill them in from the audio just generated.
"$SCRIPT_DIR/transcribe.sh" "$REPO_ROOT/outputs/evaluation/Qwen3-TTS-12Hz-1.7B-CustomVoice"
