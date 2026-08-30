#!/usr/bin/env bash
# Run transcription evaluation (WER/CER only) over pre-generated audio.
# This is called by model-specific launchers after audio generation completes.
#
# Runs in the ASR environment (.venv-asr) which is isolated from TTS dependencies
# and can process audio from all models (Orpheus, NeuTTS, OuteTTS).
#
# Usage: scripts/transcribe.sh <model-dir>
# Example: scripts/transcribe.sh outputs/evaluation/orpheus-3b-0.1-ft
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model-dir>" >&2
    echo "Example: $0 outputs/evaluation/orpheus-3b-0.1-ft" >&2
    exit 1
fi

MODEL_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

if [ ! -d "$REPO_ROOT/.venv-asr" ]; then
    echo "Warning: .venv-asr not found. Skipping transcription." >&2
    echo "  Create it with: ./setup-env.sh asr" >&2
    exit 0
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Warning: HF_TOKEN not set. Skipping transcription." >&2
    echo "  Set HF_TOKEN to enable Cohere ASR transcription." >&2
    exit 0
fi

cd "$REPO_ROOT"
export PYTHONPATH=src

echo "Running transcription evaluation (WER/CER) for $MODEL_DIR..."
.venv-asr/bin/python src/evaluate_transcription.py \
    --audio-root "$MODEL_DIR" \
    --quant-type all \
    --batch-size 8

echo "Transcription evaluation complete."
