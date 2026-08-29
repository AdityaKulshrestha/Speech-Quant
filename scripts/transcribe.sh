#!/usr/bin/env bash
# WER/CER pass over audio already written by src/evaluate.py -- no speech is regenerated.
# Runs in its own env (./setup-env.sh asr) to keep Cohere ASR's dependency stack
# isolated from the TTS model environments.
#
# Usage: scripts/transcribe.sh [audio-root] [quant-types]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

AUDIO_ROOT="${1:-$REPO_ROOT/outputs/evaluation}"
QUANT_TYPES="${2:-all}"
ASR_PYTHON="$REPO_ROOT/.venv-asr/bin/python"

if [ ! -x "$ASR_PYTHON" ]; then
    echo "WER/CER skipped: $ASR_PYTHON not found. Create it with: ./setup-env.sh asr" >&2
    exit 0
fi

PYTHONPATH="$REPO_ROOT/src" "$ASR_PYTHON" "$REPO_ROOT/src/evaluate_transcription.py" \
    --audio-root "$AUDIO_ROOT" \
    --quant-type "$QUANT_TYPES" \
    --batch-size 8
