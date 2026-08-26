#!/usr/bin/env bash
# Helper script to create and sync model-specific environments
# Uses UV_PROJECT_ENVIRONMENT to create separate venvs per model

set -e

MODELS="orpheus neutts qwen outetts"

show_help() {
    cat << EOF
Usage: ./setup-env.sh [MODEL] [--reinstall]

Create and sync a uv environment for a specific model.

Available models:
  orpheus   - Orpheus (transformers 5.15+, llmcompressor dev, torch 2.11.0)
  neutts    - NeuTTS (transformers 5.15+, llmcompressor dev, torch 2.11.0)
  qwen      - Qwen TTS (transformers 4.57.3, llmcompressor 0.9.x, torch 2.9.1)
  outetts   - OuteTTS (transformers 5.15+, llmcompressor dev, torch 2.11.0)
  all       - Create all model environments

Flags:
  --reinstall    Delete existing environment before creating

Examples:
  ./setup-env.sh orpheus              # Create/sync .venv-orpheus
  ./setup-env.sh orpheus --reinstall  # Recreate .venv-orpheus from scratch
  ./setup-env.sh all                  # Create all environments

Note: Requires --index-strategy unsafe-best-match for package resolution.

EOF
}

setup_model() {
    local model=$1
    local reinstall=$2
    local venv_name=".venv-${model}"
    local group="${model}"

    # Map model names to group names
    case "${model}" in
        qwen) group="qwen-tts" ;;
    esac

    echo "======================================"
    echo "Setting up: ${model}"
    echo "Environment: ${venv_name}"
    echo "Group: ${group}"
    echo "======================================"

    # Reinstall if requested
    if [ "${reinstall}" = "--reinstall" ]; then
        if [ -d "${venv_name}" ]; then
            echo "Removing existing environment..."
            rm -rf "${venv_name}"
        fi
    fi

    # Create virtual environment with python 3.12 if it doesn't exist
    if [ ! -d "${venv_name}" ]; then
        echo "Creating Python 3.12 environment ${venv_name}..."
        uv venv "${venv_name}" --python 3.12
    fi

    # Sync dependencies using UV_PROJECT_ENVIRONMENT
    echo "Syncing dependencies..."
    UV_PROJECT_ENVIRONMENT="${venv_name}" uv sync --group "${group}" --index-strategy unsafe-best-match

    echo ""
    echo "✓ Environment ready: ${venv_name}"
    echo ""
    echo "To activate:"
    echo "  source ${venv_name}/bin/activate"
    echo ""
    echo "To run:"
    echo "  source ${venv_name}/bin/activate"
    echo "  python src/evaluate.py --model ${model}"
    echo ""
}

# Main script
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

MODEL=$1
REINSTALL=$2

case "${MODEL}" in
    -h|--help)
        show_help
        exit 0
        ;;
    all)
        echo "Creating all environments..."
        echo ""
        for model in ${MODELS}; do
            setup_model "${model}" "${REINSTALL}"
        done
        echo "======================================"
        echo "All environments created successfully!"
        echo "======================================"
        echo ""
        echo "Available environments:"
        for model in ${MODELS}; do
            echo "  .venv-${model}"
        done
        echo ""
        ;;
    orpheus|neutts|qwen|outetts)
        setup_model "${MODEL}" "${REINSTALL}"
        ;;
    qwen-tts)
        # Accept both qwen and qwen-tts
        setup_model "qwen" "${REINSTALL}"
        ;;
    *)
        echo "Error: Unknown model '${MODEL}'"
        echo ""
        show_help
        exit 1
        ;;
esac
