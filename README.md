# Speech-Quant: Quantization Effects in Neural-Codec Speech Models

Speech-Quant studies how post-training quantization changes autoregressive text-to-speech models that generate discrete neural-codec tokens. The main question is simple: when we compress the language-model backbone, where does speech quality start to fail, and which metrics reveal the failure earliest?

This codebase compares full-precision baselines against quantized variants of Orpheus, NeuTTS, and OuteTTS. It is intended to support research experiments and paper figures, not to serve as a production TTS toolkit.

## Why This Study

Autoregressive speech models can generate long token sequences, so small distribution shifts from quantization may compound over time. Standard end metrics such as WER or perceptual quality are important, but they do not explain when the model first diverged or whether a specific codec level is more fragile.

Speech-Quant therefore records both final speech quality and token-level behavior. This makes it possible to connect quantization precision, codec-token divergence, teacher-forced distribution shift, and downstream intelligibility in one experiment.

## Setup

Speech-Quant supports three model families: Orpheus, NeuTTS, and OuteTTS. Each
uses a separate dependency group because their codec and quantization
dependencies conflict.

### Requirements

- Linux
- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Network access for Hugging Face model and dataset downloads on the first run

Set `DEVICE` when creating an environment to choose its PyTorch backend. The
default is `auto`.

### Gated Model Access

Some model checkpoints require approved access on Hugging Face. Request access
and accept the model's terms on its model page, then export a read token before
running setup or evaluation:

```bash
export HF_TOKEN="your_hugging_face_read_token"
```

The token stays in the current shell session. Do not commit it to the
repository or include it in scripts.

### Create An Environment

From the repository root, create one environment for the model you intend to
run:

```bash
./setup-env.sh orpheus
```

Replace `orpheus` with `neutts` or `outetts` as needed. Add `--reinstall` to
recreate an existing environment. Use `./setup-env.sh all` to prepare every
TTS and ASR environment.

### Why Multiple Environments

The supported models cannot safely share one environment. Orpheus installs
GPTQModel, which requires a newer `torchao`; NeuTTS pins `torchao==0.13.0` for
NeuCodec; and OuteTTS brings DAC dependencies with a different protobuf
constraint. The separate environments keep each model's codec and
quantization stack reproducible instead of resolving one model by breaking
another. The ASR environment is isolated for transcription evaluation (WER/CER)
on pre-generated audio from all models.

| environment | group | purpose |
|---|---|---|
| `.venv-orpheus` | `orpheus` | Orpheus generation + all evaluation except WER/CER |
| `.venv-neutts` | `neutts` | NeuTTS generation + all evaluation except WER/CER |
| `.venv-outetts` | `outetts` | OuteTTS generation + all evaluation except WER/CER |
| `.venv-asr` | `asr` | Transcription evaluation only (WER/CER, all models) |

## Running Experiments

Activate the matching environment and use its launcher. Each launcher generates
the full-precision baseline, runs all configured quantization variants, writes
comparison artifacts, and then starts transcription when `.venv-asr` exists.

```bash
source .venv-orpheus/bin/activate
bash scripts/run_orpheus.sh <device>

source .venv-neutts/bin/activate
bash scripts/run_neutts.sh <device>

source .venv-outetts/bin/activate
bash scripts/run_outetts.sh <device>
```

Pass the device supported by your PyTorch installation, such as `cpu` or `cuda`.

The launchers use the following quantization variants:

```text
rtn-4bit,rtn-8bit,gptq-4bit,gptq-8bit,awq-4bit,awq-8bit,sq-4bit,sq-8bit
```

The active quantization settings are defined in `src/quants/config.py`:

| quant type | precision |
|---|---|
| `rtn-4bit`, `gptq-4bit` | `W4A16` |
| `rtn-8bit`, `gptq-8bit`, `awq-8bit` | `W8A16` |
| `awq-4bit` | `W4A16_ASYM` |
| `sq-4bit` | `W4A8` |
| `sq-8bit` | `W8A8` |

Only SmoothQuant (`sq-*`) quantizes activations to 8-bit. RTN, GPTQ, and AWQ are weight-only in this repository.

### Results

A successful baseline run creates WAV files and a manifest below
`outputs/evaluation/`. A quantized run also creates one directory per
quantization type and updates:

```text
outputs/evaluation/analysis_report.xlsx
```

Inspect the workbook's `Summary`, `PerSample`, `LogProbs_KL`, and `Codec` sheets
to compare the baseline with each quantization type. Re-running a model and
quantization type replaces only its matching rows.

### Transcription Evaluation (WER/CER)

`src/evaluate.py` generates audio and computes all metrics (MCD, F0, UTMOS, token divergence, teacher-forced distributions) in the TTS environment. The WER/CER/transcript fields are filled by a separate **transcription pass** that runs in the ASR environment on pre-generated audio.

**Why separate?** The ASR environment (.venv-asr) has different dependency constraints (Cohere ASR model) that conflict with TTS model dependencies. This keeps each environment focused and reproducible.

**Setup:**

Create the ASR environment:

```bash
./setup-env.sh asr
```

**Running Transcription Evaluation:**

For all models and quants:

```bash
export HF_TOKEN="your_huggingface_token"  # Required for Cohere ASR
source .venv-asr/bin/activate
PYTHONPATH=src python src/evaluate_transcription.py \
    --audio-root outputs/evaluation \
    --batch-size 8
```

For a specific model:

```bash
PYTHONPATH=src python src/evaluate_transcription.py \
    --audio-root outputs/evaluation/orpheus-3b-0.1-ft \
    --quant-type gptq-4bit,awq-4bit
```

**How it works:**

1. Loads Cohere ASR model once
2. Processes each model's baseline + quant directories
3. Runs batch transcription (WER/CER) on all audio files
4. Updates `analysis_report.xlsx` with transcription metrics

Text normalization: Ground-truth prompts and ASR transcripts are normalized (numbers/dates expanded via NeMo, punctuation stripped, apostrophes removed, lowercased) so punctuation and casing never count as errors.

The script only updates WER/CER/transcript columns for matching `(model, quant_type)` rows; all other data (MCD, F0, UTMOS, token metrics) is preserved. Writes use an exclusive lock, so it's safe to run concurrently.

## Outputs

Each run writes generated audio, manifests, plots, and a consolidated report under `outputs/`.

The main artifact for analysis is:

```text
outputs/evaluation/analysis_report.xlsx
```

Important sheets:

| sheet | content |
|---|---|
| `Summary` | one metric per row, grouped by model and quantization type |
| `PerSample` | per-prompt WER, CER, acoustic scores, divergence scores, and transcripts |
| `LogProbs_KL` | teacher-forced per-step probability, KL/JS, argmax, and token data |
| `Codec` | codec-level divergence by sample and hierarchy level |

Rows are updated by `(model, quant_type)`. Rerunning one quantization setting replaces only that setting's rows in the report.

## Metrics Covered

Speech-Quant records metrics at three levels:

| group | metrics | computed by |
|---|---|---|
| intelligibility | WER and CER from Cohere ASR transcripts | `evaluate_transcription.py` (ASR env) |
| acoustic quality | Mel-Cepstral Distortion, F0 frame error, pitch correlation, UTMOS | `evaluate.py` (TTS env) |
| token divergence | first token divergence rate, free-run divergence, codec-level mismatch rate | `evaluate.py` (TTS env) |
| teacher-forced distributions | KL divergence, Jensen-Shannon divergence, argmax mismatch rate, top-k overlap, NLL, perplexity | `evaluate.py` (TTS env) |

The teacher-forced metrics are the cleanest way to compare model distributions because the baseline and quantized model are evaluated on the same prefix at each step. The free-run metrics remain useful for measuring how divergence appears in actual sampled generation.

## License

This project is released under the MIT License. See `LICENSE` for details.

