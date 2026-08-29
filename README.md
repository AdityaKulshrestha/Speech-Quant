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
another. The ASR environment is also isolated because it is used only after
audio generation for WER and CER scoring.

| environment | group | model / codec |
|---|---|---|
| `.venv-orpheus` | `orpheus` | Orpheus / SNAC |
| `.venv-neutts` | `neutts` | NeuTTS / NeuCodec |
| `.venv-outetts` | `outetts` | OuteTTS / DAC |
| `.venv-asr` | `asr` | Cohere ASR for WER/CER |

## Running Experiments

Activate the matching environment and use its launcher. Each launcher generates
the full-precision baseline, runs all configured quantization variants, writes
comparison artifacts, and then starts transcription when `.venv-asr` exists.

```bash
source .venv-orpheus/bin/activate
bash scripts/run_orpheus.sh

source .venv-neutts/bin/activate
bash scripts/run_neutts.sh

source .venv-outetts/bin/activate
bash scripts/run_outetts.sh
```

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

### WER / CER

`src/evaluate.py` leaves the WER, CER, and transcript fields blank. They are filled by a second pass that re-reads the audio already written under `outputs/`, so it never regenerates speech.

Each launcher calls this pass automatically after generation through
`scripts/transcribe.sh`. If `.venv-asr` does not exist, the step is skipped with
a warning. Create it before running a full experiment:

```bash
./setup-env.sh asr
```

It uses its own environment so transcription dependencies remain separate from
TTS generation environments.

To run it on its own, for every model and quantization folder found:

```bash
PYTHONPATH=src .venv-asr/bin/python src/evaluate_transcription.py \
	--audio-root outputs/evaluation \
	--quant-type all \
	--batch-size 8
```

Or narrow the scope to one model directory:

```bash
PYTHONPATH=src .venv-asr/bin/python src/evaluate_transcription.py \
	--audio-root outputs/evaluation/orpheus-3b-0.1-ft \
	--quant-type gptq-4bit,awq-4bit
```

Before scoring, both the ground-truth prompt and the ASR transcript are normalized: numbers and dates are expanded to spoken form via NeMo, punctuation is stripped, apostrophes are removed so `can't` and `cant` match, and the text is lowercased and whitespace-collapsed. Punctuation and casing therefore never count as errors.

The pass only rewrites the WER/CER, transcript, and delta columns for the matching `(model, quant_type)`; all other sheets and rows are preserved. Writes take an exclusive lock on the workbook, so this is safe to run while other jobs are queued. Raise `--batch-size` on an accelerator node — transcription is batched through a single `generate()` call per chunk.

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

| group | metrics |
|---|---|
| intelligibility | WER and CER from Cohere ASR transcripts |
| acoustic quality | Mel-Cepstral Distortion, F0 frame error, pitch correlation, UTMOS |
| token divergence | first token divergence rate, free-run divergence, codec-level mismatch rate |
| teacher-forced distributions | KL divergence, Jensen-Shannon divergence, argmax mismatch rate, top-k overlap, NLL, perplexity |

The teacher-forced metrics are the cleanest way to compare model distributions because the baseline and quantized model are evaluated on the same prefix at each step. The free-run metrics remain useful for measuring how divergence appears in actual sampled generation.

## License

This project is released under the MIT License. See `LICENSE` for details.

