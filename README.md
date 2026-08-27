# Speech-Quant: Quantization Effects in Neural-Codec Speech Models

Speech-Quant studies how post-training quantization changes autoregressive text-to-speech models that generate discrete neural-codec tokens. The main question is simple: when we compress the language-model backbone, where does speech quality start to fail, and which metrics reveal the failure earliest?

This codebase compares full-precision baselines against several quantized variants across neural-codec TTS models such as Orpheus, NeuTTS, OuteTTS, and Qwen3-TTS. It is intended to support research experiments and paper figures, not to serve as a production TTS toolkit.

## Why This Study

Autoregressive speech models can generate long token sequences, so small distribution shifts from quantization may compound over time. Standard end metrics such as WER or perceptual quality are important, but they do not explain when the model first diverged or whether a specific codec level is more fragile.

Speech-Quant therefore records both final speech quality and token-level behavior. This makes it possible to connect quantization precision, codec-token divergence, teacher-forced distribution shift, and downstream intelligibility in one experiment.

## Setup

The project uses `uv` and Python 3.12.

```bash
cd Speech-Quant
uv venv .venv-orpheus --python 3.12
source .venv-orpheus/bin/activate
UV_PROJECT_ENVIRONMENT=.venv-orpheus uv sync --group orpheus
```

### Why Separate Environments

The supported speech models depend on different codec packages, PyTorch builds, and `transformers` versions. Keeping separate environments avoids dependency conflicts and makes each model run reproducible.

Recommended environments:

| environment | dependency group | intended model family |
|---|---|---|
| `.venv-orpheus` | `orpheus` | Orpheus / SNAC |
| `.venv-neutts` | `neutts` | NeuTTS / NeuCodec |
| `.venv-outetts` | `outetts` | OuteTTS / DAC |
| `.venv-qwen` | `qwen-tts` | Qwen3-TTS |

Create another environment by changing both names:

```bash
uv venv .venv-neutts --python 3.12
source .venv-neutts/bin/activate
UV_PROJECT_ENVIRONMENT=.venv-neutts uv sync --group neutts
```

## Running Experiments

Run one of the model scripts after activating the matching environment:

```bash
source .venv-orpheus/bin/activate
bash scripts/run_orpheus.sh
```

The scripts compare one full-precision baseline against the configured quantization methods:

```text
rtn-4bit,rtn-8bit,gptq-4bit,gptq-8bit,awq-4bit,awq-8bit,sq-4bit,sq-8bit
```

You can also call the evaluator directly:

```bash
python src/evaluate.py \
	--model orpheus \
	--model-name canopylabs/orpheus-3b-0.1-ft \
	--quant-type gptq-4bit,awq-4bit,sq-8bit \
	--prompts-file src/prompts.txt \
	--num-samples 50 \
	--output-dir outputs \
	--device xpu \
	--seed 0
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
| intelligibility | WER and CER from ASR transcripts |
| acoustic quality | Mel-Cepstral Distortion, F0 frame error, pitch correlation, UTMOS |
| token divergence | first token divergence rate, free-run divergence, codec-level mismatch rate |
| teacher-forced distributions | KL divergence, Jensen-Shannon divergence, argmax mismatch rate, top-k overlap, NLL, perplexity |

The teacher-forced metrics are the cleanest way to compare model distributions because the baseline and quantized model are evaluated on the same prefix at each step. The free-run metrics remain useful for measuring how divergence appears in actual sampled generation.

## License

This project is released under the MIT License. See `LICENSE` for details.

