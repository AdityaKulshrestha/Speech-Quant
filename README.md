# Effect of Quantization on Speech Models

## Objective

This project studies the effect of post-training quantization (PTQ) on
autoregressive neural-codec speech models.

The main question is:

> How does numerical perturbation introduced by PTQ propagate through autoregressive neural codec generation into perceptual speech degradation?
> Are primary/semantic codec streams more robust than residual/acoustic streams?
> Are models with more codec streams more sensitive to PTQ?

For each model, the experiment compares a full-precision or baseline
checkpoint against one or more PTQ variants while keeping the input
text, generation settings, dataset, and evaluation procedure fixed.

The experiment is designed to separate three effects:

1.  **Language-model changes** caused by quantization.
2.  **Codec-token changes** caused by quantization.
3.  **Final speech changes** caused by the changed codec-token sequence.

The goal is not only to report whether quantized speech sounds worse,
but to identify **where and how the generated codec sequence changes**.

## What Is Being Measured

### 1. Generation performance

For each model and quantization configuration, record:

-   Total generation latency
-   Time to first token, where supported
-   Codec tokens generated per second
-   Number of generated codec tokens
-   Input token count
-   Output token count
-   Peak device memory
-   Model loading time, where useful

The same generation parameters should be used for the baseline and all
quantized variants.

### 2. Codec-token changes

The generated codec sequence is compared with the baseline sequence.

Monitor:

-   Exact token match rate
-   Token mismatch rate
-   First mismatch position
-   Number of changed token positions
-   Normalized edit distance
-   Position-wise mismatch rate
-   Mismatch rate as a function of generation position
-   Mismatch rate per codec codebook
-   Codebook-specific token accuracy
-   Token distribution changes
-   KL divergence between baseline and quantized token distributions,
    when logits are available
-   Log-probability difference for the baseline-generated token
-   Entropy difference, when logits are available

For autoregressive models, the **first mismatch position** is
particularly important. A small change early in generation can change
the remainder of the codec sequence because later tokens are conditioned
on previous generated tokens.

Therefore, token-level evaluation should distinguish between:

-   Local changes before autoregressive divergence
-   Changes after the first divergence
-   Total sequence-level divergence

### 3. Speech quality changes

Decode both baseline and quantized codec-token sequences using the same
codec decoder.

Compare:

-   Word Error Rate (WER)
-   Character Error Rate (CER), where appropriate
-   Speech/text alignment
-   Speaker similarity
-   Speech embedding similarity
-   Audio embedding similarity
-   Signal-level differences
-   Spectral differences
-   Mel-spectrogram similarity
-   PESQ, where applicable
-   STOI, where applicable
-   DNSMOS or other learned speech-quality metrics, where applicable

Metrics should be selected according to the model and supported language
rather than forcing every metric onto every model.

### 4. Baseline versus quantized audio

For each input, generate:

``` text
Input text
    |
    +--------------------+
    |                    |
Baseline model       Quantized model
    |                    |
codec tokens          codec tokens
    |                    |
    +---------+----------+
              |
        token comparison
              |
        codec decoding
              |
    +---------+----------+
    |                    |
baseline audio       quantized audio
    |                    |
    +---------+----------+
              |
        speech evaluation
```

The codec decoder must remain unchanged between the two conditions. This
isolates the effect of changing the autoregressive model.

## Experimental Principle

The baseline and quantized experiments should use the same:

-   Model checkpoint
-   Input text
-   Speaker/reference audio
-   Prompt formatting
-   Sampling parameters
-   Random seed, where deterministic comparison is possible
-   Maximum generation length
-   Codec decoder
-   Evaluation dataset
-   Evaluation metrics

Only the quantization configuration should change.

For stochastic generation, exact token-by-token comparison can be
affected by sampling. Where possible, use deterministic decoding for the
token-divergence experiment or otherwise record and control the random
seed and sampling configuration.

## Quantization

This project focuses on **post-training quantization (PTQ)**.

The initial experiments should evaluate quantization of the
autoregressive language model while keeping the speech codec unchanged.

Potential PTQ configurations include:

-   FP32 baseline
-   BF16/FP16 baseline, depending on the model
-   INT8 weight-only
-   INT4 weight-only
-   Other PTQ methods added later

Each quantization configuration should be treated as a separate
experiment.

The experiment should record:

-   Quantization method
-   Weight precision
-   Activation precision, if applicable
-   Group size, if applicable
-   Calibration dataset and number of samples, if applicable
-   Quantized parameter count
-   Model/checkpoint identifier
-   Software versions
-   Hardware
-   Generation configuration

## Important Experimental Separation

Do not quantize the codec as part of the first experiment.

The first stage should isolate:

``` text
Quantized AR model
        |
        v
Generated codec tokens
        |
        v
Original/full-precision codec decoder
        |
        v
Speech
```

This allows us to answer:

> How much speech degradation is caused solely by quantizing the
> autoregressive speech model?

Codec quantization can be studied later as a separate experiment.

## Models

The project focuses on autoregressive neural-codec speech models.

The current model list is summarized below. Hugging Face download counts are approximate and time-dependent; they are included only as an indication of current community usage.

| # | Model | Release | Parameters | Base LM / Architecture | Neural Codec | Codec Structure | HF Downloads / Month | Organization | Hugging Face |
|---:|---|---|---:|---|---|---|---:|---|---|
| 1 | **NeuTTS 2E** | Jan 2026 | ~236M | Compact speech LM | NeuCodec | Single codebook | — | Neuphonic | [HF](https://huggingface.co/neuphonic/neutts-2e) |
| 2 | **Qwen3-TTS 0.6B Base** | Jan 2026 | ~0.9B reported | Qwen3 | Qwen3-TTS-Tokenizer-12Hz | 16 codebooks, 12.5 Hz | ~400K | Alibaba / Qwen | [HF](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) |
| 3 | **Qwen3-TTS 1.7B Base** | Jan 2026 | ~2B reported | Qwen3 | Qwen3-TTS-Tokenizer-12Hz | 16 codebooks, 12.5 Hz | ~2.1M | Alibaba / Qwen | [HF](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) |
| 4 | **Fish Audio S2 Pro** | Mar 2026 | ~4B + 0.4B | Decoder-only Dual-AR | ModifiedDAC | 10 codebooks, ~21 Hz | ~264K | Fish Audio | [HF](https://huggingface.co/fishaudio/s2-pro) |
| 5 | **CSM-1B** | Mar 2025 | ~1B | Llama 3.2 1B + audio decoder | Mimi | RVQ, multi-codebook | ~237K | Sesame | [HF](https://huggingface.co/sesame/csm-1b) |
| 6 | **Orpheus 3B** | Mar 2025 | 3B | Llama 3.2 3B | SNAC | Hierarchical | ~24K | Canopy Labs | [HF](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) |
| 7 | **LLaSA 1B** | Feb 2025 | 1B | Llama 3.2 1B | XCodec2 | Single codebook | ~2K | HKUST Audio | [HF](https://huggingface.co/HKUSTAudio/Llasa-1B) |
| 8 | **LLaSA 3B** | Feb 2025 | 3B | Llama 3.2 3B | XCodec2 | Single codebook | ~700 | HKUST Audio | [HF](https://huggingface.co/HKUSTAudio/Llasa-3B) |
| 9 | **TADA 1B** | Feb 2026 | 1B | Llama 3.2 1B | TADA codec | Speech representation aligned to text tokens | ~14K | Hume AI | [HF](https://huggingface.co/HumeAI/tada-1b) |
| 10 | **TADA 3B** | Feb 2026 | 3B | Llama 3.2 3B | TADA codec | Speech representation aligned to text tokens | ~9K | Hume AI | [HF](https://huggingface.co/HumeAI/tada-3b-ml) |
| 11 | **Spark-TTS 0.5B** | Mar 2025 | ~0.5B | Qwen2.5 0.5B | BiCodec | Decoupled representation | ~900 | SparkAudio | [HF](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) |
| 12 | **Kyutai TTS 0.75B** | Sep 2025 | ~0.75B | Custom Transformer | Mimi | 16 codebooks, 12.5 Hz | ~6K | Kyutai | [HF](https://huggingface.co/kyutai/tts-0.75b-en-public) |
| 13 | **Kyutai TTS 1.8B** | Sep 2025 | ~1.8B | Custom Transformer | Mimi | 32 codebooks, 12.5 Hz | ~69K | Kyutai | [HF](https://huggingface.co/kyutai/tts-1.6b-en_fr) |
| 14 | **Fish Speech S1-mini** | 2026 | ~0.5B | Qwen-family / Dual-AR | Fish codec | Multi-codebook | ~3K | Fish Audio | [HF](https://huggingface.co/fishaudio/s1-mini) |
| 15 | **Fish Speech 1.5** | 2025 | ~3B | Qwen-family | Firefly | Multi-codebook | ~4K | Fish Audio | [HF](https://huggingface.co/fishaudio/fish-speech-1.5) |
| 16 | **Step-Audio-TTS-3B** | Feb 2025 | 3B | Step-1 | Dual-codebook tokenizer | Semantic + acoustic | ~100 | StepFun | [HF](https://huggingface.co/stepfun-ai/Step-Audio-TTS-3B) |


  -----------------------------------------------------------------------------------------------------------------------------------------------------------------
  Model                 Parameters Base LM /      Codec                      Codec structure  Organization   Hugging Face
                                   architecture                                                              
  ------------------- ------------ -------------- -------------------------- ---------------- -------------- ------------------------------------------------------
  NeuTTS-2E           \~236M total Compact speech NeuCodec                   Single codebook  Neuphonic      https://huggingface.co/neuphonic/neutts-2e
                                   LM                                                                        

  Qwen3-TTS 0.6B Base       \~0.9B Qwen3          Qwen3-TTS-Tokenizer-12Hz   16 codebooks,    Alibaba / Qwen https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base
                          reported                                           12.5 Hz                         

  Qwen3-TTS 1.7B Base         \~2B Qwen3          Qwen3-TTS-Tokenizer-12Hz   16 codebooks,    Alibaba / Qwen https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
                          reported                                           12.5 Hz                         

  Fish Audio S2 Pro     \~5B total Decoder-only / RVQ audio codec            10 codebooks,    Fish Audio     https://huggingface.co/fishaudio/s2-pro
                                   Dual-AR                                   \~21 Hz                         

  CSM-1B                \~1B class Llama 3.2 +    Mimi                       RVQ,             Sesame         https://huggingface.co/sesame/csm-1b
                                   audio decoder                             multi-codebook                  

  Orpheus 3B                    3B Llama 3.2 3B   SNAC                       Hierarchical     Canopy Labs    https://huggingface.co/canopylabs/orpheus-3b-0.1-ft

  LLaSA 1B                      1B Llama 3.2 1B   XCodec2                    Single codebook  HKUST Audio    https://huggingface.co/HKUSTAudio/Llasa-1B

  LLaSA 3B                      3B Llama 3.2 3B   XCodec2                    Single codebook  HKUST Audio    https://huggingface.co/HKUSTAudio/Llasa-3B

  TADA 1B                       1B Llama 3.2 1B   TADA codec                 Speech           Hume AI        https://huggingface.co/HumeAI/tada-1b
                                                                             representation                  
                                                                             aligned to text                 
                                                                             tokens                          

  TADA 3B                       3B Llama 3.2 3B   TADA codec                 Speech           Hume AI        https://huggingface.co/HumeAI/tada-3b-ml
                                                                             representation                  
                                                                             aligned to text                 
                                                                             tokens                          

  Spark-TTS 0.5B            \~0.5B Qwen2.5 0.5B   BiCodec                    Decoupled        SparkAudio     https://huggingface.co/SparkAudio/Spark-TTS-0.5B
                                                                             representation                  

  Kyutai TTS 0.75B         \~0.75B Custom         Mimi                       16 codebooks,    Kyutai         https://huggingface.co/kyutai/tts-0.75b-en-public
                                   Transformer                               12.5 Hz                         

  Kyutai TTS 1.8B           \~1.8B Custom         Mimi                       32 codebooks,    Kyutai         https://huggingface.co/kyutai/tts-1.6b-en_fr
                                   Transformer                               12.5 Hz                         

  Fish Speech S1-mini \~0.5B class Qwen-family /  Fish codec                 Multi-codebook   Fish Audio     https://huggingface.co/fishaudio/s1-mini
                                   Dual-AR                                                                   

  Step-Audio-TTS-3B             3B Step-1         Dual-codebook tokenizer    Semantic +       StepFun        https://huggingface.co/stepfun-ai/Step-Audio-TTS-3B
                                                                             acoustic                        
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------

### Model inclusion notes

The list is intentionally focused on autoregressive speech language
models with neural/discrete speech representations.

Models based primarily on diffusion or flow matching are outside the
initial scope.

Models with substantially different continuous audio-generation
mechanisms should be tracked separately rather than mixed into the main
codec-token experiment.

Dia is not included in the primary list because its published
architecture is described as an encoder-decoder Transformer. If the
project later broadens the scope beyond decoder-only/AR codec-LM
systems, it can be added as a separate category.

NeuTTS-2E should not be described as using the Qwen3-TTS tokenizer. It
uses Neuphonic's NeuCodec, which is a single-codebook neural codec.

## Model Popularity

Popularity should be treated as contextual information, not as a
scientific selection criterion.

For models hosted on Hugging Face, record the current Hugging Face
download count at the time the experiment list is frozen.

The download count is time-dependent and should not be presented as a
permanent property of a model.

For reproducibility, record:

``` text
model_id
date_checked
downloads_last_month
```

along with the experiment configuration.

## Dataset

The evaluation dataset should contain short speech prompts with
corresponding reference text.

For every sample, store at least:

``` text
sample_id
text
reference_audio, if required by the model
speaker_id, if required
language
```

The same samples must be used for all quantization variants of a model.

Avoid changing the dataset between baseline and quantized runs.

## Reproduce the Results

The main entry point for all experiments is:

``` bash
python benchmark.py
```

The command-line arguments and experiment configuration will be
documented here as the benchmark implementation is finalized.

For now, the expected workflow is:

1.  Install the project dependencies.
2.  Download/access the required Hugging Face model checkpoints.
3.  Select a model and quantization configuration.
4.  Run the benchmark through `benchmark.py`.
5.  Save generated codec tokens and generated audio.
6.  Compare baseline and quantized outputs.
7.  Aggregate token-level, speech-level, latency, and memory metrics.

The final benchmark command and complete list of arguments will be added
once `benchmark.py` is implemented.

## Results

Results should be stored so that each output can be traced back to:

``` text
model
model_revision
quantization_method
quantization_config
dataset
sample_id
generation_config
random_seed
hardware
software_environment
```

At minimum, each experiment should preserve:

``` text
baseline/
quantized/
metrics/
metadata/
```

Generated codec tokens should be saved before decoding. This is required
for the token-level analysis.

## Recommended Result Tables

### Model-level summary

  -----------------------------------------------------------------------------------------
  Model   Quantization      Memory   Latency   Tokens/s   WER/CER      Speaker        Audio
                                                                    similarity   similarity
  ------- -------------- --------- --------- ---------- --------- ------------ ------------

  -----------------------------------------------------------------------------------------

### Codec-token summary

  --------------------------------------------------------------------------------
  Model    Quantization        Token      First       Edit   Codebook           KL
                               match   mismatch   distance   mismatch   divergence
  -------- -------------- ---------- ---------- ---------- ---------- ------------

  --------------------------------------------------------------------------------

### Speech degradation summary

  ----------------------------------------------------------------------------
  Model     Quantization        WER/CER      Speaker        Audio     Spectral
                                  delta   similarity   similarity        delta
                                               delta        delta 
  --------- -------------- ------------ ------------ ------------ ------------

  ----------------------------------------------------------------------------

## Key Research Questions

The experiment is intended to answer the following questions:

1.  How much does PTQ change generated speech quality?
2.  How much does PTQ change the generated codec-token sequence?
3.  At what generation position does the quantized model first diverge
    from the baseline?
4.  Does an early token mismatch cause substantially larger downstream
    divergence?
5.  Are some codec codebooks more sensitive to quantization than others?
6.  Does quantization affect semantic/primary codec tokens differently
    from residual/acoustic tokens?
7.  Is token-level divergence correlated with perceptual speech
    degradation?
8.  Does the relationship between model size and quantization
    sensitivity hold across different speech models?
9.  Which PTQ precision provides the best trade-off between memory,
    latency, and speech quality?
10. Does the codec architecture change the model's sensitivity to PTQ?

## Scope

The first version of this project focuses on:

-   Autoregressive neural-codec speech models
-   Post-training quantization of the autoregressive model
-   Token-level comparison
-   Speech-quality comparison
-   Inference performance
-   Memory consumption

The following are outside the initial scope:

-   Training-time quantization
-   Quantization-aware training (QAT)
-   Codec quantization
-   Full model retraining
-   Fine-tuning after quantization
-   Diffusion/flow-matching TTS models
