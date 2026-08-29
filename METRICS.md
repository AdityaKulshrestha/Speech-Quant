# TTS Quantization Metrics: Complete Reference

This document provides comprehensive documentation for all metrics used in Speech-Quant, including calculation methods, interpretation guidelines, and analysis of quantization impacts on TTS models.

**Contents:**
- [Implemented Metrics (Phases 1-2)](#implemented-metrics)
- [Detailed Metric Specifications](#detailed-metric-specifications)
- [How to Use These Metrics](#how-to-use-these-metrics)
- [Future Metrics (Phase 3)](#phase-3-causal--sensitivity-metrics-not-yet-implemented)

---

## Implemented Metrics

### 1. Macroscopic Metrics (Phase 1)

These evaluate **perceptual quality** and **intelligibility** of generated audio.

- **Semantic Accuracy:** ASR-based Word Error Rate (WER) and Character Error Rate (CER) computed via Cohere ASR transcription
  - *Status:* Implemented in `src/evaluation/transcription.py`
  - *Use:* Measures speech intelligibility and semantic preservation

- **Acoustic Distortion:**
  - **Mel-Cepstral Distortion (MCD):** Spectral envelope degradation in dB
    - *Implementation:* `src/evaluation/acoustic_metrics.py:mel_cepstral_distortion()`
  - **F0 Frame Error:** Pitch/voicing accuracy (fraction of frames with errors)
    - *Implementation:* `src/evaluation/acoustic_metrics.py:f0_frame_error()`
  - **Pitch Pearson Correlation:** F0 contour correlation (prosody preservation)
    - *Implementation:* `src/evaluation/acoustic_metrics.py:pitch_pearson_correlation()`
  - **UTMOS:** Non-intrusive perceptual quality score [1-5]
    - *Implementation:* `src/evaluation/acoustic_metrics.py:utmos_score()` (optional, requires `speechmos`)

- **Speaker Fidelity (SECS):** Not yet implemented (see Phase 3)

### 2. Microscopic Metrics (Phase 2)

These examine **token-level impacts** of quantization on model predictions.

#### Free-Run Divergence (drift-confounded)
- **First Sampled Mismatch Position:** Index of first token disagreement
- **Final Divergence Rate (D(T)):** Cumulative token mismatch rate
- **Mean Probability Difference:** Confidence gap for baseline's chosen tokens
- **Mean KL Divergence:** Position-aligned distributional shift

#### Teacher-Forced Divergence (drift-free)
- **Teacher-Forced KL/JS Divergence:** Pure quantization-induced distribution shift
  - *Implementation:* `src/decoding/distribution.py:compare_distributions()`
- **Argmax Mismatch Rate:** Fraction of steps where top-1 tokens disagree
- **First Token Divergence Rate (FTDR):** Probability that first token diverges (deterministic)
  - *Implementation:* `src/decoding/distribution.py:first_token_divergence_rate()`
- **Top-k Jaccard Overlap:** Candidate set agreement (softer than argmax)
- **Perplexity:** How "surprised" baseline is by quantized outputs (headline quality metric)

#### Codec-Specific Divergence
- **Codebook-Wise Token Error Rate (TER_k):** Per-hierarchy-level mismatch rates
  - *Implementation:* `src/decoding/alignment.py:analyze_codebook_divergence()`
  - Both free-run and teacher-forced variants available

**Directional KL Divergence Formula:**

$$
D_{\mathrm{KL}}\left(
P_{\mathrm{FP16}} \parallel P_{\mathrm{Quant}}
\right)
=
\sum_{v \in \mathcal{V}}
P_{\mathrm{FP16}}(v)
\log\left(
\frac{P_{\mathrm{FP16}}(v)}
{P_{\mathrm{Quant}}(v)}
\right)
$$

**Codebook-Wise Token Error Rate Formula:**

$$
\mathrm{TER}_k
=
\frac{1}{T}
\sum_{t=1}^{T}
\mathbb{I}
\left[
\hat{c}_{t,k}^{\mathrm{Quant}}
\neq
\hat{c}_{t,k}^{\mathrm{FP16}}
\right]
$$

### 3. Model Efficiency Metrics

- **Model Size (MB):** Quantized model file size on disk
- **Compression Ratio:** `baseline_size / quantized_size` (higher = better compression)

---

## Phase 3: Causal & Sensitivity Metrics (Not Yet Implemented)

- **Error Amplification Factor (β):** Rate at which injected perturbation at step $t_0$ increases $D_{\mathrm{KL}}$ at step $t_0 + \Delta t$

- **Ablation Sensitivity Delta:** Marginal WER drop when quantizing layer $k$ while keeping others in FP16

---

## Detailed Metric Specifications

### Phase 1: Acoustic Metrics

#### Mel-Cepstral Distortion (MCD)

**Calculation:** DTW-aligned frame-by-frame MFCC distance:
```
MCD = (10/ln(10)) * sqrt(2 * sum((c_ref - c_hyp)^2))
```

**Range:** [0, ∞) dB, typically 5-20 dB for TTS

**Interpretation:**
- **Lower is better** (0 = identical spectral envelope)
- < 4 dB: Nearly imperceptible difference
- 4-8 dB: Noticeable but acceptable timbral change
- > 8 dB: Significant spectral distortion, audible quality loss

**What to listen for:** Timbre changes, spectral "muddiness", loss of clarity in vowels/consonants

**Divergence scope:** More = worse spectral fidelity; Less = better preservation

**Code:** `src/evaluation/acoustic_metrics.py:mel_cepstral_distortion()`

---

#### F0 Frame Error

**Calculation:** Fraction of frames with voicing mismatch OR pitch deviation > 50 cents on voiced frames

**Range:** [0, 1] (fraction)

**Interpretation:**
- **Lower is better** (0 = perfect pitch tracking)
- < 0.1: Excellent pitch preservation
- 0.1-0.3: Moderate pitch/voicing errors
- > 0.3: Significant prosodic degradation

**What to listen for:** Intonation errors, pitch jumps, voicing breaks, monotone artifacts

**Divergence scope:** More = worse pitch control; Less = better intonation

**Code:** `src/evaluation/acoustic_metrics.py:f0_frame_error()`

---

#### Pitch Pearson Correlation

**Calculation:** Pearson correlation coefficient on mutually-voiced F0 frames

**Range:** [-1, 1]

**Interpretation:**
- **Higher is better** (1 = perfect correlation)
- > 0.8: Excellent prosody preservation
- 0.5-0.8: Good correlation, minor prosodic shifts
- < 0.5: Significant prosodic divergence
- Negative: Inverted/corrupted pitch contour (severe failure)

**What to listen for:** Overall prosody preservation, question intonation, emotional tone

**Divergence scope:** More (→1) = better prosody matching; Less (→0) = degradation

**Code:** `src/evaluation/acoustic_metrics.py:pitch_pearson_correlation()`

---

### Phase 2: Token-Level Metrics

#### Teacher-Forced Argmax Mismatch Rate

**Calculation:** `mean(argmax(P_baseline) != argmax(P_quant))` at each teacher-forced step

**Range:** [0, 1] (fraction)

**Interpretation:**
- **Lower is better** (0 = always agree on top choice)
- < 0.1: Excellent agreement (Orpheus 8-bit typical)
- 0.1-0.3: Good agreement (Orpheus 4-bit typical)
- 0.3-0.5: Moderate disagreement (OuteTTS 8-bit typical)
- > 0.5: High disagreement (aggressive quants)

**What to monitor:** How often quantization changes the model's "opinion" on best token

**Divergence scope:** More = top predictions disagree more; Less = better agreement

**Code:** `src/decoding/distribution.py:compare_distributions()`

---

#### Teacher-Forced KL Divergence

**Calculation:** `D_KL(P_baseline || P_quant)` averaged over teacher-forced steps (drift-free)

**Range:** [0, ∞), unbounded

**Interpretation:**
- **Lower is better** (0 = distributions identical)
- < 0.01: Minimal impact (Orpheus 8-bit typical)
- 0.01-0.1: Low impact (good 8-bit quants)
- 0.1-0.5: Moderate impact (4-bit quants)
- > 0.5: Significant distributional shift

**This is the most reliable distributional metric** — prefer over free-run KL

**What to monitor:** True quantization impact on predictive distribution

**Divergence scope:** More = larger distributional shift; Less = better preservation

**Code:** `src/decoding/distribution.py:compare_distributions()`

---

#### Teacher-Forced Perplexity (Baseline)

**Calculation:** `exp(-mean(log(P_baseline[token_chosen_by_quant])))`

**Range:** [1, ∞)

**Interpretation:**
- **Lower is better** (closer to 1)
- Intuitive scale: perplexity of N = baseline is as surprised as uniformly choosing among N tokens
- < 5: Excellent quality (baseline finds outputs highly plausible)
- 5-20: Good to moderate quality
- > 20: Significant quality loss (outputs implausible)

**This is the headline quality metric** — directly measures quantization-induced degradation

**What to monitor:** Overall quantization quality impact

**Divergence scope:** Higher = baseline more "surprised" (worse); Lower = outputs plausible (better)

**Code:** Computed in `src/evaluate.py` teacher-forced analysis

---

#### First Token Divergence Rate (FTDR)

**Calculation:** `fraction of samples where argmax(baseline[0]) != argmax(quant[0])`

**Range:** [0, 1] (fraction of samples)

**Interpretation:**
- **Lower is better** (0 = always agree on first token)
- 0: Excellent initial stability (typical for most TTS quants)
- 0.01-0.05: Minor early divergence risk
- > 0.05: Significant trajectory corruption from step 1

**What to monitor:** Whether quantization corrupts generation from the very start

**Divergence scope:** More = first token often wrong; Less = stable initial generation

**Code:** `src/decoding/distribution.py:first_token_divergence_rate()`

---

#### Codebook Hierarchy Divergence

**Calculation:** Argmax mismatch counts bucketed by step % tokens_per_frame (teacher-forced)

**Range:** [0, 1] per level (coarse/medium/fine for RVQ codecs)

**Interpretation:**
- **Lower is better** for each level
- Coarse errors more impactful (low-freq structure)
- Fine errors less perceptible (high-freq detail)
- Look for patterns: does quantization preferentially damage one level?

**What to monitor:** Which codec levels are most affected

**Divergence scope:** Higher coarse rate = structural issues; Higher fine rate = detail loss

**Code:** `src/decoding/alignment.py:teacher_forced_hierarchy_divergence()`

---

### Metric Summary Table

| Metric | Range | Lower/Higher Better | Calculation | Typical Values |
|--------|-------|---------------------|-------------|----------------|
| **MCD** | [0, ∞) dB | Lower | DTW-aligned MFCC distance | 5-20 dB |
| **F0 Frame Error** | [0, 1] | Lower | Voicing/pitch mismatch fraction | 0.1-0.3 |
| **Pitch Pearson** | [-1, 1] | Higher | F0 contour correlation | 0.5-0.8 |
| **UTMOS** | [1, 5] | Higher | Neural MOS prediction | 3-5 |
| **TF Argmax Mismatch** | [0, 1] | Lower | Top-1 disagreement rate | 0.1-0.5 |
| **TF KL Divergence** | [0, ∞) | Lower | Drift-free distributional shift | 0.01-0.5 |
| **TF Perplexity** | [1, ∞) | Lower | Baseline surprise at outputs | 1-20 |
| **FTDR** | [0, 1] | Lower | First-token divergence fraction | 0-0.05 |
| **Top-k Jaccard** | [0, 1] | Higher | Candidate set overlap | 0.5-0.9 |
| **Codebook Rates** | [0, 1] | Lower | Level-wise token mismatch | 0.1-0.4 |
| **Compression Ratio** | [0, ∞) | Higher | Size reduction factor | 1.5-3x |

---

## How to Use These Metrics

#### Primary Metrics (Most Important)

1. **teacher_forced_perplexity_baseline** — Overall quality impact
2. **teacher_forced_argmax_mismatch_rate** — Token-level stability
3. **mean_mcd** — Acoustic spectral quality
4. **mean_pitch_pearson_correlation** — Prosody preservation

#### Secondary Metrics (Deep Analysis)

- **teacher_forced_mean_kl** — Distributional shift magnitude
- **codebook hierarchy rates** — Which codec levels degrade
- **first_token_divergence_rate** — Initial stability

#### For Audio Listening Tests

When comparing baseline vs quantized audio, listen for:

1. **Timbre/spectral (MCD):** Clarity, "muddiness", vowel quality
2. **Pitch/prosody (F0 metrics):** Intonation, monotone artifacts, question contours
3. **Intelligibility (WER/CER):** Word recognition accuracy
4. **Naturalness (UTMOS):** Overall human-like quality

#### Typical Quality Profiles

**Excellent quantization (8-bit, good algorithm):**
- teacher_forced_argmax_mismatch < 0.15
- mean_mcd < 8 dB
- mean_pitch_pearson > 0.7
- teacher_forced_perplexity_baseline < 5

**Acceptable quantization (4-bit):**
- teacher_forced_argmax_mismatch 0.2-0.3
- mean_mcd 8-15 dB
- mean_pitch_pearson 0.5-0.7
- teacher_forced_perplexity_baseline 5-10

**Degraded quantization (aggressive compression):**
- teacher_forced_argmax_mismatch > 0.3
- mean_mcd > 15 dB
- mean_pitch_pearson < 0.5
- teacher_forced_perplexity_baseline > 10

---

## Excel Report Sheet Descriptions

`outputs/evaluation/analysis_report.xlsx` contains:

### Summary Sheet
Pivoted metrics: one row per metric, one column per (model, quant_type). Use this for cross-method comparison.

### PerSample Sheet
Per-sample breakdown: MCD, F0 error, pitch correlation, WER/CER, codebook divergence, transcripts. Use for identifying outlier samples.

### LogProbs_KL Sheet
Per-step teacher-forced comparisons: baseline/quant token IDs, probabilities, KL/JS/argmax/Jaccard. Use for step-by-step distribution analysis.

### Codec Sheet
Codebook divergence by hierarchy level per sample. Use for RVQ-specific analysis (coarse/medium/fine breakdown).

### Hidden _*Data Sheets
Raw upsert sources for the display sheets. Modify these programmatically; the display sheets auto-update via pivot.

---

*For implementation details, see source files in `src/evaluation/`, `src/decoding/`, and `src/models/`. For analysis methodology, see `CLAUDE.md`.*