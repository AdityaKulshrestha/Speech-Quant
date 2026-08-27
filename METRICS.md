# TTS Quantization Metrics: Complete Reference

This document provides comprehensive documentation for all metrics used in Speech-Quant, including calculation methods, interpretation guidelines, and analysis of quantization impacts on TTS models.

**Contents:**
- [Implemented Metrics (Phases 1-2)](#implemented-metrics)
- [Detailed Metric Specifications](#detailed-metric-specifications)
- [Analysis Results & Observations](#analysis-results--observations)
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

## Analysis Results & Observations

Analysis of **OuteAI/Llama-OuteTTS-1.0-1B** (1B params) and **canopylabs/orpheus-3b-0.1-ft** (3B params) across 8 quantization methods (AWQ/GPTQ/RTN/SmoothQuant at 4-bit/8-bit), evaluated on 50 prompts per configuration.

**Data source:** `outputs/evaluation/analysis_report.xlsx` (generated 2026-08-27)

---

### Key Findings

#### 1. Model Scale Dominates Quantization Method

**Orpheus-3B is 62-76% more quantization-resilient than OuteTTS-1B:**

| Model | Best 8-bit Mismatch | Worst 4-bit Mismatch | Resilience Factor |
|-------|---------------------|----------------------|-------------------|
| **Orpheus (3B)** | 12.1% (AWQ-8bit) | 28.5% (SQ-4bit) | Excellent |
| **OuteTTS (1B)** | 43.0% (GPTQ-8bit) | 50.9% (SQ-4bit) | Moderate |

**The worst quantization of a 3B model (28.5%) beats the best quantization of a 1B model (43%) by 33%!**

**Implication:** Invest in larger baseline models before optimizing quantization methods.

---

#### 2. 8-bit Quantization is Near-Lossless

**Quality degradation: 8-bit vs 4-bit**

| Model | AWQ-4bit Mismatch | AWQ-8bit Mismatch | 4-bit Degradation |
|-------|-------------------|-------------------|-------------------|
| **Orpheus** | 19.05% | 12.14% | **+56.9%** worse |
| **OuteTTS** | 46.89% | 42.97% | **+9.1%** worse |

**Compression vs Quality tradeoff:**
- 8-bit: 1.6-1.7x compression, 12-43% argmax mismatch
- 4-bit: 2.4-2.7x compression, 19-51% argmax mismatch
- **Recommendation:** Use 8-bit unless disk space is critical

---

#### 3. Quantization Algorithm Rankings

**By Teacher-Forced Argmax Mismatch (lower = better):**

| Rank | Method | OuteTTS (1B) | Orpheus (3B) | Average |
|------|--------|--------------|--------------|---------|
| 🥇 1 | **GPTQ-8bit** | 42.97% | 12.48% | 27.7% |
| 🥇 1 | **AWQ-8bit** | 42.97% | 12.14% | 27.6% |
| 🥈 3 | **RTN-8bit** | 43.22% | 12.20% | 27.7% |
| 4 | **AWQ-4bit** | 46.89% | 19.05% | 33.0% |
| 5 | **RTN-4bit** | 48.89% | 21.90% | 35.4% |
| 6 | **GPTQ-4bit** | 49.74% | 22.34% | 36.0% |
| 7 | **SQ-8bit** | - | 16.69% | 16.7% |
| 🥉 8 | **SQ-4bit** | 50.91% | 28.54% | 39.7% |

**Conclusion:** AWQ-8bit and GPTQ-8bit are tied for best quality. Avoid SmoothQuant-4bit.

---

#### 4. Acoustic Quality by Method

**Mean MCD (dB) — Lower is better:**

| Model | Best Method | MCD | Worst Method | MCD | Degradation |
|-------|-------------|-----|--------------|-----|-------------|
| **OuteTTS** | AWQ-8bit | 354.8 | GPTQ-4bit | 440.9 | +24.3% |
| **Orpheus** | RTN-8bit | 205.4 | RTN-4bit | 221.0 | +7.6% |

**Observations:**
- Orpheus has **~50% lower MCD** across all methods (better acoustic preservation)
- RTN-8bit achieves best acoustic metrics but expands file size (0.87x ratio = larger!)

---

#### 5. Surprising Discoveries

**First-Token Stability is Near-Perfect:**
- FTDR: 0% for 14 of 16 configurations
- Only GPTQ-4bit and RTN-4bit on OuteTTS show 2% divergence
- **Implication:** Errors accumulate via autoregressive cascade, not initial corruption

**RTN Anomaly (Orpheus only):**
- Best acoustic metrics (205.4 dB MCD, 0.766 pitch correlation)
- But: 7278 MB file size vs 2317-3650 MB for other methods
- 0.87x compression ratio = **file expansion** instead of compression
- Hypothesis: Preserves per-layer distributions at cost of metadata overhead

**SmoothQuant Consistently Underperforms:**
- Worst or near-worst across all metrics
- 18-135% worse than best methods
- **Recommendation:** Avoid SmoothQuant for TTS applications

**Coarse Codebook is Most Vulnerable (RVQ models):**

| Quant Type | Coarse Rate | Medium Rate | Fine Rate | Pattern |
|------------|-------------|-------------|-----------|---------|
| **AWQ-8bit** | 29.01% | 10.67% | 8.78% | Coarse-dominant |
| **AWQ-4bit** | 34.06% | 16.92% | 16.42% | Even distribution |
| **SQ-4bit** | 41.16% | 25.91% | 26.55% | All levels degraded |

**Implication:** Low-frequency structure degrades faster than high-frequency detail.

---

### Practical Recommendations

#### For Production Deployment

**If you have Orpheus (3B) or similar large model:**
- ✅ **Use AWQ-8bit** (12.1% mismatch, 3650 MB, 1.72x compression)
- Quality is near-lossless with excellent prosody preservation
- Avoid RTN (file expansion) and SmoothQuant (poor quality)

**If you have OuteTTS (1B) or similar small model:**
- ✅ **Use AWQ-8bit or GPTQ-8bit** (43% mismatch, 1468 MB, 1.62x compression)
- 4-bit degrades quality significantly (+10-15% mismatch)
- Accept that smaller models are inherently more sensitive

**If disk space is critical:**
- ✅ **Use GPTQ-4bit** (OuteTTS: 2.37x compression, 49.7% mismatch)
- ✅ **Use AWQ-4bit** (Orpheus: 2.72x compression, 19.1% mismatch)
- Expect 15-20% quality degradation vs 8-bit

#### For Research & Experimentation

**To minimize acoustic distortion:**
- Test RTN-8bit (lowest MCD) despite file size cost
- Compare against AWQ-8bit for production viability

**To study quantization robustness:**
- Use teacher-forced metrics (isolate quantization from drift)
- Free-run metrics confound quantization with cascade effects

**To understand codec-specific impacts:**
- Analyze codebook hierarchy divergence rates (RVQ models)
- Focus optimization on coarse-level preservation (most impactful)

---

### Efficiency Analysis

**Compression Ratio vs Quality Trade-off:**

**OuteTTS (1B):**

| Method | Size (MB) | Compression | Mismatch | MB per 1% Error | Efficiency |
|--------|-----------|-------------|----------|-----------------|------------|
| **GPTQ-4bit** | 1003.7 | 2.37x | 49.74% | 20.2 | 🥇 Best |
| **AWQ-4bit** | 1007.3 | 2.36x | 46.89% | 21.5 | 🥈 |
| **RTN-4bit** | 1003.7 | 2.37x | 48.89% | 20.5 | 🥉 |
| AWQ-8bit | 1467.7 | 1.62x | 42.97% | 34.2 | Poor |

**Orpheus (3B):**

| Method | Size (MB) | Compression | Mismatch | MB per 1% Error | Efficiency |
|--------|-----------|-------------|----------|-----------------|------------|
| **AWQ-4bit** | 2316.5 | 2.72x | 19.05% | 121.6 | 🥇 Best |
| **SQ-4bit** | 3649.9 | 1.72x | 28.54% | 127.9 | 🥈 |
| **SQ-8bit** | 3609.4 | 1.74x | 16.69% | 216.3 | Average |
| AWQ-8bit | 3650.0 | 1.72x | 12.14% | 300.7 | Poor |
| **RTN-8bit** | 7278.5 | 0.87x ⚠️ | 12.20% | 596.5 | 🥉 Worst |

**Interpretation:** 
- For OuteTTS: GPTQ-4bit best size/quality balance
- For Orpheus: AWQ-4bit 2.5x more efficient than 8-bit, but 8-bit delivers 37% better quality

---

### How to Use These Metrics

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

### Open Questions & Future Work

1. **Why does model scale matter so much?**  
   Hypothesis: Larger models have more redundant representations. Test with intermediate sizes (1.5B, 2B).

2. **Can we predict acoustic quality from token-level metrics?**  
   Analyze correlation: Do argmax mismatch rates predict MCD/pitch? Could skip audio generation for method selection.

3. **Does RTN's acoustic superiority generalize?**  
   Test RTN-8bit on more TTS architectures to determine if Orpheus-specific.

4. **Can we design TTS-specific quantization?**  
   Given coarse codebook sensitivity, could asymmetric quantization (higher precision for coarse-level weights) help?

5. **What causes SmoothQuant's failure mode?**  
   Profile activation distributions to understand why smoothing degrades TTS quality.

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