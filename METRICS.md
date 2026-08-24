# Key Metrics to Implement

## 1. Macroscopic Metrics (Phase 1)

- **Semantic Accuracy:** Pass synthesized audio through a frozen Whisper model to compute ASR-WER and CER.

- **Acoustic Distortion:**
  - **Mel-Cepstral Distortion (MCD):** Measures spectral envelope degradation.
  - **$F_0$ Frame Error / Pitch Pearson Correlation:** Evaluates intonation and prosodic collapse.
  - **UTMOS / NISQA:** Non-intrusive neural perceptual audio quality estimators.

- **Speaker Fidelity:** Cosine similarity of speaker embeddings (SECS) between reference and generated audio.

## 2. Microscopic Metrics (Phase 2)

- **Directional KL Divergence:** Quantifies the shift in predictive confidence:

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

- **Codebook-Wise Token Error Rate ($\mathrm{TER}_k$):** For each Residual Vector Quantization (RVQ) level $k \in \{0, \ldots, K-1\}$, calculate the percentage of token mismatches:

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

- **First Token Divergence Rate (FTDR):** Probability that the initial token deviates from the baseline under greedy decoding, setting the autoregressive trajectory off-course.

## 3. Causal & Sensitivity Metrics (Phase 3)

- **Error Amplification Factor ($\beta$):** The rate at which an injected perturbation at step $t_0$ increases $D_{\mathrm{KL}}$ at step $t_0 + \Delta t$.

- **Ablation Sensitivity Delta:** The marginal drop in downstream WER when quantizing layer $k$ while keeping all other layers in FP16.

## 4. What Each Score Means in the Excel Summary Sheet

`outputs/evaluation/analysis_report.xlsx`'s **Summary** sheet has one metric per row and one column per (model, quant_type). Rows fall into four groups:

Each row below lists: **what the value represents**, its **range**, and **direction** (which way is "better").

### Semantic / Acoustic (from transcription + acoustic analysis)
- **baseline_wer / baseline_cer, quant_wer / quant_cer:** Word/character error rate of the baseline's and the quant model's synthesized audio, each transcribed by ASR and compared to `ground_truth_text` (after `normalize_text` normalization).
  Range: $[0, 1]$ typically (can exceed 1 if insertions outnumber reference words). **Lower = better** (0 = perfect transcription, matches ground truth exactly); higher = more transcription errors, i.e. less intelligible/correct speech.
- **mean_mcd:** Mean Mel-Cepstral Distortion between baseline and quant audio — spectral envelope degradation, in dB.
  Range: $[0, \infty)$, typically single digits to ~15dB in practice. **Lower = better** (0 = identical spectral envelope); higher = more audible timbral/spectral distortion in the quantized output.
- **mean_f0_frame_error:** Mean pitch (F0) frame error between baseline and quant audio.
  Range: $[0, 1]$ (fraction of voiced frames with a pitch error beyond the tolerance threshold). **Lower = better** (0 = pitch tracks match on every frame); higher = more pitch/intonation mismatch.
- **mean_pitch_pearson_correlation:** Mean Pearson correlation of the F0 contour between baseline and quant audio (intonation/prosody similarity).
  Range: $[-1, 1]$. **Higher = better** (1 = pitch contours move together perfectly, 0 = no linear relationship, negative = contours move in opposite directions — i.e. prosody degraded/inverted).

### Free-run divergence (baseline and quant sampled independently, aligned by raw position — confounded by drift after the first mismatch)
- **first_token_divergence_rate:** Despite the name, this is now the deterministic **FTDR** computed from teacher-forced argmax mismatches at step 0 (see next group) — *not* the free-run first-mismatch position.
  Range: $[0, 1]$ (fraction of samples). **Lower = better** (0 = baseline and quant always agree on the very first generated token); higher = quantization corrupts the trajectory from the first step more often.
- **mean_final_divergence_rate:** Average (across samples) of the free-run cumulative mismatch rate $D(T)$ at the *last* compared position — i.e. what fraction of the whole free-run sequence ended up mismatched, token-for-token, between baseline and quant.
  Range: $[0, 1]$. **Lower = better** (0 = the two sequences matched token-for-token all the way through); 1 = every position mismatched by the end.
- **mean_prob_difference:** Average of $p_{\text{baseline}} - p_{\text{quant}}$ for the baseline's own chosen free-run token, per position, then averaged over the sequence and over samples.
  Range: $[-1, 1]$ (practically small, near 0). **Closer to 0 = better** (quant assigns the token near-identical probability to baseline); positive = quant systematically under-weights the token baseline preferred (i.e. quant is less confident in / more likely to reject baseline's choice); negative = quant actually favors that token more than baseline does.
- **mean_kl_divergence:** Average $D_{\mathrm{KL}}(P_{\text{baseline}} \parallel P_{\text{quant}})$ over the free-run token distributions, aligned by raw position index. This is the "naive" KL — it drifts out of sync once the two sequences diverge, so treat it as a rough signal, not ground truth (see `teacher_forced_mean_kl` below for the clean version).
  Range: $[0, \infty)$, unbounded. **Lower = better** (0 = distributions identical at every compared position); higher = larger distributional shift (interpret relative to other quant types, not against an absolute threshold).
- **codebook_mean_divergence_rate**, **codebook_{level}_rate:** Free-run token mismatch rate overall and per RVQ/FSQ hierarchy level (e.g. coarse/medium/fine for Orpheus/SNAC).
  Range: $[0, 1]$. **Lower = better** (0 = no codebook token mismatches at that level); higher = quantization disrupts that codebook level more.

### Teacher-forced divergence (quant model's own free-run output re-scored by the full-precision baseline in one forward pass — isolates pure quantization error from drift)
- **teacher_forced_mean_kl:** Mean KL divergence between the baseline's and quant's full probability distributions at each teacher-forced step. This is the metric to trust for "how different are the two models' distributions", since both models see the identical prefix at every step.
  Range: $[0, \infty)$, unbounded. **Lower = better** (0 = distributions identical at every step); higher = quantization shifted the model's predictive distribution more.
- **teacher_forced_mean_js:** Same idea as `teacher_forced_mean_kl` but symmetric and bounded (uses Jensen-Shannon instead of KL).
  Range: $[0, \ln 2] \approx [0, 0.693]$. **Lower = better** (0 = identical distributions); $\ln 2$ = maximally different (disjoint support).
- **teacher_forced_argmax_mismatch_rate:** Fraction of teacher-forced steps where the baseline's own top-1 token differs from the quant model's own top-1 token (independent of what token was actually generated).
  Range: $[0, 1]$. **Lower = better** (0 = the two models always agree on their single most-likely token); higher = the models' "opinions" disagree more often.
- **teacher_forced_mean_top{k}_jaccard:** Mean Jaccard overlap of the top-k tokens between the two distributions per step — a softer signal than argmax mismatch.
  Range: $[0, 1]$. **Higher = better** (1 = identical top-k candidate sets at every step); 0 = no overlap at all.
- **teacher_forced_mean_nll_baseline / teacher_forced_mean_nll_quant:** Mean $-\log(\text{prob})$ of the quant model's actual chosen token, scored under the baseline model vs. under the quant model itself.
  Range: $[0, \infty)$, unbounded. **Lower = better** (0 = the scoring model assigns probability 1 to that token, i.e. fully confident/correct); higher = the scoring model finds that token increasingly implausible/surprising.
- **teacher_forced_perplexity_baseline / teacher_forced_perplexity_quant:** $\exp$ of the NLLs above. `teacher_forced_perplexity_baseline` is the headline quality metric: "how surprised is the full-precision model by what quantization actually produced" — a direct measure of quantization-induced quality loss.
  Range: $[1, \infty)$. **Lower = better**, and it has an intuitive scale: a perplexity of $n$ means the model was, on average, as "surprised" as if it had to guess uniformly among $n$ equally-likely tokens. Close to 1 = baseline finds quant's outputs highly plausible (little quality loss); large values = quant produced tokens the reference model considers implausible (significant quality degradation).
- **teacher_forced_codebook_total_mismatched_tokens / _total_tokens**, **teacher_forced_codebook_{level}_mismatched_tokens / _rate:** Same argmax-mismatch counting as `teacher_forced_argmax_mismatch_rate`, but broken down by RVQ hierarchy level, using deterministic teacher-forced comparisons (distinct from the free-run `codebook_{level}_rate` above).
  The `_tokens` counts are raw integers (no fixed range); the `_rate` columns are $[0, 1]$ with **lower = better**, same direction as `teacher_forced_argmax_mismatch_rate`.

**Note:** two different KL computations coexist in this sheet — `mean_kl_divergence` (free-run, position-aligned, drift-confounded) and `teacher_forced_mean_kl` (single-pass teacher-forced, drift-free). Don't conflate them when comparing quant methods; for both, lower is better, but their absolute magnitudes are not directly comparable to each other.