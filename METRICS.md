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