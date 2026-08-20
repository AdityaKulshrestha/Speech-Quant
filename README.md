Good — that's a smart starting point since first-token divergence gives you a clean, cheap signal before you commit to full generation pipelines. Here's how I'd expand that into a full methodology for capturing and understanding PTQ behavior on the AR model.

## 1. Sensitivity analysis (before/instead of picking a PTQ method blindly)

This tells you *where* quantization will hurt before you spend compute quantizing everything uniformly.

- **Hessian-based sensitivity** (as in GPTQ/OBQ lineage) — approximate per-layer/per-weight loss curvature to rank which layers are fragile. In codec-token AR models, expect the output projection (to codebook vocabulary) and early transformer blocks to be more sensitive than middle layers, similar to text LLMs.
- **Weight and activation outlier analysis** — plot per-channel magnitude distributions (like the SmoothQuant/LLM.int8() outlier-feature analysis). Audio-token embedding spaces often have different outlier structure than text because codebook indices don't carry the same "semantic frequency" distribution as word tokens — worth checking whether outliers cluster in specific channels or specific codebook-index ranges.
- **Layer-wise cosine similarity / drift** — compare hidden states of FP16 vs quantized model at every layer, not just output. This localizes *where in the network* error is introduced vs where it's merely propagated.

## 2. PTQ methods to actually apply and compare

Don't just pick one — a workshop paper is much stronger if it's a controlled comparison:

- **RTN (round-to-nearest)** as your naive baseline — necessary to show that "smarter" PTQ matters.
- **GPTQ** — layer-wise reconstruction using Hessian info; strong general baseline.
- **AWQ** — activation-aware, protects salient weight channels; worth testing since it's often more robust at low bit-widths.
- **SmoothQuant** — if you're also quantizing activations, not just weights, since AR codec models with long sequences will be activation/KV-bound at inference.
- **KV-cache quantization (KIVI, H2O, or simple per-channel KV quant)** — likely your most *practically relevant* result, since AR decoding over long audio sequences is KV-cache-memory-dominated. This is under-explored for audio and could be your paper's differentiator.
- Bit-width sweep: 8-bit, 4-bit, and maybe 3-bit to show a degradation curve, not just a single operating point.

## 3. Extending "first codec divergence" into a full divergence-over-time story

This is the part I'd push hardest on, since it's the most novel angle and builds directly on what you've already started.

- **Per-step divergence trajectory**: compute KL or JS divergence between FP16 and quantized next-token distributions at *every* AR step (not just the first), and plot divergence vs generation step. This shows whether errors compound, saturate, or self-correct.
- **Per-codebook-level divergence**: if your AR model predicts a codebook hierarchy (coarse/semantic token first, finer acoustic tokens after, as in VALL-E-style or RVQ-flattened models), break divergence down *by codebook depth*. Hypothesis worth testing: quantization hurts the first (semantic) codebook less than later (acoustic/fine-detail) codebooks, or vice versa — either result is a genuinely interesting finding.
- **Token-level edit distance / argmax mismatch rate** as a cheaper, more interpretable companion to KL divergence — "what fraction of generated tokens differ from the FP16 model's greedy choice at each step."
- **Cumulative divergence vs sequence length** — bucket generations by length and show whether long-form generation is disproportionately affected. This is a natural bridge to your stability/hallucination metrics below.
- **Wasserstein/earth-mover's distance** between full output token-probability distributions as an alternative to KL when distributions have disjoint support (common at very low bit-widths where quantized model assigns near-zero probability to many tokens).

## 4. Downstream behavioral evaluation (ties divergence numbers to perceptual impact)

Divergence metrics are necessary but reviewers will want to see they *matter*:

- **WER** (via Whisper) — does divergence translate into intelligibility loss?
- **Speaker similarity** (WavLM/ECAPA embeddings) — does quantization noise bleed into speaker identity, especially if it hits acoustic-detail codebooks harder?
- **UTMOS or similar naturalness proxy**
- **Repetition/hallucination/premature-stop rate** — AR models are prone to degenerate loops; check if quantization increases this, since divergence compounding could plausibly manifest exactly as runaway repetition.
- Correlate all of these against your per-step divergence curves — a scatter of "divergence at step N vs downstream WER" is a compelling figure.

## 5. Practical tooling recommendations

- Use a **fixed calibration set** across methods for fair comparison, but also run a small ablation on calibration data source (codec-token sequences vs random/text-derived) since this is an easy, cheap, and reviewer-friendly ablation.
- Log everything at the **token-probability level**, not just final audio, so you can compute all divergence metrics post-hoc without re-running generation.
- Consider **bootstrap confidence intervals** on your metrics given TTS/audio-gen eval noise — workshop reviewers increasingly expect this given how noisy MOS/WER proxies can be.

