"""
Teacher-forced distribution analysis for baseline vs quantized TTS models.

Core idea
---------
Free-running comparison confounds two effects: quantization error at step t AND
the shifted context caused by all prior mispredictions. Teacher forcing one
model on the OTHER model's own reference sequence breaks the conflation: both
models effectively see the same prefix at every step, so differences in the
output distribution are attributable to quantization rather than drift.

evaluate.py's `teacher_forced_distribution_compare()` forces the QUANTIZED
model's own free-run token sequence through the FULL-PRECISION baseline model
(one forward pass, no autoregressive loop) — measuring how well the reference
model "endorses" what quantization actually produced.

Workflow
--------
    # 1. Quantized model free run (existing flow) — audio_logits are already
    #    teacher-forced on its own sequence, so no re-run is needed for it.
    quant_out = quant_model.generate_tokens(inputs)

    # 2. Baseline (full-precision) model — teacher-forced on the quantized
    #    model's reference sequence. token_filter and vocab_slice are
    #    model-specific; pass None to collect every generated token over the
    #    full vocabulary.
    #
    #    Orpheus example:
    #      audio_start = OrpheusTTS.AUDIO_TOKEN_START
    #      audio_end   = audio_start + OrpheusTTS.TOKENS_PER_FRAME * OrpheusTTS.CODEBOOK_SIZE
    #      token_filter = lambda t: audio_start <= t < audio_end
    #      vocab_slice  = (audio_start, audio_end)
    baseline_probs, _token_ids = extract_teacher_forced_logits(
        model=baseline_model.model,
        reference_ids=quant_out.generated_ids,
        eos_token_id=OrpheusTTS.END_OF_SPEECH,
        prompt_length=quant_out.metadata["input_length"],
        token_filter=lambda t: audio_start <= t < audio_end,
        vocab_slice=(audio_start, audio_end),
        tokens_per_frame=OrpheusTTS.TOKENS_PER_FRAME,
    )

    # 3. Compare distributions; pass tokens_per_frame only for codecs that
    #    interleave multiple codec levels per frame (e.g. Orpheus/SNAC).
    results = compare_distributions(
        baseline_probs=baseline_probs,
        quant_probs=quant_out.audio_logits.float(),
        tokens_per_frame=OrpheusTTS.TOKENS_PER_FRAME,
    )
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F


@torch.inference_mode()
def extract_teacher_forced_logits(
    model: Any,
    reference_ids: torch.Tensor,
    eos_token_id: int,
    prompt_length: int,
    token_filter: Callable[[int], bool] | None = None,
    vocab_slice: tuple[int, int] | None = None,
    tokens_per_frame: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single forward pass on the full reference sequence; returns softmax probs
    (and the corresponding reference token ids) at each collected token position.

    This is the key decoupling point: the model is called once with the complete
    reference sequence, and logits are read at each selected position without
    any autoregressive generation loop.

    Args:
        model: HuggingFace causal LM in eval mode (fp16/bf16 or quantized).
        reference_ids: [1, T] int64 — complete generated_ids from the reference run.
        eos_token_id: token that marks the end of generation.
        prompt_length: number of prompt tokens before the generated portion.
        token_filter: ``(token_id) -> bool`` — collect logits only where True.
            Pass ``None`` to collect every non-EOS generated token.
            Orpheus: ``lambda t: audio_start <= t < audio_end``.
        vocab_slice: ``(start, end)`` to restrict the prob vector to a subspace.
            Pass ``None`` to return the full-vocabulary distribution.
            Orpheus: ``(audio_token_start, audio_token_end)``.
        tokens_per_frame: truncate result to complete frames. Default 1 (no-op).

    Returns:
        Tuple of:
          - ``[N, V']`` float32 probs on CPU, where N is a multiple of tokens_per_frame
            and V' is ``vocab_slice[1]-vocab_slice[0]`` when a slice is given, or the
            full vocabulary size otherwise.
          - ``[N]`` int64 reference token ids (full-vocabulary space), aligned 1:1 with
            the rows of the probs tensor.
    """
    # Single forward pass — no generation, no KV-cache stepping.
    logits = model(reference_ids).logits[0]  # [T, vocab_size]
    v_start, v_end = (0, logits.shape[-1]) if vocab_slice is None else vocab_slice
    out_size = v_end - v_start

    gen_tokens = reference_ids[0, prompt_length:]
    collected: list[torch.Tensor] = []
    collected_token_ids: list[int] = []

    for j, tok in enumerate(gen_tokens):
        tok_val = int(tok.item())
        if tok_val == eos_token_id:
            break
        if token_filter is None or token_filter(tok_val):
            # logits[p-1] predicts the token at absolute position p;
            # gen_token[j] sits at absolute position (prompt_length + j).
            step_logits = logits[prompt_length + j - 1, v_start:v_end]
            collected.append(F.softmax(step_logits.float(), dim=-1).cpu())
            collected_token_ids.append(tok_val)

    if not collected:
        return torch.zeros(0, out_size, dtype=torch.float32), torch.zeros(0, dtype=torch.long)

    # Align to complete frames so positions are comparable to baseline probs.
    n_complete = (len(collected) // tokens_per_frame) * tokens_per_frame
    probs = torch.stack(collected[:n_complete])
    token_ids = torch.tensor(collected_token_ids[:n_complete], dtype=torch.long)
    return probs, token_ids


def js_divergence_sequence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-step Jensen-Shannon divergence (symmetric, bounded in [0, ln 2]).

    Preferred over KL when one distribution can have near-zero mass where
    the other doesn't — common at aggressive bit-widths.

    Args:
        p, q: [T, V] softmax distributions (float32).

    Returns:
        [T] JS divergences.
    """
    n = min(len(p), len(q))
    p = p[:n].float().clamp(min=eps)
    q = q[:n].float().clamp(min=eps)
    m = 0.5 * (p + q)
    return 0.5 * ((p * (p / m).log()).sum(-1) + (q * (q / m).log()).sum(-1))


def topk_overlap_sequence(
    p: torch.Tensor,
    q: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """Per-step Jaccard overlap of the top-k tokens between two distributions.

    Softer than argmax mismatch: useful when quantization shifts probability
    mass without changing the single top-1 choice.

    Args:
        p, q: [T, V] softmax distributions (float32).
        k: neighbourhood size.

    Returns:
        [T] Jaccard scores in [0, 1].
    """
    n = min(len(p), len(q))
    p_idx = p[:n].topk(k, dim=-1).indices  # [T, k]
    q_idx = q[:n].topk(k, dim=-1).indices

    scores: list[float] = []
    for pt, qt in zip(p_idx, q_idx):
        pt_set = set(pt.tolist())
        qt_set = set(qt.tolist())
        union = len(pt_set | qt_set)
        scores.append(len(pt_set & qt_set) / union if union > 0 else 0.0)
    return torch.tensor(scores, dtype=torch.float32)


def _bucket_by_frame_position(
    values: torch.Tensor,
    tokens_per_frame: int,
) -> dict[int, list[float]]:
    """Group per-step scalar values by frame position (step % tokens_per_frame)."""
    buckets: dict[int, list[float]] = {pos: [] for pos in range(tokens_per_frame)}
    for t, v in enumerate(values.tolist()):
        buckets[t % tokens_per_frame].append(v)
    return buckets


def compare_distributions(
    baseline_probs: torch.Tensor,
    quant_probs: torch.Tensor,
    tokens_per_frame: int | None = None,
    topk: int = 5,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Full distribution-level comparison between baseline and quantized logits.

    When tokens_per_frame is given, metrics are additionally bucketed by frame
    position (step % tokens_per_frame) to isolate codec levels — useful for
    residual-VQ codecs like Orpheus/SNAC where each frame has multiple tokens
    at different granularities. Omit for single-token-per-step models.

    Args:
        baseline_probs: [T, V] float32 — from extract_teacher_forced_logits(),
            i.e. the model being scored via teacher forcing on the OTHER
            model's reference sequence.
        quant_probs:    [T, V] float32 — reference softmax probs, straight from
            the reference model's own free-run generation (its free-run logits
            are equivalent to teacher-forced logits on itself, since the
            reference sequence IS that model's own output).
        tokens_per_frame: bucket by frame position when set (e.g. 7 for
            Orpheus/SNAC). Pass None (default) for flat / single-token models.
        topk: neighbourhood size for Jaccard overlap metric.
        eps: numerical floor for KL/JS to avoid log(0).

    Returns:
        dict with keys:
          ``per_step``         — raw per-step metric lists
          ``by_frame_position``— per-position aggregates (empty when
                                 tokens_per_frame is None)
          ``summary``          — overall means across all steps
    """
    n = min(len(baseline_probs), len(quant_probs))
    if n == 0:
        return {"per_step": {}, "by_frame_position": {}, "summary": {"num_steps": 0}}

    bp = baseline_probs[:n].float()
    qp = quant_probs[:n].float()

    # KL(baseline || quant): measures surprise of quant relative to baseline.
    bp_c = bp.clamp(min=eps)
    qp_c = qp.clamp(min=eps)
    kl = (bp_c * (bp_c / qp_c).log()).sum(dim=-1)           # [T]

    js = js_divergence_sequence(bp, qp, eps=eps)             # [T]
    argmax_mismatch = (bp.argmax(-1) != qp.argmax(-1)).float()  # [T]
    topk_jac = topk_overlap_sequence(bp, qp, k=topk)        # [T]

    topk_key = f"mean_top{topk}_jaccard"

    by_frame_position: dict[int, dict[str, Any]] = {}
    if tokens_per_frame is not None:
        kl_by_pos = _bucket_by_frame_position(kl, tokens_per_frame)
        js_by_pos = _bucket_by_frame_position(js, tokens_per_frame)
        mm_by_pos = _bucket_by_frame_position(argmax_mismatch, tokens_per_frame)
        tk_by_pos = _bucket_by_frame_position(topk_jac, tokens_per_frame)
        for pos in range(tokens_per_frame):
            kl_v, js_v, mm_v, tk_v = kl_by_pos[pos], js_by_pos[pos], mm_by_pos[pos], tk_by_pos[pos]
            cnt = len(kl_v)
            by_frame_position[pos] = {
                "mean_kl": sum(kl_v) / cnt if cnt else 0.0,
                "mean_js": sum(js_v) / cnt if cnt else 0.0,
                "argmax_mismatch_rate": sum(mm_v) / cnt if cnt else 0.0,
                topk_key: sum(tk_v) / cnt if cnt else 0.0,
                "num_steps": cnt,
            }

    summary: dict[str, Any] = {
        "num_steps": n,
        "tokens_per_frame": tokens_per_frame,
        "mean_kl": float(kl.mean()),
        "mean_js": float(js.mean()),
        "argmax_mismatch_rate": float(argmax_mismatch.mean()),
        topk_key: float(topk_jac.mean()),
    }

    return {
        "per_step": {
            "kl": kl.tolist(),
            "js": js.tolist(),
            "argmax_mismatch": argmax_mismatch.tolist(),
            f"top{topk}_jaccard": topk_jac.tolist(),
        },
        "by_frame_position": by_frame_position,
        "summary": summary,
    }


def first_token_divergence_rate(dist_per_sample: list[dict[str, Any]]) -> float | None:
    """METRICS.md's First Token Divergence Rate (FTDR): fraction of samples whose very
    first teacher-forced audio token already has a greedy-argmax mismatch.

    Each ``dist_per_sample`` entry is one ``compare_distributions()`` result (from
    ``teacher_forced_distribution_compare`` in evaluate.py); ``per_step["argmax_mismatch"]``
    is a deterministic argmax comparison (baseline's own predicted token vs. the quant
    model's teacher-forced prediction), so this is unconfounded by sampling randomness
    or autoregressive drift — unlike the free-run ``first_divergence_position`` in
    evaluation/metrics.py, which can land much later purely because a lucky sampled
    draw happened to still agree after the underlying distributions already diverged.
    """
    diverged_first: list[bool] = []
    for entry in dist_per_sample:
        mismatches = (entry.get("per_step") or {}).get("argmax_mismatch") or []
        if mismatches:
            diverged_first.append(bool(mismatches[0]))

    if not diverged_first:
        return None
    return sum(diverged_first) / len(diverged_first)


__all__ = [
    "compare_distributions",
    "extract_teacher_forced_logits",
    "first_token_divergence_rate",
    "js_divergence_sequence",
    "topk_overlap_sequence",
]
