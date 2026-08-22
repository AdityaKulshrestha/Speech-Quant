"""
Teacher-forced distribution analysis for baseline vs quantized TTS models.

Core idea
---------
Free-running comparison confounds two effects: quantization error at step t AND
the shifted context caused by all prior mispredictions. Teacher forcing on the
FP16 reference sequence breaks the conflation: both models receive the same
prefix at every step, so differences in the output distribution are purely due
to quantization.

Workflow
--------
    # 1. FP16 free run (existing flow) — audio_logits are already teacher-forced
    #    on its own sequence, so no re-run is needed for the baseline.
    baseline_out = fp16_model.generate_tokens(inputs)

    # 2. Quantized model — teacher-forced on the FP16 reference sequence.
    quant_probs = extract_teacher_forced_logits(
        model=quant_model.model,
        reference_ids=baseline_out.generated_ids,
        audio_token_start=OrpheusTTS.AUDIO_TOKEN_START,
        audio_token_end=OrpheusTTS.AUDIO_TOKEN_START
                        + OrpheusTTS.TOKENS_PER_FRAME * OrpheusTTS.CODEBOOK_SIZE,
        eos_token_id=OrpheusTTS.END_OF_SPEECH,
        prompt_length=baseline_out.metadata["input_length"],
    )

    # 3. Compare per-frame-position distributions.
    results = compare_distributions(
        baseline_probs=baseline_out.audio_logits.float(),
        quant_probs=quant_probs,
        tokens_per_frame=OrpheusTTS.TOKENS_PER_FRAME,
    )
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


@torch.inference_mode()
def extract_teacher_forced_logits(
    model: Any,
    reference_ids: torch.Tensor,
    audio_token_start: int,
    audio_token_end: int,
    eos_token_id: int,
    prompt_length: int,
    tokens_per_frame: int = 7,
) -> torch.Tensor:
    """Single forward pass on the full reference sequence; returns audio-subspace
    softmax probs at each audio token position.

    This is the key decoupling point: the model is called once with the complete
    FP16 reference sequence, and logits are read at each audio token position
    without any autoregressive generation loop.

    Args:
        model: HuggingFace causal LM in eval mode (fp16/bf16 or quantized).
        reference_ids: [1, T] int64 — complete generated_ids from the FP16 run.
        audio_token_start: first audio vocab index (Orpheus: 128266).
        audio_token_end: exclusive upper bound of audio vocab range.
        eos_token_id: token that marks the end of the generated audio.
        prompt_length: number of prompt tokens before the generated portion.
        tokens_per_frame: truncate result to complete frames (Orpheus: 7).

    Returns:
        [num_audio_tokens, audio_vocab_size] float32 on CPU, where
        num_audio_tokens is a multiple of tokens_per_frame.
    """
    audio_size = audio_token_end - audio_token_start

    # Single forward pass — no generation, no KV-cache stepping.
    logits = model(reference_ids).logits[0]  # [T, vocab_size]

    gen_tokens = reference_ids[0, prompt_length:]
    audio_probs: list[torch.Tensor] = []

    for j, tok in enumerate(gen_tokens):
        tok_val = int(tok.item())
        if tok_val == eos_token_id:
            break
        if audio_token_start <= tok_val < audio_token_end:
            # logits[p-1] predicts the token at absolute position p;
            # gen_token[j] sits at absolute position (prompt_length + j).
            step_logits = logits[prompt_length + j - 1, audio_token_start:audio_token_end]
            audio_probs.append(F.softmax(step_logits.float(), dim=-1).cpu())

    if not audio_probs:
        return torch.zeros(0, audio_size, dtype=torch.float32)

    # Align to complete frames so positions are comparable to baseline audio_logits.
    n_complete = (len(audio_probs) // tokens_per_frame) * tokens_per_frame
    return torch.stack(audio_probs[:n_complete])


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
    tokens_per_frame: int = 7,
    topk: int = 5,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Full distribution-level comparison between baseline and quantized logits.

    All metrics are bucketed by frame position (step % tokens_per_frame) to
    isolate coarse vs fine codec levels — position 0 is coarse (L1), positions
    1-2 are mid (L2), positions 3-6 are fine (L3) for Orpheus/SNAC.

    Args:
        baseline_probs: [T, V] float32 — FP16 audio-subspace softmax probs.
            For the FP16 model, pass GenerationOutput.audio_logits directly;
            its free-run logits are equivalent to teacher-forced logits because
            the reference sequence IS the FP16 model's own output.
        quant_probs:    [T, V] float32 — from extract_teacher_forced_logits().
        tokens_per_frame: 7 for Orpheus/SNAC, 1 for FSQ/NeuTTS.
        topk: neighbourhood size for Jaccard overlap metric.
        eps: numerical floor for KL/JS to avoid log(0).

    Returns:
        dict with three keys:
          ``per_step``         — raw per-step metric lists
          ``by_frame_position``— per-position mean/count aggregates
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

    # Bucket each per-step series by frame position.
    kl_by_pos = _bucket_by_frame_position(kl, tokens_per_frame)
    js_by_pos = _bucket_by_frame_position(js, tokens_per_frame)
    mm_by_pos = _bucket_by_frame_position(argmax_mismatch, tokens_per_frame)
    tk_by_pos = _bucket_by_frame_position(topk_jac, tokens_per_frame)

    topk_key = f"mean_top{topk}_jaccard"
    by_frame_position: dict[int, dict[str, Any]] = {}
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


__all__ = [
    "compare_distributions",
    "extract_teacher_forced_logits",
    "js_divergence_sequence",
    "topk_overlap_sequence",
]
