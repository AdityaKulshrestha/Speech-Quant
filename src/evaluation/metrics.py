"""
Token-divergence metrics from METRICS.md: First Divergence Position (FDP)
and Cumulative Divergence Rate (D(t)), comparing a quantized model's
generated codec tokens against the full-precision baseline for the same
prompt.

Extended metrics:
  - codebook_ids_for_tokens: maps flat token positions to SNAC codebook (0/1/2)
  - probability_difference: per-position baseline_prob - quant_prob for chosen token
"""

import torch

# Orpheus 7-token frame layout: [L1, L2, L3, L3, L2, L3, L3]
_ORPHEUS_FRAME_CODEBOOK = [0, 1, 2, 2, 1, 2, 2]


def codebook_ids_for_tokens(
    num_tokens: int,
    tokens_per_frame: int = 7,
    frame_layout: list[int] | None = None,
) -> list[int]:
    """Codebook index for each flat token position, generic across codec families.

    Orpheus/SNAC (tokens_per_frame=7) keeps its coarse/medium/fine layout by
    default. Any other frame size (e.g. Qwen TTS's variable code-group count,
    or 1 for flat FSQ codecs) falls back to the identity layout: slot t%tpf
    maps to codebook id t%tpf.
    """
    if frame_layout is None:
        frame_layout = _ORPHEUS_FRAME_CODEBOOK if tokens_per_frame == 7 else list(range(tokens_per_frame))
    return [frame_layout[t % tokens_per_frame] for t in range(num_tokens)]


def probability_difference(
    baseline_probs: torch.Tensor,
    quant_probs: torch.Tensor,
    baseline_offsets: torch.Tensor,
) -> list[float]:
    """Per-position (p_baseline - p_quant) for the token chosen by the baseline model."""
    n = min(len(baseline_probs), len(quant_probs), len(baseline_offsets))
    diffs = []
    for i in range(n):
        idx = int(baseline_offsets[i].item())
        diffs.append(float(baseline_probs[i, idx].item()) - float(quant_probs[i, idx].item()))
    return diffs


def negative_log_likelihood(token_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-position NLL = -log(prob of the chosen/reference token)."""
    return -token_probs.clamp(min=eps).log()



def first_sampled_mismatch_position(baseline_tokens: torch.Tensor, quant_tokens: torch.Tensor) -> int | None:
    """Index of the first sampled token mismatch in free-run generation.

    Note: This is affected by both quantization error and autoregressive drift.
    For the drift-free FTDR metric, see decoding.distribution.first_token_divergence_rate().

    Returns:
        Position of first mismatch, or None if sequences match
    """
    n = min(baseline_tokens.numel(), quant_tokens.numel())
    mismatches = (baseline_tokens[:n] != quant_tokens[:n]).nonzero(as_tuple=True)[0]

    if len(mismatches) == 0:
        return None

    return mismatches[0].item()


def cumulative_divergence_rate(baseline_tokens: torch.Tensor, quant_tokens: torch.Tensor) -> list[float]:
    """D(t) for t = 1..n: running mismatch rate up to each position."""

    n = min(baseline_tokens.numel(), quant_tokens.numel())

    if n == 0:
        return []

    mismatches = (baseline_tokens[:n] != quant_tokens[:n]).float()
    running_total = torch.cumsum(mismatches, dim=0)
    position = torch.arange(1, n + 1, dtype=torch.float32)

    return (running_total / position).tolist()


def compare_sequences(baseline_tokens: torch.Tensor, quant_tokens: torch.Tensor) -> dict:
    """Free-run sequence comparison (first mismatch + D(t) divergence curve)."""

    divergence_curve = cumulative_divergence_rate(baseline_tokens, quant_tokens)

    return {
        "baseline_length": baseline_tokens.numel(),
        "quant_length": quant_tokens.numel(),
        "compared_length": min(baseline_tokens.numel(), quant_tokens.numel()),
        "first_sampled_mismatch_position": first_sampled_mismatch_position(baseline_tokens, quant_tokens),
        "final_divergence_rate": divergence_curve[-1] if divergence_curve else None,
        "divergence_curve": divergence_curve,
    }


def summarize_scores(per_sample: list[dict]) -> dict:
    """Aggregate free-run metrics (first mismatch, D(t), prob-diff, KL) across samples."""

    fdps = [s["first_sampled_mismatch_position"] for s in per_sample if s.get("first_sampled_mismatch_position") is not None]
    finals = [s["final_divergence_rate"] for s in per_sample if s.get("final_divergence_rate") is not None]
    mean_pdiffs = [s["mean_prob_difference"] for s in per_sample if s.get("mean_prob_difference") is not None]
    mean_kls = [s["mean_kl_divergence"] for s in per_sample if s.get("mean_kl_divergence") is not None]

    return {
        "num_samples": len(per_sample),
        "num_samples_with_divergence": len(fdps),
        "mean_first_sampled_mismatch_position": sum(fdps) / len(fdps) if fdps else None,
        "mean_final_divergence_rate": sum(finals) / len(finals) if finals else 0.0,
        "mean_prob_difference": sum(mean_pdiffs) / len(mean_pdiffs) if mean_pdiffs else None,
        "mean_kl_divergence": sum(mean_kls) / len(mean_kls) if mean_kls else None,
    }
