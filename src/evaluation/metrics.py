"""
Token-divergence metrics from METRICS.md: First Divergence Position (FDP)
and Cumulative Divergence Rate (D(t)), comparing a quantized model's
generated codec tokens against the full-precision baseline for the same
prompt.
"""

import torch


def first_divergence_position(baseline_tokens: torch.Tensor, quant_tokens: torch.Tensor) -> int | None:
    """Index of the first mismatched token, or None if the compared span matches fully."""

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
    """FDP + D(t) summary for one baseline/quantized token-sequence pair."""

    divergence_curve = cumulative_divergence_rate(baseline_tokens, quant_tokens)

    return {
        "baseline_length": baseline_tokens.numel(),
        "quant_length": quant_tokens.numel(),
        "compared_length": min(baseline_tokens.numel(), quant_tokens.numel()),
        "first_divergence_position": first_divergence_position(baseline_tokens, quant_tokens),
        "final_divergence_rate": divergence_curve[-1] if divergence_curve else None,
        "divergence_curve": divergence_curve,
    }


def summarize_scores(per_sample: list[dict]) -> dict:
    """Aggregate FDP/D(t) across samples for the final printed/dumped score."""

    fdps = [s["first_divergence_position"] for s in per_sample if s["first_divergence_position"] is not None]
    finals = [s["final_divergence_rate"] for s in per_sample if s["final_divergence_rate"] is not None]

    return {
        "num_samples": len(per_sample),
        "num_samples_with_divergence": len(fdps),
        "mean_first_divergence_position": sum(fdps) / len(fdps) if fdps else None,
        "mean_final_divergence_rate": sum(finals) / len(finals) if finals else 0.0,
    }
