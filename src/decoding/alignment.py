"""Alignment and codebook-divergence helpers for quantized audio codecs.

This module centralizes the logic used to compare baseline and quantized codec
streams. It is intentionally small and reusable so the main entrypoint in
``src/evaluate.py`` can stay focused on orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class CodecSpec:
    """Metadata describing the codec hierarchy used by a model."""

    family: str
    tokens_per_frame: int = 7
    hierarchy: dict[str, list[int]] | None = None


def detect_codec_family(model_name: str | None) -> str:
    """Infer whether the model uses RVQ or FSQ-style tokenization."""
    name = (model_name or "").lower()
    if any(token in name for token in ("orpheus", "snac", "rvq")):
        return "rvq"
    if any(token in name for token in ("neutts", "fsq", "neucodec")):
        return "fsq"
    return "rvq"


def get_codec_spec(model_name: str | None = None, family: str | None = None) -> CodecSpec:
    """Return the hierarchy metadata used to bucket mismatch severity by token level."""
    codec_family = (family or detect_codec_family(model_name) or "rvq").lower()
    if codec_family == "rvq":
        return CodecSpec(
            family="rvq",
            tokens_per_frame=7,
            hierarchy={
                "coarse": [0],
                "medium": [1, 2],
                "fine": [3, 4, 5, 6],
            },
        )
    return CodecSpec(
        family="fsq",
        tokens_per_frame=1,
        hierarchy={"flat": [0]},
    )


def normalize_text(text: str) -> str:
    """Lowercase + collapse whitespace to make comparisons robust."""
    return " ".join((text or "").lower().strip().split())


def levenshtein_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Return the edit distance between two token sequences."""
    rows = len(ref) + 1
    cols = len(hyp) + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_tokens = normalize_text(reference).split()
    hyp_tokens = normalize_text(hypothesis).split()
    if not ref_tokens and not hyp_tokens:
        return 0.0
    distance = levenshtein_distance(ref_tokens, hyp_tokens)
    return distance / max(len(ref_tokens), 1)


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref_chars = list(normalize_text(reference))
    hyp_chars = list(normalize_text(hypothesis))
    if not ref_chars and not hyp_chars:
        return 0.0
    distance = levenshtein_distance(ref_chars, hyp_chars)
    return distance / max(len(ref_chars), 1)


def pairwise_wer_matrix(labels: Sequence[str], text_groups: dict[str, Sequence[str]]) -> dict[str, Any]:
    """Build an aggregate WER grid among ground-truth, baseline, and quantized outputs."""
    label_order = list(labels)
    n = len(label_order)
    if not n:
        return {"labels": [], "mean_wer_matrix": []}

    lengths = {label: len(text_groups.get(label, [])) for label in label_order}
    sample_count = min(lengths.values()) if lengths else 0
    if sample_count == 0:
        return {"labels": label_order, "mean_wer_matrix": [[0.0 for _ in range(n)] for _ in range(n)]}

    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i, left in enumerate(label_order):
        left_texts = text_groups[left][:sample_count]
        for j, right in enumerate(label_order):
            right_texts = text_groups[right][:sample_count]
            matrix[i][j] = sum(
                word_error_rate(left_text, right_text)
                for left_text, right_text in zip(left_texts, right_texts)
            ) / sample_count
    return {"labels": label_order, "mean_wer_matrix": matrix}


def _as_int_list(tokens: Sequence[int] | Any) -> list[int]:
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    return [int(v) for v in tokens]


def find_audio_files(directory: Path) -> list[Path]:
    """Return all audio files under a directory."""
    audio_types = {".wav", ".flac", ".mp3", ".ogg"}
    return sorted(
        [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in audio_types],
        key=lambda p: p.name,
    )


def collect_run_dirs(audio_root: Path) -> list[Path]:
    """Return actual per-run directories that contain audio while ignoring analysis folders."""
    if not audio_root.exists():
        raise FileNotFoundError(f"Audio root not found: {audio_root}")

    child_dirs = [p for p in sorted(audio_root.iterdir()) if p.is_dir()]
    audio_child_dirs = [p for p in child_dirs if find_audio_files(p)]
    if audio_child_dirs:
        return audio_child_dirs

    candidates = [p for p in sorted(audio_root.rglob("*")) if p.is_dir() and find_audio_files(p)]
    if candidates:
        return candidates

    return [audio_root] if find_audio_files(audio_root) else []


def analyze_codebook_divergence(
    baseline_tokens: Sequence[int] | Any,
    quant_tokens: Sequence[int] | Any,
    model_name: str | None = None,
    codec_family: str | None = None,
    tokens_per_frame: int | None = None,
) -> dict[str, Any]:
    """Measure how many divergences happen at each codec hierarchy bucket.

    For RVQ-style codecs (Orpheus/SNAC), tokens are grouped into 7-token frames
    with the common coarse/medium/fine layout. For FSQ-style codecs, this is a
    flat token stream without a hierarchical sub-codebook split.
    """
    spec = get_codec_spec(model_name=model_name, family=codec_family)
    if tokens_per_frame is not None:
        spec = CodecSpec(
            family=spec.family,
            tokens_per_frame=tokens_per_frame,
            hierarchy=spec.hierarchy,
        )

    baseline_list = _as_int_list(baseline_tokens)
    quant_list = _as_int_list(quant_tokens)
    n = min(len(baseline_list), len(quant_list))
    if n == 0:
        return {
            "codec_family": spec.family,
            "tokens_per_frame": spec.tokens_per_frame,
            "total_tokens": 0,
            "divergent_tokens": 0,
            "divergence_rate": 0.0,
            "by_hierarchy": {name: {"mismatched": 0, "rate": 0.0} for name in (spec.hierarchy or {}).keys()},
        }

    baseline_trim = baseline_list[:n]
    quant_trim = quant_list[:n]
    mismatched = [i for i, (a, b) in enumerate(zip(baseline_trim, quant_trim)) if a != b]
    by_hierarchy: dict[str, dict[str, float | int]] = {}

    if spec.family == "rvq":
        hierarchy = spec.hierarchy or {"coarse": [0], "medium": [1, 2], "fine": [3, 4, 5, 6]}
        for level, positions in hierarchy.items():
            count = 0
            for idx in mismatched:
                slot = idx % spec.tokens_per_frame
                if slot in positions:
                    count += 1
            by_hierarchy[level] = {
                "mismatched": count,
                "rate": (count / max(len(mismatched), 1)) if mismatched else 0.0,
                "token_positions": positions,
            }
        total_divergent = len(mismatched)
        return {
            "codec_family": spec.family,
            "tokens_per_frame": spec.tokens_per_frame,
            "total_tokens": n,
            "divergent_tokens": total_divergent,
            "divergence_rate": total_divergent / n if n else 0.0,
            "by_hierarchy": by_hierarchy,
        }

    mismatched_total = len(mismatched)
    return {
        "codec_family": spec.family,
        "tokens_per_frame": 1,
        "total_tokens": n,
        "divergent_tokens": mismatched_total,
        "divergence_rate": mismatched_total / n if n else 0.0,
        "by_hierarchy": {
            "flat": {
                "mismatched": mismatched_total,
                "rate": mismatched_total / n if n else 0.0,
                "token_positions": [0],
            }
        },
    }


class CodebookDivergenceAnalyzer:
    """Aggregates codebook divergence statistics across a manifest pair."""

    def __init__(self, model_name: str | None = None, codec_family: str | None = None):
        self.model_name = model_name
        self.codec_family = codec_family or detect_codec_family(model_name)

    def analyze_pair(self, baseline_entry: dict[str, Any], quant_entry: dict[str, Any]) -> dict[str, Any]:
        baseline_tokens = baseline_entry.get("audio_tokens") or []
        quant_tokens = quant_entry.get("audio_tokens") or []
        return analyze_codebook_divergence(
            baseline_tokens,
            quant_tokens,
            model_name=self.model_name,
            codec_family=self.codec_family,
        )

    def analyze_manifest_pair(self, baseline_manifest: Sequence[dict[str, Any]], quant_manifest: Sequence[dict[str, Any]]) -> dict[str, Any]:
        per_sample = []
        hierarchy_totals: dict[str, dict[str, float]] = {}
        codec_family = self.codec_family
        for baseline_entry, quant_entry in zip(baseline_manifest, quant_manifest):
            sample_result = self.analyze_pair(baseline_entry, quant_entry)
            sample_result["sample_id"] = baseline_entry.get("sample_id")
            sample_result["text"] = baseline_entry.get("text")
            per_sample.append(sample_result)

            for level, payload in (sample_result.get("by_hierarchy") or {}).items():
                if level not in hierarchy_totals:
                    hierarchy_totals[level] = {"mismatched": 0, "rate": 0.0}
                hierarchy_totals[level]["mismatched"] += int(payload.get("mismatched", 0))
                hierarchy_totals[level]["rate"] += float(payload.get("rate", 0.0))

        avg_rate = {}
        for level, payload in hierarchy_totals.items():
            avg_rate[level] = payload["rate"] / max(len(per_sample), 1)

        summary = {
            "codec_family": codec_family,
            "num_samples": len(per_sample),
            "mean_divergence_rate": (
                sum(sample.get("divergence_rate", 0.0) for sample in per_sample) / max(len(per_sample), 1)
            ),
            "by_hierarchy": {
                level: {
                    "mean_rate": avg_rate.get(level, 0.0),
                    "total_mismatched": payload["mismatched"],
                }
                for level, payload in hierarchy_totals.items()
            },
        }

        return {
            "codec_family": codec_family,
            "summary": summary,
            "per_sample": per_sample,
        }


__all__ = [
    "CodecSpec",
    "CodebookDivergenceAnalyzer",
    "analyze_codebook_divergence",
    "detect_codec_family",
    "get_codec_spec",
]
