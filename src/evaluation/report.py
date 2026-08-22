"""
Consolidated multi-sheet analysis workbook (outputs/evaluation/analysis_report.xlsx).

Every evaluate.py run upserts its (model, quant_type) results into this single
shared workbook instead of writing separate scores_*.json / acoustics_*.json /
codebook_divergence.json / transcription.json files per run, so results across
models and quant types stay comparable in one place.

Sheets:
  Summary      - one metric per row, one column per (model, quant_type)
  PerSample    - one row per (model, quant_type, sample_id): aggregated per-sample stats,
                 plus ground truth / baseline / quant transcript text
  LogProbs_KL  - one row per (model, quant_type, sample_id, step): baseline and quant
                 token ids (both LLM-vocab and codec-space) and probabilities kept
                 adjacent for direct comparison, plus per-step KL/JS/argmax-match/
                 top-k-jaccard. Row-level, not bucketed by codec/frame position
                 (codebook_id is kept as an informational column). Only populated for
                 models where teacher forcing is supported.
  Codec        - one row per (model, quant_type, sample_id, hierarchy_level): codebook
                 divergence rates from the RVQ/FSQ hierarchy breakdown

"_SummaryData" is a hidden helper sheet holding the Summary metrics in long
format (metric, model, quant_type, value); it's the persisted source of truth
that the wide "Summary" sheet is pivoted from on every write.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

from evaluation.metrics import codebook_ids_for_tokens

PER_SAMPLE_SHEET = "PerSample"
LOGPROBS_SHEET = "LogProbs_KL"
CODEC_SHEET = "Codec"
SUMMARY_SHEET = "Summary"
SUMMARY_DATA_SHEET = "_SummaryData"

_KEY_COLS = ["model", "quant_type"]


def _read_sheet(report_path: Path, sheet_name: str, columns: list[str]) -> pd.DataFrame:
    if not report_path.exists():
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_excel(report_path, sheet_name=sheet_name)
    except (ValueError, KeyError):
        return pd.DataFrame(columns=columns)


def _upsert(existing: pd.DataFrame, new_rows: pd.DataFrame, model: str, quant_type: str) -> pd.DataFrame:
    """Drop any prior rows for this (model, quant_type) and append the fresh ones."""
    if not existing.empty:
        mask = (existing["model"] == model) & (existing["quant_type"] == quant_type)
        existing = existing.loc[~mask]
    return pd.concat([existing, new_rows], ignore_index=True)


def _summary_rows(
    model: str,
    quant_type: str,
    scores_summary: dict[str, Any],
    dist_summary: dict[str, Any],
    acoustics_summary: dict[str, Any],
    codebook_summary: dict[str, Any],
    baseline_transcription: dict[str, Any],
    quant_transcription: dict[str, Any],
) -> pd.DataFrame:
    metrics: dict[str, Any] = {
        "baseline_wer": baseline_transcription.get("mean_wer"),
        "baseline_cer": baseline_transcription.get("mean_cer"),
        "quant_wer": quant_transcription.get("mean_wer"),
        "quant_cer": quant_transcription.get("mean_cer"),
        "mean_mcd": acoustics_summary.get("mean_mcd"),
        "mean_f0_frame_error": acoustics_summary.get("mean_f0_frame_error"),
        "mean_pitch_pearson_correlation": acoustics_summary.get("mean_pitch_pearson_correlation"),
        "mean_first_divergence_position": scores_summary.get("mean_first_divergence_position"),
        "mean_final_divergence_rate": scores_summary.get("mean_final_divergence_rate"),
        "mean_prob_difference": scores_summary.get("mean_prob_difference"),
        "mean_kl_divergence": scores_summary.get("mean_kl_divergence"),
        "codebook_mean_divergence_rate": codebook_summary.get("mean_divergence_rate"),
    }
    if dist_summary:
        metrics["teacher_forced_mean_kl"] = dist_summary.get("mean_kl")
        metrics["teacher_forced_mean_js"] = dist_summary.get("mean_js")
        metrics["teacher_forced_argmax_mismatch_rate"] = dist_summary.get("argmax_mismatch_rate")
        for key, value in dist_summary.items():
            if key.startswith("mean_top") and key.endswith("_jaccard"):
                metrics[f"teacher_forced_{key}"] = value
    for level, payload in (codebook_summary.get("by_hierarchy") or {}).items():
        metrics[f"codebook_{level}_rate"] = payload.get("mean_rate")

    return pd.DataFrame(
        {"metric": list(metrics.keys()), "model": model, "quant_type": quant_type, "value": list(metrics.values())}
    )


def _per_sample_rows(
    model: str,
    quant_type: str,
    per_sample_scores: list[dict[str, Any]],
    acoustics_per_sample: list[dict[str, Any]],
    codebook_per_sample: list[dict[str, Any]],
    baseline_samples: list[dict[str, Any]],
    quant_samples: list[dict[str, Any]],
) -> pd.DataFrame:
    acoustics_by_id = {s["sample_id"]: s for s in acoustics_per_sample}
    codebook_by_id = {s["sample_id"]: s for s in codebook_per_sample}

    rows = []
    for i, score in enumerate(per_sample_scores):
        sample_id = score["sample_id"]
        acoustics = acoustics_by_id.get(sample_id, {})
        codebook = codebook_by_id.get(sample_id, {})
        baseline_t = baseline_samples[i] if i < len(baseline_samples) else {}
        quant_t = quant_samples[i] if i < len(quant_samples) else {}

        rows.append(
            {
                "model": model,
                "quant_type": quant_type,
                "sample_id": sample_id,
                "ground_truth_text": score.get("text"),
                "baseline_transcript": baseline_t.get("transcript"),
                "quant_transcript": quant_t.get("transcript"),
                "baseline_wer": baseline_t.get("wer"),
                "baseline_cer": baseline_t.get("cer"),
                "quant_wer": quant_t.get("wer"),
                "quant_cer": quant_t.get("cer"),
                "mcd": acoustics.get("mcd"),
                "f0_frame_error": acoustics.get("f0_frame_error"),
                "pitch_pearson_correlation": acoustics.get("pitch_pearson_correlation"),
                "first_divergence_position": score.get("first_divergence_position"),
                "final_divergence_rate": score.get("final_divergence_rate"),
                "mean_prob_difference": score.get("mean_prob_difference"),
                "mean_kl_divergence": score.get("mean_kl_divergence"),
                "codebook_divergence_rate": codebook.get("divergence_rate"),
                "codec_family": codebook.get("codec_family"),
                "tokens_per_frame": codebook.get("tokens_per_frame"),
            }
        )
    return pd.DataFrame(rows)


def _logprobs_kl_rows(
    model: str,
    quant_type: str,
    dist_per_sample: list[dict[str, Any]],
    audio_token_start: int | None,
) -> pd.DataFrame:
    """One row per teacher-forced generation step (not bucketed by codec level).

    baseline/quant token ids and probs are kept adjacent so they're directly
    comparable: baseline_* is the actual reference token and the baseline
    model's own probability of it; quant_* is the quantized model's argmax
    prediction at that same step and its probability of the reference token.
    """
    rows = []
    for entry in dist_per_sample:
        sample_id = entry.get("sample_id")
        text = entry.get("text")
        per_step = entry.get("per_step") or {}
        summary = entry.get("summary") or {}
        tokens_per_frame = summary.get("tokens_per_frame") or 1

        baseline_token_ids = entry.get("step_baseline_token_id") or []
        quant_token_ids = entry.get("step_quant_token_id") or []
        baseline_prob = entry.get("step_baseline_prob") or []
        quant_prob = entry.get("step_quant_prob") or []
        kl = per_step.get("kl") or []
        js = per_step.get("js") or []
        argmax_mismatch = per_step.get("argmax_mismatch") or []
        topk_key = next((k for k in per_step if k.startswith("top") and k.endswith("_jaccard")), None)
        topk_jaccard = per_step.get(topk_key) or [] if topk_key else []

        n = min(
            len(baseline_token_ids), len(quant_token_ids), len(baseline_prob), len(quant_prob),
            len(kl), len(js), len(argmax_mismatch),
        )
        codebook_ids = codebook_ids_for_tokens(n, tokens_per_frame=tokens_per_frame)

        for step in range(n):
            baseline_llm_id = baseline_token_ids[step]
            quant_llm_id = quant_token_ids[step]
            rows.append(
                {
                    "model": model,
                    "quant_type": quant_type,
                    "sample_id": sample_id,
                    "ground_truth_text": text,
                    "step": step,
                    "codebook_id": codebook_ids[step],
                    "llm_token_id_baseline": baseline_llm_id,
                    "llm_token_id_quant": quant_llm_id,
                    "audio_codec_token_id_baseline": (
                        baseline_llm_id - audio_token_start if audio_token_start is not None else None
                    ),
                    "audio_codec_token_id_quant": (
                        quant_llm_id - audio_token_start if audio_token_start is not None else None
                    ),
                    "baseline_prob": baseline_prob[step],
                    "quant_prob": quant_prob[step],
                    "kl": kl[step],
                    "js": js[step],
                    "argmax_match": not bool(argmax_mismatch[step]),
                    "top5_jaccard": topk_jaccard[step] if step < len(topk_jaccard) else None,
                }
            )
    return pd.DataFrame(rows)


def _codec_rows(model: str, quant_type: str, codebook_per_sample: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for entry in codebook_per_sample:
        for level, payload in (entry.get("by_hierarchy") or {}).items():
            rows.append(
                {
                    "model": model,
                    "quant_type": quant_type,
                    "sample_id": entry.get("sample_id"),
                    "ground_truth_text": entry.get("text"),
                    "codec_family": entry.get("codec_family"),
                    "tokens_per_frame": entry.get("tokens_per_frame"),
                    "hierarchy_level": level,
                    "mismatched": payload.get("mismatched"),
                    "rate": payload.get("rate"),
                    "token_positions": ",".join(str(p) for p in payload.get("token_positions", [])),
                    "total_tokens": entry.get("total_tokens"),
                    "divergent_tokens": entry.get("divergent_tokens"),
                    "sample_divergence_rate": entry.get("divergence_rate"),
                }
            )
    return pd.DataFrame(rows)


def _style_sheet(ws) -> None:
    """Bold header row, freeze it, and auto-size columns for readability."""
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.freeze_panes = "A2"
    for col_idx, col_cells in enumerate(ws.columns, start=1):
        best_len = max((len(str(c.value)) for c in col_cells if c.value is not None), default=8)
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max(best_len + 2, 10), 60)


def update_report(
    report_path: Path,
    model: str,
    quant_type: str,
    scores_summary: dict[str, Any],
    per_sample_scores: list[dict[str, Any]],
    dist_summary: dict[str, Any],
    dist_per_sample: list[dict[str, Any]],
    acoustics_summary: dict[str, Any],
    acoustics_per_sample: list[dict[str, Any]],
    codebook_divergence: dict[str, Any],
    baseline_transcription: dict[str, Any],
    quant_transcription: dict[str, Any],
    audio_token_start: int | None = None,
) -> None:
    """Upsert one (model, quant_type) run's results into the shared report workbook."""
    codebook_summary = codebook_divergence.get("summary", {})
    codebook_per_sample = codebook_divergence.get("per_sample", [])

    summary_long = _upsert(
        _read_sheet(report_path, SUMMARY_DATA_SHEET, ["metric", *_KEY_COLS, "value"]),
        _summary_rows(model, quant_type, scores_summary, dist_summary, acoustics_summary,
                      codebook_summary, baseline_transcription, quant_transcription),
        model, quant_type,
    )
    per_sample = _upsert(
        _read_sheet(report_path, PER_SAMPLE_SHEET, [*_KEY_COLS, "sample_id"]),
        _per_sample_rows(model, quant_type, per_sample_scores, acoustics_per_sample, codebook_per_sample,
                          baseline_transcription.get("samples", []), quant_transcription.get("samples", [])),
        model, quant_type,
    )
    logprobs_kl = _upsert(
        _read_sheet(report_path, LOGPROBS_SHEET, [*_KEY_COLS, "sample_id", "step"]),
        _logprobs_kl_rows(model, quant_type, dist_per_sample, audio_token_start),
        model, quant_type,
    )
    codec = _upsert(
        _read_sheet(report_path, CODEC_SHEET, [*_KEY_COLS, "sample_id", "hierarchy_level"]),
        _codec_rows(model, quant_type, codebook_per_sample),
        model, quant_type,
    )

    summary_wide = (
        summary_long.pivot_table(index="metric", columns=["model", "quant_type"], values="value", aggfunc="first")
        if not summary_long.empty
        else pd.DataFrame()
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        summary_wide.to_excel(writer, sheet_name=SUMMARY_SHEET)
        per_sample.to_excel(writer, sheet_name=PER_SAMPLE_SHEET, index=False)
        logprobs_kl.to_excel(writer, sheet_name=LOGPROBS_SHEET, index=False)
        codec.to_excel(writer, sheet_name=CODEC_SHEET, index=False)
        summary_long.to_excel(writer, sheet_name=SUMMARY_DATA_SHEET, index=False)

        for name in (PER_SAMPLE_SHEET, LOGPROBS_SHEET, CODEC_SHEET):
            _style_sheet(writer.sheets[name])
        writer.sheets[SUMMARY_SHEET].freeze_panes = "B3"
        for cell in writer.sheets[SUMMARY_SHEET][2]:
            cell.font = Font(bold=True)
        writer.sheets[SUMMARY_DATA_SHEET].sheet_state = "hidden"


__all__ = ["update_report"]
