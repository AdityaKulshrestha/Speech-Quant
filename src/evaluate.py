"""
Core benchmark entry point.

Loads a TTS model, generates codec-token audio for a set of English
prompts, and saves the resulting waveforms plus a manifest of
per-sample metadata.

If --quant-type is "none" (default), only the full-precision baseline
runs. If a quant flavour is given (see quants/config.py), the baseline
and the quantized model are both run on the same prompts, and the
generated codec-token sequences are compared using the FDP / D(t)
metrics from METRICS.md. Results are upserted into a single consolidated
workbook at <output-dir>/evaluation/analysis_report.xlsx.
"""

import argparse
import contextlib
import json
import math
import random
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import set_seed

from decoding.alignment import CodebookDivergenceAnalyzer, detect_codec_family
from decoding.distribution import compare_distributions, extract_teacher_forced_logits
from evaluation.acoustic_metrics import compare_acoustics_manifest, summarize_acoustics
from evaluation.report import update_report
from evaluation.metrics import (
    codebook_ids_for_tokens,
    compare_sequences,
    negative_log_likelihood,
    probability_difference,
    summarize_scores,
)
from quants.config import QUANT_CONFIGS
from visualization.heatmap import (
    save_codebook_heatmap,
    save_prob_diff_heatmap,
    save_token_heatmap,
)

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent

MODEL_REGISTRY = {
    "orpheus": ("models.orpheus_model", "OrpheusTTS"),
    "neutts": ("models.neutts_model", "NeuTTSModel"),
    "outetts": ("models.outetts_model", "OuteTTSModel"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate speech samples for the quantization experiment."
        )
    )

    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY),
        default="orpheus",
        help="Which TTS model to run.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="HF checkpoint id/path. Defaults to the model's built-in default.",
    )
    parser.add_argument(
        "--quant-type",
        default="none",
        help=(
            "Comma-separated quantization flavours to compare against the full-precision "
            "baseline, e.g. 'gptq-4bit,rtn-8bit,awq-4bit,sq-8bit' (see quants/config.py "
            "for the full list). The baseline is generated once and every listed flavour "
            "is compared against that same baseline run. 'none' (default) runs only the "
            "baseline; mixing 'none' into a list with other flavours just ignores it."
        ),
    )
    parser.add_argument(
        "--prompts-file",
        default=str(SRC_DIR / "prompts.txt"),
        help="Text file with one English prompt per line.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of prompts to generate, taken from the top of --prompts-file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs"),
        help="Directory to write generated audio, manifests and scores.",
    )
    parser.add_argument(
        "--voice",
        default="tara",
        help=(
            "Speaker/voice tag (Orpheus: voice style; NeuTTS: emily/paul/sophie/steven; "
            "OuteTTS: unused, zero-shot or model-default voice)."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run inference on.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Random seed, reset before generating each sample so the "
            "baseline and quantized runs draw the same random numbers."
        ),
    )

    return parser.parse_args()


def load_prompts(prompts_file: str, num_samples: int) -> list[str]:
    path = Path(prompts_file)

    with open(path) as f:
        prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        raise ValueError(f"No prompts found in {path}")

    if num_samples > len(prompts):
        raise ValueError(
            f"--num-samples={num_samples} exceeds the {len(prompts)} "
            f"prompts available in {path}"
        )

    return prompts[:num_samples]


def parse_quant_types(raw: str) -> list[str]:
    """Split --quant-type's comma-separated value into a validated, ordered,
    de-duplicated list of real quant flavours ('none' entries are dropped)."""
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("--quant-type must not be empty.")

    unknown = [v for v in values if v not in QUANT_CONFIGS]
    if unknown:
        raise ValueError(
            f"Unknown --quant-type value(s) {unknown}. Available: {sorted(QUANT_CONFIGS)}"
        )

    seen: set[str] = set()
    quant_types = []
    for v in values:
        if v == "none" or v in seen:
            continue
        seen.add(v)
        quant_types.append(v)

    return quant_types


def build_model(args: argparse.Namespace, quant_type: str):
    module_name, class_name = MODEL_REGISTRY[args.model]
    model_cls = getattr(import_module(module_name), class_name)

    kwargs = {"device": args.device, "quant_type": quant_type}
    if args.model_name:
        kwargs["model_name"] = args.model_name

    return model_cls(**kwargs)


def synchronize(device: str) -> None:
    """Block until outstanding device work finishes, for accurate wall-clock timing."""

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.startswith("xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()


@contextlib.contextmanager
def deterministic_disabled():
    """Temporarily lift torch's deterministic-algorithms enforcement.

    Weight packing (e.g. `scatter_add_` in compressed-tensors' `pack_to_int32`)
    has no deterministic kernel and raises under `use_deterministic_algorithms`.
    Packing is exact bit manipulation, so relaxing it here does not affect
    reproducibility of generation, which is what determinism is enabled for.
    """

    was_enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=warn_only)


def run_model(model, prompts: list[str], args: argparse.Namespace, run_dir: Path) -> list[dict]:
    """Generate audio for every prompt with `model` and save it under run_dir."""

    run_dir.mkdir(parents=True, exist_ok=True)

    # Warmup: 200 tokens gives enough headroom for at least one complete audio
    # frame (7 for Orpheus, 1 for NeuTTS) without timing the full generation.
    print(f"[{run_dir.name}] Warming up...")
    model.generate_audio(
        prompts[0],
        voice=args.voice,
        max_new_tokens=min(200, args.max_new_tokens),
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    synchronize(args.device)

    manifest = []

    torch.use_deterministic_algorithms(True)

    for idx, text in enumerate(prompts):
        sample_id = f"sample_{idx:03d}"
        print(f"[{run_dir.name}/{sample_id}] Generating: {text!r}")

        # Reset the seed per sample (not just once before the loop) so the
        # baseline and quantized runs draw the same random numbers at every
        # sample, not only the first.
        sample_seed = args.seed + idx
        random.seed(sample_seed)
        np.random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)
        elif args.device.startswith("xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed_all(sample_seed)
        set_seed(sample_seed)

        synchronize(args.device)
        start = time.perf_counter()

        output = model.generate_audio(
            text,
            voice=args.voice,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

        synchronize(args.device)
        elapsed = time.perf_counter() - start

        audio_path = run_dir / f"{sample_id}.wav"
        sf.write(str(audio_path), output.audio.float().numpy(), output.sampling_rate)

        # Codebook IDs and per-token probabilities
        num_tok = output.audio_tokens.numel()
        tokens_per_frame = getattr(model, "TOKENS_PER_FRAME", 7)
        cb_ids = codebook_ids_for_tokens(num_tok, tokens_per_frame=tokens_per_frame)

        audio_probs_path = None
        token_probs: list[float] = []
        audio_token_start = getattr(model, "AUDIO_TOKEN_START", None)

        if output.audio_logits is not None and audio_token_start is not None:
            offsets = (output.audio_tokens - audio_token_start).cpu().long()
            n = min(len(output.audio_logits), len(offsets))
            token_probs = [
                float(output.audio_logits[i, int(offsets[i].item())].item())
                for i in range(n)
            ]
            audio_probs_path = run_dir / f"{sample_id}_audio_probs.npy"
            np.save(str(audio_probs_path), output.audio_logits.numpy())

        generated_ids_path = run_dir / f"{sample_id}_generated_ids.pt"
        torch.save(output.generated_ids.cpu(), str(generated_ids_path))

        manifest.append(
            {
                "sample_id": sample_id,
                "text": text,
                "model_name": model.model_name,
                "quant_type": model.quant_type,
                "voice": args.voice,
                "audio_path": str(audio_path),
                "num_audio_tokens": num_tok,
                "generation_seconds": elapsed,
                "audio_tokens": output.audio_tokens.tolist(),
                "codebook_ids": cb_ids,
                "token_probs": token_probs,
                "audio_token_start": audio_token_start,
                "audio_probs_path": str(audio_probs_path) if audio_probs_path else None,
                "generated_ids_path": str(generated_ids_path),
                **output.metadata,
            }
        )

    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(manifest)} samples to {run_dir}")

    return manifest


def compare_runs(baseline_manifest: list[dict], quant_manifest: list[dict]) -> list[dict]:
    """Compute token-level sequence comparison metrics for matched prompt runs."""
    per_sample_scores = []

    for baseline_entry, quant_entry in zip(baseline_manifest, quant_manifest):
        baseline_tokens = torch.tensor(baseline_entry["audio_tokens"])
        quant_tokens = torch.tensor(quant_entry["audio_tokens"])

        score = compare_sequences(baseline_tokens, quant_tokens)
        score["sample_id"] = baseline_entry["sample_id"]
        score["text"] = baseline_entry["text"]

        b_path = baseline_entry.get("audio_probs_path")
        q_path = quant_entry.get("audio_probs_path")
        ats = baseline_entry.get("audio_token_start")

        if b_path and q_path and ats is not None:
            b_probs = torch.from_numpy(np.load(b_path))
            q_probs = torch.from_numpy(np.load(q_path))
            offsets = (baseline_tokens - ats).long()

            pdiff = probability_difference(b_probs, q_probs, offsets)

            # KL divergence: inlined to avoid duplication with distribution.py
            n = min(len(b_probs), len(q_probs))
            p = b_probs[:n].float().clamp(min=1e-8)
            q = q_probs[:n].float().clamp(min=1e-8)
            kl_vals = (p * (p / q).log()).sum(dim=-1).tolist()

            score["prob_difference"] = pdiff
            score["mean_prob_difference"] = sum(pdiff) / len(pdiff) if pdiff else 0.0
            score["kl_divergence"] = kl_vals
            score["mean_kl_divergence"] = sum(kl_vals) / len(kl_vals) if kl_vals else 0.0

        per_sample_scores.append(score)

    return per_sample_scores


def teacher_forced_distribution_compare(
    baseline_manifest: list[dict],
    quant_manifest: list[dict],
    baseline_model,
    quant_model,
) -> list[dict]:
    """Teacher-forced distribution comparison: isolates pure quantization error.

    Correct order (per arXiv:2405.02404 "Statistically-Lossless Quantization"):
        1. Full-precision (baseline) model generates freely → produces reference sequence
        2. Freeze that reference sequence
        3. Quantized model is teacher-forced on the frozen sequence
        4. Compute D_KL(P_baseline || P_quant) at each position

    Reason: In free (autoregressive) generation, a single divergent token shifts
    the entire continuation, making it impossible to attribute later differences
    to quantization error versus the natural consequence of conditioning on a
    different prefix. Teacher-forcing isolates the per-position effect of
    quantization: each reported divergence reflects a position where quantization
    alone caused the model to prefer a different token, independent of any
    earlier divergence.

    Reference: "Statistically-Lossless Quantization of Large Language Models"
               arXiv:2405.02404, Section 3.2

    Each per-sample result carries step-level detail (aligned with
    result["per_step"]): the baseline model's own token at each step (from its
    free-run generation), the quantized model's predicted (argmax) token at that
    same step (from teacher forcing), each model's probability of the reference
    token, and the corresponding per-step negative log-likelihood (NLL) — so
    callers can build a full row-level table instead of only frame-bucketed
    summaries.
    """
    _required = ("model", "AUDIO_TOKEN_START", "VOCAB_AUDIO_TOKEN_START", "END_OF_SPEECH", "TOKENS_PER_FRAME", "CODEBOOK_SIZE")
    if not all(hasattr(baseline_model, a) for a in _required):
        return []
    if not all(hasattr(quant_model, a) for a in _required):
        return []

    # VOCAB_AUDIO_TOKEN_START is the real tokenizer-vocab offset of the audio/codec
    # token block. All models must define this explicitly (no fallback).
    audio_start = baseline_model.VOCAB_AUDIO_TOKEN_START
    audio_end = audio_start + baseline_model.TOKENS_PER_FRAME * baseline_model.CODEBOOK_SIZE
    per_sample = []

    for baseline_entry, quant_entry in zip(baseline_manifest, quant_manifest):
        # Load BASELINE's reference sequence (the frozen ground truth)
        baseline_gids_path = baseline_entry.get("generated_ids_path")
        baseline_probs_path = baseline_entry.get("audio_probs_path")
        if not baseline_gids_path or not baseline_probs_path:
            continue
        if not Path(baseline_gids_path).exists() or not Path(baseline_probs_path).exists():
            continue

        # BASELINE's own generated sequence (the reference)
        reference_ids = torch.load(baseline_gids_path, map_location=quant_model.device, weights_only=True)

        # BASELINE's own probabilities (from its free-run generation)
        baseline_probs = torch.from_numpy(np.load(baseline_probs_path)).float()

        # QUANT model teacher-forced on BASELINE's sequence
        quant_probs, token_ids = extract_teacher_forced_logits(
            model=quant_model.model,
            reference_ids=reference_ids,
            eos_token_id=quant_model.END_OF_SPEECH,
            prompt_length=baseline_entry["input_length"],
            token_filter=lambda t: audio_start <= t < audio_end,
            vocab_slice=(audio_start, audio_end),
            tokens_per_frame=quant_model.TOKENS_PER_FRAME,
        )

        result = compare_distributions(
            baseline_probs=baseline_probs,
            quant_probs=quant_probs,
            tokens_per_frame=baseline_model.TOKENS_PER_FRAME,
        )

        # step_* arrays line up 1:1 with result["per_step"]'s kl/js/argmax_mismatch lists.
        # token_ids are the baseline's reference tokens (from its free-run generation)
        n = min(len(baseline_probs), len(quant_probs), len(token_ids))
        offsets = (token_ids[:n] - audio_start).long()
        idx = torch.arange(n)
        quant_argmax = quant_probs[:n].argmax(dim=-1)
        result["step_baseline_token_id"] = token_ids[:n].tolist()  # baseline's actual token
        result["step_quant_token_id"] = (quant_argmax + audio_start).tolist()  # quant's predicted token
        step_baseline_prob = baseline_probs[:n][idx, offsets]
        step_quant_prob = quant_probs[:n][idx, offsets]
        result["step_baseline_prob"] = step_baseline_prob.tolist()
        result["step_quant_prob"] = step_quant_prob.tolist()
        result["step_baseline_nll"] = negative_log_likelihood(step_baseline_prob).tolist()
        result["step_quant_nll"] = negative_log_likelihood(step_quant_prob).tolist()
        if n:
            result["summary"]["mean_nll_baseline"] = float(sum(result["step_baseline_nll"]) / n)
            result["summary"]["mean_nll_quant"] = float(sum(result["step_quant_nll"]) / n)
            result["summary"]["perplexity_baseline"] = math.exp(result["summary"]["mean_nll_baseline"])
            result["summary"]["perplexity_quant"] = math.exp(result["summary"]["mean_nll_quant"])

        result["sample_id"] = baseline_entry["sample_id"]
        result["text"] = baseline_entry["text"]
        per_sample.append(result)

    return per_sample


def _aggregate_distribution_summary(per_sample: list[dict]) -> dict:
    """Average per-sample distribution summaries and aggregate by_frame_position."""
    summaries = [s["summary"] for s in per_sample if s.get("summary")]
    if not summaries:
        return {}

    scalar_keys = [k for k in summaries[0] if k not in ("num_steps", "tokens_per_frame")]
    agg: dict = {
        "num_samples": len(summaries),
        "tokens_per_frame": summaries[0].get("tokens_per_frame"),
        **{k: sum(s[k] for s in summaries) / len(summaries) for k in scalar_keys},
    }

    all_pos = [s.get("by_frame_position", {}) for s in per_sample if s.get("by_frame_position")]
    if all_pos:
        positions = sorted(all_pos[0].keys())
        metric_keys = [k for k in all_pos[0][positions[0]] if k != "num_steps"]
        agg["by_frame_position"] = {
            pos: {k: sum(pd[pos][k] for pd in all_pos if pos in pd) / len(all_pos) for k in metric_keys}
            for pos in positions
        }

    return agg


def _run_single_quant_comparison(
    args: argparse.Namespace,
    quant_type: str,
    baseline_model,
    baseline_manifest: list[dict],
    prompts: list[str],
    output_dir: Path,
    analysis_dir: Path,
) -> None:
    """Quantize with *quant_type*, generate, and compare against the already-
    generated *baseline_manifest* — the same baseline run is reused for every
    quant_type in the --quant-type list, never regenerated per flavour."""

    quant_model = build_model(args, quant_type=quant_type)
    # Quantization/packing runs non-deterministic kernels; determinism is
    # restored right after loading, before any generation happens.
    with deterministic_disabled():
        quant_model.load()
    quant_manifest = run_model(quant_model, prompts, args, output_dir / quant_type)
    quant_model.unload()

    # Teacher-force the quantized model on the baseline's reference sequence
    # (one model resident at a time) to measure pure quantization error without
    # autoregressive drift contamination.
    baseline_model.load()
    quant_model.load()
    dist_per_sample = teacher_forced_distribution_compare(
        baseline_manifest, quant_manifest, baseline_model, quant_model
    )
    quant_model.unload()
    baseline_model.unload()

    per_sample_scores = compare_runs(baseline_manifest, quant_manifest)
    summary = summarize_scores(per_sample_scores)
    dist_summary = _aggregate_distribution_summary(dist_per_sample)

    acoustics_per_sample = compare_acoustics_manifest(baseline_manifest, quant_manifest)
    acoustics_summary = summarize_acoustics(acoustics_per_sample)

    print(f"\n=== {quant_type} vs baseline ===")
    print(json.dumps(summary, indent=2))
    if dist_summary:
        print("Distribution analysis (teacher-forced):")
        print(json.dumps(dist_summary, indent=2))
    print("Acoustic distortion (MCD / F0):")
    print(json.dumps(acoustics_summary, indent=2))

    save_token_heatmap(
        baseline_manifest, quant_manifest,
        analysis_dir / f"heatmap_{quant_type}.png",
    )
    save_codebook_heatmap(
        baseline_manifest, quant_manifest,
        analysis_dir / f"codebook_{quant_type}.png",
    )
    save_prob_diff_heatmap(
        per_sample_scores,
        analysis_dir / f"prob_diff_{quant_type}.png",
    )

    empty_transcription = {"samples": []}

    divergence_analyzer = CodebookDivergenceAnalyzer(
        model_name=getattr(baseline_model, "model_name", None),
        codec_family=detect_codec_family(getattr(baseline_model, "model_name", None)),
        tokens_per_frame=getattr(baseline_model, "TOKENS_PER_FRAME", None),
    )
    divergence_summary = divergence_analyzer.analyze_manifest_pair(
        baseline_manifest[: min(len(baseline_manifest), len(quant_manifest))],
        quant_manifest[: min(len(baseline_manifest), len(quant_manifest))],
    )

    report_path = Path(args.output_dir) / "evaluation" / "analysis_report.xlsx"
    update_report(
        report_path,
        model=getattr(baseline_model, "model_name", None),
        quant_type=quant_type,
        scores_summary=summary,
        per_sample_scores=per_sample_scores,
        dist_summary=dist_summary,
        dist_per_sample=dist_per_sample,
        acoustics_summary=acoustics_summary,
        acoustics_per_sample=acoustics_per_sample,
        codebook_divergence=divergence_summary,
        baseline_transcription=empty_transcription,
        quant_transcription=empty_transcription,
        audio_token_start=(
            getattr(baseline_model, "VOCAB_AUDIO_TOKEN_START", None)
            if getattr(baseline_model, "VOCAB_AUDIO_TOKEN_START", None) is not None
            else getattr(baseline_model, "AUDIO_TOKEN_START", None)
        ),
    )
    print(f"Consolidated report updated: {report_path}")


def run_experiment(args: argparse.Namespace) -> None:
    quant_types = parse_quant_types(args.quant_type)
    prompts = load_prompts(args.prompts_file, args.num_samples)

    baseline_model = build_model(args, quant_type="none")
    model_slug = Path(baseline_model.model_name).name
    output_dir = Path(args.output_dir) / "evaluation" / model_slug
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    baseline_model.load()
    baseline_manifest = run_model(baseline_model, prompts, args, output_dir / "baseline")
    baseline_model.unload()

    if not quant_types:
        return

    print(f"Comparing quant flavours {quant_types} against the same baseline run.")
    for quant_type in quant_types:
        _run_single_quant_comparison(
            args, quant_type, baseline_model, baseline_manifest,
            prompts, output_dir, analysis_dir,
        )


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
