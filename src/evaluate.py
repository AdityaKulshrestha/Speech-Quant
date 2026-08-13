"""
Core benchmark entry point.

Loads a TTS model, generates codec-token audio for a set of English
prompts, and saves the resulting waveforms plus a manifest of
per-sample metadata.

If --quant-type is "none" (default), only the full-precision baseline
runs. If a quant flavour is given (see quants/config.py), the baseline
and the quantized model are both run on the same prompts, and the
generated codec-token sequences are compared using the FDP / D(t)
metrics from METRICS.md. The comparison scores are printed and dumped
to <output-dir>/scores_<quant-type>.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from evaluation.metrics import (
    codebook_ids_for_tokens,
    compare_sequences,
    kl_divergence_sequence,
    probability_difference,
    summarize_scores,
)
from models.orpheus_model import OrpheusTTS
from models.neutts_model import NeuTTSModel
from quants.config import QUANT_CONFIGS
from visualization.heatmap import (
    save_codebook_heatmap,
    save_prob_diff_heatmap,
    save_token_heatmap,
)

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent

MODEL_REGISTRY = {
    "orpheus": OrpheusTTS,
    "neutts": NeuTTSModel,
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
        choices=sorted(QUANT_CONFIGS),
        default="none",
        help=(
            "Quantization flavour to compare against the full-precision "
            "baseline (see quants/config.py). 'none' runs only the baseline."
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
        help="Speaker/voice tag (Orpheus: voice style; NeuTTS: emily/paul/sophie/steven).",
    )
    parser.add_argument(
        "--device",
        default="xpu",
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


def build_model(args: argparse.Namespace, quant_type: str):
    model_cls = MODEL_REGISTRY[args.model]

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


def run_model(model, prompts: list[str], args: argparse.Namespace, run_dir: Path) -> list[dict]:
    """Generate audio for every prompt with `model` and save it under run_dir."""

    run_dir.mkdir(parents=True, exist_ok=True)

    # Warm up: first call pays for CUDA/XPU context init and kernel
    # compilation, which would otherwise skew sample_000's timing.
    print(f"[{run_dir.name}] Warming up...")
    model.generate_audio(
        prompts[0],
        voice=args.voice,
        max_new_tokens=min(32, args.max_new_tokens),
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    synchronize(args.device)

    manifest = []

    for idx, text in enumerate(prompts):
        sample_id = f"sample_{idx:03d}"
        print(f"[{run_dir.name}/{sample_id}] Generating: {text!r}")

        # Reset the seed per sample so the baseline and quantized runs
        # draw the same random numbers for a fair token-level comparison.
        torch.manual_seed(args.seed + idx)

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
        cb_ids = codebook_ids_for_tokens(num_tok)

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
                **output.metadata,
            }
        )

    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(manifest)} samples to {run_dir}")

    return manifest


def main() -> None:
    args = parse_args()

    prompts = load_prompts(args.prompts_file, args.num_samples)

    output_dir = Path(args.output_dir)

    baseline_model = build_model(args, quant_type="none")
    baseline_model.load()
    baseline_manifest = run_model(baseline_model, prompts, args, output_dir / "baseline")
    baseline_model.unload()

    if args.quant_type == "none":
        return

    quant_model = build_model(args, quant_type=args.quant_type)
    quant_model.load()
    quant_manifest = run_model(quant_model, prompts, args, output_dir / args.quant_type)
    quant_model.unload()

    per_sample_scores = []

    for baseline_entry, quant_entry in zip(baseline_manifest, quant_manifest):
        baseline_tokens = torch.tensor(baseline_entry["audio_tokens"])
        quant_tokens = torch.tensor(quant_entry["audio_tokens"])

        score = compare_sequences(baseline_tokens, quant_tokens)
        score["sample_id"] = baseline_entry["sample_id"]
        score["text"] = baseline_entry["text"]

        # Probability difference and KL divergence
        b_path = baseline_entry.get("audio_probs_path")
        q_path = quant_entry.get("audio_probs_path")
        ats = baseline_entry.get("audio_token_start")

        if b_path and q_path and ats is not None:
            b_probs = torch.from_numpy(np.load(b_path))
            q_probs = torch.from_numpy(np.load(q_path))
            offsets = (baseline_tokens - ats).long()

            pdiff = probability_difference(b_probs, q_probs, offsets)
            kl_vals = kl_divergence_sequence(b_probs, q_probs)

            score["prob_difference"] = pdiff
            score["mean_prob_difference"] = sum(pdiff) / len(pdiff) if pdiff else 0.0
            score["kl_divergence"] = kl_vals
            score["mean_kl_divergence"] = sum(kl_vals) / len(kl_vals) if kl_vals else 0.0

        per_sample_scores.append(score)

    summary = summarize_scores(per_sample_scores)

    print(f"\n=== {args.quant_type} vs baseline ===")
    print(json.dumps(summary, indent=2))

    scores_path = output_dir / f"scores_{args.quant_type}.json"
    with open(scores_path, "w") as f:
        json.dump({"summary": summary, "per_sample": per_sample_scores}, f, indent=2)

    print(f"Scores: {scores_path}")

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    save_token_heatmap(
        baseline_manifest, quant_manifest,
        analysis_dir / f"heatmap_{args.quant_type}.png",
    )
    save_codebook_heatmap(
        baseline_manifest, quant_manifest,
        analysis_dir / f"codebook_{args.quant_type}.png",
    )
    save_prob_diff_heatmap(
        per_sample_scores,
        analysis_dir / f"prob_diff_{args.quant_type}.png",
    )


if __name__ == "__main__":
    main()
