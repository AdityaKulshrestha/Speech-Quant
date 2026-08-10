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

import soundfile as sf
import torch

from evaluation.metrics import compare_sequences, summarize_scores
from models.orpheus_model import OrpheusTTS
from quants.config import QUANT_CONFIGS

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent

MODEL_REGISTRY = {
    "orpheus": OrpheusTTS,
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
        help="Speaker/voice tag used for prompt formatting.",
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

        manifest.append(
            {
                "sample_id": sample_id,
                "text": text,
                "model_name": model.model_name,
                "quant_type": model.quant_type,
                "voice": args.voice,
                "audio_path": str(audio_path),
                "num_audio_tokens": output.audio_tokens.numel(),
                "generation_seconds": elapsed,
                "audio_tokens": output.audio_tokens.tolist(),
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

        per_sample_scores.append(score)

    summary = summarize_scores(per_sample_scores)

    print(f"\n=== {args.quant_type} vs baseline ===")
    print(json.dumps(summary, indent=2))

    scores_path = output_dir / f"scores_{args.quant_type}.json"
    with open(scores_path, "w") as f:
        json.dump({"summary": summary, "per_sample": per_sample_scores}, f, indent=2)

    print(f"Scores: {scores_path}")


if __name__ == "__main__":
    main()
