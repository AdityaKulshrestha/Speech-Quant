"""
Core benchmark entry point.

Loads a TTS model, generates codec-token audio for a set of English
prompts, and saves the resulting waveforms plus a manifest of
per-sample metadata. Quantization is not implemented yet; --quant-type
is recorded in the manifest for future use but does not change the
model that gets loaded.
"""

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch

from models.orpheus_model import OrpheusTTS

SRC_DIR = Path(__file__).resolve().parent

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
        default="none",
        help=(
            "Quantization configuration to use. Not implemented yet: "
            "recorded in the manifest only, the default full-precision "
            "model is always loaded."
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
        default=str(SRC_DIR / "outputs"),
        help="Directory to write generated audio and the run manifest.",
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
        default=None,
        help="Optional random seed for reproducible sampling.",
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


def build_model(args: argparse.Namespace):
    if args.quant_type != "none":
        print(
            f"Warning: --quant-type={args.quant_type!r} was requested, but "
            "quantization is not implemented yet. Loading the default "
            "full-precision model instead."
        )

    model_cls = MODEL_REGISTRY[args.model]

    kwargs = {"device": args.device}
    if args.model_name:
        kwargs["model_name"] = args.model_name

    return model_cls(**kwargs)


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    prompts = load_prompts(args.prompts_file, args.num_samples)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args)
    model.load()

    manifest = []

    for idx, text in enumerate(prompts):
        sample_id = f"sample_{idx:03d}"
        print(f"[{sample_id}] Generating: {text!r}")

        start = time.perf_counter()

        output = model.generate_audio(
            text,
            voice=args.voice,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

        elapsed = time.perf_counter() - start

        audio_path = output_dir / f"{sample_id}.wav"
        sf.write(str(audio_path), output.audio.float().numpy(), output.sampling_rate)

        manifest.append(
            {
                "sample_id": sample_id,
                "text": text,
                "model": args.model,
                "model_name": model.model_name,
                "quant_type": args.quant_type,
                "voice": args.voice,
                "audio_path": str(audio_path),
                "num_audio_tokens": output.audio_tokens.numel(),
                "generation_seconds": elapsed,
                **output.metadata,
            }
        )

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    model.unload()

    print(f"Wrote {len(manifest)} samples to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
