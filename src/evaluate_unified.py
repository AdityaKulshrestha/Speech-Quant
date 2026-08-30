#!/usr/bin/env python3
"""Unified evaluation: Transcription (WER/CER) + UTMOS on pre-generated audio.

This script runs in the ASR environment and processes audio from all TTS models
(Orpheus, NeuTTS, OuteTTS) without needing their model-specific dependencies.

Usage:
    # Process all models and quants
    export HF_TOKEN="your_huggingface_token"
    source .venv-asr/bin/activate
    PYTHONPATH=src python src/evaluate_unified.py --audio-root outputs/evaluation

    # Process specific model
    PYTHONPATH=src python src/evaluate_unified.py \\
        --audio-root outputs/evaluation/orpheus-3b-0.1-ft

    # Process specific quant types
    PYTHONPATH=src python src/evaluate_unified.py \\
        --audio-root outputs/evaluation \\
        --quant-type gptq-4bit,awq-4bit
"""

import argparse
import json
from pathlib import Path
import sys

import triton.language  # noqa: F401 - torch._dynamo expects triton.language

from evaluation.report import update_transcription_and_utmos_report
from evaluation.transcription import evaluate_transcription_run, load_asr_model
from evaluation.acoustic_metrics import evaluate_utmos_run

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified evaluation: transcription (WER/CER) + UTMOS on pre-generated audio."
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "evaluation",
        help=(
            "Either outputs/evaluation (processes all models), or one model directory "
            "containing baseline/manifest.json and quant folders."
        ),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Workbook to update. Defaults to <outputs>/evaluation/analysis_report.xlsx.",
    )
    parser.add_argument(
        "--quant-type",
        default="all",
        help="Comma-separated quant folders to process, or 'all' for every manifest folder.",
    )
    parser.add_argument(
        "--asr-model",
        default=None,
        help="Cohere ASR checkpoint. Defaults to evaluation.transcription.DEFAULT_MODEL.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for transcription.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="ASR sampling rate.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language for transcription.",
    )
    parser.add_argument(
        "--skip-transcription",
        action="store_true",
        help="Skip transcription (WER/CER), only run UTMOS.",
    )
    parser.add_argument(
        "--skip-utmos",
        action="store_true",
        help="Skip UTMOS, only run transcription.",
    )
    return parser.parse_args()


def load_manifest(run_dir: Path) -> list[dict]:
    """Load manifest.json from a run directory."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def discover_model_dirs(audio_root: Path) -> list[Path]:
    """Find all model directories with baseline/manifest.json."""
    if (audio_root / "baseline" / "manifest.json").exists():
        return [audio_root]
    if not audio_root.exists():
        raise FileNotFoundError(f"Audio root not found: {audio_root}")
    model_dirs = [
        p for p in sorted(audio_root.iterdir())
        if (p / "baseline" / "manifest.json").exists()
    ]
    if not model_dirs:
        raise FileNotFoundError(
            f"No model folders with baseline/manifest.json found under {audio_root}"
        )
    return model_dirs


def quant_dirs_for(model_dir: Path, quant_type: str) -> list[Path]:
    """Get list of quant directories to process."""
    if quant_type == "all":
        return [
            p for p in sorted(model_dir.iterdir())
            if p.is_dir()
            and p.name not in {"baseline", "analysis"}
            and (p / "manifest.json").exists()
        ]
    names = [name.strip() for name in quant_type.split(",") if name.strip()]
    if not names:
        raise ValueError("--quant-type must not be empty")
    dirs = [model_dir / name for name in names]
    missing = [str(path) for path in dirs if not (path / "manifest.json").exists()]
    if missing:
        raise FileNotFoundError(f"Missing quant manifest(s): {missing}")
    return dirs


def report_path_for(audio_root: Path, explicit: Path | None) -> Path:
    """Determine the report path."""
    if explicit is not None:
        return explicit
    if audio_root.name == "evaluation":
        return audio_root / "analysis_report.xlsx"
    return audio_root.parent / "analysis_report.xlsx"


def main() -> int:
    args = parse_args()

    print("=" * 80)
    print("UNIFIED EVALUATION: Transcription + UTMOS")
    print("=" * 80)

    # Discover model directories
    model_dirs = discover_model_dirs(args.audio_root)
    print(f"\nFound {len(model_dirs)} model(s):")
    for model_dir in model_dirs:
        print(f"  - {model_dir.name}")

    # Report path
    report_path = report_path_for(args.audio_root, args.report_path)
    print(f"\nReport: {report_path}")

    # Load ASR model once for all transcriptions
    asr_processor, asr_model = None, None
    if not args.skip_transcription:
        print("\n" + "-" * 80)
        print("Loading ASR model...")
        try:
            asr_processor, asr_model = load_asr_model(
                model_name=args.asr_model, device_map="auto"
            )
            print(f"✓ ASR model loaded on {asr_model.device}")
        except Exception as e:
            print(f"✗ Failed to load ASR model: {e}")
            print("  Skipping transcription. Set HF_TOKEN if needed.")
            args.skip_transcription = True

    # Process each model
    for model_dir in model_dirs:
        print("\n" + "=" * 80)
        print(f"Processing: {model_dir.name}")
        print("=" * 80)

        # Get baseline + quant dirs
        baseline_dir = model_dir / "baseline"
        quant_dirs = quant_dirs_for(model_dir, args.quant_type)

        print(f"\nBaseline: {baseline_dir.name}")
        print(f"Quants: {', '.join(d.name for d in quant_dirs)}")

        # Load manifests
        try:
            baseline_manifest = load_manifest(baseline_dir)
        except FileNotFoundError as e:
            print(f"✗ {e}")
            continue

        model_name = baseline_manifest[0]["model_name"]
        print(f"Model: {model_name}")

        # Process baseline
        print("\n" + "-" * 80)
        print("Baseline:")
        print("-" * 80)

        if not args.skip_transcription:
            print("Running transcription...")
            baseline_trans = evaluate_transcription_run(
                manifest=baseline_manifest,
                processor=asr_processor,
                model=asr_model,
                batch_size=args.batch_size,
                sampling_rate=args.sampling_rate,
                language=args.language,
            )
            print(f"✓ Transcribed {len(baseline_trans['samples'])} samples")
        else:
            baseline_trans = None

        if not args.skip_utmos:
            print("Computing UTMOS...")
            baseline_utmos = evaluate_utmos_run(baseline_manifest)
            print(f"✓ Scored {len(baseline_utmos['samples'])} samples")
        else:
            baseline_utmos = None

        # Update report for baseline
        print("Updating report...")
        update_transcription_and_utmos_report(
            report_path=report_path,
            model=model_name,
            quant_type="baseline",
            transcription_results=baseline_trans,
            utmos_results=baseline_utmos,
        )
        print("✓ Report updated for baseline")

        # Process each quant
        for quant_dir in quant_dirs:
            print("\n" + "-" * 80)
            print(f"Quant: {quant_dir.name}")
            print("-" * 80)

            try:
                quant_manifest = load_manifest(quant_dir)
            except FileNotFoundError as e:
                print(f"✗ {e}")
                continue

            if not args.skip_transcription:
                print("Running transcription...")
                quant_trans = evaluate_transcription_run(
                    manifest=quant_manifest,
                    processor=asr_processor,
                    model=asr_model,
                    batch_size=args.batch_size,
                    sampling_rate=args.sampling_rate,
                    language=args.language,
                )
                print(f"✓ Transcribed {len(quant_trans['samples'])} samples")
            else:
                quant_trans = None

            if not args.skip_utmos:
                print("Computing UTMOS...")
                quant_utmos = evaluate_utmos_run(quant_manifest)
                print(f"✓ Scored {len(quant_utmos['samples'])} samples")
            else:
                quant_utmos = None

            # Update report
            print("Updating report...")
            update_transcription_and_utmos_report(
                report_path=report_path,
                model=model_name,
                quant_type=quant_dir.name,
                transcription_results=quant_trans,
                utmos_results=quant_utmos,
            )
            print(f"✓ Report updated for {quant_dir.name}")

    print("\n" + "=" * 80)
    print("UNIFIED EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Report: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
