#!/usr/bin/env python3
"""Run Cohere ASR WER/CER on existing generated audio and upsert the report.

This is intentionally separate from src/evaluate.py: generation/quantization writes
manifests and non-ASR metrics first, then this script can be run later against the
saved audio folders to fill transcript, WER, and CER fields in the same workbook.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import triton.language  # noqa: F401 - torch._dynamo expects triton.language to be attached.

from evaluation.report import update_transcription_report
from evaluation.transcription import evaluate_transcription_run, load_asr_model

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append Cohere-ASR WER/CER results to analysis_report.xlsx.")
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "evaluation",
        help=(
            "Either outputs/evaluation, or one model output directory containing "
            "baseline/manifest.json and quant folders."
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
        help="Comma-separated quant folders to transcribe, or 'all' for every non-baseline manifest folder.",
    )
    parser.add_argument(
        "--asr-model",
        default=None,
        help="Cohere ASR checkpoint id/path. Defaults to evaluation.transcription.DEFAULT_MODEL.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--language", default="en")
    return parser.parse_args()


def load_manifest(run_dir: Path) -> list[dict]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def discover_model_dirs(audio_root: Path) -> list[Path]:
    if (audio_root / "baseline" / "manifest.json").exists():
        return [audio_root]
    if not audio_root.exists():
        raise FileNotFoundError(f"Audio root not found: {audio_root}")
    model_dirs = [p for p in sorted(audio_root.iterdir()) if (p / "baseline" / "manifest.json").exists()]
    if not model_dirs:
        raise FileNotFoundError(
            f"No model output folders with baseline/manifest.json found under {audio_root}"
        )
    return model_dirs


def quant_dirs_for(model_dir: Path, quant_type: str) -> list[Path]:
    if quant_type == "all":
        return [
            p for p in sorted(model_dir.iterdir())
            if p.is_dir() and p.name not in {"baseline", "analysis"} and (p / "manifest.json").exists()
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
    if explicit is not None:
        return explicit
    if audio_root.name == "evaluation":
        return audio_root / "analysis_report.xlsx"
    return audio_root.parent / "analysis_report.xlsx"


def main() -> None:
    args = parse_args()
    report_path = report_path_for(args.audio_root, args.report_path)
    asr_processor, asr_model = load_asr_model(args.asr_model) if args.asr_model else load_asr_model()

    for model_dir in discover_model_dirs(args.audio_root):
        baseline_dir = model_dir / "baseline"
        baseline_manifest = load_manifest(baseline_dir)
        if not baseline_manifest:
            raise ValueError(f"Empty baseline manifest: {baseline_dir / 'manifest.json'}")

        model_name = baseline_manifest[0].get("model_name") or model_dir.name
        prompts = [entry.get("text", "") for entry in baseline_manifest]
        baseline_transcription = evaluate_transcription_run(
            baseline_dir,
            prompts,
            asr_model,
            asr_processor,
            batch_size=args.batch_size,
            sampling_rate=args.sampling_rate,
            language=args.language,
        )

        for quant_dir in quant_dirs_for(model_dir, args.quant_type):
            quant_type = quant_dir.name
            print(f"[{model_dir.name}/{quant_type}] Transcribing and updating {report_path}")
            quant_transcription = evaluate_transcription_run(
                quant_dir,
                prompts,
                asr_model,
                asr_processor,
                batch_size=args.batch_size,
                sampling_rate=args.sampling_rate,
                language=args.language,
            )
            update_transcription_report(
                report_path,
                model=model_name,
                quant_type=quant_type,
                baseline_transcription=baseline_transcription,
                quant_transcription=quant_transcription,
            )

    print(f"Transcription metrics upserted into {report_path}")


if __name__ == "__main__":
    main()
