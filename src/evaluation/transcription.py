#!/usr/bin/env python3
"""Batch transcription WER/CER evaluation for quantized speech models.

This script:
  1. scans generated audio outputs for each quantized model under an output root,
  2. transcribes those waveforms in batches using the Cohere ASR model,
  3. compares each transcript against the original prompt text,
  4. aggregates WER/CER per quant and writes the result to a JSON file.

Example:
  python src/evaluation/transcription.py \
      --audio-root outputs \
      --prompts-file src/prompts.txt \
      --output-json src/evaluation/transcription.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio


DEFAULT_MODEL = "CohereLabs/cohere-transcribe-03-2026"
DEFAULT_SAMPLE_RATE = 16000


def normalize_text(text: str) -> str:
    """Normalize text for robust WER/CER comparison."""
    return " ".join((text or "").lower().strip().split())


def levenshtein_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Minimum edit distance between two token sequences."""
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
    denom = max(len(ref_tokens), 1)
    return distance / denom


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref_chars = list(normalize_text(reference))
    hyp_chars = list(normalize_text(hypothesis))
    if not ref_chars and not hyp_chars:
        return 0.0
    distance = levenshtein_distance(ref_chars, hyp_chars)
    denom = max(len(ref_chars), 1)
    return distance / denom


def load_prompts(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    prompts = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if text:
                prompts.append(text)

    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def collect_quant_dirs(audio_root: Path) -> list[Path]:
    """Collect quant directories to evaluate.

    If the root contains one or more subdirectories, each subdirectory is treated as
    a separate quant run. Otherwise the root itself is treated as one run.
    """
    if not audio_root.exists():
        raise FileNotFoundError(f"Audio root not found: {audio_root}")

    child_dirs = sorted([p for p in audio_root.iterdir() if p.is_dir()])
    if child_dirs:
        return child_dirs
    return [audio_root]


def find_audio_files(directory: Path) -> list[Path]:
    wavs = [
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}
    ]
    return sorted(wavs, key=lambda p: p.name)


def transcribe_batch(
    model: CohereAsrForConditionalGeneration,
    processor,
    audio_paths: Sequence[Path],
    batch_size: int,
    sampling_rate: int = DEFAULT_SAMPLE_RATE,
    language: str = "en",
) -> list[str]:
    """Batch-transcribe a list of audio files using the Cohere ASR model."""
    if not audio_paths:
        return []

    results: list[str] = []
    for start in range(0, len(audio_paths), batch_size):
        chunk = audio_paths[start : start + batch_size]
        audios = [load_audio(str(path), sampling_rate=sampling_rate) for path in chunk]
        inputs = processor(
            audios,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            language=language,
        )

        audio_chunk_index = inputs.get("audio_chunk_index")
        inputs = inputs.to(model.device, dtype=getattr(model, "dtype", torch.float32))

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
            )

        texts = processor.decode(
            outputs,
            skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language=language,
        )

        if isinstance(texts, list):
            results.extend(texts)
        else:
            results.append(texts)

    return results


def load_asr_model(model_name: str = DEFAULT_MODEL, device_map: str = "auto"):
    """Load the Cohere ASR processor/model pair with the correct dtype for the runtime."""
    processor = AutoProcessor.from_pretrained(model_name)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=(
            torch.bfloat16
            if (torch.xpu.is_available() or torch.cuda.is_available())
            else torch.float32
        ),
    )
    return processor, model


def evaluate_transcription_run(
    run_dir: Path,
    prompts: Sequence[str],
    asr_model: CohereAsrForConditionalGeneration,
    processor,
    batch_size: int,
    sampling_rate: int = DEFAULT_SAMPLE_RATE,
    language: str = "en",
) -> dict:
    """Evaluate one audio run against the prompt list and return aggregate WER/CER metrics."""
    audio_files = find_audio_files(run_dir)
    if not audio_files:
        raise FileNotFoundError(f"No audio files found under {run_dir}")

    n_allowed = min(len(prompts), len(audio_files))
    selected_files = audio_files[:n_allowed]
    transcripts = transcribe_batch(
        asr_model,
        processor,
        selected_files,
        batch_size=batch_size,
        sampling_rate=sampling_rate,
        language=language,
    )
    if len(transcripts) != n_allowed:
        raise ValueError(
            f"Expected {n_allowed} transcripts for {run_dir}, got {len(transcripts)}."
        )

    entries = []
    wer_values = []
    cer_values = []
    for prompt, transcript in zip(prompts[:n_allowed], transcripts):
        wer = word_error_rate(prompt, transcript)
        cer = character_error_rate(prompt, transcript)
        wer_values.append(wer)
        cer_values.append(cer)
        entries.append({"prompt": prompt, "transcript": transcript, "wer": wer, "cer": cer})

    return {
        "run": run_dir.name,
        "num_samples": n_allowed,
        "mean_wer": sum(wer_values) / len(wer_values) if wer_values else 0.0,
        "mean_cer": sum(cer_values) / len(cer_values) if cer_values else 0.0,
        "transcripts": transcripts,
        "samples": entries,
    }


def evaluate_quant_dir(
    quant_dir: Path,
    prompts: Sequence[str],
    model: CohereAsrForConditionalGeneration,
    processor,
    batch_size: int,
    sampling_rate: int = DEFAULT_SAMPLE_RATE,
) -> dict:
    """Evaluate a single quant directory against its prompt list."""
    audio_files = find_audio_files(quant_dir)
    if not audio_files:
        raise FileNotFoundError(f"No audio files found under {quant_dir}")

    if len(audio_files) < len(prompts):
        raise ValueError(
            f"Found {len(audio_files)} audio files in {quant_dir}, but need at least {len(prompts)} "
            f"for the prompt list."
        )

    subset = audio_files[: len(prompts)]
    transcripts = transcribe_batch(
        model=model,
        processor=processor,
        audio_paths=subset,
        batch_size=batch_size,
        sampling_rate=sampling_rate,
    )

    if len(transcripts) != len(prompts):
        raise ValueError(
            f"Expected {len(prompts)} transcripts for {quant_dir}, got {len(transcripts)}. "
            f"This usually means the batch decode returned fewer items than expected."
        )

    score_rows = []
    wer_values = []
    cer_values = []

    for prompt, transcript in zip(prompts, transcripts):
        wer = word_error_rate(prompt, transcript)
        cer = character_error_rate(prompt, transcript)
        wer_values.append(wer)
        cer_values.append(cer)
        score_rows.append(
            {
                "prompt": prompt,
                "transcript": transcript,
                "wer": wer,
                "cer": cer,
            }
        )

    return {
        "quant": quant_dir.name,
        "num_samples": len(subset),
        "mean_wer": sum(wer_values) / len(wer_values) if wer_values else 0.0,
        "mean_cer": sum(cer_values) / len(cer_values) if cer_values else 0.0,
        "samples": score_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute WER/CER for ASR transcriptions across quant runs.")
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "outputs",
        help="Root directory containing per-quant output folders (default: <repo>/outputs).",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "prompts.txt",
        help="Text file containing one ground-truth prompt per line.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Cohere ASR model ID used for transcription.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of audio files to transcribe in one batch for faster throughput.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Target sampling rate used by the ASR pipeline.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code passed to the processor and decoder.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).resolve().parent / "transcription.json",
        help="Path for the aggregated output JSON.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_file)
    print(f"Loaded {len(prompts)} ground-truth prompts from {args.prompts_file}")

    processor, model = load_asr_model(args.model, device_map="auto")

    quant_dirs = collect_quant_dirs(args.audio_root)
    print(f"Evaluating {len(quant_dirs)} quant runs under {args.audio_root}")

    results = []
    for quant_dir in quant_dirs:
        print(f"\n[{quant_dir.name}] Transcribing batch ...")
        quant_result = evaluate_quant_dir(
            quant_dir=quant_dir,
            prompts=prompts,
            model=model,
            processor=processor,
            batch_size=args.batch_size,
            sampling_rate=args.sampling_rate,
        )
        results.append(quant_result)
        print(
            f"[{quant_dir.name}] mean WER={quant_result['mean_wer']:.4f}, "
            f"mean CER={quant_result['mean_cer']:.4f}"
        )

    aggregate = {
        "model": args.model,
        "audio_root": str(args.audio_root),
        "prompts_file": str(args.prompts_file),
        "language": args.language,
        "batch_size": args.batch_size,
        "sampling_rate": args.sampling_rate,
        "results": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as fh:
        json.dump(aggregate, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"\nSaved aggregate transcription metrics to {args.output_json}")


if __name__ == "__main__":
    main()
