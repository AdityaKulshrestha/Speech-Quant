#!/usr/bin/env python3
"""Batch transcription WER/CER evaluation for quantized speech models.

This script evaluates generated audio under each quantized output directory by:
  1. batching audio files into Cohere ASR requests,
  2. comparing the transcript to the original prompt text,
  3. aggregating WER and CER per quantized run,
  4. writing a final JSON summary to the evaluation output folder.

Example:
  python src/evaluation/transcriptiont.py \
      --audio-root outputs \
      --prompts-file src/prompts.txt \
      --output-json evaluation/transcription.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio

DEFAULT_MODEL = "CohereLabs/cohere-transcribe-03-2026"
DEFAULT_SAMPLE_RATE = 16000


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


def load_prompts(path: Path) -> list[str]:
    """Return the list of ground-truth prompts from a text file."""
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    prompts = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if text:
                prompts.append(text)

    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def find_audio_files(directory: Path) -> list[Path]:
    """Return all audio files under a directory, sorted by file name."""
    audio_types = {".wav", ".flac", ".mp3", ".ogg"}
    return sorted(
        [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in audio_types],
        key=lambda p: p.name,
    )


def collect_quant_dirs(audio_root: Path) -> list[Path]:
    """Return the actual per-quant directories that contain audio; ignore analysis/meta folders."""
    if not audio_root.exists():
        raise FileNotFoundError(f"Audio root not found: {audio_root}")

    child_dirs = [p for p in sorted(audio_root.iterdir()) if p.is_dir()]
    audio_child_dirs = [p for p in child_dirs if find_audio_files(p)]
    if audio_child_dirs:
        return audio_child_dirs

    # fall back to nested directories only when no direct child run folder is present
    candidates = [p for p in sorted(audio_root.rglob("*")) if p.is_dir() and find_audio_files(p)]
    if candidates:
        return candidates

    return [audio_root] if find_audio_files(audio_root) else []


def transcribe_batch(
    model: CohereAsrForConditionalGeneration,
    processor,
    audio_paths: Sequence[Path],
    batch_size: int,
    sampling_rate: int = DEFAULT_SAMPLE_RATE,
    language: str = "en",
) -> list[str]:
    """Batch-transcribe a sequence of audio files with the Cohere ASR model."""
    if not audio_paths:
        return []

    transcripts: list[str] = []
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
        inputs = inputs.to(model.device, dtype=getattr(model, "dtype", torch.float16))

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
            )

        decoded = processor.decode(
            outputs,
            skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language=language,
        )

        if isinstance(decoded, list):
            transcripts.extend(decoded)
        else:
            transcripts.append(decoded)

    return transcripts


def evaluate_quant_dir(
    quant_dir: Path,
    prompts: Sequence[str],
    model: CohereAsrForConditionalGeneration,
    processor,
    batch_size: int,
    sampling_rate: int = DEFAULT_SAMPLE_RATE,
    language: str = "en",
) -> dict:
    """Evaluate one quant run against the prompt list and compute aggregate metrics."""
    audio_files = find_audio_files(quant_dir)
    if not audio_files:
        raise FileNotFoundError(f"No audio files found under {quant_dir}")

    if len(audio_files) < len(prompts):
        raise ValueError(
            f"Found {len(audio_files)} audio files in {quant_dir}, but need at least {len(prompts)} for the prompt list."
        )

    selected_files = audio_files[: len(prompts)]
    transcripts = transcribe_batch(
        model=model,
        processor=processor,
        audio_paths=selected_files,
        batch_size=batch_size,
        sampling_rate=sampling_rate,
        language=language,
    )

    if len(transcripts) != len(prompts):
        raise ValueError(
            f"Expected {len(prompts)} transcripts for {quant_dir}, got {len(transcripts)}. "
            "This usually means the model returned fewer results than expected."
        )

    entries = []
    wer_values = []
    cer_values = []

    for prompt, transcript in zip(prompts, transcripts):
        wer = word_error_rate(prompt, transcript)
        cer = character_error_rate(prompt, transcript)
        wer_values.append(wer)
        cer_values.append(cer)
        entries.append(
            {
                "prompt": prompt,
                "transcript": transcript,
                "wer": wer,
                "cer": cer,
            }
        )

    return {
        "quant": quant_dir.name,
        "num_samples": len(selected_files),
        "mean_wer": sum(wer_values) / len(wer_values) if wer_values else 0.0,
        "mean_cer": sum(cer_values) / len(cer_values) if cer_values else 0.0,
        "transcripts": transcripts,
        "samples": entries,
    }


def pairwise_wer_matrix(labels: Sequence[str], text_groups: dict[str, Sequence[str]]) -> dict:
    """Build an aggregate WER grid among ground-truth, baseline, and quant output texts."""
    label_order = list(labels)
    n = len(label_order)
    if n == 0:
        return {"labels": [], "mean_wer_matrix": []}

    lengths = {label: len(text_groups.get(label, [])) for label in label_order}
    sample_count = min(lengths.values())
    if sample_count == 0:
        return {"labels": label_order, "mean_wer_matrix": [[0.0 for _ in range(n)] for _ in range(n)]}

    for label in label_order:
        if label not in text_groups:
            raise KeyError(f"Missing text group for label: {label}")
        if len(text_groups[label]) < sample_count:
            raise ValueError(
                f"Text group '{label}' has {len(text_groups[label])} entries, expected at least {sample_count}."
            )

    mean_matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    for idx in range(sample_count):
        for i, left in enumerate(label_order):
            left_text = text_groups[left][idx]
            for j, right in enumerate(label_order):
                right_text = text_groups[right][idx]
                score = word_error_rate(left_text, right_text)
                mean_matrix[i][j] += score

    for i in range(n):
        for j in range(n):
            mean_matrix[i][j] /= sample_count

    return {
        "labels": label_order,
        "mean_wer_matrix": mean_matrix,
    }


def infer_default_output_json(audio_root: Path, repo_root: Path) -> Path:
    """Default to outputs/evaluation/<model_name>/transcription.json when a model folder is passed."""
    if audio_root == repo_root / "outputs":
        return repo_root / "outputs" / "evaluation" / "aggregate" / "transcription.json"

    model_name = audio_root.name
    return repo_root / "outputs" / "evaluation" / model_name / "transcription.json"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Batch-transcribe generated audio and compare it with the original prompts using WER and CER."
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=repo_root / "outputs",
        help="Root containing per-quant output folders with generated audio files.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=repo_root / "src" / "prompts.txt",
        help="Ground-truth prompt list with one prompt per line.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Cohere ASR model name to use for transcription.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of audio files to transcribe in each batch for faster processing.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Limit evaluation to the first N prompt lines. Defaults to the number of available audio files in each folder.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Sampling rate used for audio loading and ASR inference.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code passed to the ASR processor.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to the aggregated transcription JSON summary. Defaults to outputs/evaluation/<model_name>/transcription.json.",
    )
    args = parser.parse_args()
    args.output_json = args.output_json or infer_default_output_json(args.audio_root, repo_root)
    return args


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts_file)
    print(f"Loaded {len(prompts)} ground-truth prompts from {args.prompts_file}")

    processor = AutoProcessor.from_pretrained(args.model)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        args.model,
        device_map="auto",
    )

    quant_dirs = collect_quant_dirs(args.audio_root)
    print(f"Evaluating {len(quant_dirs)} quant run(s) under {args.audio_root}")

    if args.num_prompts is not None:
        prompts = prompts[: args.num_prompts]
        print(f"Using only the first {len(prompts)} prompt(s) from {args.prompts_file}")

    results = []
    texts_by_label = {"ground_truth": list(prompts)}
    for quant_dir in quant_dirs:
        audio_files = find_audio_files(quant_dir)
        n_allowed = min(len(prompts), len(audio_files))
        prompt_slice = prompts[:n_allowed]
        print(f"\n[{quant_dir.name}] batch transcription ({n_allowed}/{len(prompts)} prompts, {len(audio_files)} audio files) ...")
        quant_result = evaluate_quant_dir(
            quant_dir=quant_dir,
            prompts=prompt_slice,
            model=model,
            processor=processor,
            batch_size=args.batch_size,
            sampling_rate=args.sampling_rate,
            language=args.language,
        )
        results.append(quant_result)
        texts_by_label[quant_dir.name] = list(quant_result["transcripts"][:n_allowed])
        print(
            f"[{quant_dir.name}] mean WER={quant_result['mean_wer']:.4f}, "
            f"mean CER={quant_result['mean_cer']:.4f}"
        )

    # Keep the pairwise grid aligned to the matching prompt/audio slice used by the run.
    matched_count = min(len(v) for v in texts_by_label.values()) if texts_by_label else 0
    if matched_count:
        texts_by_label = {k: v[:matched_count] for k, v in texts_by_label.items()}

    label_order = list(texts_by_label.keys())
    pairwise_grid = pairwise_wer_matrix(label_order, texts_by_label)

    aggregate = {
        "model": args.model,
        "audio_root": str(args.audio_root),
        "prompts_file": str(args.prompts_file),
        "language": args.language,
        "batch_size": args.batch_size,
        "sampling_rate": args.sampling_rate,
        "results": results,
        "pairwise_wer_grid": pairwise_grid,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as fh:
        json.dump(aggregate, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"\nSaved aggregate transcription metrics to {args.output_json}")


if __name__ == "__main__":
    main()
