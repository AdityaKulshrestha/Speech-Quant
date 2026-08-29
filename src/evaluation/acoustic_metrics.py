"""
Phase 1 (Macroscopic) acoustic-distortion metrics from METRICS.md:
Mel-Cepstral Distortion (MCD), F0 Frame Error / Pitch Pearson Correlation, and
UTMOS between baseline and quantized waveforms for the same prompt.

MCD/F0 operate on raw audio rather than codec tokens, so they are naturally
codec-agnostic and apply the same way to Orpheus, NeuTTS, and OuteTTS.

Phase 1's remaining metric (NISQA, SECS) is intentionally not implemented
here; see METRICS.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio

_utmos_predictor = None  # lazily loaded, cached across calls (torch.hub download is expensive)


def _load_audio(path: str | Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if target_sr is not None and sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, sr


def mel_cepstral_distortion(
    reference: np.ndarray,
    hypothesis: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
) -> float:
    """Mel-cepstral distortion (dB) between two waveforms of the same prompt.

    MFCCs (excluding the 0th/energy coefficient) stand in for MCEPs, DTW-aligned
    frame-by-frame, following the standard MCD formula:
        MCD = (10 / ln(10)) * sqrt(2 * sum((c_ref - c_hyp)^2)), averaged over frames.
    """
    ref_mfcc = librosa.feature.mfcc(y=reference, sr=sr, n_mfcc=n_mfcc)[1:]
    hyp_mfcc = librosa.feature.mfcc(y=hypothesis, sr=sr, n_mfcc=n_mfcc)[1:]

    _, wp = librosa.sequence.dtw(X=ref_mfcc, Y=hyp_mfcc, metric="euclidean")
    diffs = ref_mfcc[:, wp[:, 0]] - hyp_mfcc[:, wp[:, 1]]
    frame_dist = np.sqrt((diffs**2).sum(axis=0))
    return float((10.0 / np.log(10.0)) * np.sqrt(2.0) * frame_dist.mean())


def extract_f0(
    audio: np.ndarray,
    sr: int,
    fmin: float = 65.0,
    fmax: float = 400.0,
    frame_length: int = 1024,
) -> np.ndarray:
    """Frame-wise F0 contour (Hz, 0 for unvoiced frames) via librosa's pYIN."""
    f0, _voiced_flag, _voiced_prob = librosa.pyin(
        audio, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length
    )
    return np.nan_to_num(f0, nan=0.0)


def f0_frame_error(
    reference: np.ndarray,
    hypothesis: np.ndarray,
    sr: int,
    cent_threshold: float = 50.0,
) -> float:
    """Fraction of frames with a voicing mismatch or a pitch error above
    `cent_threshold` cents (on mutually-voiced frames)."""
    ref_f0 = extract_f0(reference, sr)
    hyp_f0 = extract_f0(hypothesis, sr)
    n = min(len(ref_f0), len(hyp_f0))
    if n == 0:
        return 0.0
    ref_f0, hyp_f0 = ref_f0[:n], hyp_f0[:n]

    ref_voiced = ref_f0 > 0
    hyp_voiced = hyp_f0 > 0
    voicing_mismatch = ref_voiced != hyp_voiced

    both_voiced = ref_voiced & hyp_voiced
    cents = np.zeros(n)
    cents[both_voiced] = 1200.0 * np.abs(np.log2(hyp_f0[both_voiced] / ref_f0[both_voiced]))
    pitch_error = both_voiced & (cents > cent_threshold)

    return float((voicing_mismatch | pitch_error).mean())


def pitch_pearson_correlation(
    reference: np.ndarray,
    hypothesis: np.ndarray,
    sr: int,
) -> float | None:
    """Pearson correlation between F0 contours on mutually-voiced frames.

    Returns None when fewer than 2 mutually-voiced frames are available.
    """
    ref_f0 = extract_f0(reference, sr)
    hyp_f0 = extract_f0(hypothesis, sr)
    n = min(len(ref_f0), len(hyp_f0))
    if n == 0:
        return None
    ref_f0, hyp_f0 = ref_f0[:n], hyp_f0[:n]

    voiced = (ref_f0 > 0) & (hyp_f0 > 0)
    if voiced.sum() < 2:
        return None

    corr = np.corrcoef(ref_f0[voiced], hyp_f0[voiced])[0, 1]
    return float(corr) if np.isfinite(corr) else None


def utmos_score(audio_path: str | Path) -> float | None:
    """Perceptual quality score via UTMOS (1=bad, 5=excellent).

    Non-intrusive metric that predicts human Mean Opinion Score. Loads the
    tarepan/SpeechMOS `utmos22_strong` model via torch.hub (there is no PyPI
    package for this model; the `speechmos` package on PyPI is an unrelated
    Microsoft AECMOS/DNSMOS/PLCMOS suite).

    Args:
        audio_path: Path to audio file

    Returns:
        UTMOS score in [1, 5] range, or None if scoring fails
    """
    global _utmos_predictor
    try:
        if _utmos_predictor is None:
            _utmos_predictor = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
            )

        # Load audio at 16kHz (UTMOS requirement) using soundfile instead of torchaudio
        audio_np, sr = _load_audio(audio_path, target_sr=16000)

        # Convert numpy array to torch tensor [1, samples] for UTMOS model
        audio = torch.from_numpy(audio_np).unsqueeze(0)

        score = _utmos_predictor(audio, sr=16000)
        return float(score)

    except Exception as e:
        print(f"UTMOS scoring failed: {e}")
        return None


def compare_acoustics(reference_path: str | Path, hypothesis_path: str | Path) -> dict[str, Any]:
    """MCD + F0 + UTMOS for one baseline/quantized audio pair."""
    reference, sr = _load_audio(reference_path)
    hypothesis, _ = _load_audio(hypothesis_path, target_sr=sr)

    return {
        "mcd": mel_cepstral_distortion(reference, hypothesis, sr),
        "f0_frame_error": f0_frame_error(reference, hypothesis, sr),
        "pitch_pearson_correlation": pitch_pearson_correlation(reference, hypothesis, sr),
        "utmos_baseline": utmos_score(reference_path),
        "utmos_quant": utmos_score(hypothesis_path),
    }


def compare_acoustics_manifest(
    baseline_manifest: list[dict[str, Any]],
    quant_manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run compare_acoustics over every matched baseline/quantized sample pair."""
    per_sample = []
    for baseline_entry, quant_entry in zip(baseline_manifest, quant_manifest):
        result = compare_acoustics(baseline_entry["audio_path"], quant_entry["audio_path"])
        result["sample_id"] = baseline_entry["sample_id"]
        result["text"] = baseline_entry["text"]
        per_sample.append(result)
    return per_sample


def summarize_acoustics(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate MCD/F0/UTMOS metrics across samples for the final score block."""
    mcds = [s["mcd"] for s in per_sample if s.get("mcd") is not None]
    fers = [s["f0_frame_error"] for s in per_sample if s.get("f0_frame_error") is not None]
    corrs = [s["pitch_pearson_correlation"] for s in per_sample if s.get("pitch_pearson_correlation") is not None]
    utmos_base = [s["utmos_baseline"] for s in per_sample if s.get("utmos_baseline") is not None]
    utmos_quant = [s["utmos_quant"] for s in per_sample if s.get("utmos_quant") is not None]

    return {
        "num_samples": len(per_sample),
        "mean_mcd": sum(mcds) / len(mcds) if mcds else None,
        "mean_f0_frame_error": sum(fers) / len(fers) if fers else None,
        "mean_pitch_pearson_correlation": sum(corrs) / len(corrs) if corrs else None,
        "mean_utmos_baseline": sum(utmos_base) / len(utmos_base) if utmos_base else None,
        "mean_utmos_quant": sum(utmos_quant) / len(utmos_quant) if utmos_quant else None,
        "utmos_degradation": (sum(utmos_base) / len(utmos_base) - sum(utmos_quant) / len(utmos_quant))
                             if utmos_base and utmos_quant else None,
        "num_samples_with_pitch_correlation": len(corrs),
    }


__all__ = [
    "compare_acoustics",
    "compare_acoustics_manifest",
    "extract_f0",
    "f0_frame_error",
    "mel_cepstral_distortion",
    "pitch_pearson_correlation",
    "summarize_acoustics",
    "utmos_score",
]
