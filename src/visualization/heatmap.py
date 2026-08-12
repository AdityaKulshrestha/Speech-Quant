"""
Per-token match heatmap: green = baseline and quantized token agree,
red = they differ, light-gray = position beyond the shorter sequence.
"""

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def save_token_heatmap(
    baseline_manifest: List[dict],
    quant_manifest: List[dict],
    output_path: str | Path,
) -> None:
    n_samples = len(baseline_manifest)

    match_arrays = []
    for b_entry, q_entry in zip(baseline_manifest, quant_manifest):
        b = np.array(b_entry["audio_tokens"])
        q = np.array(q_entry["audio_tokens"])
        n = min(len(b), len(q))
        match_arrays.append((b[:n] == q[:n]).astype(float))

    max_len = max(len(a) for a in match_arrays) if match_arrays else 1

    # NaN marks positions beyond the shorter sequence (masked → gray)
    matrix = np.full((n_samples, max_len), np.nan)
    for i, arr in enumerate(match_arrays):
        matrix[i, : len(arr)] = arr

    masked = np.ma.masked_invalid(matrix)

    cmap = mcolors.ListedColormap(["#e74c3c", "#2ecc71"])  # red=0, green=1
    cmap.set_bad("whitesmoke")

    fig_w = max(12, max_len / 60)
    fig_h = max(3, n_samples * 0.7 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="none")

    # Minimal colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=[0.25, 0.75], fraction=0.02, pad=0.02)
    cbar.ax.set_yticklabels(["Mismatch", "Match"])

    ax.set_xlabel("Token position")
    ax.set_ylabel("Sample")
    ax.set_yticks(range(n_samples))
    ax.set_yticklabels([e["sample_id"] for e in baseline_manifest], fontsize=8)
    ax.set_title("Token match heatmap  ·  green = correct  ·  red = incorrect")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Heatmap: {output_path}")
