"""
Loads or quantizes a model using gptqmodel GPTQ.

Two paths:
  - model_id already contains quantize_config.json  →  load pre-quantized.
  - model_id is a float checkpoint  →  quantize on the fly with GPTQ
    (calibration data auto-fetched from allenai/c4 if not supplied).
"""

from pathlib import Path
from typing import List, Optional, Union

import torch

from .config import QUANT_CONFIGS


def _is_pre_quantized(model_id: str) -> bool:
    """True when model_id is a local directory that already holds GPTQ weights."""
    return (Path(model_id) / "quantize_config.json").exists()


def _fetch_calibration_data(n_samples: int = 1024) -> List[str]:
    """Download a small slice of C4 for GPTQ calibration (mirrors test.py)."""
    from datasets import load_dataset

    return (
        load_dataset(
            "allenai/c4",
            data_files="en/c4-train.00001-of-01024.json.gz",
            split="train",
        )
        .select(range(n_samples))["text"]
    )


def quantize_model(
    model_id: str,
    quant_type: str,
    calibration: Optional[Union[List[str], List[dict]]] = None,
    batch_size: int = 1,
    device: Optional[str] = None,
) -> torch.nn.Module:
    """
    Return an HF torch.nn.Module ready for inference.

    If *model_id* is a pre-quantized GPTQ directory the weights are loaded
    directly.  Otherwise the float model is quantized with GPTQ using
    *calibration* (or C4 if omitted) and the result is returned in memory
    without saving to disk.
    """

    if quant_type not in QUANT_CONFIGS:
        raise ValueError(
            f"Unknown quant_type={quant_type!r}. "
            f"Available: {sorted(QUANT_CONFIGS)}"
        )

    if quant_type == "none":
        raise ValueError("quantize_model() must not be called for quant_type='none'.")

    config_fn = QUANT_CONFIGS[quant_type]
    if config_fn is None:
        raise NotImplementedError(
            f"quant_type={quant_type!r} is registered but has no config. "
            "Add one in quants/config.py."
        )

    from gptqmodel import GPTQModel

    load_kwargs = {}
    if device is not None:
        load_kwargs["device"] = device

    if _is_pre_quantized(model_id):
        gptq_model = GPTQModel.load(model_id, **load_kwargs)
    else:
        if calibration is None:
            calibration = _fetch_calibration_data()
        gptq_model = GPTQModel.load(model_id, quantize_config=config_fn(), **load_kwargs)
        gptq_model.quantize(calibration, batch_size=batch_size)

    return gptq_model.model
