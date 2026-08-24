"""
Loads or quantizes a model using gptqmodel GPTQ.

Resolution order:
  1. Cache hit: <QUANT_STORE>/<model_slug>/ exists  →  load from cache.
  2. model_id is already a pre-quantized local dir  →  load directly.
  3. Float model  →  GPTQ-quantize, save to cache, return.

Using BACKEND.GPTQ_TORCH (pure-PyTorch kernel) so quantized models work on
"""

import re
from pathlib import Path
from typing import List, Optional, Union

import torch

from .config import QUANT_CONFIGS

# One sub-directory per (model_id, quant_type) pair lives here.
_QUANT_STORE = Path(__file__).resolve().parents[2] / "quant_models"


def _model_slug(model_id: str, quant_type: str) -> str:
    """Build a safe directory name from model_id + quant_type."""
    slug = re.sub(r"[/\\:]+", "--", model_id.strip("/\\"))
    slug = re.sub(r"[\s.]+", "-", slug).strip("-")
    return f"{slug}__{quant_type}"


def _cache_path(model_id: str, quant_type: str, store: Path = _QUANT_STORE) -> Path:
    return store / _model_slug(model_id, quant_type)


def _is_pre_quantized(path: Union[str, Path]) -> bool:
    """True when *path* is a local directory that already holds GPTQ weights."""
    return (Path(path) / "quantize_config.json").exists()


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
    store_dir: Optional[Path] = None,
) -> torch.nn.Module:
    """
    Return an HF torch.nn.Module ready for inference on *device*.

    The model is always explicitly moved to *device* after loading so that
    all parameter tensors (including norm weights) land on the same device.
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
    from gptqmodel.utils.backend import BACKEND

    store = Path(store_dir) if store_dir else _QUANT_STORE
    cache = _cache_path(model_id, quant_type, store)

    # ------------------------------------------------------------------ load
    if _is_pre_quantized(cache):
        print(f"Loading cached GPTQ model: {cache}")
        load_src = str(cache)
    elif _is_pre_quantized(model_id):
        print(f"Loading pre-quantized model: {model_id}")
        load_src = model_id
    else:
        load_src = None

    if load_src is not None:
        # device_map={"": device} loads every buffer/param directly onto the
        # target device, avoiding the meta-tensor error that occurs when calling
        # .to() on lazily-initialised QuantLinear buffers after the fact.
        load_device_map = {"": device} if device else {"": "cpu"}
        gptq_model = GPTQModel.load(
            load_src,
            backend=BACKEND.GPTQ_TORCH,
            device_map=load_device_map,
        )
        return gptq_model.model

    # --------------------------------------------------------------- quantize
    if calibration is None:
        calibration = _fetch_calibration_data()

    print(f"Quantizing {model_id!r} → {cache}")
    gptq_model = GPTQModel.load(model_id, quantize_config=config_fn())
    gptq_model.quantize(calibration, batch_size=batch_size)

    cache.mkdir(parents=True, exist_ok=True)
    gptq_model.save(str(cache))
    print(f"Saved quantized model: {cache}")

    # Reload from the just-saved cache so device_map is applied uniformly.
    load_device_map = {"": device} if device else {"": "cpu"}
    gptq_model = GPTQModel.load(
        str(cache),
        backend=BACKEND.GPTQ_TORCH,
        device_map=load_device_map,
    )
    return gptq_model.model

