"""
Quantizes or loads a cached quantized model for TTS model backbones.

Two backends:
  - "llm_compressor" (default — every --quant-type alias in quants/config.py
    uses this today): applies RTN / GPTQ / AWQ / SmoothQuant via
    llmcompressor.oneshot(), saving/loading compressed-tensors checkpoints.
  - "gptq": the original gptqmodel-based implementation, kept as an internal
    alternate entrypoint (quantize_model(..., backend="gptq")) — not wired
    to any --quant-type value right now.

Resolution order (both backends):
  1. Cache hit: <QUANT_STORE>/<model_slug>__<backend>__<quant_type>/ exists
     → load from cache.
  2. model_id is already a pre-quantized local dir → load directly.
  3. Float model → quantize, save to cache, return.
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Union

import torch

from .config import QUANT_CONFIGS

# One sub-directory per (model_id, backend, quant_type) triple lives here.
_QUANT_STORE = Path(__file__).resolve().parents[2] / "quant_models"

# gptqmodel-backend configs, keyed by the same quant_type aliases as
# quants/config.py so both backends share one nomenclature.
_GPTQMODEL_CONFIGS = {
    "gptq-4bit": {"bits": 4, "group_size": 128},
    "gptq-8bit": {"bits": 8, "group_size": 128},
}


def _model_slug(model_id: str, quant_type: str, backend: str) -> str:
    """Build a safe directory name from model_id + backend + quant_type."""
    slug = re.sub(r"[/\\:]+", "--", model_id.strip("/\\"))
    slug = re.sub(r"[\s.]+", "-", slug).strip("-")
    return f"{slug}__{backend}__{quant_type}"


def _cache_path(model_id: str, quant_type: str, backend: str, store: Path = _QUANT_STORE) -> Path:
    return store / _model_slug(model_id, quant_type, backend)


def _fetch_calibration_data(n_samples: int = 512) -> List[str]:
    """Download a small slice of C4 for calibration (mirrors test.py)."""
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
    backend: str = "llm_compressor",
    num_calibration_samples: int = 512,
    max_seq_length: int = 2048,
) -> torch.nn.Module:
    """
    Return an HF torch.nn.Module ready for inference on *device*.

    backend: "llm_compressor" (default) or "gptq". Every model adapter in
    src/models/ calls this without passing backend=, so they all use
    llm-compressor unless a caller explicitly opts into the gptqmodel path.
    """

    if quant_type not in QUANT_CONFIGS:
        raise ValueError(
            f"Unknown quant_type={quant_type!r}. "
            f"Available: {sorted(QUANT_CONFIGS)}"
        )

    if quant_type == "none":
        raise ValueError("quantize_model() must not be called for quant_type='none'.")

    spec = QUANT_CONFIGS[quant_type]
    if spec is None:
        raise NotImplementedError(
            f"quant_type={quant_type!r} is registered but has no config. "
            "Add one in quants/config.py."
        )

    if backend == "llm_compressor":
        return _quantize_llm_compressor(
            model_id, quant_type, spec, calibration,
            num_calibration_samples, max_seq_length, device, store_dir,
        )
    if backend == "gptq":
        return _quantize_gptqmodel(model_id, quant_type, calibration, batch_size, device, store_dir)

    raise ValueError(f"Unknown backend={backend!r}. Available: 'llm_compressor', 'gptq'.")


# ---------------------------------------------------------------- llm_compressor

def _is_llm_compressor_quantized(path: Union[str, Path]) -> bool:
    """True when *path* is a local dir already holding a compressed-tensors checkpoint."""
    config_path = Path(path) / "config.json"
    if not config_path.exists():
        return False
    try:
        cfg = json.loads(config_path.read_text())
    except (OSError, ValueError):
        return False
    return "quantization_config" in cfg


def _build_calibration_dataset(calibration, tokenizer, max_seq_length: int):
    """Tokenize raw calibration text into the HF Dataset oneshot() expects."""
    from datasets import Dataset

    texts = [ex.get("text", "") if isinstance(ex, dict) else ex for ex in calibration]
    ds = Dataset.from_dict({"text": texts})

    def _tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=True,
        )

    return ds.map(_tokenize, remove_columns=["text"])


def _build_recipe(spec):
    """Translate a quants/config.py QuantSpec into an llm-compressor recipe."""
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

    algorithm, scheme = spec["algorithm"], spec["scheme"]
    ignore = ["lm_head"]

    if algorithm == "rtn":
        return QuantizationModifier(targets="Linear", scheme=scheme, ignore=ignore)
    if algorithm == "gptq":
        return GPTQModifier(targets="Linear", scheme=scheme, ignore=ignore)
    if algorithm == "awq":
        # AWQModifier smooths using its built-in Llama mapping (every model
        # this repo quantizes is a plain Llama-family backbone); the actual
        # weight quantization is done by the QuantizationModifier that follows.
        return [AWQModifier(), QuantizationModifier(targets=["Linear"], scheme=scheme, ignore=ignore)]
    if algorithm == "sq":
        return [
            SmoothQuantModifier(smoothing_strength=0.8),
            GPTQModifier(targets="Linear", scheme=scheme, ignore=ignore),
        ]
    raise ValueError(f"Unknown algorithm={algorithm!r} in quant spec.")


def _quantize_llm_compressor(
    model_id: str,
    quant_type: str,
    spec,
    calibration,
    num_calibration_samples: int,
    max_seq_length: int,
    device: Optional[str],
    store_dir: Optional[Path],
) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmcompressor import oneshot

    store = Path(store_dir) if store_dir else _QUANT_STORE
    cache = _cache_path(model_id, quant_type, "llmcompressor", store)
    load_device_map = {"": device} if device else {"": "cpu"}

    if _is_llm_compressor_quantized(cache):
        print(f"Loading cached llm-compressor model: {cache}")
        return AutoModelForCausalLM.from_pretrained(str(cache), device_map=load_device_map)
    if _is_llm_compressor_quantized(model_id):
        print(f"Loading pre-quantized model: {model_id}")
        return AutoModelForCausalLM.from_pretrained(model_id, device_map=load_device_map)

    print(f"Quantizing {model_id!r} with llm-compressor ({quant_type}) → {cache}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    if calibration is None:
        calibration = _fetch_calibration_data(num_calibration_samples)
    dataset = _build_calibration_dataset(calibration, tokenizer, max_seq_length)

    oneshot(
        model=model,
        dataset=dataset,
        recipe=_build_recipe(spec),
        max_seq_length=max_seq_length,
        num_calibration_samples=len(dataset),
    )

    cache.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cache), save_compressed=True)
    tokenizer.save_pretrained(str(cache))
    print(f"Saved quantized model: {cache}")

    return AutoModelForCausalLM.from_pretrained(str(cache), device_map=load_device_map)


# ---------------------------------------------------------------- gptq (alternate)

def _quantize_gptqmodel(
    model_id: str,
    quant_type: str,
    calibration,
    batch_size: int,
    device: Optional[str],
    store_dir: Optional[Path],
) -> torch.nn.Module:
    """Original gptqmodel-based backend. Internal alternate, not wired to the CLI."""
    from gptqmodel import GPTQConfig, GPTQModel
    from gptqmodel.utils.backend import BACKEND

    if quant_type not in _GPTQMODEL_CONFIGS:
        raise NotImplementedError(
            f"backend='gptq' only supports {sorted(_GPTQMODEL_CONFIGS)}, got {quant_type!r}."
        )

    store = Path(store_dir) if store_dir else _QUANT_STORE
    cache = _cache_path(model_id, quant_type, "gptqmodel", store)
    load_device_map = {"": device} if device else {"": "cpu"}

    def _is_pre_quantized(path: Union[str, Path]) -> bool:
        return (Path(path) / "quantize_config.json").exists()

    if _is_pre_quantized(cache):
        print(f"Loading cached GPTQ model: {cache}")
        return GPTQModel.load(str(cache), backend=BACKEND.GPTQ_TORCH, device_map=load_device_map).model
    if _is_pre_quantized(model_id):
        print(f"Loading pre-quantized model: {model_id}")
        return GPTQModel.load(model_id, backend=BACKEND.GPTQ_TORCH, device_map=load_device_map).model

    if calibration is None:
        calibration = _fetch_calibration_data()

    print(f"Quantizing {model_id!r} with gptqmodel ({quant_type}) → {cache}")
    quant_cfg = GPTQConfig(**_GPTQMODEL_CONFIGS[quant_type])
    gptq_model = GPTQModel.load(model_id, quantize_config=quant_cfg)
    gptq_model.quantize(calibration, batch_size=batch_size)

    cache.mkdir(parents=True, exist_ok=True)
    gptq_model.save(str(cache))
    print(f"Saved quantized model: {cache}")

    return GPTQModel.load(str(cache), backend=BACKEND.GPTQ_TORCH, device_map=load_device_map).model

