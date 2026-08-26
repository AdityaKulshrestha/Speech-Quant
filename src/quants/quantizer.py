"""
Quantizes or loads a cached quantized model for TTS model backbones.

Uses llm_compressor for all quantization methods (RTN, GPTQ, AWQ, SmoothQuant),
applying them via llmcompressor.oneshot() and saving/loading compressed-tensors
checkpoints.

Resolution order:
  1. Cache hit: <QUANT_STORE>/<model_slug>__<quant_type>/ exists
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

# One sub-directory per (model_id, quant_type) pair lives here.
_QUANT_STORE = Path(__file__).resolve().parents[2] / "quant_models"


def _model_slug(model_id: str, quant_type: str) -> str:
    """Build a safe directory name from model_id + quant_type."""
    slug = re.sub(r"[/\\:]+", "--", model_id.strip("/\\"))
    slug = re.sub(r"[\s.]+", "-", slug).strip("-")
    return f"{slug}__{quant_type}"


def _cache_path(model_id: str, quant_type: str, store: Path = _QUANT_STORE) -> Path:
    return store / _model_slug(model_id, quant_type)


def compute_dir_size_mb(path: Path) -> float:
    """Compute size in MB of weight files in directory."""
    weight_files = list(path.glob("*.safetensors")) or list(path.glob("*.bin"))
    if not weight_files:
        weight_files = [p for p in path.glob("*") if p.is_file() and p.name != "quant_stats.json"]
    return sum(f.stat().st_size for f in weight_files) / (1024 ** 2)


def compute_model_size_mb(model: torch.nn.Module) -> float:
    """Compute uncompressed model size in MB (all parameters and buffers)."""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.element_size() * param.numel()
    for buf in model.buffers():
        total_bytes += buf.element_size() * buf.numel()
    return total_bytes / (1024 ** 2)


def _fetch_calibration_data(n_samples: Optional[int] = None) -> List[str]:
    """Fetch TTS-domain calibration data with comprehensive phonetic coverage.

    Uses Harvard Sentences (nineninesix/harvard-sentences-tts-benchmark),
    which covers all English phonemes systematically. This ensures quantization
    calibration matches the TTS inference domain.

    Args:
        n_samples: Number of calibration samples needed. If None, uses all dataset samples.

    Returns:
        List of text strings for calibration
    """
    from datasets import load_dataset

    # Load Harvard sentences: 720 phonetically-balanced sentences for TTS
    ds = load_dataset("nineninesix/harvard-sentences-tts-benchmark", split="train")
    texts = list(ds["text"])

    if n_samples is not None:
        if len(texts) < n_samples:
            repetitions = (n_samples // len(texts)) + 1
            texts = (texts * repetitions)[:n_samples]
        else:
            texts = texts[:n_samples]

    print(f"Calibration: {len(texts)} Harvard sentences (phonetically balanced, TTS-domain)")
    return texts


def quantize_model(
    model_id: str,
    quant_type: str,
    calibration: Optional[Union[List[str], List[dict]]] = None,
    batch_size: int = 1,
    device: Optional[str] = None,
    store_dir: Optional[Path] = None,
    num_calibration_samples: Optional[int] = None,
    max_seq_length: int = 2048,
) -> torch.nn.Module:
    """
    Return an HF torch.nn.Module ready for inference on *device*.

    Uses llm_compressor for all quantization methods (RTN, GPTQ, AWQ, SmoothQuant).
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

    return _quantize_llm_compressor(
        model_id, quant_type, spec, calibration,
        num_calibration_samples, max_seq_length, device, store_dir,
    )


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
        # AWQ: auto-detects architecture, smooths activations, then quantizes weights
        return [
            AWQModifier(),
            QuantizationModifier(targets=["Linear"], scheme=scheme, ignore=ignore)
        ]

    if algorithm == "sq":
        # SmoothQuant: smooth activations (strength=0.8), then quantize weights+activations
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
    num_calibration_samples: Optional[int],
    max_seq_length: int,
    device: Optional[str],
    store_dir: Optional[Path],
) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmcompressor import oneshot

    store = Path(store_dir) if store_dir else _QUANT_STORE
    cache = _cache_path(model_id, quant_type, store)
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

    # Measure size before quantization
    fp_size_mb = compute_model_size_mb(model)

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

    # Measure actual compressed size on disk (after save_compressed=True packs 4-bit/8-bit weights)
    quant_size_mb = compute_dir_size_mb(cache)
    compression_ratio = fp_size_mb / quant_size_mb if quant_size_mb > 0 else 1.0

    # Save size metadata
    (cache / "quant_stats.json").write_text(json.dumps({
        "fp_size_mb": round(fp_size_mb, 2),
        "quant_size_mb": round(quant_size_mb, 2),
        "compression_ratio": round(compression_ratio, 2),
        "quant_type": quant_type,
    }, indent=2))

    print(f"Saved quantized model: {cache}")
    print(f"Model size: {fp_size_mb:.1f} MB → {quant_size_mb:.1f} MB (compression: {compression_ratio:.2f}×)")

    return AutoModelForCausalLM.from_pretrained(str(cache), device_map=load_device_map)

