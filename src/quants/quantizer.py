"""
Wrapper that applies a quant flavour from quants/config.py to a loaded
torch model, in place, using torchao's quantize_.
"""

import torch

from .config import QUANT_CONFIGS


def quantize_model(model: torch.nn.Module, quant_type: str) -> torch.nn.Module:
    """Quantize `model` in place according to `quant_type`. No-op for "none"."""

    if quant_type not in QUANT_CONFIGS:
        raise ValueError(
            f"Unknown quant_type={quant_type!r}. "
            f"Available: {sorted(QUANT_CONFIGS)}"
        )

    if quant_type == "none":
        return model

    config_fn = QUANT_CONFIGS[quant_type]

    if config_fn is None:
        raise NotImplementedError(
            f"quant_type={quant_type!r} is registered but has no config yet. "
            "Add one in quants/config.py."
        )

    from torchao.quantization import quantize_

    quantize_(model, config_fn())

    return model
