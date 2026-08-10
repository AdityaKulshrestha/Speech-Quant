"""
Quant-flavour registry.

Maps a --quant-type name to a zero-arg callable that builds a torchao
config. `None` means the flavour is recognized (so it shows up in
--help/choices) but has no config wired up yet.
"""

from typing import Callable, Optional

QUANT_CONFIGS: dict[str, Optional[Callable]] = {
    "none": None,
}

try:
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Int4WeightOnlyConfig,
        Int8WeightOnlyConfig,
        PerRow,
    )

    QUANT_CONFIGS["int4"] = lambda: Int4WeightOnlyConfig(
        group_size=32,
        int4_packing_format="plain_int32",
    )
    QUANT_CONFIGS["int8"] = lambda: Int8WeightOnlyConfig()
    QUANT_CONFIGS["fp8"] = lambda: Float8DynamicActivationFloat8WeightConfig(
        granularity=PerRow()
    )

except ImportError:
    # torchao not installed: keep the names selectable, but unusable
    # until quantizer.py raises a clear error.
    QUANT_CONFIGS["int4"] = None
    QUANT_CONFIGS["int8"] = None
    QUANT_CONFIGS["fp8"] = None
