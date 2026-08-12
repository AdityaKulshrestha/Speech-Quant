"""
Quant-flavour registry.

Maps a --quant-type name to a zero-arg callable that builds a
gptqmodel GPTQConfig. `None` means the flavour is recognized (so it
shows up in --help/choices) but has no config wired up yet.
"""

from typing import Callable, Optional

QUANT_CONFIGS: dict[str, Optional[Callable]] = {
    "none": None,
}

try:
    from gptqmodel import GPTQConfig

    QUANT_CONFIGS["gptq-4bit"] = lambda: GPTQConfig(bits=4, group_size=128)
    QUANT_CONFIGS["gptq-8bit"] = lambda: GPTQConfig(bits=8, group_size=128)

except ImportError:
    # gptqmodel not installed: keep the names selectable but unusable
    # until quantizer.py raises a clear error.
    QUANT_CONFIGS["gptq-4bit"] = None
    QUANT_CONFIGS["gptq-8bit"] = None
