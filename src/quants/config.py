"""
Quant-flavour registry (see quants/quantizer.py for how these are applied).

Alias format: "{algorithm}-{bits}bit" where algorithm is one of
rtn / gptq / awq / sq (SmoothQuant) and bits is 4 or 8 — this exact string is
what --quant-type takes (comma-separated for multiple), what quant_models/
cache directories embed, and what ends up in the analysis_report.xlsx
(model, quant_type) columns. Keep it identical everywhere.

All quantization methods use llm_compressor (llmcompressor.oneshot(), see
quants/quantizer.py).

rtn/gptq/awq are weight-only (activations stay fp16/bf16: W4A16/W8A16).
sq (SmoothQuant) additionally quantizes activations to int8 (W4A8/W8A8),
since enabling accurate low-bit activation quantization is its whole point.
"""

from typing import Optional, TypedDict


class QuantSpec(TypedDict):
    algorithm: str  # "rtn" | "gptq" | "awq" | "sq"
    bits: int  # 4 | 8
    scheme: str  # compressed_tensors preset scheme name


QUANT_CONFIGS: dict[str, Optional[QuantSpec]] = {
    "none": None,
    "rtn-4bit": {"algorithm": "rtn", "bits": 4, "scheme": "W4A16"},
    "rtn-8bit": {"algorithm": "rtn", "bits": 8, "scheme": "W8A16"},
    "gptq-4bit": {"algorithm": "gptq", "bits": 4, "scheme": "W4A16"},
    "gptq-8bit": {"algorithm": "gptq", "bits": 8, "scheme": "W8A16"},
    "awq-4bit": {"algorithm": "awq", "bits": 4, "scheme": "W4A16_ASYM"},
    "awq-8bit": {"algorithm": "awq", "bits": 8, "scheme": "W8A16"},
    "sq-4bit": {"algorithm": "sq", "bits": 4, "scheme": "W4A8"},
    "sq-8bit": {"algorithm": "sq", "bits": 8, "scheme": "W8A8"},
}
