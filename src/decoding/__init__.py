"""Decoding and alignment utilities for sequence-level codec analysis."""

from .alignment import (
    CodecSpec,
    CodebookDivergenceAnalyzer,
    detect_codec_family,
    get_codec_spec,
)
from .distribution import (
    compare_distributions,
    extract_teacher_forced_logits,
    js_divergence_sequence,
    topk_overlap_sequence,
)

__all__ = [
    "CodecSpec",
    "CodebookDivergenceAnalyzer",
    "detect_codec_family",
    "get_codec_spec",
    "compare_distributions",
    "extract_teacher_forced_logits",
    "js_divergence_sequence",
    "topk_overlap_sequence",
]
