"""
NeuTTS model adapter for the Speech-Quant BaseTTS interface.

Torch backbone only: no GGUF, no ONNX, no streaming, no watermarking.
Targets BPE-format models (neutts-nano and family).
Speaker references are loaded from the bundled NeuTTS2E sample directory.
"""

from pathlib import Path
from typing import Any, Optional

import torch
from neucodec import NeuCodec
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

from .base import BaseTTS, GenerationOutput

# Bundled speaker references shipped with the neutts package.
import neutts as _neutts_pkg
_SAMPLE_DIR = Path(_neutts_pkg.__file__).parent / "samples"


class _SpeechLogitCapture(LogitsProcessor):
    """Records audio-subspace softmax probs at every generation step."""

    def __init__(self, speech_start: int, speech_end: int) -> None:
        self.speech_start = speech_start
        self.speech_end = speech_end
        self.step_probs: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        s = scores[0, self.speech_start : self.speech_end].float()
        self.step_probs.append(torch.softmax(s, dim=-1).cpu().half())
        return scores


class NeuTTSModel(BaseTTS):
    """
    NeuTTS (torch backbone) + NeuCodec wrapped for the BaseTTS interface.

    AUDIO_TOKEN_START = 0: codec codes are 0-based (no offset, unlike Orpheus).
    Speaker is selected via the `voice` kwarg (maps to bundled NeuTTS2E speakers).
    """

    AUDIO_TOKEN_START = 0
    SAMPLE_RATE = 24_000
    SPEAKERS = ("emily", "paul", "sophie", "steven")

    def __init__(
        self,
        model_name: str = "neuphonic/neutts-nano",
        codec_name: str = "neuphonic/neucodec",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        quant_type: str = "none",
    ) -> None:
        self.model_name = model_name
        self.codec_name = codec_name
        self.device = device
        self.dtype = dtype
        self.quant_type = quant_type

        self.model = None
        self.tokenizer = None
        self.codec = None

        self._speaker_cache: dict[str, tuple[torch.Tensor, str]] = {}
        self._speech_start: int = 0
        self._speech_end: int = 0

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        print(f"Loading NeuTTS backbone: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.quant_type.startswith("gptq"):
            from quants.quantizer import quantize_model

            print(f"GPTQ loading ({self.quant_type}): {self.model_name}")
            self.model = quantize_model(
                self.model_name, self.quant_type, device=self.device
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=self.dtype
            ).to(self.device)

        self.model.eval()

        print(f"Loading NeuCodec: {self.codec_name}")
        self.codec = NeuCodec.from_pretrained(self.codec_name).eval().to(self.device)

        # Resolve the speech token range in the vocabulary.
        self._speech_start = self.tokenizer.convert_tokens_to_ids("<|speech_0|>")
        for size in (8192, 4096, 2048, 1024):
            last = self.tokenizer.convert_tokens_to_ids(f"<|speech_{size - 1}|>")
            if last != self.tokenizer.unk_token_id:
                self._speech_end = self._speech_start + size
                break
        else:
            self._speech_end = self._speech_start + 4096

        print("NeuTTS loaded.")

    def _speaker(self, name: str) -> tuple[torch.Tensor, str]:
        """Load bundled speaker codes + transcript (cached)."""
        if name not in self.SPEAKERS:
            raise ValueError(f"Unknown speaker '{name}'. Available: {self.SPEAKERS}")
        if name not in self._speaker_cache:
            codes = torch.load(_SAMPLE_DIR / f"{name}.pt", weights_only=True)
            text = (_SAMPLE_DIR / f"{name}.txt").read_text().strip()
            self._speaker_cache[name] = (codes, text)
        return self._speaker_cache[name]

    # ---------------------------------------------------- BaseTTS interface

    def prepare_input(
        self,
        text: str,
        voice: str = "emily",
        **kwargs,
    ) -> dict[str, Any]:
        codes, ref_text = self._speaker(voice)

        ts = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
        te = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")
        sg = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")

        text_ids = self.tokenizer.encode(
            f"{ref_text} {text}".strip(), add_special_tokens=False
        )
        codes_str = "".join(f"<|speech_{i}|>" for i in codes.tolist())
        code_ids = self.tokenizer.encode(codes_str, add_special_tokens=False)

        prompt = [ts] + text_ids + [te, sg] + code_ids
        ids = torch.tensor(prompt, dtype=torch.long).unsqueeze(0).to(self.device)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    @torch.inference_mode()
    def generate_tokens(
        self,
        inputs: dict[str, Any],
        max_new_tokens: int = 1200,
        temperature: float = 1.0,
        top_k: int = 50,
        **kwargs,
    ) -> GenerationOutput:
        eos_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        capture = _SpeechLogitCapture(self._speech_start, self._speech_end)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_id,
            logits_processor=LogitsProcessorList([capture]),
        )

        input_len = inputs["input_ids"].shape[-1]
        audio_tokens, audio_logits = self._extract_audio(
            generated_ids, capture.step_probs, input_len
        )

        return GenerationOutput(
            generated_ids=generated_ids,
            audio_tokens=audio_tokens,
            audio_logits=audio_logits,
            metadata={
                "input_length": input_len,
                "generated_length": generated_ids.shape[-1],
                "num_audio_tokens": audio_tokens.numel(),
            },
        )

    def _extract_audio(
        self,
        generated_ids: torch.Tensor,
        step_probs: list[torch.Tensor],
        input_len: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        gen = generated_ids[0, input_len:]
        speech_mask = (gen >= self._speech_start) & (gen < self._speech_end)
        indices = speech_mask.nonzero(as_tuple=True)[0].tolist()

        # Store raw codec codes (0-based), not vocab IDs.
        audio_tokens = (gen[speech_mask] - self._speech_start).cpu()
        probs = [step_probs[i] for i in indices if i < len(step_probs)]
        audio_logits = torch.stack(probs) if probs else None
        return audio_tokens, audio_logits

    def decode_audio(
        self, audio_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        codes = audio_tokens.long().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            recon = self.codec.decode_code(codes).cpu()
        return recon[0, 0, :], self.SAMPLE_RATE

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.codec = None
        self._speaker_cache.clear()
