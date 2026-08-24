"""
OuteTTS model adapter for the Speech-Quant BaseTTS interface.

OuteTTS 1.0 (OuteAI/Llama-OuteTTS-1.0-1B) is a plain Llama-3.2-1B causal LM
(no custom package needed for the LM itself) that predicts two interleaved
DAC codebook streams as ordinary vocabulary tokens: `<|c1_0..1024|>` then
`<|c2_0..1024|>`, verified (via the actual tokenizer) to occupy one
contiguous 2050-token vocab block with c1 immediately followed by c2 — the
same "single shared vocab slice, N tokens per frame" layout Orpheus/SNAC
uses, so this model gets full teacher-forced distribution comparison
support (unlike QwenTTSModel).

This intentionally does NOT depend on the `outetts` PyPI package: that
package hard-pins transformers==4.52.3 plus a large stack of unrelated
dependencies (llama-cpp-python, openai-whisper, pygame, mecab, uroman) we
don't need — we only use its documented special-token vocabulary (see
https://github.com/edwko/OuteTTS/blob/main/outetts/version/v3/tokens.py)
and drive the transformers backend + DAC codec directly, the same way this
repo's other models do. Only the lightweight `descript-audio-codec` (`dac`)
package is required for the codec (no torch version pin), installable into
the main `.venv`.

Zero-shot generation only (no speaker-reference voice cloning): the model
card confirms word alignment/timing/pitch/energy features are inferred by
the model itself at generation time and are only required in the prompt
for voice-cloning/training, not for plain text-to-speech.
"""

from typing import Any, Optional

import torch
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

from .base import BaseTTS, GenerationOutput


class AudioLogitCapture(LogitsProcessor):
    """Captures audio-subspace softmax probs at every generation step."""

    def __init__(self, audio_start: int, audio_end: int):
        self.audio_start = audio_start
        self.audio_end = audio_end
        self.step_probs: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        audio_slice = scores[0, self.audio_start:self.audio_end].half()
        self.step_probs.append(audio_slice)  # stay on device; bulk .cpu() after generation
        return scores


class _DacCodec:
    """Minimal decode-only wrapper around `ibm-research/DAC.speech.v1.0`.

    Reimplements just the `decode()` path of outetts's `DacInterface` (see
    module docstring) without the `outetts` package's unrelated dependencies.
    """

    SAMPLE_RATE = 24_000
    REPO_ID = "ibm-research/DAC.speech.v1.0"
    WEIGHTS_FILE = "weights_24khz_1.5kbps_v1.0.pth"

    def __init__(self, device: str) -> None:
        import dac

        model_path = hf_hub_download(repo_id=self.REPO_ID, filename=self.WEIGHTS_FILE)
        self.model = dac.DAC.load(model_path).to(device).eval()

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: [1, num_codebooks, T] long -> waveform [1, 1, T']."""
        z = self.model.quantizer.from_codes(codes)[0]
        return self.model.decode(z)


class OuteTTSModel(BaseTTS):
    """
    OuteTTS 1.0 (Llama backbone) + DAC codec wrapped for the BaseTTS interface.

    AUDIO_TOKEN_START = 0: audio_tokens are de-offset relative to the shared
    c1+c2 vocab block (VOCAB_AUDIO_TOKEN_START holds the real vocab offset),
    same convention as NeuTTS/Llasa. TOKENS_PER_FRAME=2 (c1, c2 interleaved
    per DAC frame), CODEBOOK_SIZE=1025 per codebook.
    """

    AUDIO_TOKEN_START = 0
    CODEBOOK_SIZE = 1025  # DAC.speech.v1.0: 1025 entries per codebook (0..1024)
    TOKENS_PER_FRAME = 2  # c1, c2 interleaved
    SAMPLE_RATE = 24_000

    def __init__(
        self,
        model_name: str = "OuteAI/Llama-OuteTTS-1.0-1B",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        quant_type: str = "none",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.quant_type = quant_type

        self.model = None
        self.tokenizer = None
        self.codec = None

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        print(f"Loading OuteTTS backbone: {self.model_name}")
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

        print("Loading DAC codec: ibm-research/DAC.speech.v1.0")
        self.codec = _DacCodec(self.device)

        # Resolve special tokens dynamically (mirrors outetts's own
        # get_audio_token_map()) instead of hardcoding ids.
        conv = self.tokenizer.convert_tokens_to_ids
        self._bos = conv("<|im_start|>")
        self._eos = conv("<|im_end|>")
        self._text_start = conv("<|text_start|>")
        self._text_end = conv("<|text_end|>")
        self._audio_start_tok = conv("<|audio_start|>")
        self._audio_end_tok = conv("<|audio_end|>")

        c1_start = conv("<|c1_0|>")
        c1_end = conv(f"<|c1_{self.CODEBOOK_SIZE - 1}|>") + 1
        c2_start = conv("<|c2_0|>")
        c2_end = conv(f"<|c2_{self.CODEBOOK_SIZE - 1}|>") + 1
        if (
            c1_end - c1_start != self.CODEBOOK_SIZE
            or c2_end - c2_start != self.CODEBOOK_SIZE
            or c2_start != c1_end
        ):
            raise ValueError(
                "Unexpected OuteTTS c1/c2 vocab layout: expected two contiguous "
                f"{self.CODEBOOK_SIZE}-entry blocks with c2 immediately after c1, "
                f"got c1=[{c1_start},{c1_end}) c2=[{c2_start},{c2_end})."
            )
        self.VOCAB_AUDIO_TOKEN_START = c1_start
        self._audio_start = c1_start
        self._audio_end = c2_end
        # audio_end doubles as the teacher-forcing/generic-pipeline EOS marker
        # (training data always emits it right before the real <|im_end|>).
        self.END_OF_SPEECH = self._audio_end_tok

        print("OuteTTS loaded.")

    # ---------------------------------------------------- BaseTTS interface

    def prepare_input(
        self,
        text: str,
        **kwargs,
    ) -> dict[str, Any]:
        prompt = f"<|im_start|>\n<|text_start|>{text}<|text_end|>\n<|audio_start|>\n"
        input_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        input_ids = input_ids.to(self.device)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    @torch.inference_mode()
    def generate_tokens(
        self,
        inputs: dict[str, Any],
        max_new_tokens: int = 1200,
        temperature: float = 0.4,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        **kwargs,
    ) -> GenerationOutput:
        capture = AudioLogitCapture(self._audio_start, self._audio_end)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_return_sequences=1,
            eos_token_id=[self._audio_end_tok, self._eos],
            logits_processor=LogitsProcessorList([capture]),
        )

        audio_tokens = self.extract_audio_tokens(generated_ids)
        audio_logits = self._extract_audio_logits(
            capture.step_probs, generated_ids, inputs["input_ids"].shape[-1]
        )

        return GenerationOutput(
            generated_ids=generated_ids,
            audio_tokens=audio_tokens,
            audio_logits=audio_logits,
            metadata={
                "input_length": inputs["input_ids"].shape[-1],
                "generated_length": generated_ids.shape[-1],
                "num_audio_tokens": audio_tokens.numel(),
            },
        )

    def _extract_audio_logits(
        self,
        step_probs: list[torch.Tensor],
        generated_ids: torch.Tensor,
        input_len: int,
    ) -> Optional[torch.Tensor]:
        gen_tokens = generated_ids[0, input_len:]

        audio_probs: list[torch.Tensor] = []
        for i, tok in enumerate(gen_tokens.tolist()):
            if tok == self._audio_end_tok or tok == self._eos:
                break
            if self._audio_start <= tok < self._audio_end and i < len(step_probs):
                audio_probs.append(step_probs[i])

        n_complete = (len(audio_probs) // self.TOKENS_PER_FRAME) * self.TOKENS_PER_FRAME
        if n_complete == 0:
            return None
        stacked = torch.stack(audio_probs[:n_complete]).float()
        return torch.softmax(stacked, dim=-1).cpu().half()

    def extract_audio_tokens(self, generated_ids: torch.Tensor) -> torch.Tensor:
        """Flat, de-offset [c1_0, c2_0, c1_1, c2_1, ...] stream (interspersed
        word/text/time/feature tokens are dropped by the range filter)."""
        row = generated_ids[0]
        mask = (row >= self._audio_start) & (row < self._audio_end)
        tokens = (row[mask] - self._audio_start).long()
        n_complete = (tokens.numel() // self.TOKENS_PER_FRAME) * self.TOKENS_PER_FRAME
        return tokens[:n_complete]

    def decode_audio(
        self, audio_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        if audio_tokens.numel() == 0:
            raise ValueError("No audio tokens were generated.")

        tokens = audio_tokens.reshape(-1, self.TOKENS_PER_FRAME)  # [T, 2]
        c1 = tokens[:, 0]
        c2 = tokens[:, 1] - self.CODEBOOK_SIZE
        codes = torch.stack([c1, c2], dim=0).unsqueeze(0).long().to(self.device)  # [1, 2, T]

        with torch.no_grad():
            audio = self.codec.decode(codes).cpu()

        return audio[0, 0, :], self.SAMPLE_RATE

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.codec = None
