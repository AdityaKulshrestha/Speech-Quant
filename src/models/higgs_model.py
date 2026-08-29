"""
Higgs Audio v3 model adapter for the Speech-Quant BaseTTS interface.

Wraps multimodalart/higgs-audio-v3-tts-4b-transformers (a `trust_remote_code`
port of bosonai/higgs-audio-v3-tts-4b to plain transformers): a Qwen3-4B
backbone plus a fused multi-codebook audio embedding/head (8 codebooks,
vocab 1026 incl. BOC/EOC) driven by Higgs' delay pattern, decoded to a 24kHz
waveform via the `higgs_audio_v2_tokenizer` codec.

The model's public `generate_speech()` method does prompt-building,
autoregressive sampling and codec decoding all in one call, with no exposed
seam between "generate tokens" and "decode audio" (see
HiggsMultimodalQwen3ForConditionalGeneration.generate_speech in the model's
remote code). To fit this repo's generate_tokens/decode_audio split (needed
for FDP/D(t)/KL token-level comparisons), this adapter reimplements
`generate_speech`'s sampling loop using the same private building blocks
(`_build_prompt_ids`, `_prefill_embeds`, `_decode_codes`, and the delay-pattern
helpers from the model's own dynamically-loaded module) instead of calling it
directly — the same "reimplement a decode-only seam" approach OuteTTSModel
uses for its DAC codec wrapper.

Requires transformers>=5.5 (see model card); no separate venv/torch pin
needed, unlike the old QwenTTSModel this replaces.
"""

from importlib import import_module
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseTTS, GenerationOutput


class HiggsAudioV3Model(BaseTTS):
    """
    Higgs Audio v3 (Qwen3-4B backbone + fused multi-codebook head/embedding).

    AUDIO_TOKEN_START = 0: codes are 0-based (no vocabulary offset, the audio
    head/embedding are separate fused modules, not part of the text vocab).
    TOKENS_PER_FRAME = num_codebooks (8), CODEBOOK_SIZE = per-codebook vocab
    (1026, incl. BOC/EOC specials) — both resolved from config after load().

    Deliberately does not define VOCAB_AUDIO_TOKEN_START/END_OF_SPEECH: the
    fused audio head has no plain forward(input_ids)->logits over a shared
    LM vocab to teacher-force against, so teacher_forced_distribution_compare
    in evaluate.py skips this model, the same way it already skips
    QwenTTSModel.
    """

    AUDIO_TOKEN_START = 0
    SAMPLE_RATE = 24_000  # updated from model config after load()

    def __init__(
        self,
        model_name: str = "multimodalart/higgs-audio-v3-tts-4b-transformers",
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

        # Delay-pattern helpers from the model's own remote-code module.
        self._apply_delay_pattern = None
        self._reverse_delay_pattern = None
        self._SamplerState = None
        self._sampler_step = None

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        if self.quant_type != "none":
            raise NotImplementedError(
                f"Quantization ({self.quant_type}) for HiggsAudioV3Model is not yet "
                "supported. The Qwen3 backbone plus fused multi-codebook audio "
                "head/embedding needs a dedicated quantization recipe."
            )

        print(f"Loading Higgs Audio v3: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True, dtype=self.dtype
        ).to(self.device).eval()

        self.SAMPLE_RATE = self.model.config.sample_rate
        self.TOKENS_PER_FRAME = self.model.num_codebooks
        self.CODEBOOK_SIZE = self.model.codebook_vocab_size

        module = import_module(type(self.model).__module__)
        self._apply_delay_pattern = module.apply_delay_pattern
        self._reverse_delay_pattern = module.reverse_delay_pattern
        self._SamplerState = module._SamplerState
        self._sampler_step = module._sampler_step

        print("Higgs Audio v3 loaded.")

    # ---------------------------------------------------- BaseTTS interface

    def prepare_input(self, text: str, **kwargs) -> dict[str, Any]:
        # Zero-shot only (no reference_audio/reference_text voice cloning);
        # --voice is unused here, same as OuteTTS/Llasa/Qwen.
        prompt_ids = self.model._build_prompt_ids(
            self.tokenizer, text, num_ref_tokens=0, reference_text=None
        )
        inputs_embeds = self.model._prefill_embeds(prompt_ids, None)
        return {"inputs_embeds": inputs_embeds}

    @torch.inference_mode()
    def generate_tokens(
        self,
        inputs: dict[str, Any],
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float | None = None,
        top_k: int | None = None,
        **kwargs,
    ) -> GenerationOutput:
        N = self.TOKENS_PER_FRAME
        inputs_embeds = inputs["inputs_embeds"]

        out = self.model.model(inputs_embeds=inputs_embeds, use_cache=True)
        past = out.past_key_values
        hidden_last = out.last_hidden_state[:, -1, :]
        position = inputs_embeds.shape[1]

        state = self._SamplerState(num_codebooks=N)
        rows: list[torch.Tensor] = []
        step_probs: list[torch.Tensor] = []

        for _ in range(max_new_tokens):
            logits_NV = self.model.audio_head(hidden_last).to(torch.float32)[0]  # [N, V]
            codes_N = self._sampler_step(
                logits_NV, state, temperature=temperature, top_p=top_p, top_k=top_k
            )
            if state.generation_done:
                break
            rows.append(codes_N.cpu())
            step_probs.append(logits_NV.softmax(dim=-1).half().cpu())  # [N, V]

            step_embed = self.model.audio_embedding(codes_N.unsqueeze(0)).unsqueeze(1)
            cache_pos = torch.tensor([position], device=self.device)
            out = self.model.model(
                inputs_embeds=step_embed.to(inputs_embeds.dtype),
                past_key_values=past,
                use_cache=True,
                cache_position=cache_pos,
            )
            past = out.past_key_values
            hidden_last = out.last_hidden_state[:, -1, :]
            position += 1

        if len(rows) < N:
            raise RuntimeError(
                "Higgs Audio v3 generated no complete codec frame "
                f"(only {len(rows)} rows, need >= {N})."
            )

        delayed_LN = torch.stack(rows, dim=0)              # [L, N]
        audio_tokens = delayed_LN.reshape(-1).long()        # flat, slot t%N == codebook
        audio_logits = torch.cat(step_probs, dim=0)          # [L*N, V], same flattening

        return GenerationOutput(
            generated_ids=audio_tokens,
            audio_tokens=audio_tokens,
            audio_logits=audio_logits,
            metadata={
                "input_length": inputs_embeds.shape[1],
                "num_audio_tokens": audio_tokens.numel(),
                "delayed_frames": len(rows),
            },
        )

    def decode_audio(self, audio_tokens: torch.Tensor) -> tuple[torch.Tensor, int]:
        N = self.TOKENS_PER_FRAME
        L = audio_tokens.numel() // N
        delayed_LN = audio_tokens[: L * N].view(L, N)
        codes_TN = self._reverse_delay_pattern(delayed_LN)
        audio = self.model._decode_codes(codes_TN)
        return audio, self.SAMPLE_RATE
