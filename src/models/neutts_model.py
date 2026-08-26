"""
NeuTTS model adapter for the Speech-Quant BaseTTS interface.

Torch backbone only: no GGUF, no ONNX, no streaming, no watermarking.
BPE-format models only (neutts-2e). Speaker references loaded from the
bundled neutts package samples directory.
"""

import re
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



class _SpeechLogitCapture(LogitsProcessor):
    """Records audio-subspace softmax probs at every generation step."""

    def __init__(self, speech_start: int, speech_end: int) -> None:
        self.speech_start = speech_start
        self.speech_end = speech_end
        self.step_probs: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        s = scores[0, self.speech_start : self.speech_end].half()
        self.step_probs.append(s)  # stay on device; bulk .cpu() after generation
        return scores


class NeuTTSModel(BaseTTS):
    """
    NeuTTS (torch backbone) + NeuCodec wrapped for the BaseTTS interface.

    AUDIO_TOKEN_START = 0: codec codes are 0-based (no offset, unlike Orpheus).
    Speaker selected via the 'voice' kwarg; maps to bundled NeuTTS2E speakers.
    """

    AUDIO_TOKEN_START = 0
    SAMPLE_RATE = 24_000
    SPEAKERS = ("emily", "paul", "sophie", "steven")
    TOKENS_PER_FRAME = 1  # flat FSQ codec: no RVQ frame structure

    def __init__(
        self,
        model_name: str = "neuphonic/neutts-2e",
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

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=self.dtype
        ).to(self.device)

        self.model.eval()

        # Reject phoneme-format models; only BPE is supported here.
        neuphonic_cfg = getattr(self.model.config, "neuphonic", None) or {}
        input_fmt = neuphonic_cfg.get("input_format", "phonemes")
        if input_fmt != "BPE":
            raise ValueError(
                f"{self.model_name!r} uses input_format={input_fmt!r}. "
                "NeuTTSModel only supports BPE-format models (e.g. neuphonic/neutts-2e)."
            )

        print(f"Loading NeuCodec: {self.codec_name}")
        self.codec = NeuCodec.from_pretrained(self.codec_name).eval().to(self.device)

        # Resolve speech token vocabulary range.
        self._speech_start = self.tokenizer.convert_tokens_to_ids("<|speech_0|>")
        # unk_token_id is None for this tokenizer, so use 'is not None' check.
        for size in (131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024):
            last = self.tokenizer.convert_tokens_to_ids(f"<|speech_{size - 1}|>")
            if last is not None:
                self._speech_end = self._speech_start + size
                break
        else:
            self._speech_end = self._speech_start + 65536
        print(
            f"Speech token range: [{self._speech_start}, {self._speech_end}) "
            f"codebook_size={self._speech_end - self._speech_start}"
        )

        # Constants for teacher-forced distribution comparison (single flat
        # codebook, unlike Orpheus's 7-token RVQ frame). VOCAB_AUDIO_TOKEN_START
        # is the real tokenizer-vocab offset (unlike AUDIO_TOKEN_START=0, which
        # is relative to the already-de-offset regex-extracted audio_tokens).
        self.VOCAB_AUDIO_TOKEN_START = self._speech_start
        self.CODEBOOK_SIZE = self._speech_end - self._speech_start
        self.END_OF_SPEECH = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

        print("NeuTTS loaded.")

    def _speaker(self, name: str) -> tuple[torch.Tensor, str]:
        if name not in self.SPEAKERS:
            raise ValueError(f"Unknown speaker '{name}'. Available: {self.SPEAKERS}")
        if name not in self._speaker_cache:
            voices = Path(__file__).parents[2] / "voices"
            pt  = voices / f"{name}.pt"
            txt = voices / f"{name}.txt"
            if not pt.exists() or not txt.exists():
                import shutil, subprocess, tempfile
                with tempfile.TemporaryDirectory() as tmp:
                    print("Cloning neutts to fetch speaker voices...")
                    subprocess.run(
                        ["git", "clone", "--depth=1",
                         "https://github.com/neuphonic/neutts.git", tmp],
                        check=True,
                    )
                    shutil.copytree(Path(tmp) / "samples", voices, dirs_exist_ok=True)
                print(f"Voices saved to {voices}")
            codes = torch.load(pt, weights_only=True, map_location="cpu")
            self._speaker_cache[name] = (codes, txt.read_text(encoding="utf-8").strip())
        return self._speaker_cache[name]

    # ---------------------------------------------------- BaseTTS interface

    def prepare_input(
        self,
        text: str,
        voice: str = "emily",
        **kwargs,
    ) -> dict[str, Any]:
        codes, ref_text = self._speaker(voice)

        # Normalise whitespace — mirrors reference normalize_text().
        text = re.sub(r"\s+", " ", text.strip())
        ref_text = re.sub(r"\s+", " ", ref_text.strip())

        ts = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
        te = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")
        sg = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
        tr = self.tokenizer.convert_tokens_to_ids("<|TEXT_REPLACE|>")
        sr = self.tokenizer.convert_tokens_to_ids("<|SPEECH_REPLACE|>")

        text_ids = self.tokenizer.encode(
            f"{ref_text} {text}", add_special_tokens=False
        )

        # Encode the template to capture any BOS / framing the tokeniser adds.
        base_ids = self.tokenizer.encode("<|TEXT_REPLACE|><|SPEECH_REPLACE|>")
        tr_idx = base_ids.index(tr)
        sr_idx = base_ids.index(sr)

        codes_str = "".join(f"<|speech_{int(i)}|>" for i in codes.tolist())
        code_ids = self.tokenizer.encode(codes_str, add_special_tokens=False)

        prompt = (
            base_ids[:tr_idx]
            + [ts] + text_ids + [te]
            + base_ids[tr_idx + 1 : sr_idx]
            + [sg] + code_ids
        )
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

        input_len = inputs["input_ids"].shape[-1]
        # Cap total sequence length at 2048 — mirrors reference max_length=2048.
        max_length = min(input_len + max_new_tokens, 2048)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            max_length=max_length,
            eos_token_id=eos_id,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            use_cache=True,
            min_new_tokens=50,
            logits_processor=LogitsProcessorList([capture]),
        )

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
        new_ids = generated_ids[0, input_len:].cpu()

        # Decode to text and extract codec IDs via regex — mirrors reference exactly.
        generated_text = self.tokenizer.decode(new_ids.tolist(), add_special_tokens=False)
        speech_ids = [int(x) for x in re.findall(r"<\|speech_(\d+)\|>", generated_text)]
        audio_tokens = torch.tensor(speech_ids, dtype=torch.long)

        # Align step_probs to speech positions for logit capture.
        speech_mask = (new_ids >= self._speech_start) & (new_ids < self._speech_end)
        indices = speech_mask.nonzero(as_tuple=True)[0].tolist()
        probs = [step_probs[i] for i in indices if i < len(step_probs)]
        if probs:
            stacked = torch.stack(probs).float()
            audio_logits = torch.softmax(stacked, dim=-1).cpu().half()
        else:
            audio_logits = None
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
