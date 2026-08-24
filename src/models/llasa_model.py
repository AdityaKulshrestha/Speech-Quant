"""
Llasa model adapter for the Speech-Quant BaseTTS interface.

Llasa (HKUSTAudio/Llasa-1B) is a plain Llama-3.2-1B-Instruct causal LM that
predicts XCodec2 speech tokens as ordinary vocabulary tokens (`<|s_0|>` ..
`<|s_65535|>`), exactly like NeuTTS's flat-FSQ layout. XCodec2 is a *single*
vector-quantizer codec (not RVQ despite the multi-codebook naming elsewhere):
50 tokens/sec, 16kHz output, one token per frame.

Requires the `xcodec2` package, which hard-pins torch==2.5.0/torchaudio==2.5.0
transitively via `torchtune`. Run this model from a dedicated `.venv-llasa`
environment (same recipe as `.venv-qwen` — link the main `.venv`'s
site-packages via a `.pth` file, then install `xcodec2` with `--no-deps` so
it doesn't drag in a conflicting torch build):
    uv venv .venv-llasa --python 3.12.13
    echo "<repo>/.venv/lib/python3.12/site-packages" > \\
        .venv-llasa/lib/python3.12/site-packages/_main_venv_link.pth
    uv pip install --python .venv-llasa/bin/python3.12 --no-deps xcodec2==0.1.3
    uv pip install --python .venv-llasa/bin/python3.12 \\
        "einops==0.8.0" "torchtune>=0.3.1" "vector-quantize-pytorch==1.17.8"
    # torchtune pulls in a CUDA torch==2.13.0 + nvidia-* wheels that shadow the
    # linked xpu torch build; remove those so the linked build is used instead:
    uv pip uninstall --python .venv-llasa/bin/python3.12 \\
        torch torchdata triton nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc \\
        nvidia-cuda-runtime nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile \\
        nvidia-curand nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 \\
        nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx \\
        huggingface-hub safetensors tokenizers pandas pyarrow protobuf omegaconf \\
        kagglehub kagglesdk multiprocess xxhash dill regex tiktoken sentencepiece
    # xcodec2's own from_pretrained() (nested Wav2Vec2BertModel.from_pretrained
    # inside its __init__) breaks under transformers>=5's meta-device guard
    # ("using from_pretrained with a meta device context manager..."), so pin
    # an older transformers LOCALLY in this venv (shadows the linked 5.15):
    uv pip install --python .venv-llasa/bin/python3.12 "transformers==4.57.3"

Known upstream quirk (not fixed here): `HKUSTAudio/xcodec2`'s published
safetensors store Snake-activation params as `act.beta`, but xcodec2==0.1.3
and 0.1.5's modeling code both reference `act.bias` — those params load as
freshly-initialized instead of from the checkpoint (harmless warning, minor
potential codec-quality regression, not something we patch here).
"""


import re
from typing import Any, Optional

import torch
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


class LlasaModel(BaseTTS):
    """
    Llasa (Llama-3.2 backbone) + XCodec2 wrapped for the BaseTTS interface.

    AUDIO_TOKEN_START = 0: codec codes are 0-based (no offset), same
    convention as NeuTTSModel. Single flat codebook (TOKENS_PER_FRAME=1),
    so teacher-forced distribution comparison is fully supported.
    """

    AUDIO_TOKEN_START = 0
    CODEBOOK_SIZE = 65536
    TOKENS_PER_FRAME = 1  # XCodec2: single vector quantizer, no RVQ frame structure
    SAMPLE_RATE = 16_000  # XCodec2 only supports 16kHz speech

    def __init__(
        self,
        model_name: str = "HKUSTAudio/Llasa-1B",
        codec_name: str = "HKUSTAudio/xcodec2",
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

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        print(f"Loading Llasa backbone: {self.model_name}")
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

        try:
            from xcodec2.modeling_xcodec2 import XCodec2Model
        except ImportError:
            raise ImportError(
                "xcodec2 is not installed. LlasaModel requires the dedicated "
                ".venv-llasa environment — see this module's docstring."
            )

        print(f"Loading XCodec2: {self.codec_name}")
        self.codec = XCodec2Model.from_pretrained(self.codec_name).eval().to(self.device)

        # Resolve speech token vocabulary range (mirrors NeuTTS's dynamic
        # resolution instead of hardcoding — robust to tokenizer additions).
        speech_start = self.tokenizer.convert_tokens_to_ids("<|s_0|>")
        speech_end = self.tokenizer.convert_tokens_to_ids(f"<|s_{self.CODEBOOK_SIZE - 1}|>") + 1
        if speech_end - speech_start != self.CODEBOOK_SIZE:
            raise ValueError(
                f"Unexpected Llasa speech-token vocabulary layout: "
                f"[{speech_start}, {speech_end}) != {self.CODEBOOK_SIZE} entries."
            )
        self.AUDIO_TOKEN_START = 0
        self.VOCAB_AUDIO_TOKEN_START = speech_start
        self._speech_start = speech_start
        self._speech_end = speech_end
        self.END_OF_SPEECH = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

        print("Llasa loaded.")

    # ---------------------------------------------------- BaseTTS interface

    def prepare_input(
        self,
        text: str,
        **kwargs,
    ) -> dict[str, Any]:
        text = re.sub(r"\s+", " ", text.strip())
        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"

        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors="pt",
            continue_final_message=True,
        ).to(self.device)

        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    @torch.inference_mode()
    def generate_tokens(
        self,
        inputs: dict[str, Any],
        max_new_tokens: int = 1200,
        temperature: float = 0.8,
        top_p: float = 1.0,
        **kwargs,
    ) -> GenerationOutput:
        capture = _SpeechLogitCapture(self._speech_start, self._speech_end)

        input_len = inputs["input_ids"].shape[-1]
        # Cap total sequence length at 2048 — mirrors reference max_length=2048.
        max_length = min(input_len + max_new_tokens, 2048)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            eos_token_id=self.END_OF_SPEECH,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
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

        speech_mask = (new_ids >= self._speech_start) & (new_ids < self._speech_end)
        audio_tokens = (new_ids[speech_mask] - self._speech_start).long()

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
