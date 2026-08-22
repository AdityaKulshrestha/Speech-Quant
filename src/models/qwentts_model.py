"""
Qwen3 TTS model adapter for the Speech-Quant BaseTTS interface.

Wraps Qwen3TTSModel (custom_voice type) using its internal generate step
to capture codec codes as audio_tokens before decoding.

qwen_tts==0.1.1 pins transformers==4.57.3 exactly and breaks under this repo's
main transformers version (multiple incompatible API changes: check_model_inputs,
PreTrainedConfig special-token defaults, RoPE init registry). Run this model with
the dedicated `.venv-qwen` environment instead of the main `.venv`:
    uv venv .venv-qwen --python 3.12.13
    echo "<repo>/.venv/lib/python3.12/site-packages" > \\
        .venv-qwen/lib/python3.12/site-packages/_main_venv_link.pth
    uv pip install --python .venv-qwen/bin/python3.12 \\
        "transformers==4.57.3" "qwen_tts @ git+https://github.com/QwenLM/Qwen3-TTS"
    # then remove the CUDA torch/torchaudio/nvidia-* + gradio/demo-CLI extras
    # qwen_tts pulls in as required deps, so the linked xpu torch build is used.
"""

import re
from typing import Any, Optional

import torch
import transformers

from .base import BaseTTS, GenerationOutput

_REQUIRED_TRANSFORMERS_PREFIX = "4.57."


class QwenTTSModel(BaseTTS):
    """
    Qwen3 TTS (custom_voice) + speech tokenizer wrapped for BaseTTS.

    audio_tokens = flattened codec codes (1D).  num_quantizers is stored
    in metadata so decode_audio can reshape before passing to the decoder.
    """

    AUDIO_TOKEN_START = 0   # codes are 0-based (no vocabulary offset)
    SAMPLE_RATE = 24_000    # updated from model after load()

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        quant_type: str = "none",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.quant_type = quant_type

        self._qwen = None           # Qwen3TTSModel wrapper
        self._num_quantizers = 1    # resolved after load()

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        if not transformers.__version__.startswith(_REQUIRED_TRANSFORMERS_PREFIX):
            raise RuntimeError(
                f"QwenTTSModel requires transformers=={_REQUIRED_TRANSFORMERS_PREFIX}x "
                f"(found {transformers.__version__}). Run this model from the dedicated "
                ".venv-qwen environment, not the main .venv — see this module's docstring."
            )
        try:
            from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
        except ImportError:
            raise ImportError(
                "qwen_tts is not installed. Install with:\n"
                "  pip install git+https://github.com/QwenLM/Qwen3-TTS"
            )

        print(f"Loading Qwen3 TTS: {self.model_name}")
        self._qwen = Qwen3TTSModel.from_pretrained(
            self.model_name,
            device_map={"": self.device},
            dtype=self.dtype,
        )

        if self.quant_type.startswith("gptq"):
            raise NotImplementedError(
                "GPTQ quantization for QwenTTSModel is not yet supported. "
                "The inner LM must be quantized before wrapping with Qwen3TTSModel."
            )

        # Validate custom_voice model type
        if self._qwen.model.tts_model_type != "custom_voice":
            raise ValueError(
                f"{self.model_name!r} is not a custom_voice model "
                f"(got {self._qwen.model.tts_model_type!r}). "
                "QwenTTSModel only supports custom_voice models."
            )

        self.SAMPLE_RATE = self._qwen.model.speech_tokenizer.get_model_type(
        ) and self._qwen.model.speech_tokenizer.model.get_output_sample_rate() or 24_000

        # Frame size for token-level metrics (FDP/D(t)/codebook-divergence);
        # KL/distribution comparison isn't supported here since the talker has
        # no plain forward(input_ids)->logits API to teacher-force against.
        talker_config = self._qwen.model.config.talker_config
        self.TOKENS_PER_FRAME = talker_config.num_code_groups
        self.CODEBOOK_SIZE = talker_config.vocab_size

        print("Qwen3 TTS loaded.")
        print("Supported speakers:", self._qwen.get_supported_speakers())

    # ---------------------------------------------------- BaseTTS interface

    def prepare_input(
        self,
        text: str,
        voice: str = "Ethan",
        language: str = "English",
        **kwargs,
    ) -> dict[str, Any]:
        text = re.sub(r"\s+", " ", text.strip())
        # Tokenise the text into the assistant-prompt format.
        input_ids = self._qwen._tokenize_texts(
            [self._qwen._build_assistant_text(text)]
        )
        return {"input_ids": input_ids, "speaker": voice, "language": language}

    @torch.inference_mode()
    def generate_tokens(
        self,
        inputs: dict[str, Any],
        max_new_tokens: int = 2048,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        **kwargs,
    ) -> GenerationOutput:
        gen_kwargs = self._qwen._merge_generate_kwargs(
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )

        # Call the inner LM directly to get codec codes before decoding.
        talker_codes_list, _ = self._qwen.model.generate(
            input_ids=inputs["input_ids"],
            instruct_ids=[None],
            speakers=[inputs["speaker"]],
            languages=[inputs["language"]],
            non_streaming_mode=True,
            **gen_kwargs,
        )

        codes = talker_codes_list[0].cpu()      # (T,) or (T, Q)
        num_q = codes.shape[-1] if codes.dim() == 2 else 1
        self._num_quantizers = num_q
        audio_tokens = codes.view(-1).long()    # flatten to 1D

        return GenerationOutput(
            generated_ids=audio_tokens,
            audio_tokens=audio_tokens,
            audio_logits=None,
            metadata={
                "num_audio_tokens": audio_tokens.numel(),
                "num_quantizers": num_q,
                "codes_len": codes.shape[0],
            },
        )

    def decode_audio(
        self, audio_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        num_q = self._num_quantizers
        codes_len = audio_tokens.numel() // num_q
        # Reshape back to (T, Q) for the decoder.
        codes = audio_tokens[:codes_len * num_q].view(codes_len, num_q).to(self.device)

        wavs, fs = self._qwen.model.speech_tokenizer.decode(
            [{"audio_codes": codes}]
        )
        wav = torch.from_numpy(wavs[0]).float()
        return wav, int(fs)

    def unload(self) -> None:
        self._qwen = None
