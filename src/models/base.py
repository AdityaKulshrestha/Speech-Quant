from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class GenerationOutput:
    """
    Output of the autoregressive generation stage.

    generated_ids:
        Complete output from the language model, including prompt tokens.

    audio_tokens:
        Extracted neural codec tokens, before codec decoding.

    audio:
        Optional decoded waveform.
    """

    generated_ids: torch.Tensor
    audio_tokens: torch.Tensor

    audio: Optional[torch.Tensor] = None
    sampling_rate: Optional[int] = None
    # (num_audio_tokens, audio_vocab_size) float16 — audio-subspace softmax probs
    audio_logits: Optional[torch.Tensor] = None

    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTTS(ABC):

    # Subclasses set this from their constructor and apply it in load()
    # by calling self.quantize(self.quant_type) once self.model exists.
    quant_type: str = "none"

    @abstractmethod
    def load(self) -> None:
        """Load the language model, tokenizer and codec."""
        raise NotImplementedError

    def quantize(self, quant_type: str) -> None:
        """Quantize self.model in place according to quant_type (quants/config.py)."""

        from quants.quantizer import quantize_model

        self.model = quantize_model(self.model, quant_type)

    @abstractmethod
    def prepare_input(self, text: str, **kwargs) -> dict[str, Any]:
        """Convert user input into model-specific inputs."""
        raise NotImplementedError

    @abstractmethod
    def generate_tokens(
        self,
        inputs: dict[str, Any],
        **generation_kwargs,
    ) -> GenerationOutput:
        """
        Autoregressively generate codec tokens.

        This should NOT decode the tokens into audio.
        """
        raise NotImplementedError

    @abstractmethod
    def decode_audio(
        self,
        audio_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Decode codec tokens into waveform."""
        raise NotImplementedError

    def generate_audio(
        self,
        text: str,
        **kwargs,
    ) -> GenerationOutput:

        inputs = self.prepare_input(text, **kwargs)

        output = self.generate_tokens(
            inputs,
            **kwargs,
        )

        if output.audio_tokens.numel() == 0:
            raise RuntimeError(
                f"No audio tokens generated for prompt {text!r}. "
                "The model stopped before producing audio — try increasing "
                "max_new_tokens or check the prompt/voice settings."
            )

        audio, sampling_rate = self.decode_audio(
            output.audio_tokens
        )

        output.audio = audio
        output.sampling_rate = sampling_rate

        return output

    def unload(self) -> None:
        """Release model resources."""

        for attr in ("model", "tokenizer", "codec"):
            if hasattr(self, attr):
                setattr(self, attr, None)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()