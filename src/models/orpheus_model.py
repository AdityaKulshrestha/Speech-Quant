from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

from snac import SNAC

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


class OrpheusTTS(BaseTTS):

    # ---------------------------------------------------------
    # Orpheus special tokens
    # ---------------------------------------------------------

    START_OF_TEXT = 128000
    END_OF_TEXT = 128009

    START_OF_SPEECH = 128257
    END_OF_SPEECH = 128258

    START_OF_HUMAN = 128259
    END_OF_HUMAN = 128260

    START_OF_AI = 128261
    END_OF_AI = 128262

    PAD_TOKEN = 128263

    AUDIO_TOKEN_START = 128266
    VOCAB_AUDIO_TOKEN_START = 128266
    # Orpheus audio tokens occupy LLM vocab positions 128266+ directly
    # (unlike NeuTTS/OuteTTS which store de-offset 0+ tokens).

    # SNAC
    CODEBOOK_SIZE = 4096
    TOKENS_PER_FRAME = 7

    SAMPLE_RATE = 24000

    def __init__(
        self,
        model_name: str = "canopylabs/orpheus-3b-0.1-ft",
        tokenizer_name: str | None = None,
        snac_model_name: str = "hubertsiuzdak/snac_24khz",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        quant_type: str = "none",
    ):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.snac_model_name = snac_model_name

        self.device = device
        self.dtype = dtype
        self.quant_type = quant_type

        self.model = None
        self.tokenizer = None
        self.codec = None

    # =========================================================
    # Loading
    # =========================================================

    def load(self) -> None:

        print(f"Loading Orpheus model: {self.model_name}")

        if self.quant_type != "none":
            from quants.quantizer import quantize_model

            print(f"Quantizing Orpheus ({self.quant_type}): {self.model_name}")
            self.model = quantize_model(
                self.model_name, self.quant_type, device=self.device
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
            ).to(self.device)

        self.model.eval()

        print("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name
        )

        print(
            f"Loading SNAC codec: "
            f"{self.snac_model_name}"
        )

        self.codec = SNAC.from_pretrained(
            self.snac_model_name
        ).to(self.device)

        self.codec.eval()

        print("Orpheus loaded.")

    # =========================================================
    # Input preparation
    # =========================================================

    def prepare_input(
        self,
        text: str,
        voice: str = "tara",
        **kwargs,
    ) -> dict[str, Any]:

        prompt = f"{voice}: {text}"

        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).input_ids

        # SOH SOT text EOT EOH
        start_token = torch.tensor(
            [[self.START_OF_HUMAN]],
            dtype=torch.long,
        )

        end_tokens = torch.tensor(
            [[
                self.END_OF_TEXT,
                self.END_OF_HUMAN,
            ]],
            dtype=torch.long,
        )

        input_ids = torch.cat(
            [
                start_token,
                input_ids,
                end_tokens,
            ],
            dim=1,
        )

        attention_mask = torch.ones_like(
            input_ids
        )

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(
                self.device
            ),
        }

    # =========================================================
    # Autoregressive generation
    # =========================================================

    @torch.inference_mode()
    def generate_tokens(
        self,
        inputs: dict[str, Any],
        max_new_tokens: int = 1200,
        temperature: float = 0.6,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        **kwargs,
    ) -> GenerationOutput:

        audio_end = self.AUDIO_TOKEN_START + self.TOKENS_PER_FRAME * self.CODEBOOK_SIZE
        capture = AudioLogitCapture(self.AUDIO_TOKEN_START, audio_end)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=1,
            eos_token_id=self.END_OF_SPEECH,
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
    ) -> torch.Tensor | None:
        """Match captured per-step probs to audio token positions."""
        gen_tokens = generated_ids[0, input_len:]

        sos_pos = (gen_tokens == self.START_OF_SPEECH).nonzero(as_tuple=True)[0]
        start_idx = int(sos_pos[-1].item()) + 1 if len(sos_pos) > 0 else 0

        audio_probs: list[torch.Tensor] = []
        for i in range(start_idx, len(gen_tokens)):
            tok = gen_tokens[i].item()
            if tok == self.END_OF_SPEECH:
                break
            if tok >= self.AUDIO_TOKEN_START and i < len(step_probs):
                audio_probs.append(step_probs[i])

        n_complete = (len(audio_probs) // self.TOKENS_PER_FRAME) * self.TOKENS_PER_FRAME
        if n_complete == 0:
            return None
        # Single device→CPU transfer + softmax for all audio steps at once
        stacked = torch.stack(audio_probs[:n_complete]).float()
        return torch.softmax(stacked, dim=-1).cpu().half()

    # =========================================================
    # Extract Orpheus codec tokens
    # =========================================================

    def extract_audio_tokens(
        self,
        generated_ids: torch.Tensor,
    ) -> torch.Tensor:

        """
        Extract the SNAC tokens from the LLM output.

        Input:
            generated_ids:
                [batch, sequence_length]

        Returns:
            [num_audio_tokens]

        These are still Orpheus token IDs, i.e. they
        have the 128266 offset and codebook offsets.
        """

        if generated_ids.ndim != 2:
            raise ValueError(
                "generated_ids must have shape "
                "[batch, sequence_length]"
            )

        # We currently support one sample.
        row = generated_ids[0]

        # Find the final <|audio|> token.
        audio_start = (
            row == self.START_OF_SPEECH
        ).nonzero(as_tuple=True)[0]

        if len(audio_start) > 0:
            start_idx = audio_start[-1].item() + 1
            row = row[start_idx:]

        # Remove EOS speech token.
        row = row[
            row != self.END_OF_SPEECH
        ]

        # Keep only actual audio tokens.
        row = row[
            row >= self.AUDIO_TOKEN_START
        ]

        # Seven tokens constitute one SNAC frame.
        num_complete_frames = (
            row.numel()
            // self.TOKENS_PER_FRAME
        )

        row = row[
            : num_complete_frames
            * self.TOKENS_PER_FRAME
        ]

        return row

    # =========================================================
    # Convert Orpheus tokens -> SNAC codebooks
    # =========================================================

    def tokens_to_snac_codes(
        self,
        audio_tokens: torch.Tensor,
    ) -> list[torch.Tensor]:

        """
        Convert Orpheus's interleaved 7-token representation
        into the three SNAC codebooks.

        Orpheus layout:

            frame:
                L1[0]
                L2[0]
                L3[0]
                L3[1]
                L2[1]
                L3[2]
                L3[3]
        """

        if audio_tokens.numel() == 0:
            raise ValueError(
                "No audio tokens were generated."
            )

        tokens = (
            audio_tokens
            - self.AUDIO_TOKEN_START
        )

        num_frames = (
            tokens.numel()
            // self.TOKENS_PER_FRAME
        )

        tokens = tokens.reshape(
            num_frames,
            self.TOKENS_PER_FRAME,
        )

        layer_1 = []
        layer_2 = []
        layer_3 = []

        for frame in tokens:

            layer_1.append(
                frame[0]
            )

            layer_2.extend(
                [
                    frame[1]
                    - self.CODEBOOK_SIZE,

                    frame[4]
                    - 4 * self.CODEBOOK_SIZE,
                ]
            )

            layer_3.extend(
                [
                    frame[2]
                    - 2 * self.CODEBOOK_SIZE,

                    frame[3]
                    - 3 * self.CODEBOOK_SIZE,

                    frame[5]
                    - 5 * self.CODEBOOK_SIZE,

                    frame[6]
                    - 6 * self.CODEBOOK_SIZE,
                ]
            )

        return [
            torch.stack(layer_1),
            torch.stack(layer_2),
            torch.stack(layer_3),
        ]

    # =========================================================
    # Decode
    # =========================================================

    @torch.inference_mode()
    def decode_audio(
        self,
        audio_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:

        codes = self.tokens_to_snac_codes(
            audio_tokens
        )

        codes = [
            code.unsqueeze(0)
            .to(self.device)
            .long()
            for code in codes
        ]

        audio = self.codec.decode(
            codes
        )

        # [1, 1, samples] -> [samples]
        audio = (
            audio
            .detach()
            .squeeze()
            .cpu()
        )

        return audio, self.SAMPLE_RATE