# import torch

# from transformers import (
#     TorchAoConfig,
#     AutoModelForCausalLM,
#     AutoTokenizer,
# )

# from torchao.quantization import Int4WeightOnlyConfig
# from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_

# # Create quantization configuration


# # MODEL_NAME = "Qwen/Qwen3.5-2B"
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


# # ---------------------------------------------------------
# # Quantization configuration
# # ---------------------------------------------------------

# quantization_config = TorchAoConfig(
#     quant_type=Int4WeightOnlyConfig(
#         group_size=32,
#         int4_packing_format="tile_packed_to_4d",
#         int4_choose_qparams_algorithm="hqq",
#     )
# )
# # quantization_config = TorchAoConfig(
# #     quant_type=Float8DynamicActivationFloat8WeightConfig(
# #         granularity=PerRow()
# #         )
# #     )


# # ---------------------------------------------------------
# # Load tokenizer
# # ---------------------------------------------------------

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# # ---------------------------------------------------------
# # Load quantized model
# # ---------------------------------------------------------

# print("Loading model...")

# quantized_model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     dtype="auto",
#     device_map="auto",
#     # quantization_config=quantization_config,
# )

# quantized_model.eval()
# # This works, hf based backend is not working
# quantize_(quantized_model, Int4WeightOnlyConfig(group_size=32, int4_packing_format="plain_int32"))
# # quantize_(quantized_model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))


# print("Model loaded.")


# # ---------------------------------------------------------
# # Sample prompts
# # ---------------------------------------------------------

# prompts = [
#     "Explain how a transformer works in simple terms.",
#     "Write a Python function to calculate the Fibonacci sequence.",
#     "What are the main differences between CPU and GPU architectures?",
#     "Explain why quantization can affect the output of a language model.",
#     "Give me three practical applications of large language models.",
# ]


# # ---------------------------------------------------------
# # Generation
# # ---------------------------------------------------------

# def generate_response(
#     prompt,
#     max_new_tokens=256,
#     enable_thinking=False,
# ):
#     messages = [
#         {
#             "role": "user",
#             "content": prompt,
#         }
#     ]

#     # Qwen3 chat template
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#         enable_thinking=enable_thinking,
#     )

#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#     )

#     # Move inputs to the same device as the model
#     inputs = {
#         key: value.to(quantized_model.device)
#         for key, value in inputs.items()
#     }

#     with torch.no_grad():
#         outputs = quantized_model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.8,
#             top_k=20,
#             min_p=0,
#             pad_token_id=tokenizer.eos_token_id,
#         )

#     # Remove prompt tokens
#     generated_ids = outputs[0][inputs["input_ids"].shape[1]:]

#     response = tokenizer.decode(
#         generated_ids,
#         skip_special_tokens=True,
#     )

#     return response


# # ---------------------------------------------------------
# # Generate samples
# # ---------------------------------------------------------

# for i, prompt in enumerate(prompts, start=1):

#     print("=" * 80)
#     print(f"PROMPT {i}")
#     print("=" * 80)

#     print(prompt)

#     print("\nRESPONSE")
#     print("-" * 80)

#     response = generate_response(
#         prompt,
#         max_new_tokens=256,
#         enable_thinking=False,
#     )

#     print(response)
#     print()


from datasets import load_dataset
from gptqmodel import GPTQConfig, GPTQModel

model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
  ).select(range(1024))["text"]

quant_config = GPTQConfig(bits=4, group_size=128)

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match GPU/VRAM specs to speed up quantization
model.quantize(calibration_dataset, batch_size=1)

model.save(quant_path)