import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define quantized model path
quantized_path = "./phi3_model"

# Load tokenizer and model
print(f"Loading model from {quantized_path}...")
tokenizer = AutoTokenizer.from_pretrained(quantized_path)
model = AutoModelForCausalLM.from_pretrained(
    quantized_path,
    torch_dtype=torch.float32,  # Use float32 for CPU compatibility
    device_map="cpu",          # Explicitly set to CPU
    load_in_4bit=False         # Avoid bitsandbytes
)

# Prepare model for quantization
model.eval()

# Custom 4-bit quantization function
def quantize_4bit(module):
    if isinstance(module, nn.Linear):  # Use isinstance
        weight = module.weight.data
        max_val = weight.abs().max()
        if max_val == 0:
            return
        scale = max_val / 7  # 4-bit: [-7, 7] for 16 levels
        quantized_weight = torch.round(weight / scale).clamp(-7, 7)
        module.weight.data = quantized_weight * scale
        if hasattr(module, 'bias') and module.bias is not None:
            bias = module.bias.data
            max_val_b = bias.abs().max()
            if max_val_b == 0:
                return
            scale_b = max_val_b / 7
            quantized_bias = torch.round(bias / scale_b).clamp(-7, 7)
            module.bias.data = quantized_bias * scale_b

# Re-quantize to 4-bit
print("Re-quantizing linear layers to 4-bit precision...")
model.apply(quantize_4bit)

# Prepare input prompt
prompt = "What is the capital of India?"
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

# Perform inference
print("Generating response...")
outputs = model.generate(
    **inputs,
    max_length=50,
    num_return_sequences=1,
    do_sample=False
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print response
print(f"Prompt: {prompt}")
print(f"Response: {response}")