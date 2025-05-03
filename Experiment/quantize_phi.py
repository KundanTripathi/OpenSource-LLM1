import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Define paths
input_path = "./phi3_model"
output_path = "./phi3_quantized_8bit"

# Set quantization engine
torch.backends.quantized.engine = 'qnnpack'

# Load model and tokenizer
print(f"Loading model from {input_path}...")
tokenizer = AutoTokenizer.from_pretrained(input_path)
model = AutoModelForCausalLM.from_pretrained(
    input_path,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# Prepare model for quantization
model.eval()

# Function to count linear layers and total layers
def count_layers(model):
    linear_count = 0
    total_count = 0
    for module in model.modules():
        total_count += 1
        if type(module) is nn.Linear:  # Direct type comparison
            linear_count += 1
    return linear_count, total_count

# Count layers before quantization
linear_count, total_count = count_layers(model)
print(f"Total layers: {total_count}")
print(f"Linear layers (nn.Linear): {linear_count}")
print(f"Fraction of layers quantized: {linear_count / total_count:.2%}")

# Apply dynamic quantization to linear layers
print("Quantizing linear layers to 8-bit precision (qint8)...")
try:
    
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec= {nn.Linear},  # Quantize only nn.Linear layers
        dtype=torch.qint8     # Use 8-bit integer quantization
    )
except Exception as e:
    print(f"Quantization failed: {e}")
    raise

# Create output directory
os.makedirs(output_path, exist_ok=True)

# Save quantized model and tokenizer
print(f"Saving quantized model to {output_path}...")
quantized_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print(f"Quantized model and tokenizer saved to {output_path}")