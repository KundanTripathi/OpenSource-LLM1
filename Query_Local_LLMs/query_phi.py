from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define local path
local_path = "./phi3_model"

# Load model and tokenizer from local path
print(f"Loading model from {local_path}...")
tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForCausalLM.from_pretrained(
    local_path,
    torch_dtype=torch.float32,  # Use float32 for CPU
    device_map="cpu"           # Explicitly set to CPU
)

# Prepare input prompt
prompt = "What is capital of India?"
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