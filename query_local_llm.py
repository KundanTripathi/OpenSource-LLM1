from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define local path
local_path = "./distilgpt2_model"

# Load model and tokenizer from local path
print(f"Loading model from {local_path}...")
tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForCausalLM.from_pretrained(
    local_path,
    torch_dtype=torch.float32,  # Use float32 for CPU
    device_map="cpu"           # Explicitly set to CPU
)

# Prepare input prompt
prompt = "Share Shakespear's most famous quote"
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

# Perform inference
print("Generating response...")
outputs = model.generate(
    **inputs,
    max_length=100,
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    temperature=1.0,
    pad_token_id=tokenizer.eos_token_id
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print response
print(f"Prompt: {prompt}")
print(f"Response: {response}")