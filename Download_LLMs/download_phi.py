# This script downloads a pre-trained model from Hugging Face and saves it locally.

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set. Please set it to your Hugging Face token.")



login(token=hf_token)


# Define model and local path
model_name = "microsoft/phi-3-mini-4k-instruct"
local_path = "./phi3_model"

# Download model and tokenizer
print(f"Downloading model {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Use float32 for CPU compatibility
    device_map="cpu"           # Explicitly set to CPU
)

# Create local directory if it doesn't exist
os.makedirs(local_path, exist_ok=True)

# Save model and tokenizer locally
print(f"Saving model to {local_path}...")
model.save_pretrained(local_path)
tokenizer.save_pretrained(local_path)
print(f"Model and tokenizer saved to {local_path}")