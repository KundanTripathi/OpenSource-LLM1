from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login


class ModelLoader:
    def __init__(self, model_name: str, local_path: str, model_params: dict):
        self.model_name = model_name
        self.local_path = local_path
        self.tokenizer = None
        self.model = None
        self.model_params = model_params

    def _load_api(self):
        # Load environment variables from .env file
        load_dotenv()

        hf_token = os.getenv("HF_TOKEN")

        if hf_token is None:
            raise ValueError("HF_TOKEN environment variable not set. Please set it to your Hugging Face token.")
        return hf_token
    
    def download_model(self):

        hf_token = self._load_api()

        login(token=hf_token)

        # Download model and tokenizer
        print(f"Downloading model {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
        #    quantization_config=BitsAndBytesConfig(self.model_params["quantization_config"]),
            torch_dtype=torch.float32, 
            device_map=self.model_params["device_map"]
        )

        # Create local directory if it doesn't exist
        os.makedirs(self.local_path, exist_ok=True)

        # Save model and tokenizer locally
        print(f"Saving model to {self.local_path}...")
        self.model.save_pretrained(self.local_path)
        self.tokenizer.save_pretrained(self.local_path)
        print(f"Model and tokenizer saved to {self.local_path}")

class ModelInference:
    def __init__(self, local_path: str, output_params: dict):
        self.local_path = local_path
        self.tokenizer = None
        self.model = None
        self.output_params = output_params

    def load_model(self):
        # Load model and tokenizer from local path
        print(f"Loading model from {self.local_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.local_path,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu"
        )
    
    def generate_response(self, prompt: str):
        # Prepare input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")

        # Perform inference
        print("Generating response...")
        outputs = self.model.generate(
            **inputs,
            max_length=self.output_params["max_length"],
            num_return_sequences=self.output_params["num_return_sequences"],
            do_sample=self.output_params["do_sample"],
            top_k=self.output_params["top_k"],
            temperature=self.output_params["temperature"],
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Print response
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
