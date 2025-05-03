import yaml
import os
from utils import ModelInference

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_name = config["model_name"]
local_path = config["local_path"]
output_params = config["output_params"]
prompt = config["prompt"]

model_infer = ModelInference(local_path, output_params)
model_infer.load_model()
model_infer.generate_response(prompt)