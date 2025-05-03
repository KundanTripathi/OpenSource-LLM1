import yaml
import os
from utils import ModelLoader

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_name = config["model_name"]
local_path = config["local_path"]
model_params = config["model_params"]

model_loader = ModelLoader(model_name, local_path, model_params)
model_loader.download_model()

