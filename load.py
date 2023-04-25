import argparse
import os
import subprocess
from transformers import pipeline
import torch

parser = argparse.ArgumentParser(description='Start the Flask app with the specified model type.')
parser.add_argument('model_path', help='Path to model on HuggingFace')
args = parser.parse_args()

model_path = args.model_path
if not model_path:
    raise ValueError(f"Invalid model path: {args.model_path}")

# Load the pipeline
instruct_pipeline = pipeline(
    model=model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)

# Save the pipeline model to a file
instruct_pipeline.model.save_pretrained("saved_model")

# Start the Flask app with nohup
subprocess.Popen(
    ["nohup", "python3", "app.py"],
    stdout=open("log.txt", "w"),
    stderr=subprocess.STDOUT,
    preexec_fn=os.setpgrp
)
