import argparse
import os
import pickle
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

# Save the pipeline object to a file
with open("pipeline.pkl", "wb") as f:
    pickle.dump(instruct_pipeline, f)

# Start the Flask app with nohup
subprocess.Popen(
    ["nohup", "python3", "app.py"],
    stdout=open("log.txt", "w"),
    stderr=subprocess.STDOUT,
    preexec_fn=os.setpgrp
)
