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

# Start the Flask app with nohup
subprocess.Popen(
    ["nohup", "python3", "app.py"],
    env=dict(os.environ, INSTRUCT_PIPELINE=instruct_pipeline),
    stdout=open("log.txt", "w"),
    stderr=subprocess.STDOUT,
    preexec_fn=os.setpgrp
)
    