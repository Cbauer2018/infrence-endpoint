import argparse
import os
import subprocess
import sys
from flask import Flask, request, jsonify
import torch
from transformers import pipeline

app = Flask(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Start the Flask app with the specified model type.')
parser.add_argument('model_type', help='Model type to use')
args = parser.parse_args()

# Model type to model path in huggingface mapping
MODEL_PATHS = {
    "Dolly V2 7B": "databricks/dolly-v2-7b",
    "StableLM 7B": "stabilityai/stablelm-tuned-alpha-7b",
}

# Get the model path based on the model type
model_path = MODEL_PATHS.get(args.model_type)
if not model_path:
    raise ValueError(f"Invalid model type: {args.model_type}")

# Load the pipeline when the app starts
instruct_pipeline = pipeline(
    model=model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)

@app.route('/', methods=['POST'])
def inference():
    prompt = request.json.get('prompt')

    if prompt:
        result = instruct_pipeline(prompt)
        return jsonify({"result": result})
    else:
        return jsonify({"error": "No prompt provided"}), 400

def run_app():
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    flask_process = subprocess.Popen(
        [sys.executable, "-u", "-m", "flask", "run", "--host", "0.0.0.0", "--port", "5000"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    print(f"Flask app started with process ID {flask_process.pid}")