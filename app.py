from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import argparse
import os
import subprocess

app = Flask(__name__)

model_path = os.environ.get('MODEL_PATH')
if not model_path:
    raise ValueError("MODEL_PATH environment variable is not set")


# Load the pipeline
instruct_pipeline = pipeline(
    model=model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "Model is loaded and API is running."})

@app.route('/', methods=['POST'])
def inference():
    prompt = request.json.get('prompt')

    if prompt:
        result = instruct_pipeline(prompt)
        return jsonify({"result": result})
    else:
        return jsonify({"error": "No prompt provided"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


