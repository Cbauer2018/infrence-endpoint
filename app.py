from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import argparse
import os
import subprocess
from multiprocessing import Process, Value

app = Flask(__name__)

parser = argparse.ArgumentParser(description='Start the Flask app with the specified model type.')
parser.add_argument('model_path', help='Path to model on HuggingFace')
args = parser.parse_args()

model_path = args.model_path
if not model_path:
    raise ValueError(f"Invalid model path: {args.model_path}")

# Global variable to store the model loading status
model_loaded = Value('b', False)

def load_model():
    global model_loaded, instruct_pipeline
    # Load the pipeline
    instruct_pipeline = pipeline(
        model=model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    with model_loaded.get_lock():
        model_loaded.value = True

@app.route('/status', methods=['GET'])
def status():
    if model_loaded.value:
        return jsonify({"status": "Model is loaded and API is running."})
    else:
        return jsonify({"status": "Model is loading. Please wait."})

@app.route('/', methods=['POST'])
def inference():
    if not model_loaded.value:
        return jsonify({"error": "Model is still loading. Please wait and try again."}), 503

    prompt = request.json.get('prompt')

    if prompt:
        result = instruct_pipeline(prompt)
        return jsonify({"result": result})
    else:
        return jsonify({"error": "No prompt provided"}), 400

if __name__ == '__main__':
    # Start the Flask API in a separate process
    api_process = Process(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000})
    api_process.start()

    # Load the model in the main process
    load_model()

