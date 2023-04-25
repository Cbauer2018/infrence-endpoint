from flask import Flask, request, jsonify
import torch
from transformers import pipeline
import argparse
import threading

app = Flask(__name__)

parser = argparse.ArgumentParser(description='Start the Flask app with the specified model type.')
parser.add_argument('model_path', help='Path to model on HuggingFace')
args = parser.parse_args()

model_path = args.model_path
if not model_path:
    raise ValueError(f"Invalid model path: {args.model_path}")


global model_loaded,instruct_pipeline
model_loaded = False
instruct_pipeline = None

def load_pipeline():
    global model_loaded, instruct_pipeline
    instruct_pipeline = pipeline(
        model=model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    model_loaded = True

@app.route('/status', methods=['GET'])
def status():
    if model_loaded:
        return jsonify({"status": "Model is loaded and API is running."})
    else:
        return jsonify({"status": "Model is not loaded yet."})

@app.route('/', methods=['POST'])
def inference():

    if not model_loaded:
        return jsonify({"error": "Model is not loaded yet."}), 400
    
    prompt = request.json.get('prompt')

    if prompt:
        result = instruct_pipeline(prompt)
        return jsonify({"result": result})
    else:
        return jsonify({"error": "No prompt provided"}), 400

if __name__ == '__main__':
    load_thread = threading.Thread(target=load_pipeline)
    load_thread.start()
    app.run(host='0.0.0.0', port=5000)


