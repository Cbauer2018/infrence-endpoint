import argparse
from flask import Flask, request, jsonify
import torch
from transformers import pipeline

app = Flask(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Start the Flask app with the specified model type.')
parser.add_argument('model_path', help='Path to model on HuggingFace')
args = parser.parse_args()


# Get the model path based on the model type
model_path = args.model_path

if not model_path:
    raise ValueError(f"Invalid model path: {args.model_path}")

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)