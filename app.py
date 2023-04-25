import pickle
from flask import Flask, request, jsonify
import torch
from transformers import pipeline

app = Flask(__name__)

# Load the pipeline from the file
with open("pipeline.pkl", "rb") as f:
    instruct_pipeline = pickle.load(f)

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
