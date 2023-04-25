import argparse
from flask import Flask, request, jsonify
import torch
from transformers import pipeline

app = Flask(__name__)

# Global variable to hold the pipeline object
instruct_pipeline = None

@app.route('/', methods=['POST'])
def inference():
    prompt = request.json.get('prompt')

    if prompt:
        result = instruct_pipeline(prompt)
        return jsonify({"result": result})
    else:
        return jsonify({"error": "No prompt provided"}), 400

def start_app(pipeline_object):
    global instruct_pipeline
    instruct_pipeline = pipeline_object
    app.run(host='0.0.0.0', port=5000)
