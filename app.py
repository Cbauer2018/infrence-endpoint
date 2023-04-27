from flask import Flask, request, jsonify, Response
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import argparse
import threading
import io

app = Flask(__name__)

parser = argparse.ArgumentParser(description='Start the Flask app with the specified model type.')
parser.add_argument('model_path', help='Path to model on HuggingFace')
parser.add_argument('pipeline_type', help='Pipeline type to use')
args = parser.parse_args()

model_path = args.model_path
pipeline_type = args.pipeline_type
if not model_path:
    raise ValueError(f"Invalid model path: {args.model_path}")


global model_loaded,hf_pipeline
model_loaded = False
hf_pipeline = None

def load_pipeline():
    global model_loaded, hf_pipeline
    if pipeline_type == "text-generation":

        hf_pipeline = pipeline(
            model=model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
    elif pipeline_type == "text-to-image":
        hf_pipeline= StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        hf_pipeline = hf_pipeline.to("cuda")

    model_loaded = True

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"model_loaded": model_loaded})

@app.route('/', methods=['POST'])
def inference():

    if not model_loaded:
        return jsonify({"error": "Model is not loaded yet."}), 400
    
    prompt = request.json.get('prompt')


    if prompt:
        
        if pipeline_type == "text-to-image":
            result = hf_pipeline(prompt).images[0]
            img_io = io.BytesIO()
            result.save(img_io, 'JPEG')
            img_io.seek(0)
            return Response(img_io.getvalue(), content_type='image/jpeg')
        else:
            result = hf_pipeline(prompt)
            return jsonify({"result": result})
    else:
        return jsonify({"error": "No prompt provided"}), 400

if __name__ == '__main__':
    load_thread = threading.Thread(target=load_pipeline)
    load_thread.start()
    app.run(host='0.0.0.0', port=5000)


