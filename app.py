from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = Flask(__name__)

# Load the pipeline model from the file
model = AutoModelForCausalLM.from_pretrained("saved_model")
tokenizer = AutoTokenizer.from_pretrained("saved_model")

instruct_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
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
