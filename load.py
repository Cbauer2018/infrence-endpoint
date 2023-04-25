from transformers import pipeline
import torch
import argparse
# Parse command-line arguments
parser = argparse.ArgumentParser(description='Start the Flask app with the specified model type.')
parser.add_argument('model_path', help='Path to model on HuggingFace')
args = parser.parse_args()


# Get the model path based on the model type
model_path = args.model_path

if not model_path:
    raise ValueError(f"Invalid model path: {args.model_path}")

if __name__ == '__main__':
    # Load the pipeline when the app starts
    instruct_pipeline = pipeline(
    model=model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
    