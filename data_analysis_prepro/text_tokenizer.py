import os
import torch
from transformers import AutoTokenizer, AutoModel

# Set your API token as an environment variable
# os.environ["HUGGINGFACE_TOKEN"] = "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" #read
os.environ["HF_TOKEN"] = "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" # write

model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Load the tokenizer and model with authentication
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
model = AutoModel.from_pretrained(model_name, token=True)

def encode_descriptions(descriptions):
    # Tokenize the descriptions
    inputs = tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt")

    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Example descriptions
descriptions = [
    "This is a Magic card with special abilities.",
    "Another card that has a unique ability.",
    "This card allows you to draw extra cards."
]

embeddings = encode_descriptions(descriptions)
print("Embeddings:\n", embeddings)
