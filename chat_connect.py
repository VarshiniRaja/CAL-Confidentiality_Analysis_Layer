from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import login
import os

app = Flask(__name__)
print("Start2")
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token)
print("Start3")

# Load trained model and tokenizer
model = BertForSequenceClassification.from_pretrained("VarshiniRaja/my-confidentiality-model")
tokenizer = BertTokenizer.from_pretrained("VarshiniRaja/my-confidentiality-model")
model.eval()  # Set model to evaluation mode

# Function to classify confidentiality
def classify_confidentiality(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=-1).item()
    return predicted_label

@app.route('/receive_chat_data', methods=['POST'])
def receive_chat_data():
    data = request.json
    user_message = data.get("user_message", "")
    print("Received Chat Data:", data)
    
    # Infer confidentiality level
    confidentiality_level = classify_confidentiality(user_message)
    print(confidentiality_level)
    
    response = {
        "status": "success",
        "cal": confidentiality_level
    }
    return jsonify(response), 200

import os

if __name__ == '__main__':
    print("Start1")
    pass  # Render will use gunicorn to run the app

