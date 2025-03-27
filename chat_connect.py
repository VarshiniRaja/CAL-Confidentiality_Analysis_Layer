from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import login
import os

app = Flask(__name__)
print("Start2")

# Login to Hugging Face
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Hugging Face token not found!")

print("Start3")

# Load model & tokenizer
model = BertForSequenceClassification.from_pretrained("VarshiniRaja/my-confidentiality-model")
tokenizer = BertTokenizer.from_pretrained("VarshiniRaja/my-confidentiality-model")
model.eval()

def classify_confidentiality(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=-1).item()
    return predicted_label

@app.route('/', methods=['GET'])
def home():
    return "App is running!", 200

@app.route('/receive_chat_data', methods=['POST'])
def receive_chat_data():
    data = request.json
    user_message = data.get("user_message", "")
    print("Received Chat Data:", data)
    
    confidentiality_level = classify_confidentiality(user_message)
    print("Confidentiality Level:", confidentiality_level)
    
    return jsonify({"status": "success", "cal": confidentiality_level}), 200
