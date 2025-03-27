from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

