# 🔒 Confidentiality Analysis Layer

This project introduces a **Confidentiality Analysis Layer** that evaluates user messages and predicts their **confidentiality level** on a scale of 0 (Public) to 4 (Highly Confidential). It is designed to be integrated into chatbot pipelines, especially in sensitive environments like corporate or legal communications.

---

## 📌 Key Features

- ⚙️ Fine-tuned **BERT model** trained on domain-specific prompts.
- 📊 Five-level confidentiality classification.
- ☁️ **Model hosted on Hugging Face Hub** for portability and easy access.
- 🌐 Deployed via **Render** for always-on public endpoint.
- 🧠 Designed to be plugged into chatbot systems for real-time analysis.

---

## 🧠 Model

- **Architecture**: BERT (base-uncased)
- **Task**: Sequence Classification (`Prompt` → `Confidentiality Level`)
- **Classes**:
  - `0` → Public
  - `1` → Internal
  - `2` → Confidential
  - `3` → Strictly Confidential
  - `4` → Highly Confidential

📦 The trained model is available on **Hugging Face**:  
👉 [Hugging Face Model Repo](https://huggingface.co/VarshiniRaja/my-confidentiality-model)

To load it:
```python
from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("VarshiniRaja/my-confidentiality-model")
tokenizer = BertTokenizer.from_pretrained("VarshiniRaja/my-confidentiality-model")
