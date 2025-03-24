import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "slm"
classification_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
text_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
category_labels = ["Non-Toxic", "Toxic"]

def predict_text_category(input_text):
    encoded_input = text_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        model_output = classification_model(**encoded_input)
    probabilities = torch.nn.functional.softmax(model_output.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    return category_labels[predicted_class]

st.title("Toxicity Checker")
user_input = st.text_area("Your text", "")
if st.button("Check"):
    classification_result = predict_text_category(user_input)
    st.write(f"Detected Category: {classification_result}")

