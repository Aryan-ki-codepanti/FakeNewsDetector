# app.py
import os
import torch
import torch.nn as nn
import streamlit as st
from transformers import BertTokenizerFast, BertModel
from model import BERT_Arch  # import your class definition here
import time
from download_model import download_model

# Load model
model_path = 'model.pt'


st.set_page_config(
    page_title="Fake News Detector - Know if its true", page_icon="📰")


if not os.path.exists(model_path):
    download_model(model_path)


bert = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

model = BERT_Arch(bert)
model.load_state_dict(torch.load(
    model_path, map_location=torch.device('cpu')), strict=False)
model.eval()

st.title("📰 Fake News Detector (BERT Based)")

user_input = st.text_area("Enter News Headline or Sentence")

if st.button("Predict"):
    # with st.spinner("Running BERT model..."):
    if user_input.strip():

        with st.spinner("Running BERT model..."):
            time.sleep(1.5)
            tokens = tokenizer.encode_plus(user_input, max_length=15, truncation=True,
                                           padding='max_length', return_tensors='pt')
            with torch.no_grad():
                outputs = model(tokens['input_ids'], tokens['attention_mask'])
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = torch.max(probs).item()
                label = "FAKE" if pred == 1 else "REAL"
        if label == "FAKE":
            st.error(
                f"Prediction: {label} (Confidence: {confidence:.2f})")
        else:
            st.success(
                f"Prediction: {label} (Confidence: {confidence:.2f})")
    else:
        st.warning("Please enter some text to classify.")
