import streamlit as st
import joblib
import os

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../models/phishing_model.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.set_page_config(page_title="Phishing Email Detector")

st.title("AI-Powered Phishing Email Detector")

model = load_model()

email_text = st.text_area(
    "Paste email content below:",
    height=250
)

if st.button("Analyze Email"):
    if not email_text.strip():
        st.warning("Please enter email content.")
    else:
        pred = model.predict([email_text])[0]
        probs = model.predict_proba([email_text])[0]

        if pred == 1:
            st.error("🚨 PHISHING EMAIL DETECTED")
        else:
            st.success("✅ LEGITIMATE EMAIL")

        st.write("Prediction Confidence:", probs)
