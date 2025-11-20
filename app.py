import streamlit as st
import joblib
import re
import string

@st.cache_resource
def load_model():
    model = joblib.load("LR_model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    return model, vectorizer

model, vectorizer = load_model()

def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

st.title("📰 Fake News Detection App")

input_text = st.text_area("Enter news text:")

if st.button("Check"):
    cleaned = wordopt(input_text)
    transformed = vectorizer.transform([cleaned])
    result = model.predict(transformed)[0]

    if result == 1:
        st.success("🟢 REAL NEWS")
    else:
        st.error("🔴 FAKE NEWS")
