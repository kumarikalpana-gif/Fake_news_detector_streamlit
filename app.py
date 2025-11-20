import streamlit as st
import pickle
import requests
import io
import re
import string

# -----------------------------
#   Helper: Load from Google Drive
# -----------------------------
def load_from_drive(url):
    response = requests.get(url)
    return pickle.load(io.BytesIO(response.content))

# -----------------------------
#   Text Cleaning
# -----------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text.strip()

def output_label(n):
    return "Fake News" if n == 0 else "Real News"

# -------------------------------------------------------
#     🔗 DIRECT GOOGLE DRIVE LINK — REPLACE FILE IDs
# -------------------------------------------------------

DT_URL = "https://drive.google.com/uc?export=download&id=1SkeIK3kWZ6_G5_hw1LOOk4-YvLXsh7Hy"
LR_URL = "https://drive.google.com/uc?export=download&id=1AKrSa2DJo84kISyZQGL2U7mnx-BSe7M5"
GB_URL = "https://drive.google.com/uc?export=download&id=1k3y1BtA6WQRVi_c6J9TXylp-MK8WqqZo"
RF_URL = "https://drive.google.com/uc?export=download&id=1jSoECG-xbqLRdyCOnPygoHoyZRsEkvTJ"
VEC_URL = "https://drive.google.com/uc?export=download&id=1fXqPhXO4xeMLQM8ChvIOnS-cf3iRPiV1"

# -------------------------------------------------------
#            LOAD ALL MODELS ONLY ONCE
# -------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    models['vectorizer'] = load_from_drive(VEC_URL)
    models['LR'] = load_from_drive(LR_URL)
    models['DT'] = load_from_drive(DT_URL)
    models['GB'] = load_from_drive(GB_URL)
    models['RF'] = load_from_drive(RF_URL)
    return models

models = load_models()

# -------------------------------------------------------
#            STREAMLIT USER INTERFACE
# -------------------------------------------------------

st.set_page_config(page_title='Fake News Detector', layout='centered')

st.title("📰 Fake News Detector")
st.write("Paste any news text below to check whether it is *Fake* or *Real*.")

news_text = st.text_area("Enter News Text:", height=250)

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text first!")
    else:
        cleaned = clean_text(news_text)
        vec = models['vectorizer'].transform([cleaned])

        lr = models['LR'].predict(vec)[0]
        dt = models['DT'].predict(vec)[0]
        gb = models['GB'].predict(vec)[0]
        rf = models['RF'].predict(vec)[0]

        st.subheader("🔍 Individual Model Predictions")
        st.write(f"• Logistic Regression: {output_label(lr)}")
        st.write(f"• Decision Tree: {output_label(dt)}")
        st.write(f"• Gradient Boosting: {output_label(gb)}")
        st.write(f"• Random Forest: {output_label(rf)}")

        votes = [lr, dt, gb, rf]
        majority = 1 if sum(votes) >= 2 else 0

        st.markdown(f"### 🏆 Final Majority Result: *{output_label(majority)}*")

st.markdown("---")
st.caption("✔ Models hosted on Google Drive\n✔ App built using scikit-learn + Streamlit")
