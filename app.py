import streamlit as st
import joblib
import requests
import io
import re
import string

# ---------------------
# HELPERS
# ---------------------

def wordopt(text: str) -> str:
    text = str(text)
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text.strip()


def output_label(n: int) -> str:
    return "Fake News" if n == 0 else "Not A Fake News"


# ---------------------
# LOAD MODELS FROM HUGGINGFACE
# ---------------------

@st.cache_resource
def load_models():
    models = {}

    base_url = "https://huggingface.co/kkalpana/fake_news_models/resolve/main/"

    def load_from_url(filename):
        url = base_url + filename
        response = requests.get(url)
        response.raise_for_status()
        return joblib.load(io.BytesIO(response.content))

    models["LR"] = load_from_url("LR_model.pkl")
    models["DT"] = load_from_url("DT_model.pkl")
    models["GB"] = load_from_url("GB_model.pkl")
    models["RF"] = load_from_url("RF_model.pkl")
    models["vectorizer"] = load_from_url("vectorizer.pkl")

    return models


models = load_models()


# ---------------------
# STREAMLIT UI
# ---------------------

st.set_page_config(page_title='Fake News Detector', layout='centered')
st.title("📰 Fake News Detection")
st.write("Paste a news article snippet and check if it's fake or not.")

news_text = st.text_area("Enter news article text", height=250)

if st.button("Check"):
    if not news_text.strip():
        st.warning("Please enter some text.")
    else:
        processed = wordopt(news_text)
        vect = models["vectorizer"].transform([processed])

        lr_pred = models["LR"].predict(vect)[0]
        dt_pred = models["DT"].predict(vect)[0]
        gb_pred = models["GB"].predict(vect)[0]
        rf_pred = models["RF"].predict(vect)[0]

        st.subheader("Predictions")
        st.write("- Logistic Regression:", output_label(int(lr_pred)))
        st.write("- Decision Tree:", output_label(int(dt_pred)))
        st.write("- Gradient Boosting:", output_label(int(gb_pred)))
        st.write("- Random Forest:", output_label(int(rf_pred)))

        # Majority vote
        votes = [lr_pred, dt_pred, gb_pred, rf_pred]
        majority = 1 if sum(votes) >= 2 else 0

        st.markdown(f"**Majority Vote:** {output_label(majority)}")

st.caption("Models loaded from HuggingFace.")
