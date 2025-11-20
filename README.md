# Fake News Detector (Streamlit)

A lightweight Streamlit web app that uses a trained Logistic Regression model and TF-IDF vectorizer to classify news as **Real** or **Fake**.

---

## How to Prepare Models

Train your model locally and save these two files:

* `LR_model.joblib`
* `vectorizer.joblib`

(If you have `.pkl` files, convert them using joblib.)

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy on Streamlit Cloud

1. Upload to GitHub:

   * `app.py`
   * `LR_model.joblib`
   * `vectorizer.joblib`
   * `requirements.txt`
   * `README.md`
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Choose your repo → select `app.py` → Deploy.

---

## Usage

Enter news text → click **Check** → view prediction:

* 🟢 Real News
* 🔴 Fake News
