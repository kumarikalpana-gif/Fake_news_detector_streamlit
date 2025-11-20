# Fake News Detector (Streamlit)


This repo contains a Streamlit web app for detecting fake news. It expects pre-trained scikit-learn models and a vectorizer saved as `.pkl` files inside `models/`.


## How to prepare models


1. Train models locally in your notebook following your existing script.
2. Save the models with pickle into `models/`:
- `LR_model.pkl`
- `DT_model.pkl`
- `GB_model.pkl`
- `RF_model.pkl`
- `vectorizer.pkl`


Alternatively, you can commit the `.pkl` files into the repository (not recommended for large files).


## Run locally


```bash
pip install -r requirements.txt
streamlit run app.py