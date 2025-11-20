import streamlit as st
import pickle
import re
import string


# ---- helpers ----


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


# ---- load models ----


@st.cache_resource
def load_models():
models = {}
models['LR'] = pickle.load(open('models/LR_model.pkl', 'rb'))
models['DT'] = pickle.load(open('models/DT_model.pkl', 'rb'))
models['GB'] = pickle.load(open('models/GB_model.pkl', 'rb'))
models['RF'] = pickle.load(open('models/RF_model.pkl', 'rb'))
models['vectorizer'] = pickle.load(open('models/vectorizer.pkl', 'rb'))
return models


models = load_models()


# ---- Streamlit UI ----


st.set_page_config(page_title='Fake News Detector', layout='centered')
st.title('📰 Fake News Detection')
st.write('Paste the article text (or a news snippet) below and click **Check** to predict whether it is fake or not.')


news_text = st.text_area('Enter news article text', height=250)


if st.button('Check'):
if not news_text or news_text.strip() == '':
st.warning('Please enter some text to analyze.')
else:
processed = wordopt(news_text)
vect = models['vectorizer'].transform([processed])


lr_pred = models['LR'].predict(vect)[0]
dt_pred = models['DT'].predict(vect)[0]
gb_pred = models['GB'].predict(vect)[0]
rf_pred = models['RF'].predict(vect)[0]


st.subheader('Predictions')
st.write('- Logistic Regression:', output_label(int(lr_pred)))
st.write('- Decision Tree:', output_label(int(dt_pred)))
st.write('- Gradient Boosting:', output_label(int(gb_pred)))
st.write('- Random Forest:', output_label(int(rf_pred)))


# Simple majority vote
votes = [int(lr_pred), int(dt_pred), int(gb_pred), int(rf_pred)]
majority = 1 if sum(votes) >= 2 else 0
st.markdown('**Majority vote:** **{}**'.format(output_label(majority)))


st.markdown('---')
st.caption('Model built with scikit-learn. Preprocess similarly to training: lowercase, remove punctuation and numbers.')