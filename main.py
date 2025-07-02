#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


@st.cache_resource
def load_or_train_model():
    model_path = 'spam_model.pkl'
    vectorizer_path = 'vectorizer.pkl'

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    else:

        data = pd.read_csv('spam.csv', encoding='latin-1', na_values=['null'])
        data.columns = ['label', 'text']
        data['text'] = data['text'].str.lower()


        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(data['text'])
        y = data['label']


        model = MultinomialNB()
        model.fit(X, y)


        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)

    return vectorizer, model

vectorizer, model = load_or_train_model()



st.title("Spam Detector App")
st.write("Enter a message below to check if it's **Spam** or **Not Spam**.")

user_input = st.text_area("Your message:")

if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter a message first.")
    else:

        user_text = user_input.lower()
        user_vec = vectorizer.transform([user_text])
        prediction = model.predict(user_vec)[0]

        if prediction.lower() == "spam":
            st.error(f"This message is predicted to be **SPAM**.")
        else:
            st.success(f"This message is predicted to be **NOT SPAM**.")

