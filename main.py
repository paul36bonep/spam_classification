#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import joblib
import os

#@st.cache_resource

def load_model():
    model_path = "model.pkl"
    vectorizer_path = "vector.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("Model files not found. Please run train.py first.")
        st.stop()

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return vectorizer, model
    
try:
    
    with st.spinner("Loading model..."):
        vectorizer, model = load_model()
    st.success("Model loaded successfully.")
    
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


st.title("Spam Detector App")
st.write("Enter a message below to check if it's **SPAM** or **NOT SPAM**.")

user_input = st.text_area("Your message:")

if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:

        cleaned_text = user_input.lower()
        vectorized_input = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_input)[0]

        if prediction.lower() == "spam":
            st.error("This message is predicted to be **SPAM**.")
        else:
            st.success("This message is predicted to be **NOT SPAM**.")

