#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib


data = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
data.columns = ["label", "text"]
data["text"] = data["text"].str.lower()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["text"])
y = data["label"]


model = MultinomialNB()
model.fit(X, y)

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vector.pkl")