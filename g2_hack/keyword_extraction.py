from rake_nltk import Rake
import nltk
from keyphrase_vectorizers import KeyphraseCountVectorizer
from nltk.corpus import stopwords
import os
import joblib
import json
import torch
# nltk.download("stopwords")
# nltk.download('punkt')
# r=Rake()

def keygen(file_path):
    kw_model = joblib.load('keybert_model.pkl')
    # cd = os.getcwd()
    # wd = os.path.join(cd,'scraped_cleaned')
    # files = os.listdir(wd)
    # for file1 in files:
        # file = os.path.join(wd,file1)
    with open(str(file_path),'r',encoding='utf-8') as f:
        z1 = f.read()
        x1 = json.loads(z1)
        text = x1['full_text']
        keywords = kw_model.extract_keywords(text,top_n=20,vectorizer = KeyphraseCountVectorizer(),diversity=0.8)
        return keywords
