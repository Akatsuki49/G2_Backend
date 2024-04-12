from keyphrase_vectorizers import KeyphraseCountVectorizer
from keybert import KeyBERT
import joblib
kw_model = KeyBERT(model="paraphrase-mpnet-base-v2")
joblib.dump(kw_model, 'keybert_model.pkl')
