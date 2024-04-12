import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# Download stopwords if not already downloaded
# nltk.download('stopwords')
# nltk.download('punkt')


def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters except for whitespace, exclamation marks, and full stops
    text = re.sub(r'[^\w\s.!]', '', text)
    return text


def remove_stopwords(text):
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    tokens = [
        word for word in tokens if word not in stopwords.words('english')]
    # Join tokens back into text
    return ' '.join(tokens)


def sentence_tokenization(text):
    return sent_tokenize(text)


def clean_text_data(input_text):
    cleaned_data = {}
    cleaned_data['text'] = remove_stopwords(clean_text(input_text))
    cleaned_data['text_sentences'] = sentence_tokenization(
        cleaned_data['text'])
    return cleaned_data


# Example usage:
# input_text = "Your input text goes here."
# cleaned_text_data = clean_text_data(input_text)
# print(cleaned_text_data)
