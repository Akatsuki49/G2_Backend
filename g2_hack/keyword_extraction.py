# import nltk
from keyphrase_vectorizers import KeyphraseCountVectorizer
# from nltk.corpus import stopwords
# import os
from data_cleaner import clean_text_data
import redis
import joblib
from openai import OpenAI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import json
import google.generativeai as genai
from spiders.product_description_spider import scrape
from groq import Groq
import spacy
from textrank import trank
import yake

# import torch
# nltk.download("stopwords")
# nltk.download('punkt')
# r=Rake()


def keygen(text):
    client = Groq(
        api_key='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',
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": text + '\n\n\n Extract top 20 keyphrases in the given text. Seperate them by commas and within single square brackets. ',
            }
        ],
        model="llama3-70b-8192",
        temperature=0.1
    )
    ans1 = chat_completion.choices[0].message.content

    # client = OpenAI(api_key="sk-proj-bU08tiIRs1Hm4F5KRE9pT3BlbkFJb9xzSGt37w9JhGt36WlD")

    # response = client.completions.create(
    # model="gpt-3.5-turbo-instruct",
    # prompt= text + '\n\n\n Extract top 20 keyphrases in the given text. Seperate them by commas and within single square brackets. ',
    # temperature=0.1,
    # )
    # print(response.choices[0].text)

    api_key = 'FWxPbAliDOaNC2DdFohRTCFd9UxeeUNo'
    model = "open-mixtral-8x22b"

    client = MistralClient(api_key=api_key)

    chat_response1 = client.chat(
        temperature=0.1,
        model=model,
        messages=[ChatMessage(role="user", content= text + '\n\n\n Extract only top 20 keyphrases in the given text. Seperate them by commas and within square brackets. Keep the keyphrases length about 2-3 words')]
    )

    ans2 = chat_response1.choices[0].message.content

# Access your API key as an environment variable.
    genai.configure(api_key="AIzaSyDG8Ocml8L2RtuWRbeVpesAhOIQz9qAVec")
    # Choose a model that's appropriate for your use case.
    model = genai.GenerativeModel('gemini-1.5-pro')


    prompt = text + '\n\n\n Give top 20 keyphrases in the given text. Seperate them by commas and within square brackets. Keep the keyphrases length of 2-3 words'

    response = model.generate_content(prompt)

    def text_extract(m1):
        t1 = str(m1)
        l1 = t1.index('[')
        l2 = t1.index(']')
        z1 = t1[l1:l2+1]
        # print(z1)
        string_list = z1[1:-1]
        elements = string_list.split(", ")
        elements = [elem.strip() for elem in elements]
        elements = [elem.lower() for elem in elements]
        return ', '.join(elements)
        # print(elements)

    def similar(a1,a2,a3):
        client = Groq(
            api_key='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',
        )
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": a1 + '\nThis is the first set of data\n' + a2 + '\nThis is the second set of data\n' + a3 + '\nThis is the third set of data\n\n\n Extract top 12 keyphrases which occurs and is similar in all sets of data. Seperate them by commas and within single square brackets. ',
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.1
        )
        k1 = chat_completion.choices[0].message.content
        return k1

    x1 = text_extract(ans1)
    x2 = text_extract(ans2)
    x3 = text_extract(response.text)
    s1 = similar(x1,x2,x3)
    s2 = text_extract(s1)
    return s2

def custom_keygen(text):

    def text_extract(m1):
        t1 = str(m1)
        l1 = t1.index('[')
        l2 = t1.index(']')
        z1 = t1[l1:l2+1]
        # print(z1)
        string_list = z1[1:-1]
        elements = string_list.split(", ")
        elements = [elem.strip() for elem in elements]
        elements = [elem.lower() for elem in elements]
        return ', '.join(elements)

    kw1 = trank(text)
    kw_model = joblib.load('keybert_model.pkl')
    kw21 = kw_model.extract_keywords(
        text, top_n=20, vectorizer=KeyphraseCountVectorizer(), diversity=0.8)
    kw22 = [keyphrase for keyphrase, score in kw21]
    kw2 = ', '.join(kw22)
    kw_extractor = yake.KeywordExtractor(lan="english",n=3,top=20)
    keywords = kw_extractor.extract_keywords(text)
    kw31 = [keyphrase for keyphrase, score in keywords]
    kw3 = ', '.join(kw31)
    client = Groq(
    api_key='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',
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": kw1 + '\nThis is the first set of data\n' + kw2 + '\nThis is the second set of data\n' + kw3 + '\nThis is the third set of data\n\n\n Extract top 12 keyphrases which occurs and is similar in all sets of data. Seperate them by commas and within single square brackets. ',
            }
        ],
        model="mixtral-8x7b-32768",
        temperature=0.1
    )
    k1 = chat_completion.choices[0].message.content
    final = text_extract(k1)
    return final
    # print(kw3)


if __name__ == "__main__":
    r = redis.Redis(host='localhost', port=6379, db=0)

    url = "https://www.chattechnologies.com/"
    data_full_context = r.hgetall(f"scraped:{url}")
    decoded_data = {key.decode('utf-8'): value.decode('utf-8')
                    for key, value in data_full_context.items()}
    scraped_data = scrape(url)
    # fn = 'scraped_data/www.chattechnologies.com.json'
    # with open(fn, 'r', encoding='utf-8') as file:
    #     data = json.load(file)

    full_context1 = clean_text_data(scraped_data)
    full_context = scraped_data['full_text']
    keygen(str(full_context1))
    custom_keygen(str(full_context))
    # print(key_phrases)
    # print(data_full_context)

