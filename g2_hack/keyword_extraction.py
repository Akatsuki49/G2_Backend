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
from vertexai.preview import generative_models
from vertexai.preview.generative_models import (
    HarmCategory,
    HarmBlockThreshold)
from google.cloud.aiplatform_v1beta1.types.content import SafetySetting

# import torch
# nltk.download("stopwords")
# nltk.download('punkt')
# r=Rake()

r = redis.Redis(host='localhost', port=6379, db=0)


class keygen_models:
    def __init__(self, text, url):
        self.text = text
        self.url = url

    def get_from_groq(self):
        if r.exists(f"keyphrases_groq:{self.url}"):
            str = r.get(f"keyphrases_groq:{self.url}").decode('utf-8')
            str_lst = str.strip('[]').split(', ')
            return str_lst

        client = Groq(
            api_key='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',
        )
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": self.text + '\n\n\n Extract top 20 keyphrases in the given text. Separate them by commas and within single square brackets. I dont want any extra text. Just the formatted list of keyphrases.',
                }
            ],
            model="llama3-70b-8192",
            temperature=0.1
        )
        ans1 = chat_completion.choices[0].message.content

        r.set(f"keyphrases_groq:{self.url}", ans1)

        return ans1

    def get_from_mistral(self):
        if r.exists(f"keyphrases_mistral:{self.url}"):
            str = r.get(f"keyphrases_mistral:{self.url}").decode('utf-8')
            str_lst = str.strip('[]').split(', ')
            return str_lst

        api_key = 'FWxPbAliDOaNC2DdFohRTCFd9UxeeUNo'
        model = "open-mixtral-8x22b"

        client = MistralClient(api_key=api_key)

        chat_response1 = client.chat(
            temperature=0.1,
            model=model,
            messages=[ChatMessage(role="user", content=self.text +
                                  '\n\n\n Extract only top 20 keyphrases in the given text. Separate them by commas and within square brackets. Keep the keyphrases length about 2-3 words. I dont want any extra text. Just the formatted list of keyphrases.')]
        )

        ans2 = chat_response1.choices[0].message.content

        r.set(f"keyphrases_mistral:{self.url}", ans2)
        return ans2

    # def get_from_gemini(self):
    #     if r.exists(f"keyphrases_gemini:{self.url}"):
    #         str = r.get(f"keyphrases_gemini:{self.url}").decode('utf-8')
    #         str_lst = str.strip('[]').split(', ')
    #         return str_lst

    #     genai.configure(api_key="AIzaSyDG8Ocml8L2RtuWRbeVpesAhOIQz9qAVec")
    #     model = genai.GenerativeModel('gemini-1.5-pro')

    #     prompt = self.text + '\n\n\n Give top 20 keyphrases in the given text. Separate them by commas and within square brackets. Keep the keyphrases length of 2-3 words. I dont want any extra text. Just the formatted list of keyphrases.'

    #     safety_config = {
    #         generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    #         generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    #         generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    #         generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    #     }
    #     response = model.generate_content(
    #         prompt, safety_settings=safety_config)
    #     print(response)
    #     response = response.text

    #     r.set(f"keyphrases_gemini:{self.url}", response)

    #     return response

    def get_from_textrank(self):
        if r.exists(f"keyphrases_textrank:{self.url}"):
            str = r.get(f"keyphrases_textrank:{self.url}").decode('utf-8')
            str_lst = str.strip('[]').split(', ')
            return str_lst

        kw = trank(self.text)
        r.set(f"keyphrases_textrank:{self.url}", kw)

        return kw

    def get_from_yake(self):
        if r.exists(f"keyphrases_yake:{self.url}"):
            str = r.get(f"keyphrases_yake:{self.url}").decode('utf-8')
            str_lst = str.strip('[]').split(', ')
            return str_lst

        kw_extractor = yake.KeywordExtractor(lan="english", n=3, top=20)
        keywords = kw_extractor.extract_keywords(self.text)
        kw_ = [keyphrase for keyphrase, score in keywords]
        kw = ', '.join(kw_)

        r.set(f"keyphrases_yake:{self.url}", kw)

        return kw

    def get_from_keybert(self):
        if r.exists(f"keyphrases_keybert:{self.url}"):
            str = r.get(f"keyphrases_keybert:{self.url}").decode('utf-8')
            str_lst = str.strip('[]').split(', ')
            return str_lst

        kw_model = joblib.load('keybert_model.pkl')
        kw_ = kw_model.extract_keywords(
            self.text, top_n=20, vectorizer=KeyphraseCountVectorizer(), diversity=0.8)
        kw_ = [keyphrase for keyphrase, score in kw_]
        kw = ', '.join(kw_)

        r.set(f"keyphrases_keybert:{self.url}", kw)
        return kw

    def text_extract(self, m1):
        t1 = str(m1)
        l1 = t1.index('[')
        l2 = t1.index(']')
        z1 = t1[l1:l2 + 1]
        string_list = z1[1:-1]
        elements = string_list.split(", ")
        elements = [elem.strip() for elem in elements]
        elements = [elem.lower() for elem in elements]
        return ', '.join(elements)

    def similar(self, a1, a2, a3):
        str_a1 = ','.join(a1)
        str_a2 = ','.join(a2)
        str_a3 = ','.join(a3)

        client = Groq(
            api_key='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',
        )
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": str_a1 + '\nThis is the first set of data\n' + str_a2 + '\nThis is the second set of data\n' + str_a3 + '\nThis is the third set of data\n\n\n Extract top 12 keyphrases which occur and are similar in all sets of data. Separate them by commas and within single square brackets. ',
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.1
        )
        k1 = chat_completion.choices[0].message.content
        return k1

    def main_model(self):
        if r.exists(f"keyphrases:{self.url}"):
            str = r.get(f"keyphrases:{self.url}").decode('utf-8')
            str_lst = str.strip('[]').split(', ')
            return str_lst

        ans1 = self.get_from_groq()
        ans2 = self.get_from_mistral()
        ans3 = self.get_from_keybert()

        s1 = self.similar(ans1, ans2, ans3)
        s2 = self.text_extract(s1)

        r.set(f"keyphrases:{self.url}", s2)
        return s2


def eval_main_model(urls):

    avg_score = 0.0
    for url in urls:
        print('Scraping')
        scraped_data = scrape(url)
        # print(type(scraped_data))

        if (type(scraped_data) == str):
            decoded_data = {key.decode('utf-8'): value.decode('utf-8')
                            for key, value in scraped_data.items()}
        else:
            decoded_data = scraped_data

        full_context = clean_text_data(decoded_data)
        full_context = full_context['full_text']

        KeygenModels = keygen_models(full_context, url)

        print("getting from main model")
        KW_lst_main = KeygenModels.main_model()

        print("getting from groq")
        kw1 = KeygenModels.get_from_groq()
        print("getting from mistral")
        kw2 = KeygenModels.get_from_mistral()
        # print("getting from gemini")
        # kw3 = KeygenModels.get_from_gemini()
        print("getting from textrank")
        kw4 = KeygenModels.get_from_textrank()
        print("getting from yake")
        kw5 = KeygenModels.get_from_yake()
        print("getting from keybert")
        kw6 = KeygenModels.get_from_keybert()

        # Now gotta write a custom eval fnc based on these:
        score = eval_lst(KW_lst_main, [kw1, kw2, kw4, kw5, kw6])

        avg_score += score

    avg_score = avg_score / len(urls)
    print(avg_score)


def eval_lst(main_lst, _bigLst):

    print(main_lst)
    total_avg = 0
    for lst in _bigLst:
        print(lst)
        score = 0
        for kw in lst:
            if kw in main_lst:
                score += 1
        total_avg += score / len(lst)

    total_avg = total_avg / len(_bigLst)

    return total_avg


if __name__ == "__main__":

    # urls = ['https://www.telesign.com/products/trust-engine', 'https://www.litzia.com/professional-it-services/',
    #         'https://www.chattechnologies.com/', 'https://inita.com/', 'https://aim-agency.com/']

    urls = ['https://www.telesign.com/products/trust-engine']

    eval_main_model(urls)
    # data_full_context = r.hgetall(f"scraped:{url}")
    # decoded_data = {key.decode('utf-8'): value.decode('utf-8')
    #                 for key, value in data_full_context.items()}

    # full_context1 = clean_text_data(decoded_data)
    # full_context = decoded_data['full_text']
