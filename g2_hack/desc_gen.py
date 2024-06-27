import sys
import warnings
import os
from openai import OpenAI
import google.generativeai as genai
from keyword_extraction import keygen
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import redis
from data_cleaner import clean_text_data


def generate_summary_keybert_gpt(input_data):
    warnings.filterwarnings("ignore")
    OPENAI_API_KEY = os.environ['OpenAI_API_KEY_G2']
    client = OpenAI(
        api_key=OPENAI_API_KEY)
    keywords = keygen(str(input_data))
    prompt = f'''
    {keywords}

    Using the above information, generate a short 1-2 lines of description, and make sure that the description generated has most of the text from the above information provided. Keep the description very professional and do not disclose information about any place in the world
    '''

    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0.1,
        max_tokens=80
    )
    # print(response.choices[0].text.strip())

    return response.choices[0].text.strip()


def generate_summary_keybert_gemini(input_data):
    GEMINI_API_KEY = os.environ['Gemini_API_KEY_G2']
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    keywords = keygen(str(input_data))
    prompt = f'''
    {keywords}

    Using the above information, generate a short 1-2 lines of description, and make sure that the description generated has most of the text from the above information provided. Keep the description very professional and do not disclose information about any place in the world
    '''
    # Create a generation_config object
    generation_config = genai.GenerationConfig(
        max_output_tokens=80,  # Set the maximum number of tokens to generate
        temperature=0.1  # Set the temperature for the generation
    )
    response = model.generate_content(
        prompt, generation_config=generation_config)

    return response.text


def generate_summary_keybert_mixtral(input_data):
    MIXTRAL_API_KEY = os.environ['Mistral_API_KEY_G2']
    # model = "mistral-large-latest"
    model = "open-mixtral-8x22b"

    client = MistralClient(api_key=MIXTRAL_API_KEY)
    keywords = keygen(str(input_data))

    prompt = f'''
    {keywords}

    Using the above information, generate a short 1-2 lines of description, and make sure that the description generated has most of the text from the above information provided. Keep the description very professional and do not disclose information about any place in the world
    '''

    chat_response = client.chat(
        model=model,
        messages=[ChatMessage(role="user", content=prompt)],
    )

    return chat_response.choices[0].message.content


# if __name__ == "__main__":
#     r = redis.Redis(host='localhost', port=6379, db=0)
#     url = "https://www.chattechnologies.com/"
#     data = r.hgetall(f"scraped:{url}")
#     decoded_data = {key.decode('utf-8'): value.decode('utf-8')
#                     for key, value in data.items()}
#     cleaned_data = clean_text_data(decoded_data)

#     summary_list = []

#     print("Sending to Gemini")

#     for i in range(10):
#         summary = generate_summary_keybert_gemini(cleaned_data['full_text'])
#         summary_list.append(summary)

#     print("Sending to GPT")

#     for i in range(10):
#         summary = generate_summary_keybert_gpt(cleaned_data['full_text'])
#         summary_list.append(summary)

#     print("Sending to MixTral")

#     for i in range(10):
#         summary = generate_summary_keybert_mixtral(cleaned_data['full_text'])
#         summary_list.append(summary)

#     print("Writing to file")

#     with open('summaries.txt', 'w') as f:
#         for item in summary_list:
#             f.write("%s\n" % item)
