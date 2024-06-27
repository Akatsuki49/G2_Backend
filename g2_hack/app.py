from flask import Flask, request, jsonify
import threading
from spiders.product_description_spider import scrape
from data_cleaner import clean_text_data
from desc_gen import generate_summary_keybert_gpt, generate_summary_keybert_gemini, generate_summary_keybert_mixtral
import subprocess
from werkzeug.serving import make_server
import shutil
import os
import redis
import json
import sys

app = Flask(__name__)
r = redis.Redis(host='localhost', port=6379, decode_responses=True)


def save_app():
    # Open the app.py file in append mode
    with open('app.py', 'a') as file:
        # Add a comment to the file
        file.write('\n# This is a new comment added by save_app function.\n')

    # Read the file contents
    with open('app.py', 'r') as file:
        lines = file.readlines()

    # Remove the last line if it matches the line we added
    if lines[-1].strip() == '# This is a new comment added by save_app function.':
        lines.pop()

    # Save the modified contents back to the file
    with open('app.py', 'w') as file:
        file.writelines(lines)


@app.route('/summarize', methods=['POST'])
def summarize():
    url = request.form['url']

    if (r.exists(f"summary:{url}") == True):
        return jsonify({"summary": r.get(f"summary:{url}")})

    if (r.exists(f"scraped:{url}") == False):
        scrape(url)

    cw = os.getcwd()
    # fd = os.path.join(cw, 'scraped_data')
    # for file in os.listdir(fd):
    #     fs = os.path.join(fd, file)

    data = r.hgetall(f"scraped:{url}")  # this is stored as bytes

    cleaned_data = clean_text_data(data)
    # print(cleaned_data)

    # run a shell command: python keyword_model.pyqs
    loc = os.path.join(cw, 'keybert_model.pkl')
    if not os.path.exists(loc):
        subprocess.run([sys.executable, 'keyword_model.py'])
    summary = generate_summary_keybert_gpt(cleaned_data['full_text'])
    # summary = generate_summary_keybert_gemini(cleaned_data['full_text'])
    # summary = generate_summary_keybert_mixtral(cleaned_data['full_text'])
    r.set(f"summary:{url}", summary)
    save_app()
    return jsonify({"summary": summary})


@app.route('/scrape', methods=["POST"])
def get_scraped_data():
    url = request.form['url']
    if (r.exists(f"scraped:{url}")):
        return jsonify({"text": r.hgetall(f"scraped:{url}")})
    scrape(url)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
