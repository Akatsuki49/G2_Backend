
from flask import Flask, request, jsonify
from spiders.product_description_spider import scrape
from data_cleaner import clean_text_data
from desc_gen import generate_summary
import subprocess
import os


app = Flask(__name__)


@app.route('/summarize', methods=['POST'])
def summarize():
    print("hi")
    url = request.form['url']
    scrape(url)
    cw = os.getcwd()
    fd = os.path.join(cw,'spiders/scraped_data')
    for file in fd:
        fs = os.path.join(fd,file)
    cleaned_data = clean_text_data(fs)
    # run a shell command: python keyword_model.py
    subprocess.run(['python', 'keyword_model.py'])
    summary = generate_summary(cleaned_data['full_text'])
    return jsonify({'summary': summary})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
