
from flask import Flask, request, jsonify
from spiders.product_description_spider import scrape_website
from data_cleaner import clean_text_data
from desc_gen import generate_summary
import subprocess


app = Flask(__name__)


@app.route('/summarize', methods=['POST'])
def summarize():
    print("hi")
    url = request.form['url']
    scraped_data = scrape_website(url)
    cleaned_data = clean_text_data(scraped_data)
    # run a shell command: python keyword_model.py
    subprocess.run(['python', 'keyword_model.py'])
    summary = generate_summary(cleaned_data)
    return jsonify({'summary': summary})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
