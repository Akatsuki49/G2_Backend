# Product Description Scraper

## To run 

1. Install the requirements

  ```bash
  pip install -r requirements.txt
  ```
2. Setup redis on docker:

i) Install the docker image of redis

ii) ```
   docker run -d -p 6379:6379 --name rdb redis
     ```
     
iii) ```
   docker start rdb
     ```

2. Navigate to spiders directory inside g2_hack and run - 

  **Run the scraper :**

   ```bash
   scrapy crawl product_description
   ```

3. After you are sure that scraper works properly for all sites, navigate to g2_hack dir - 

  ** Run data_cleaner file:**

   ```bash
   python data_cleaner.py
   ```

4. To run the description generator model

  **First Run :**

  ```bash
  python keyword_model.py
  ```

  **Then run :**

  ```bash 
  python desc_gen.py <path_to_your_cleaned_json_file>
  ```

5. desc.py now has added support of gemini, mistral as well. ive generated 10 summaries each from all 3 models and put them in summaries.txt file. These will be used as reference summaries for eval
