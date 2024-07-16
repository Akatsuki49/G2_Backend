import scrapy
from scrapy.crawler import CrawlerProcess
from newspaper import Article
import os
import json
import redis


def scrape(url):
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)

    if (r.exists(f"scraped:{url}")):
        print("hahahhhh luffy")
        return r.hgetall(f"scraped:{url}")

    class ProductDescriptionSpider(scrapy.Spider):
        name = 'product_description'

        def start_requests(self):
            # url = sys.argv[1]
            yield scrapy.Request(url=url, callback=self.parse, errback=self.errback)

        def parse(self, response):
            if response.status == 200:
                main_content = self.extract_full_text(response)

                about_us_link = response.css(
                    'a::attr(href)').re_first(r'/about-us/?$')
                our_work_link = response.css(
                    'a::attr(href)').re_first(r'/our-work/?$')

                if about_us_link:
                    about_us_url = response.urljoin(about_us_link)
                    yield scrapy.Request(url=about_us_url, callback=self.parse_about_us, meta={'main_content': main_content, 'url': response.url})
                elif our_work_link:
                    our_work_url = response.urljoin(our_work_link)
                    yield scrapy.Request(url=our_work_url, callback=self.parse_our_work, meta={'main_content': main_content, 'url': response.url})
                else:
                    self.log(f"No 'About Us' or 'Our Work' link found on {
                             response.url}")
                    self.parse_content(response.url, main_content)
            else:
                self.log(
                    f"Received non-200 response code: {response.status} for URL: {response.url}")

        def parse_about_us(self, response):
            main_content = response.meta['main_content']
            url = response.meta['url']
            about_us_content = self.extract_full_text(response)

            full_text = main_content + '\n\n' + about_us_content
            self.parse_content(url, full_text)

        def parse_our_work(self, response):
            main_content = response.meta['main_content']
            url = response.meta['url']
            our_work_content = self.extract_full_text(response)

            full_text = main_content + '\n\n' + our_work_content
            self.parse_content(url, full_text)

        def parse_content(self, url, full_text):
            # Extract the title, text, authors, and publish date using newspaper3k
            article = Article(url)
            article.download()
            article.parse()
            title = article.title
            text = article.text
            authors = article.authors
            publish_date = article.publish_date

            # Save the extracted data to a JSON file
            # self.save_to_file(url, title, text, full_text,
            #                   authors, publish_date)
            self.save_to_redis(url, title, text, full_text,
                               authors, publish_date)

            # Print the extracted data to the console
            self.display_data(url, title, text, full_text,
                              authors, publish_date)

        def extract_full_text(self, response):
            text_elements = response.css(
                'p::text, li::text, h1::text, h2::text, h3::text, h4::text, h5::text, '
                'h6::text, span::text, div::text, td::text, th::text'
            ).getall()
            full_text = '. '.join([p.strip()
                                  for p in text_elements if p.strip()])
            return full_text

        def save_to_redis(self, url, title, text, full_text, authors, publish_date):
            authors_json = json.dumps(authors)
            data = {
                'title': title,
                'text': text,
                'full_text': full_text,
                # 'authors': authors_json,
                # 'publish_date': publish_date
            }

            r.hset(f"scraped:{url}", mapping=data)
            print(f"Saved data to Redis!")

        def display_data(self, url, title, text, full_text, authors, publish_date):
            # Print the extracted data to the console in a formatted manner
            print("=" * 80)
            print(f"URL: {url}")
            print(f"Title: {title}")
            print("\nText:")
            print(text)
            print("\nFull Text:")
            self.print_full_text(full_text)
            print(f"\nAuthors: {', '.join(authors)}")
            print(f"Published: {publish_date}")
            print("=" * 80)
            print()

        def print_full_text(self, full_text):
            # Print the full text in a consolidated, paragraph-based format
            paragraphs = [p for p in full_text.split('\n\n') if p.strip()]
            for paragraph in paragraphs:
                print(paragraph)
                print()

        def errback(self, failure):
            # Log error message
            self.log(f"Request failed with exception: {failure}")

            # Print error message to console
            print(f"Request failed with exception: {failure}")

            # You can also retry the request if needed
            if failure.check(HttpError):
                response = failure.value.response
                self.log(f"HttpError on {response.url}")

    process = CrawlerProcess()
    process.crawl(ProductDescriptionSpider)
    process.start()

    return r.hgetall(f"scraped:{url}")

# import scrapy
# from scrapy.crawler import CrawlerProcess
# from newspaper import Article
# import os
# import json

# def scrape(url):

#     class ProductDescriptionSpider(scrapy.Spider):
#         name = 'product_description'

#         def start_requests(self):
#             yield scrapy.Request(url=url, callback=self.parse, errback=self.errback)

#         def parse(self, response):
#             if response.status == 200:
#                 main_content = self.extract_full_text(response)

#                 about_us_link = response.css('a::attr(href)').re_first(r'/about-us/?$')
#                 our_work_link = response.css('a::attr(href)').re_first(r'/our-work/?$')

#                 if about_us_link:
#                     about_us_url = response.urljoin(about_us_link)
#                     yield scrapy.Request(url=about_us_url, callback=self.parse_about_us, meta={'main_content': main_content, 'url': response.url})
#                 elif our_work_link:
#                     our_work_url = response.urljoin(our_work_link)
#                     yield scrapy.Request(url=our_work_url, callback=self.parse_our_work, meta={'main_content': main_content, 'url': response.url})
#                 else:
#                     self.log(f"No 'About Us' or 'Our Work' link found on {response.url}")
#                     self.parse_content(response.url, main_content)
#             else:
#                 self.log(f"Received non-200 response code: {response.status} for URL: {response.url}")

#         def parse_about_us(self, response):
#             main_content = response.meta['main_content']
#             url = response.meta['url']
#             about_us_content = self.extract_full_text(response)

#             full_text = main_content + '\n\n' + about_us_content
#             self.parse_content(url, full_text)

#         def parse_our_work(self, response):
#             main_content = response.meta['main_content']
#             url = response.meta['url']
#             our_work_content = self.extract_full_text(response)

#             full_text = main_content + '\n\n' + our_work_content
#             self.parse_content(url, full_text)

#         def parse_content(self, url, full_text):
#             article = Article(url)
#             article.download()
#             article.parse()
#             title = article.title
#             text = article.text
#             authors = article.authors
#             publish_date = article.publish_date

#             self.save_to_file(url, title, text, full_text, authors, publish_date)
#             self.display_data(url, title, text, full_text, authors, publish_date)

#         def extract_full_text(self, response):
#             text_elements = response.css(
#                 'p::text, li::text, h1::text, h2::text, h3::text, h4::text, h5::text, '
#                 'h6::text, span::text, div::text, td::text, th::text'
#             ).getall()
#             full_text = '. '.join([p.strip() for p in text_elements if p.strip()])
#             return full_text

#         def save_to_file(self, url, title, text, full_text, authors, publish_date):
#             output_dir = 'scraped_data'
#             os.makedirs(output_dir, exist_ok=True)

#             domain = url.split('//')[-1].split('/')[0]

#             filename = f"{domain}.json"
#             file_path = os.path.join(output_dir, filename)

#             data = {
#                 'title': title,
#                 'text': text,
#                 'full_text': full_text,
#                 'authors': authors,
#                 'publish_date': publish_date
#             }
#             with open(file_path, 'w', encoding='utf-8') as f:
#                 json.dump(data, f, ensure_ascii=False, indent=2)

#             print(f"Saved data to {file_path}")

#         def display_data(self, url, title, text, full_text, authors, publish_date):
#             print("=" * 80)
#             print(f"URL: {url}")
#             print(f"Title: {title}")
#             print("\nText:")
#             print(text)
#             print("\nFull Text:")
#             self.print_full_text(full_text)
#             print(f"\nAuthors: {', '.join(authors)}")
#             print(f"Published: {publish_date}")
#             print("=" * 80)
#             print()

#         def print_full_text(self, full_text):
#             paragraphs = [p for p in full_text.split('\n\n') if p.strip()]
#             for paragraph in paragraphs:
#                 print(paragraph)
#                 print()

#         def errback(self, failure):
#             # Log error message
#             self.log(f"Request failed with exception: {failure}")

#             # Print error message to console
#             print(f"Request failed with exception: {failure}")

#             # You can also retry the request if needed
#             # if failure.check(HttpError):
#             #     response = failure.value.response
#             #     self.log(f"HttpError on {response.url}")

#     process = CrawlerProcess()
#     process.crawl(ProductDescriptionSpider)
#     process.start()

# scrape("https://www.g2.com/")
