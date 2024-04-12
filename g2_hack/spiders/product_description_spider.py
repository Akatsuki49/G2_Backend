import scrapy
from newspaper import Article
from scrapy.crawler import CrawlerProcess
import os
import json


def scrape_website(url):
    class ProductDescriptionSpider(scrapy.Spider):
        name = 'product_description'

        def start_requests(self):
            yield scrapy.Request(url=url, callback=self.parse)

        def parse(self, response):
            # Extract the main content from the URL
            main_content = self.extract_full_text(response)

            # Find the "About Us" link
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
                self.log(
                    f"No 'About Us' or 'Our Work' link found on {response.url}")
                yield self.parse_content(response.url, main_content)

        def parse_about_us(self, response):
            main_content = response.meta['main_content']
            url = response.meta['url']
            about_us_content = self.extract_full_text(response)

            full_text = main_content + '\n\n' + about_us_content
            yield self.parse_content(url, full_text)

        def parse_our_work(self, response):
            main_content = response.meta['main_content']
            url = response.meta['url']
            our_work_content = self.extract_full_text(response)

            full_text = main_content + '\n\n' + our_work_content
            yield self.parse_content(url, full_text)

        def parse_content(self, url, full_text):
            # Extract the title, text, authors, and publish date using newspaper3k
            article = Article(url)
            article.download()
            article.parse()
            title = article.title
            text = article.text
            authors = article.authors
            publish_date = article.publish_date

            # Return the scraped data
            return {
                'url': url,
                'title': title,
                'text': text,
                'full_text': full_text,
                'authors': authors,
                'publish_date': publish_date
            }

        def extract_full_text(self, response):
            # Use Scrapy's built-in selectors to extract the complete text content of the website
            text_elements = response.css(
                'p::text, li::text, h1::text, h2::text, h3::text, h4::text, h5::text, '
                'h6::text, span::text, div::text, td::text, th::text'
            ).getall()
            full_text = '. '.join([p.strip()
                                  for p in text_elements if p.strip()])
            return full_text

    process = CrawlerProcess(settings={
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
    })

    results = []

    def collect_output(item, response, spider):
        results.append(item)

    process.crawl(ProductDescriptionSpider)
    process.start(stop_after_crawl=True)

    return results


# Example usage:
# url = "https://www.chattechnologies.com/"
# scraped_data = scrape_website(url)
# print(len(scraped_data))
