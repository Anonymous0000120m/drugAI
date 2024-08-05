import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import argparse

class WebCrawler:
    def __init__(self, base_url, max_pages=100, output_dir='downloaded_pages'):
        self.base_url = base_url
        self.max_pages = max_pages
        self.visited = set()
        self.pages_crawled = 0
        self.output_dir = output_dir
        self.pages_file = os.path.join(self.output_dir, 'pages.txt')
        
        os.makedirs(self.output_dir, exist_ok=True)

    def download_page(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()  # 抛出HTTP错误
            page_content = response.text
            page_title = self.get_page_title(page_content)
            filename = self.generate_filename(url, page_title)
            self.save_page(filename, page_content)
            return page_content
        except requests.RequestException as e:
            print(f"Failed to retrieve page: {url} - Error: {e}")
        return None

    def get_page_title(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        title_tag = soup.find('title')
        return title_tag.text.strip() if title_tag else 'untitled'

    def generate_filename(self, url, title):
        # Sanitize filename
        parsed_url = urlparse(url)
        sanitized_title = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in title).rstrip()
        filename = f"{sanitized_title}_{parsed_url.netloc}.html"
        return os.path.join(self.output_dir, filename)

    def save_page(self, filename, content):
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Page saved: {filename}")
        self.save_link(filename)

    def save_link(self, url):
        with open(self.pages_file, 'a') as f:
            f.write(url + "\n")

    def extract_links(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        links = {urljoin(self.base_url, anchor['href']) for anchor in soup.find_all('a', href=True)}
        return {link for link in links if self.is_valid_link(link)}

    def is_valid_link(self, url):
        parsed_url = urlparse(url)
        return parsed_url.scheme in ('http', 'https') and url not in self.visited

    def crawl(self, url):
        if self.pages_crawled >= self.max_pages:
            return

        print(f"Crawling: {url}")
        page_content = self.download_page(url)
        if page_content:
            self.visited.add(url)
            self.pages_crawled += 1
            links = self.extract_links(page_content)
            for link in links:
                if self.pages_crawled >= self.max_pages:
                    break
                if link not in self.visited:
                    self.crawl(link)

    def start(self):
        self.crawl(self.base_url)


def main():
    parser = argparse.ArgumentParser(description="Web Crawler")
    parser.add_argument("url", help="The base URL of the website to crawl.")
    parser.add_argument("--max-pages", type=int, default=100, help="The maximum number of pages to crawl.")
    parser.add_argument("--output-dir", default="downloaded_pages", help="Directory to save downloaded pages.")
    args = parser.parse_args()
    
    crawler = WebCrawler(args.url, max_pages=args.max_pages, output_dir=args.output_dir)
    crawler.start()


if __name__ == "__main__":
    main()

#python script.py https://example.com --max-pages 10 --output-dir my_downloads
