import pandas as pd
import requests
from bs4 import BeautifulSoup
from time import sleep
import random
import csv

articles_df = pd.read_csv("articles-list-links.csv")
url_col = 'Article URL'  

def fetch_article_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=12)
        soup = BeautifulSoup(r.content, "html.parser")
        paragraphs = soup.find_all('p')
        text = "\n".join([p.get_text() for p in paragraphs if len(p.get_text()) > 40])
        return text.strip()
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""

articles_df['Article Text'] = ""

for i, row in articles_df.iterrows():
    url = row[url_col]
    print(f"Scraping [{i+1}/{len(articles_df)}]: {url}")
    text = fetch_article_text(url)
    articles_df.at[i, 'Article Text'] = text
    sleep(random.uniform(1, 2.5))  # Be nice to servers, add delay

articles_df.to_csv("data-raw/articles-raw.csv", index=False, escapechar='\\', quoting=csv.QUOTE_ALL)