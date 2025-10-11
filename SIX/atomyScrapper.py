import random
from typing import List, Dict, Optional
import re
import requests
import os


from bs4 import BeautifulSoup
import json
import time

from pythonProject.LeetCode.natas11 import DEFAULT_TIMEOUT, BACKOFF_BASE
BASE_URL = "https://tr.atomy.com/main"

BASE_URL2 = 'https://tr.atomy.com/category?dispCtgNo=2504003356&sortType=POPULAR'
session = requests.Session()

session.headers.update({
    "Accept" : "*/*",
    "Accept-Encoding" : "gzip, deflate, br, zstd",
    "Accept-Language" : "tr-TR,tr;q=0.8",
    "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
})

HEADERS = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "tr-TR,tr;q=0.8",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
}

# response = session.get(BASE_URL)
# html = response.text
# soup = BeautifulSoup(html,"html.parser")
# print(soup)

DEFAULT_TIMEOUT
BACKOFF_BASE
MAX_RETRIES = 5
#BRING URL
def fetch_url(url:str) -> Optional[requests.Response]:
    for attempt in (1,MAX_RETRIES+1):
        try:
            res = session.get(url, timeout = DEFAULT_TIMEOUT)
            if res.status_code == 200:
                return res
            else:
                print(f"[UYARI] {url} -> HTTP {res.status_code}")
        except Exception as e:
            print(f"[HATA] {url} {e} hatasi")
    backoff_time = BACKOFF_BASE ** attempt + random.uniform(0,0.5)
    time.sleep(backoff_time)
    return None

def find_product(url):
    response = fetch_url(url)
    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    results = []
    ul = soup.select_one("ul.gdsList-wrap")

    for product in ul.select("li"):
        info = product.select_one("div.gdInfo")
        urun_ad = info.select_one("span.title")
        musteri_fiyat = info.select_one(".gdsPrice .prc .prc_ori b")

        link = info.select_one("a")["href"]
        results.append({
            "urun_ad": urun_ad.get_text(strip=False),
            "musteri_fiyat" : musteri_fiyat.get_text(strip=False),
            "link" : link
        })


    return results



def find_categories(url):
    print("Finding categories...")
    categories = []

    html = fetch_url(url).text
    soup = BeautifulSoup(html,"html.parser")
    # Find all the <a> tags inside the list items
    links = soup.select("li.swiper-slide > a")

    for link in links:
        # Find the image tag within the link
        # alt_text = img_tag['alt'].strip()
        # if alt_text:
        #     categories.append(alt_text)
        img_tag = link.select_one("img")
        onclick_attr = link.get('onclick')

        if img_tag and img_tag.has_attr('alt') and onclick_attr:
            name = img_tag['alt'].strip()

            match = re.search(r"linkUrl:'(.*?)'",onclick_attr)
            if match:
                category_url = match.group(1)

                if category_url.startswith('/'):
                    full_url = f"https://tr.atomy.com{category_url}"
                else:
                    full_url = category_url

                if name and full_url:
                    categories.append({"name":name, "url": full_url})
    return categories



def main():
    categories = find_categories(BASE_URL)

    if not categories:
        print("No categories found. Exiting.")
        return
    print(f"Found {len(categories)} categories. Now scraping products for each...")
    all_data = {}

    for idx,category in enumerate(categories):
        cat_name = category['name']
        cat_url = category['url']


        print(f"--- Scrapping category: {idx+1}:{cat_name} ---")
        # products = find_product(cat_url)
        # all_data[cat_name] = products
        time.sleep(1)
    wanted_cats = [cat.lower().strip() for cat in input("wanted categories >> ").split(",")]
    for idx,category in enumerate(categories):
        cat_name = category['name'].lower()
        cat_url = category['url']
        # print(wanted_cats)
        if cat_name in wanted_cats:
            print(f"--- Scrapping category: {idx + 1}:{cat_name} ---")
            products = find_product(cat_url)
            all_data[cat_name] = products
            time.sleep(2)
            print("\n --- All scraped data --- ")
            print(json.dumps(all_data, indent = 4, ensure_ascii=False))
            output_file = './data/atomyProducts.jsonl'
            with open(output_file, "w", encoding="utf-8") as f:
                json_data = json.dumps(all_data, indent=4, ensure_ascii=False)
                f.write(json_data)
                print(f"Saved in {output_file} ")



if __name__ == "__main__":
    main()
