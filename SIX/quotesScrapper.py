import csv
import time
import random

from pathlib import Path
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://quotes.toscrape.com/"
START_URL = f"{BASE_URL}/"

#oturum acma istegi by request
session=requests.Session()
session.headers.update({
    "User-Agent":"Educatioanl Scraper"
})

DEFAULT_TIMEOUT = 10
MAX_RETRIES = 3
BACKOFF_BASE = 1.5
#s1 urli getir kontrol et
def fetch(url:str)-> Optional[requests.Response]:
    for attempt in range(1,MAX_RETRIES+1):
        try:
            response = session.get(url, timeout = DEFAULT_TIMEOUT)
            if response.status_code == 200:
                return response
            else:
                print(f"[UYARI] {url} -> HTTP {response.status_code}")
        except:
            print(f"[HATA] {url} istek hatasi")

        backoff_time = BACKOFF_BASE ** attempt + random.uniform(0,0.5)
        time.sleep(backoff_time)
    print("give up!")
    return None
#s2 kazimaya basla
def quotes_parser(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    results = []

    for quote in soup.select("div.quote"):
        text = quote.select_one("span.text")
        author = quote.select_one("small.author")
        tags = quote.select("div.tags a.tag")

        results.append({
            "text" : text.get_text(strip=True) if text else "",
            "author" : author.get_text(strip=True) if author else "",
            "tags" : ','.join(t.get_text(strip=True) for t in tags) if tags else ""
        })
    return results

#s3 sonraki sayfalara bak
def find_next_page(html:str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    next_link = soup.select_one("li.next > a")
    if not next_link or not next_link.get("href"):
        return None
    else:
        return BASE_URL + next_link["href"]

#s4 csv formatinda kaydet

def save_to_csv(rows: List[Dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = ["text", "author", "tags"]
    write_headers = not out_path.exists()

    with out_path.open("a", newline = "",encoding = "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames = field_names)
        if write_headers:
            writer.writeheader()
        writer.writerows(rows)

#s5
def crawl_all_quotes(start_url: str = START_URL, out_csv: str = "./data/output.csv" ):
    current_url = start_url
    csv_path = Path(out_csv)
    total = 0
    page_no = 1

    print(f"Veri kazima basladi...")

    while current_url:
        print(f"[GET] {page_no} kaziniyor")
        resp = fetch(current_url)
        if resp is None:
            print("Durdu!")
            break

        quotes = quotes_parser(resp.text)

        if quotes:
            save_to_csv(quotes, csv_path)
            total+=1
            print(f"[OK] {page_no} : {len(quotes)} kayit eklendi")
        else:
            print("Bu sayfa da soz yok")

        time.sleep(random.uniform(0.8,1.6))

        next_url = find_next_page(resp.text)
        if next_url:
            current_url = next_url
            page_no+=1
        else:
            print("Bitti")
            break

if __name__ == "__main__":
    crawl_all_quotes()