import random
from textwrap import indent
import json
import requests
from bs4 import BeautifulSoup
import time
import os
from typing import List,Optional,Dict
from pythonProject.LeetCode.natas11 import DEFAULT_TIMEOUT, BACKOFF_BASE, MAX_RETRIES

BASE_URL = 'https://www.sahibinden.com/kategori-vitrin?viewType=Gallery&category=89789'
# BASE_URL = 'https://www.sahibinden.com/klasik-araclar-klasik-otomobiller-lincoln-continental'
# session = requests.Session()
# HEADERS = {
#     "Accept" : "application/json, text/javascript, */*; q=0.01",
#     "Accept-Encoding" : "gzip,deflate,br,zstd",
#     "Accept-Language" : "tr-TR,tr;q=0.6",
#     "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
# }
##GET HEADERS
def get_headers() -> Dict[str, str]:
    """Return a random set of headers to avoid detection"""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/118.0"
        "Educational-Agent"
    ]

    return {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "tr-TR,tr;q=0.8,en-US;q=0.5,en;q=0.3",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Referer": "https://www.sahibinden.com/",
        "Cache-Control": "max-age=0"
    }


def fetch_url(url:str) -> Optional[requests.Response]:
    session = requests.Session()

    for attempt in range(1,MAX_RETRIES+1):
        try:
            # headers = get_headers()
            time.sleep(1.5 + random.uniform(0, 1.5))
            response = session.get(url, timeout = DEFAULT_TIMEOUT,headers = get_headers(),allow_redirects=True)
            if response.status_code == 403:
                print(f"[{attempt}/{MAX_RETRIES}] 403 Forbidden - Trying with different headers...")
                continue
            if response.status_code == 200:
                return response
            else:
                print(f"[{attempt}/{MAX_RETRIES}] HTTP {response.status_code} - Retrying...")
                print(f"[HATA] {url} -> HTTP {response.status_code}")
        except Exception as e:
            print(f"[{attempt}/{MAX_RETRIES}] Request failed: {str(e)}")

        backoff_time = BACKOFF_BASE ** attempt + random.uniform(0,0.5)
        time.sleep(backoff_time)
    print("Give up!!")
    return None

# fetch_url(BASE_URL)
def extract_land(url):
    res = fetch_url(url)
    if not res:
        return []
    soup = BeautifulSoup(res.text, "html.parser")
    lands = soup.find_all("tr", class_ = "searchResultsGalleryRow")
    land_data = []
    for index, land in enumerate(lands):
        try:
            ilan_ad = land.find("a", class_ = "classifiedTitle").text.strip()
            ilan_fiyat = land.find("div", class_ = "searchResultsPriceValue").text.strip()

            yer_zmn = land.find("div", class_ = "searchResultsGallerySubContent")
            details = {}

            for detail in yer_zmn.find_all("div"):
                if "searchResultsGalleryAttributeTitle" in str(detail):
                    etiket = detail.find("span", class_ = "searchResultsGalleryAttributeTitle").text.strip()
                    value = detail.get_text(strip=True).replace(etiket, "").strip(": ").strip()
                    details[etiket] = value
            land_info = {
                "ilan_ad": ilan_ad,
                "fiyat": ilan_fiyat,
                "tarih": details.get("İlan Tarihi", "Bilgi yok"),
                "konum": details.get("İl / İlçe", "Bilgi yok")
            }

            land_data.append(land_info)

        except AttributeError as e:
            print(f"Hata: {e} - İlan bilgileri çıkarılamadı")
            continue


    return land_data


def save_to_json(data: List[Dict],filename:str = "arsaIlnlari.json") -> None:
    with open(filename,"w",encoding = "utf-8") as f:
        json.dump(data, f, ensure_ascii = False, indent = 2)
    print(f"Veriler {filename} dosyasina kaydedildi")

def main():
    print("İlanlar çekiliyor, lütfen bekleyin...")
    lands = extract_land(BASE_URL)

    if not lands:
        print("Hiç ilan bulunamadı veya erişim engellendi.")
        return
    for idx, land in enumerate(lands,1):
        print(f"\n ---ILAN {idx} ----")
        print(f"Başlık: {land['ilan_ad']}")
        print(f"Fiyat: {land['fiyat']}")
        print(f"Tarih: {land['tarih']}")
        print(f"Konum: {land['konum']}")
    save_to_json(lands)
if __name__ == "__main__":
    main()