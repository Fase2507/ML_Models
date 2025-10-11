import queue

from bs4 import BeautifulSoup
import requests
import os
import time
import json


# BASE_URL = 'https://www.walmart.com/ip/SAMSUNG-27-Odyssey-G55C-QHD-165Hz-1ms-MPRT-Curved-Gaming-Monitor-LS27CG556ENXZA/5329046686?classType=REGULAR&athbdg=L1600'
BASE_URL = 'https://www.walmart.com/sp/track?bt=1&eventST=click&plmt=sb-search-top~desktop~&pos=2&tax=3944_1089430_3951_8835131_1737838&rdf=1&rd=https%3A%2F%2Fwww.walmart.com%2Fip%2FHP-Stream-14-inch-Windows-Laptop-Intel-Processor-N150-4GB-128GB-eMMC-Pink-12-mo-Microsoft-365-included%2F13982958746%3FadsRedirect%3Dtrue&adUid=02075da7-3c4b-48da-872c-f716aaa14a99&mloc=sb-search-top&pltfm=desktop&pgId=computers&pt=search&spQs=QA2x2W6r00wWwRpURLk6p1FrcB_lZzEIS0VNnYCm4ygLWldAm4LawgCDJG-5-_Qg0PfTvbUeftqc1-EAOfU5FoiJgYY6QmM9Jst7e8Fgji6gHpH_d-ZAXrQPlDx-shoItTUqm-IgGuZhH3nf2zsRbd-hzTmVVbkzh1MCjp6ahOsZwxiWS4e7veNg78iJEyk62vY0DVaoruKkBavJfUmmz62_OGwIwvmO4kr67Oh1h4DEpaHuBXgaqX8iCTVgPzlc&storeId=3081&specificity=broad&specificityScore=0.1503576&couponState=na&bkt=ace1_default%7Cace2_default%7Cace3_default&/ip/HP-Stream-14-inch-Windows-Laptop-Intel-Processor-N150-4GB-128GB-eMMC-Pink-12-mo-Microsoft-365-included/13982958746'
HEADERS = {
    "Accept" : "*/*",
    "Accept-Encoding" : "gzip, deflate, br, zstd",
    "Accept-Language" : "en-US,en;q=0.9",
    "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
}
wanted_products = [product.lower().strip() for product in input(">>").split(",")]

# search_queries = ["computer","monitor","laptops","mouse","desktops","keyboard"]
product_queue = queue.Queue()
seen_urls = set()

def get_product_links(query="monitor",page_no = 1):
    search_url = f"https://www.walmart.com/search?q={query}&page={page_no}"
    response = requests.get(search_url, headers=HEADERS)

    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all('a', href=True)

    product_links = []

    for link in links:
        link_href = link["href"]
        if '/ip/' in link_href:
            if "https" in link_href:
                full_url = link_href
            else:
                full_url = "https://walmart.com" + link_href
            product_links.append(full_url)

    return product_links


def extract_product(url):
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    script_tag = soup.find("script", id = "__NEXT_DATA__")

    data = json.loads(script_tag.string)#u can go token counter and find how many tokens are here.
    initial_data = data["props"]["pageProps"]["initialData"]["data"]
    product_data = initial_data["product"]
    reviews_data = initial_data.get('reviews', {})

    product_info = {
        "price" : product_data["priceInfo"]["currentPrice"]["price"],
        "review_count" : reviews_data.get("totalReviewCount", 0),
        "avg_rating" : reviews_data.get("averageOverallRating", 0),
        "product_name" : product_data["name"],
        "brand" : product_data.get("brand",""),
        "availability" : product_data["availabilityStatus"],
        "image_url" : product_data["imageInfo"]["thumbnailUrl"],
        "url" : url
    }

    return product_info

def main():
   OUTPUT_FILE = "./data/product_info.jsonl"

   with open(OUTPUT_FILE, "w") as f:
        while wanted_products:
            current_query = wanted_products.pop(0)
            print("\n\nCurrent query ", current_query, "\n\n")
            page_no = 1

            while True:

                links = get_product_links(current_query,page_no)
                if not links or page_no>=3:
                    break

                for link in links:
                    if link not in seen_urls:
                        product_queue.put(link)
                        seen_urls.add(link)

                while not product_queue.empty():
                    product_url = product_queue.get()
                    try:
                        product_info = extract_product(product_url)
                        if product_info:
                            f.write(json.dumps(product_info)+"\n")
                    except Exception as e:
                        print(f"failed to process URL {product_url} Error {e} ")
                page_no += 1
                print(f"Searching page {page_no}")
if __name__ == "__main__":
    main()
