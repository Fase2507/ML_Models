import requests
import time
import random
from typing import Optional
import undetected_chromedriver as uc  # pip install undetected-chromedriver


# Method 1: Undetected ChromeDriver (Most Effective)
def try_undetected_chrome(url: str) -> Optional[str]:
    """Using undetected-chromedriver which is specifically designed to bypass detection"""
    try:
        print("🔄 Trying undetected-chromedriver...")

        options = uc.ChromeOptions()
        # options.add_argument('--headless')  # Uncomment for headless mode
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        driver = uc.Chrome(options=options)
        driver.get(url)

        # Wait for Cloudflare to finish
        time.sleep(15)

        # Check if we bypassed the challenge
        page_source = driver.page_source
        driver.quit()

        if 'searchResultsGalleryRow' in page_source:
            print("✅ Undetected Chrome bypassed Cloudflare!")
            return page_source
        elif 'Olağandışı bir durum' in page_source:
            print("❌ Still blocked by Cloudflare")
            return None
        else:
            print("⚠️ Got different page structure, checking...")
            return page_source

    except Exception as e:
        print(f"❌ Undetected Chrome error: {str(e)}")
        return None


# Method 2: Using Different Endpoints/Pages
def try_mobile_version(base_url: str) -> str:
    """Try mobile version which often has less protection"""
    mobile_urls = [
        base_url.replace('www.', 'm.'),
        base_url + '&mobile=1',
        base_url.replace('viewType=Gallery', 'viewType=List')
    ]
    return mobile_urls


# Method 3: Using Proxy Rotation
def get_proxy_list():
    """Get a list of free proxies (replace with premium service for better results)"""
    return [
        # Add your proxy list here
        # Format: {'http': 'http://ip:port', 'https': 'https://ip:port'}
    ]


def try_with_proxy_rotation(url: str, proxies_list: list) -> Optional[requests.Response]:
    """Try request with different proxies"""
    import cloudscraper

    for i, proxy in enumerate(proxies_list):
        try:
            print(f"Trying proxy {i + 1}/{len(proxies_list)}: {proxy['http']}")

            scraper = cloudscraper.create_scraper(
                browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
            )

            response = scraper.get(url, proxies=proxy, timeout=30)

            if response.status_code == 200 and 'searchResultsGalleryRow' in response.text:
                print(f"✅ Proxy {i + 1} successful!")
                return response

        except Exception as e:
            print(f"❌ Proxy {i + 1} failed: {str(e)}")
            continue

    return None


# Method 4: Session-based approach with cookies
def try_session_with_cookies(url: str) -> Optional[requests.Response]:
    """Try to establish a session first"""
    try:
        import cloudscraper

        scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
        )

        # First, visit the main page to get cookies
        print("🍪 Getting session cookies...")
        scraper.get('https://www.sahibinden.com/', timeout=30)
        time.sleep(5)

        # Then visit category pages
        category_url = 'https://www.sahibinden.com/kategori/emlak'
        scraper.get(category_url, timeout=30)
        time.sleep(3)

        # Finally try our target URL
        print("🎯 Trying target URL with established session...")
        response = scraper.get(url, timeout=30)

        if response.status_code == 200:
            return response

    except Exception as e:
        print(f"❌ Session approach error: {str(e)}")

    return None


# Method 5: Try different URLs/endpoints
def try_alternative_urls():
    """Different URL patterns that might work"""
    base_patterns = [
        'https://www.sahibinden.com/kategori-vitrin?viewType=List&category=89789',  # List view
        'https://www.sahibinden.com/emlak/arsa',  # Direct category
        'https://www.sahibinden.com/emlak/arsa?pagingOffset=0',  # With offset
        'https://www.sahibinden.com/kategori-vitrin?category=89789',  # No viewType
    ]
    return base_patterns


# Method 6: FlareSolverr (External service)
def try_flaresolverr(url: str, flaresolverr_url: str = 'http://localhost:8191') -> Optional[dict]:
    """Use FlareSolverr service to bypass Cloudflare"""
    try:
        payload = {
            "cmd": "request.get",
            "url": url,
            "maxTimeout": 60000
        }

        response = requests.post(f"{flaresolverr_url}/v1", json=payload)

        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'ok':
                return data.get('solution', {})

        return None

    except Exception as e:
        print(f"❌ FlareSolverr error: {str(e)}")
        return None


# Method 7: Complete bypass strategy
def comprehensive_bypass_attempt(target_url: str) -> Optional[str]:
    """Try all methods in order"""

    methods = [
        ("Undetected Chrome", lambda: try_undetected_chrome(target_url)),
        ("Session with Cookies", lambda: try_session_with_cookies(target_url)),
    ]

    # Try alternative URLs first
    urls_to_try = [target_url] + try_alternative_urls()

    for url in urls_to_try:
        print(f"\n🎯 Trying URL: {url}")

        for method_name, method_func in methods:
            print(f"🔄 Method: {method_name}")
            try:
                result = method_func()
                if result:
                    if isinstance(result, str):
                        return result
                    elif hasattr(result, 'text'):
                        return result.text
            except Exception as e:
                print(f"❌ {method_name} failed: {str(e)}")
                continue

    return None


# Usage example
if __name__ == "__main__":
    target_url = 'https://www.sahibinden.com/kategori-vitrin?viewType=Gallery&category=89789'

    # Install required package first:
    # pip install undetected-chromedriver

    result = comprehensive_bypass_attempt(target_url)

    if result:
        print("✅ Successfully bypassed Cloudflare!")
        with open('successful_response.html', 'w', encoding='utf-8') as f:
            f.write(result)
        print("📄 Response saved to successful_response.html")
    else:
        print("❌ All bypass methods failed")
        print("\n🔧 Next steps:")
        print("1. Try using a premium proxy service")
        print("2. Use FlareSolverr with Docker")
        print("3. Consider using residential proxies")
        print("4. Try during off-peak hours")
        print("5. Use a VPN from a different country")