import os
import random
import time
from typing import List, Dict, Optional

from asgiref.timeout import timeout
from bs4 import BeautifulSoup
import requests
print("the skills u don't have")
filtered_skills = [skill.lower().strip() for skill in input("DON'T FORGET PUT COMMA ',' >>").split(',')]
print(f"Filtering {filtered_skills} ...")

BASE_URL = "https://m.timesjobs.com/mobile/jobs-search-result.html?jobsSearchCriteria=Information%20Technology&cboPresFuncArea=35"
#
session = requests.Session()
session.headers.update({
    "User-Agent" : "Educational-Agent"
})

DEFAULT_TIMEOUT = 15
MAX_RETRIES = 5
BACKOFF_BASE = 1.5

def fetch_url(url:str)-> Optional[requests.Response]:
    for attempt in range(1,MAX_RETRIES+1):
        try:
            response = session.get(url, timeout = DEFAULT_TIMEOUT)
            if response.status_code == 200:
                return response
            else:
                print(f"[UYARI] {url} -> HTTP {response.status_code}")
        except:
            print(f"[HATA] {url} istek hatasi ")
        backoff_time = BACKOFF_BASE ** attempt + random.uniform(0,0.5)
        time.sleep(backoff_time)
    print("give up!")
    return None

def find_jobs(url):
    # html_text = requests.get(BASE_URL).text
    html_text = fetch_url(url).text
    soup = BeautifulSoup(html_text, 'lxml')

    jobs = soup.find_all('div', class_ = 'srp-listing')

    for index,job in enumerate(jobs):

        publish_date = job.find('span', class_ = 'posting-time').text
        company_name = job.find('span',class_ = 'srp-comp-name').text.strip()

        skill_container = job.find('div', class_ = 'srp-keyskills')
        skills = [skill.text.strip() for skill in skill_container.find_all('a', class_ = 'srphglt')]
        key_skills = ' | '.join(skills).lower()

        more_info = job.find('a', class_ = 'srp-apply-new')['href']

        location = job.find('div', class_ = 'srp-loc').text
        job_entry = (f"""
                     Company name: {company_name},
                     Required skills: {key_skills},
                     Puslish date: {publish_date},
                     Location: {location},
                     More info: {more_info}
                 """)
        # with open("./data/unfiltered_jobs.txt", "w") as file:
        #     file.write(job_entry)
        if not any(skill in key_skills for skill in filtered_skills):
          print(f"Company name: {company_name.strip()}")
          print(f"Required skills: {key_skills.strip()}")
          print(f"Publish date: {publish_date.strip()}")
          print(f"Location: {location.strip()}")
          print(f"More info: {more_info}")
          job_entry = (f"""
      Company name: {company_name},
      Required skills: {key_skills}, 
      Puslish date: {publish_date},
      Location: {location},
      More info: {more_info} """)
          os.makedirs('./data/posts',exist_ok = True)
          with open(f"./data/posts/{index}.txt", "w") as file:
            file.write(job_entry)
            print(f"file saved {index}.txt")
        print('')
if __name__ == "__main__":
    while True:
        find_jobs(BASE_URL)
        time_wait = 10
        print(f"Waiting {time_wait} minutes...")#each three minutes it gets jobs
        time.sleep(time_wait * 60)
