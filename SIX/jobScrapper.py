import os
import time

from bs4 import BeautifulSoup
import requests
print("the skills u don't have")
filtered_skills = [skill.lower().strip() for skill in input(">>").split(',')]
print(f"Filtering {filtered_skills} ...")

def find_jobs():
    html_text = requests.get("https://m.timesjobs.com/mobile/jobs-search-result.html?jobsSearchCriteria=Information%20Technology&cboPresFuncArea=35").text
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
        with open("./data/unfiltered_jobs.txt", "a") as file:
            file.write(job_entry)
        if not any(skill in key_skills for skill in filtered_skills):
          print(f"Company name: {company_name.strip()}")
          print(f"Required skills: {key_skills.strip()}")
          print(f"Publish date: {publish_date.strip()}")
          print(f"Location: {location.strip()}")
          print(f"More info: {more_info}")
          job_entry = (f""" Company name: {company_name},
          Required skills: {key_skills}, 
          Puslish date: {publish_date},
          Location: {location},
          More info: {more_info} """)
          os.makedirs('./data/posts',exist_ok = True)
          with open(f"./data/posts/{index}.txt", "a") as file:
            file.write(job_entry)
            print(f"file saved {index}.txt")
        print('')
if __name__ == "__main__":
    while True:
        find_jobs()
        time_wait = 10
        print(f"Waiting {time_wait} minutes...")#each three minutes it gets jobs
        time.sleep(time_wait * 60)
