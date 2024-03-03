#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import glob
from pandarallel import pandarallel
from bs4 import BeautifulSoup, SoupStrainer
import requests
import random
import os
import time

# In[2]:
os.makedirs("data/github_clean/linked_issues", exist_ok=True)

pd.set_option('display.max_columns', None)


# In[3]:


def grabIssue(repo, issue_number):
    try:
        scrape_url = f"https://github.com/{repo}/issues/{issue_number}"
        product = SoupStrainer('a')
        sesh = requests.Session() 
        page = sesh.get(scrape_url)
        page_text = str(page.text)
        if "Please wait a few minutes before you try again" in page_text:
            print('pausing, rate limit hit')
            time.sleep(120)
            page = sesh.get(scrape_url)
            page_text = str(page.text)
            
        soup = BeautifulSoup(page.content,parse_only = product,features="html.parser")
        pr_links = soup.find_all("a", attrs={"data-hovercard-type":'pull_request'})
        pr_link = pr_links[0]['href']
        return pr_link
    except Exception as e:
        error = str(e)
        return error


# In[7]:

files = glob.glob('data/inputs/linked_issues/*')

random.shuffle(files)

for file in files:
    check_issues = pd.read_csv(file, index_col = 0)
    fname = file.split("/")[-1]
    if fname in os.listdir('data/github_clean/linked_issues'):
        continue
    else:
        while check_issues.dropna().shape[0] != check_issues.shape[0]:
            na_df = check_issues[check_issues['linked_pr'].isna()]
            print(f"{na_df.shape[0]} remaining in {file}")
            # update file
            apply_inds = na_df.sample(min(10, na_df.shape[0])).index
            check_issues.loc[apply_inds, 'linked_pr'] = check_issues.loc[apply_inds].apply(
                lambda x: grabIssue(x['repo_name'], x['potential_issues']), axis = 1)
            ## update with new changes
            check_issues.to_csv(f'data/github_clean/linked_issues/{fname}')