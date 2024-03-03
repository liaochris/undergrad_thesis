#!/usr/bin/env python
# coding: utf-8

# In[322]:
import sys
sys.executable

import ast
import pandas as pd
import numpy as np
from pygit2 import Object, Repository, GIT_SORT_TIME
from pygit2 import init_repository, Patch
from colorama import Fore
from tqdm import tqdm
import swifter
from pandarallel import pandarallel
from bs4 import BeautifulSoup, SoupStrainer
import requests
from urllib.request import urlopen
import re
import os
import time
import requests 
import random

def grabCommits(repo, pr_commits_url):
    try:
        pull_info = "/".join(pr_commits_url.split("/")[-3:-1]).replace("pulls","pull")
        scrape_url = f"https://github.com/{repo}/{pull_info}/commits"
        product = SoupStrainer('div', {'id': 'commits_bucket'})
        sesh = requests.Session() 
        page = sesh.get(scrape_url)
        page_text = str(page.text)
        if "Please wait a few minutes before you try again" in page_text:
            print('pausing, rate limit hit')
            time.sleep(120)
        soup = BeautifulSoup(page.content,parse_only = product,features="html.parser")
        commits = soup.find_all("a", attrs={"id":re.compile(r'commit-details*')})
        commit_urls = [c['href'].split("/")[-1] for c in commits]
        return commit_urls
    except Exception as e:
        error = str(e)
        return error


if __name__ == '__main__':   
    pandarallel.initialize(progress_bar=True)

    val = sys.argv[1]
    folder = sys.argv[2]
    if int(val) < 10:
        val = f"0{val}"
    if int(val) < 100:
        val = f"0{val}"
    if int(val) < 1000:
        val = f"0{val}"

    if folder != "github_data_2324":
        name = 'prEvent'
    else:
        name = 'pullRequestEvent'
    
    if f'prEventCommits000000000{val}.csv' not in os.listdir(f'data/github_clean/{folder}'):
        if folder != 'github_data_2324':
            df_part = pd.read_csv(f'data/github_clean/{folder}/{name}00000000{val}.csv', index_col = 0)
        else:
            df_part = pd.read_parquet(f'data/github_clean/{folder}/{name}00000000{val}.parquet')
        
        df_part['partition'] = val    
        df_part['commit_list'] = df_part.parallel_apply(lambda x: grabCommits(x['repo_name'], x['pr_commits_url']), axis = 1 )
        df_part.to_csv(f'data/github_clean/{folder}/prEventCommits00000000{val}.csv')
    else:
        print(f"prEventCommits000000000{val}.csv is already made for {folder}")

    