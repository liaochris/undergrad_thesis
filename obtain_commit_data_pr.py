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

def grabCommits(repo, pr_commits_url):
    try:
        pull_info = "/".join(pr_commits_url.split("/")[-3:-1]).replace("pulls","pull")
        scrape_url = f"https://github.com/{repo}/{pull_info}/commits"

        product = SoupStrainer('div', {'id': 'commits_bucket'})
    
        page = urlopen(scrape_url)
        soup = BeautifulSoup(page.read(),parse_only = product,features="html.parser")
        commits = soup.find_all("a", attrs={"id":re.compile(r'commit-details*')})
        commit_urls = [c['href'].split("/")[-1] for c in commits]
        return commit_urls
    except Exception as e:
        return e.args[0] if len(e.args)>0 else "error"

def grabCommitsBetter(repo, pr_commits_url):
    res = grabCommits(repo, pr_commits_url)
    i = 1
    while type(res) != list:
        time.sleep(i)
        res = grabCommits(repo, pr_commits_url)
        i+=1
    return res

if __name__ == '__main__':   
    pandarallel.initialize(progress_bar=True)

    val = sys.argv[1]
    if int(val) < 10:
        val = f"0{val}"

    if f'prEventCommits0000000000{val}.csv' not in os.listdir('data/github_clean'):
        df_part = pd.read_csv(f'data/github_clean/prEvent0000000000{val}.csv', index_col = 0)
        df_part['partition'] = val    
        df_part['commit_list'] = df_part.parallel_apply(lambda x: grabCommitsBetter(x['repo_name'], x['pr_commits_url']), axis = 1)    
    
        df_part.to_csv(f'data/github_clean/prEventCommits0000000000{val}.csv')
    else:
        print(f"prEventCommits0000000000{val}.csv is already made")

    