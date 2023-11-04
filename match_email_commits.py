#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.executable


# In[2]:


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
import time
import random
import subprocess
import os


# In[3]:


pandarallel.initialize(progress_bar=True)


# In[9]:


df_emails = pd.DataFrame()
i = 0
for file in os.listdir('data/github_commits/parquet/'):
    try:
        df_file = pd.read_parquet(f'data/github_commits/parquet/{file}')
        df_file = df_file[['commit author email', 'commmitter email', 'actor_login']].drop_duplicates()
        df_emails = pd.concat([df_emails, df_file])
        i+=1
        if i%100 == 0:
            print(i)
    except:
        print(file)


# In[5]:


emails1 = df_emails['commit author email'].unique().tolist()
emails2 = df_emails['commmitter email'].unique().tolist()
emails1.extend(emails2)
emails1 = list(set(emails1))


# In[56]:


username = "liaochris"
token = 


# In[61]:


def getCommits(email):
    api_url = f"https://api.github.com/search/users?q={email}"
    with requests.get(api_url, auth=(username,token)) as url:
        data = url.json()
    print(data)
    if "API rate limit exceeded" in data.get('message', "no message"):
        return "pause"
    if data.get('total_count', 0) == 0:
        return []
    data_items = data['items']
    return [[d['login'], d['id'], d['type']] for d in data_items]


# In[62]:


count = 0
email_data_list = []

start = time.time()
for email in emails1:
    email_data = getCommits(email)
    if email_data != "pause":
        email_data_list.append(email_data)
    count+=1

    if email_data == "pause":
        diff = time.time() - start
        sleep_time = 1 if diff > 3600 else int(3601 - diff)
        print(f"Pausing for {sleep_time} seconds after having obtained {count}") 
        time.sleep(sleep_time)
        start = time.time()
        email_data = getCommits(email)
        email_data_list.append(email_data)


# In[60]:


pd.concat([pd.DataFrame(emails1), pd.DataFrame(email_data_list)], axis = 1).to_csv(
    'data/github_commits/csv/commit_associated_emails.csv')

