#!/usr/bin/env python
# coding: utf-8

# In[199]:


import pandas as pd
import urllib.request, json 
from tqdm import tqdm
import requests
import os
import sys
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
username = "liaochris"
token = os.environ['token']

file = sys.argv[1].replace('data/github_raw/github_data_2324/','')
df = pd.read_parquet(f'data/github_raw/github_data_2324/{file}', engine = 'pyarrow')

if f'{file}' in os.listdir('data/github_clean/github_data_2324/'):
    print(f"Already cleaned {file}")
    quit()

print(f"Starting to clean {file}")
# In[158]:

tqdm.pandas()

df_push = df[df['type'] == 'PushEvent']

def getCommits(repo_info, before, head, original_urls):
    api_url = f"https://api.github.com/repos/{repo_info}/compare/{before}...{head}"
    try:
        with requests.get(api_url, auth=(username,token)) as url:
            data = url.json()
        commits = data['commits']
        time.sleep(.1)
        return [c['url'] for c in commits]
    except:
        return original_urls


# In[65]:
df_push['push_size'] = pd.to_numeric(df_push['push_size'])
if "contributors" not in sys.argv[1]:
    min20_commits = df_push.query('push_size>20').index
    start = time.time()
    print(len(min20_commits))
    df_push.loc[min20_commits, 'commit_urls'] = df_push.query('push_size>20').apply(lambda x: getCommits(x['repo_name'], x['push_before'], x['push_head'], x['commit_urls']), axis = 1)
    end = time.time()
    print(3600/((end-start)/len(min20_commits)))
# In[66]:


df_push[['type', 'created_at', 'repo_id', 'repo_name', 'actor_id', 'actor_login', 'org_id', 'org_login',
         'push_id', 'push_size', 'push_size_distinct', 'push_before', 'push_head', 'push_ref', 'commit_urls']].to_parquet(
    f'data/github_clean/github_data_2324/{file}')
