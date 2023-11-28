#!/usr/bin/env python
# coding: utf-8

# In[98]:


import os
#import cudf.pandas
import pandas as pd
import numpy as np
from ast import literal_eval
from collections import OrderedDict, defaultdict
import yaml
import requests
import time
#cudf.pandas.install()

os.environ['NUMEXPR_MAX_THREADS'] = '48'
os.environ['NUMEXPR_NUM_THREADS'] = '36'


# In[2]:


pd.set_option('display.max_columns', None)


# In[3]:


get_ipython().run_cell_magic('time', '', "df_pr = pd.concat([pd.read_csv('data/merged_data/filtered_github_data_large/merged_commit_pr.csv', index_col = 0),\n                   pd.read_csv('data/merged_data/github_data_pre_18/merged_commit_pr.csv', index_col = 0)])\n\npr_cols = df_pr.columns[0:24].tolist() + df_pr.columns[47:63].tolist() + ['commit_actor_id_list']\npr_data = df_pr[pr_cols].drop_duplicates()\n\ncommit_cols = [df_pr.columns[2]] + df_pr.columns[24:47].tolist() + ['repo_name'] + ['pr_number'] + ['repo_id']\ncommit_data = df_pr[commit_cols].drop_duplicates()")


# In[4]:


for col in ['created_at', 'pr_merged_at', 'pr_closed_at', 'pr_updated_at']:
    pr_data[col] = pd.to_datetime(pr_data[col])
for col in ['pr_assignees', 'pr_requested_reviewers', 'pr_requested_teams']:
    pr_data[col] = pr_data[col].apply(lambda x: literal_eval(x) if type(x) == str else x).apply(lambda x: x if len(x)== 0 else [ele['id'] if type(ele) == dict else ele for ele in x])
for col in ['pr_label', 'pr_actor_id_list', 'pr_assignees_list', 'pr_requested_reviewers_list','pr_requested_teams_list', 
            'pr_actors', 'pr_commit_actors', 'all_pr_actors', 'pr_orgs', 'pr_commit_orgs']:
    pr_data[col] = pr_data[col].apply(lambda x: literal_eval(x) if type(x) != list else x)

if (pr_data['commit_actor_id_list'].apply(lambda x: type(x) == list).mean() != 1):
    pr_data['commit_actor_id_list'] = pr_data['commit_actor_id_list'].apply(lambda x: literal_eval(x) if not pd.isnull(x) else [])


# In[5]:


# error correction
pr_data['pr_orgs'] = pr_data['pr_actor_id_list'].apply(lambda x: [ele.split("|")[2].strip() for ele in x if ele.split("|")[2].strip()  != 'NAN ORG'])
pr_data['pr_commit_orgs'] = pr_data['commit_actor_id_list'].apply(lambda x: [ele.split("|")[1].strip() for ele in x if ele.split("|")[1].strip() != 'NAN ORG'] if type(x) == list else [])
pr_data['all_pr_orgs'] = (pr_data['pr_orgs']+pr_data['pr_commit_orgs']).apply(lambda x: list(set([ele for ele in x if ele != 'NAN ORG'])))


# In[6]:


df_pr_actor = pd.concat([pd.read_csv('data/merged_data/filtered_github_data_large/pr_actor.csv', index_col = 0),
                   pd.read_csv('data/merged_data/github_data_pre_18/pr_actor.csv', index_col = 0)])


# In[7]:


df_committers = pd.concat([
    commit_data[['pr_number', 'repo_name', 'repo_id', 'commit author name', 'commit author email']].drop_duplicates().rename(
        {'commit author name': 'name', 'commit author email': 'email'}, axis = 1),
    commit_data[['pr_number', 'repo_name','repo_id', 'committer name', 'commmitter email']].drop_duplicates().rename(
        {'committer name': 'name', 'commmitter email': 'email'}, axis = 1)]).dropna()



df_committers_uq = df_committers[['name', 'email']].drop_duplicates()


# In[55]:


commit_data['commit author details'] = commit_data['commit author name'] + "_"+commit_data['commit author email']
author_emails_nodup = commit_data[~commit_data['commit author details'].duplicated()]
author_emails_nodup['commit_repo'] = author_emails_nodup['commit sha'] + "_" + author_emails_nodup['repo_name']
dict_author_emails = author_emails_nodup[['commit author details', 'commit_repo']].set_index('commit author details').to_dict()['commit_repo']


# In[56]:


commit_data['committer details'] = commit_data['committer name'] + "_"+commit_data['commmitter email']
committer_emails_nodup = commit_data[~commit_data['committer details'].duplicated()]
committer_emails_nodup['commit_repo'] = committer_emails_nodup['commit sha'] + "_" + committer_emails_nodup['repo_name']
dict_committer_emails = committer_emails_nodup[['committer details', 'commit_repo']].set_index('committer details').to_dict()['commit_repo']


# In[74]:


df_committers_uq['commit_repo'] = df_committers_uq.apply(
    lambda x: dict_author_emails.get(x['name']+"_"+x['email'], np.nan), axis = 1)
df_committers_uq['user_type'] = df_committers_uq['commit_repo'].apply(
    lambda x: 'author' if not pd.isnull(x) else 'committer')
df_committers_uq['commit_repo'] = df_committers_uq.apply(
    lambda x: dict_committer_emails[x['name']+"_"+x['email']] 
    if pd.isnull(x['commit_repo']) else x['commit_repo'], axis = 1)


# In[95]:


username = "liaochris"
token = os.environ['token']


# In[101]:


def getCommits(repo_info, sha, user_type):
    api_url = f"https://api.github.com/repos/{repo_info}/commits/{sha}"
    with requests.get(api_url, auth=(username,token)) as url:
        try:
            data = url.json()
            info = data[user_type]
            time.sleep(.75)
            if info != None:
                return [info['login'], info['id'], info['type'], info['site_admin']]
            return []
        except:
            print(data)
            time.sleep(20)
            try:
                data = url.json()
                info = data[user_type]
                time.sleep(.75)
                if info != None:
                    return [info['login'], info['id'], info['type'], info['site_admin']]
                return []
            except:
                return 'failure'


# In[104]:
ncount = 1000
df_committers_uq.reset_index(drop = True, inplace = True)
indices = np.array_split(df_committers_uq.index, ncount)
start=0
for i in np.arange(start, 100, 1):
    print(f"Iter {i}")
    df_committers_uq.loc[indices[i], 'committer_info'] = df_committers_uq.loc[indices[i]].apply(lambda x: getCommits(x['commit_repo'].split("_")[1], x['commit_repo'].split("_")[0],x['user_type']), axis = 1)
    df_committers_uq.to_csv('data/merged_data/committers_info.csv')
    

# In[ ]: