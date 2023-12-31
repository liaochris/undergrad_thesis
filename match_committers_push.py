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
import random

os.environ['NUMEXPR_MAX_THREADS'] = '48'
os.environ['NUMEXPR_NUM_THREADS'] = '36'


# In[2]:


pd.set_option('display.max_columns', None)


# In[3]:


df_push = pd.concat([pd.read_csv('data/merged_data/filtered_github_data_large/merged_commit_push.csv', index_col = 0),
     pd.read_csv('data/merged_data/github_data_pre_18/merged_commit_push.csv', index_col = 0)]).drop_duplicates()

## redo indices to get repo_name
push_cols = df_push.columns[0:11].tolist() + df_push.columns[25:28].tolist()
push_data = df_push[push_cols].drop_duplicates().reset_index(drop = True)

commit_cols = [df_push.columns[2]] + df_push.columns[11:25].tolist() + df_push.columns[28:32].tolist() + ['push_id'] + ['repo_name']
commit_data = df_push[commit_cols].drop_duplicates().reset_index(drop = True)
# In[4]:

subprocess.run(['ls', '-lh', 'data/github_raw/github_data_pre_18'])

for col in ['created_at']:
    push_data[col] = pd.to_datetime(push_data[col])


push_data.to_csv('data/merged_data/cleaned_push_data.csv')
commit_data.to_csv('data/merged_data/cleaned_push_commit_data.csv')

# In[7]:


df_committers = pd.concat([
    commit_data[['push_id', 'repo_name', 'repo_id', 'commit author name', 'commit author email']].drop_duplicates().rename(
        {'commit author name': 'name', 'commit author email': 'email'}, axis = 1),
    commit_data[['push_id', 'repo_name','repo_id', 'committer name', 'commmitter email']].drop_duplicates().rename(
        {'committer name': 'name', 'commmitter email': 'email'}, axis = 1)]).dropna()

df_committers_uq = df_committers[['name', 'email']].drop_duplicates()


# In[55]:


commit_data['commit author details'] = commit_data['commit author name'] + "_"+commit_data['commit author email']
commit_data['commit_repo'] = commit_data['commit sha'] + "_" + commit_data['repo_name']
df_author_emails = commit_data[
    commit_data.apply(lambda x: pd.isnull(x['commit_repo']) or pd.isnull(x['commit author details']), axis = 1)].groupby(
    'commit author details')[['commit_repo']].agg(list)
df_author_emails['commit_repo'] = df_author_emails['commit_repo'].apply(lambda x: random.sample(x, min(5, len(x))))
dict_author_emails = df_author_emails.to_dict()['commit_repo']


# In[56]:


commit_data['committer details'] = commit_data['committer name'] + "_"+commit_data['commmitter email']
df_committer_emails = commit_data[
    commit_data.apply(lambda x: pd.isnull(x['commit_repo']) or pd.isnull(x['committer details']), axis = 1)].groupby(
    'committer details')[['commit_repo']].agg(list)
df_committer_emails['commit_repo'] = df_committer_emails['commit_repo'].apply(lambda x: random.sample(x, min(5, len(x))))
dict_committer_emails = df_committer_emails.to_dict()['commit_repo']


# In[74]:


df_committers_uq['commit_repo'] = df_committers_uq.apply(
    lambda x: dict_author_emails.get(x['name']+"_"+x['email'], np.nan), axis = 1)
df_committers_uq['user_type'] = df_committers_uq['commit_repo'].apply(
    lambda x: 'author' if type(x) == list else 'committer')
df_committers_uq['commit_repo'] = df_committers_uq.apply(
    lambda x: dict_committer_emails[x['name']+"_"+x['email']] 
    if type(x['commit_repo']) != list else x['commit_repo'], axis = 1)


# In[95]:

username = "liaochris"
token = os.environ['token']


# In[101]:


def getCommits(commit_repo, user_type):
    success = False
    i = 0
    while (not success) or i < len(commit_repo):
        repo_info = commit_repo[i].split("_")[1]
        sha = commit_repo[i].split("_")[0]
        api_url = f"https://api.github.com/repos/{repo_info}/commits/{sha}"
        with requests.get(api_url, auth=(username,token)) as url:
            try:
                data = url.json()
                info = data[user_type]
                time.sleep(.75)
                if info != None:
                    success = True
                    return [info['login'], info['id'], info['type'], info['site_admin']]
                i+=1
            except:
                print(data)
                i+=1
        return np.nan

# In[104]:
ncount = 1000
df_committers_uq.reset_index(drop = True, inplace = True)
indices = np.array_split(df_committers_uq.index, ncount)
start=2
for i in np.arange(start, ncount, 1):
    print(f"Iter {i}")
    df_committers_uq.loc[indices[i], 'committer_info'] = df_committers_uq.loc[indices[i]].apply(
        lambda x: getCommits(x['commit_repo'],x['user_type']), axis = 1)
    df_committers_uq.to_csv('data/merged_data/committers_info_push.csv')
    

# same email
email_info_dict = df_committers_uq[['email', 'committer_info']].dropna().drop_duplicates().astype(str).set_index('email').to_dict()['committer_info']
df_committers_uq['committer_info'] = df_committers_uq.apply(lambda x: email_info_dict.get(x['email'], np.nan) if \
    type(x) != list else x['committer_info'], axis = 1)

# email trick
ends_with_ind  = df_committers_uq[df_committers_uq.apply(lambda x: 
    (x['email'].endswith("@users.noreply.github.com") if not pd.isnull(x['email']) else False) and \
    pd.isnull(x['committer_info']), axis = 1)].index
df_committers_uq.loc[ends_with_ind, 'committer_info'] = df_committers_uq.loc[ends_with_ind, 'email'].apply(lambda x: x.replace("@users.noreply.github.com","").replace("@","").split("+"))
df_committers_uq['committer_info'] = df_committers_uq['committer_info'].apply(lambda x: str(x) if type(x) == list else x)

# same name, same repo -> fill in
df_committers_uq['repo'] = df_committers_uq['commit_repo'].apply(lambda x: x.split("_")[1])
df_committers_uq['name_repo'] = df_committers_uq['name']+"_"+df_committers_uq['repo']
name_repo_info = df_committers_uq[['name_repo', 'committer_info']].drop_duplicates().dropna().set_index('name_repo').to_dict()['committer_info']
df_committers_uq['committer_info'] = df_committers_uq.apply(lambda x: name_repo_info.get(x['name_repo'], np.nan) if \
    pd.isnull(x['committer_info']) else x['committer_info'], axis = 1)
df_committers_uq.drop(['name_repo', 'repo'], axis = 1, inplace = True)
# same name, same organization of repo -> fill in

df_committers_uq['committer_info'] = df_committers_uq['committer_info'].apply(lambda x: literal_eval(x) if type(x) == str else x)
df_committers_uq['committer_info'] = df_committers_uq['committer_info'].apply(lambda x: np.nan if type(x) == list and len(x) == 0 else x)


## full info grab
val_inds = df_committers_uq[df_committers_uq['committer_info'].apply(lambda x: type(x) == list and len(x) == 4)].index
df_committers_uq.loc[val_inds, 'actor_login'] = df_committers_uq.loc[val_inds, 'committer_info'].apply(lambda x: x[0])
df_committers_uq.loc[val_inds, 'actor_id'] = df_committers_uq.loc[val_inds, 'committer_info'].apply(lambda x: x[1])
df_committers_uq.loc[val_inds, 'user_type'] = df_committers_uq.loc[val_inds, 'committer_info'].apply(lambda x: x[2])
df_committers_uq.loc[val_inds, 'site_admin'] = df_committers_uq.loc[val_inds, 'committer_info'].apply(lambda x: x[3])

## username and id
three_ind = df_committers_uq[df_committers_uq['committer_info'].apply(lambda x: type(x) == list and len(x)==3)]['committer_info'].index
df_committers_uq.loc[three_ind, 'committer_info'] = pd.Series([[7869818, 'rayrrr']])

val_inds = df_committers_uq[df_committers_uq['committer_info'].apply(lambda x: type(x) == list and len(x) == 2 and x[0] != '{ID}')].index
# fix corrupted cases
df_committers_uq.loc[val_inds, 'actor_login'] = df_committers_uq.loc[val_inds, 'committer_info'].apply(lambda x: x[1])
df_committers_uq.loc[val_inds, 'actor_id'] = df_committers_uq.loc[val_inds, 'committer_info'].apply(lambda x: pd.to_numeric(x[0]))


# just username or just id
val_inds = df_committers_uq[df_committers_uq['committer_info'].apply(lambda x: type(x) == list and len(x) == 1 and 
    pd.isnull(pd.to_numeric(x[0], errors = 'coerce')))].index
df_committers_uq.loc[val_inds, 'actor_login'] = df_committers_uq.loc[val_inds, 'committer_info'].apply(lambda x: x[0])

df_committers_uq.drop('commit_repo', axis = 1).to_csv('data/merged_data/committers_info_push.csv')