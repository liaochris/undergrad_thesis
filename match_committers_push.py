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
import glob

os.environ['NUMEXPR_MAX_THREADS'] = '48'
os.environ['NUMEXPR_NUM_THREADS'] = '36'


# In[2]:


pd.set_option('display.max_columns', None)


# In[3]:
def tryReadParquet(file, cols):
    try:
        return pd.read_parquet(file)[cols]
    except:
        return pd.DataFrame()

commit_list = glob.glob('data/github_commits/parquet/filtered_github_data/*_push_*.parquet')
commit_list.extend(glob.glob('data/github_commits/parquet/github_data_pre_18/*_push_*.parquet'))
commit_list.extend(glob.glob('data/github_commits/parquet/github_data_2324/*_push_*.parquet'))

commit_cols = ['push_id', 'repo_name', 'repo_id', 'commit author name', 'commit author email', 'committer name', 'commmitter email', 'commit sha']

df_push = pd.concat([tryReadParquet(ele, commit_cols) for ele in commit_list]).drop_duplicates()

# In[4]:


# In[7]:


df_committers = df_push.dropna()

df_committers_uq = pd.concat([
    df_committers[['commit author name', 'commit author email']].rename(
        {'commit author name':'name', 'commit author email': 'email'}, axis = 1).drop_duplicates(),
    df_committers[['committer name', 'commmitter email']].rename(
        {'committer name':'name', 'commmitter email': 'email'}, axis = 1).drop_duplicates()]).drop_duplicates()


# In[55]:


df_committers['commit author details'] = df_committers['commit author name'] + "_"+df_committers['commit author email']
df_committers['commit_repo'] = df_committers['commit sha'] + "_" + df_committers['repo_name']
df_author_emails = df_committers[
    df_committers.apply(lambda x: not pd.isnull(x['commit_repo']) and not pd.isnull(x['commit author details']), axis = 1)].groupby(
    'commit author details')[['commit_repo']].agg(list)
df_author_emails['commit_repo'] = df_author_emails['commit_repo'].apply(lambda x: random.sample(x, min(5, len(x))))
dict_author_emails = df_author_emails.to_dict()['commit_repo']


# In[56]:


df_committers['committer details'] = df_committers['committer name'] + "_"+df_committers['commmitter email']
df_committer_emails = df_committers[
     df_committers.apply(lambda x: not pd.isnull(x['commit_repo']) and not pd.isnull(x['committer details']), axis = 1)].groupby(
    'committer details')[['commit_repo']].agg(list)
df_committer_emails['commit_repo'] = df_committer_emails['commit_repo'].apply(lambda x: random.sample(x, min(5, len(x))))
dict_committer_emails = df_committer_emails.to_dict()['commit_repo']


# In[74]:


df_committers_uq['commit_repo'] = df_committers_uq.apply(
    lambda x: dict_author_emails.get(x['name']+"_"+x['email'], np.nan), axis = 1)
df_committers_uq['user_type'] = df_committers_uq['commit_repo'].apply(
    lambda x: 'author' if type(x) == list else 'committer')
df_committers_uq['commit_repo'] = df_committers_uq.apply(
    lambda x: dict_committer_emails.get(x['name']+"_"+x['email'], x['commit_repo']) 
    if type(x['commit_repo']) != list else x['commit_repo'], axis = 1)


# In[95]:

username = "liaochris"
token = os.environ['token']

#existing_pr = pd.read_csv('data/merged_data/committers_info_pr.csv', index_col = 0)[['name','email','committer_info']].dropna().drop_duplicates()
#existing_push = pd.read_csv('data/merged_data/committers_info_push.csv', index_col = 0)[['name','email','committer_info']].dropna().drop_duplicates()
#df_existing = pd.concat([existing_pr,existing_push]).drop_duplicates()
#df_committers_uq = pd.merge(df_committers_uq, df_existing, how = 'left')
#df_committers_uq['committer_info'] = df_committers_uq['committer_info'].apply(lambda x: literal_eval(x) if not pd.isnull(x) else x)
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
indices = np.array_split(df_committers_uq[df_committers_uq['existing'].isna()].index, ncount)
start=0
for i in np.arange(start, ncount, 1):
    print(f"Iter {i}")
    df_committers_uq.loc[indices[i], 'committer_info'] = df_committers_uq.loc[indices[i]].apply(
        lambda x: getCommits(x['commit_repo'],x['user_type']), axis = 1)
    df_committers_uq.to_parquet('data/merged_data/committers_info_push.parquet')

# same email
email_info_dict = df_committers_uq[['email', 'committer_info']].dropna().astype(str).drop_duplicates().set_index('email').to_dict()['committer_info']
df_committers_uq['committer_info'] = df_committers_uq.apply(lambda x: email_info_dict.get(x['email'], np.nan) if \
    type(x) != list else x['committer_info'], axis = 1)

# email trick
ends_with_ind = df_committers_uq[df_committers_uq.apply(lambda x: 
    (x['email'].endswith("@users.noreply.github.com") if not pd.isnull(x['email']) else False) and \
    pd.isnull(x['committer_info']), axis = 1)].index
df_committers_uq.loc[ends_with_ind, 'committer_info'] = df_committers_uq.loc[ends_with_ind, 'email'].apply(lambda x: x.replace("@users.noreply.github.com","").replace("@","").split("+"))
df_committers_uq['committer_info'] = df_committers_uq['committer_info'].apply(lambda x: str(x) if type(x) == list else x)

# same name, same repo -> fill in
df_committers_uq['repo'] = df_committers_uq['commit_repo'].apply(lambda x: x[0].split("_")[1])
df_committers_uq['name_repo'] = df_committers_uq['name']+"_"+df_committers_uq['repo']
name_repo_info = df_committers_uq[['name_repo', 'committer_info']].astype(str).drop_duplicates().dropna().set_index('name_repo').to_dict()['committer_info']
df_committers_uq['committer_info'] = df_committers_uq.apply(lambda x: name_repo_info.get(str(x['name_repo']), np.nan) if \
    type(x['committer_info']) != list else x['committer_info'], axis = 1)
df_committers_uq.drop(['name_repo', 'repo'], axis = 1, inplace = True)
# same name, same organization of repo -> fill in

df_committers_uq['committer_info'] = df_committers_uq['committer_info'].apply(lambda x: np.nan if type(x) == list and len(x) == 0 else x)


## full info grab
df_committers_uq['committer_info'] = df_committers_uq['committer_info'].apply(lambda x: literal_eval(x) if x != 'nan' else x)

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
df_committers_uq.loc[val_inds, 'actor_id'] = df_committers_uq.loc[val_inds, 'committer_info'].apply(lambda x: pd.to_numeric(x[0], errors = 'coerce'))


# just username or just id
val_inds = df_committers_uq[df_committers_uq['committer_info'].apply(lambda x: type(x) == list and len(x) == 1 and 
    pd.isnull(pd.to_numeric(x[0], errors = 'coerce')))].index
df_committers_uq.loc[val_inds, 'actor_login'] = df_committers_uq.loc[val_inds, 'committer_info'].apply(lambda x: x[0])

df_committers_uq.drop('commit_repo', axis = 1).sort_values(['name','email']).to_parquet(
    'data/merged_data/committers_info_push.parquet', engine = 'fastparquet')
