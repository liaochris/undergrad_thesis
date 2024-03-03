#!/usr/bin/env python
# coding: utf-8

# ## Step 1: Exogenous Shock
# 1. Create measure of code quality
#    
#     a. Measure of user-side code quality
#    
#     b. Measure of maintainer-side code quality
# 3. Analyze contributions made
# 4. Hypothesize how my instrument/exogenous affects either, and examine the empirical effect
# 
# ## Step 2: 
# How do I actually perform analysis?
# 1. Measure 1: Compare within repository groups
# 2. Measure 2: Find some other way to weight what a "download" means? 

# In[1]:


import os
import pandas as pd
import ast
#from pandarallel import pandarallel
import glob
import json
import sys
import numpy as np
from itertools import chain
import dask.dataframe as da
import numpy as np


# In[2]:


#pandarallel.initialize(progress_bar=False)


# In[3]:




# In[4]:


folder = sys.argv[1]
print("reading push data")
df_push = pd.DataFrame()
file_count = np.array([int(ele.replace("pushEvent000000000","").replace(".csv","").replace(".parquet","")) for ele in os.listdir(f'data/github_clean/{folder}/') if 'pushEvent000000000' in ele])

for i in range(max(file_count)+1):
    if int(i) < 10:
        i = f"0{i}"
    if int(i) < 100:
        i = f"0{i}"
    if int(i) < 1000:
        i = f"0{i}"
    
    try:
        if folder == 'github_data_2324':
            df_push_i = pd.read_parquet(f'data/github_clean/{folder}/pushEvent00000000{i}.parquet')    
        else:
            df_push_i = pd.read_csv(f'data/github_clean/{folder}/pushEvent00000000{i}.csv', index_col = 0)
        df_push = pd.concat([df_push_i[['type', 'created_at', 'repo_name', 'repo_id', 'actor_id', 'org_id', 'push_id',
                                        'push_size', 'push_size_distinct', 'push_before', 'push_head']],
                             df_push])
    except:
        print(f'data/github_clean/{folder}/pushEvent00000000{i} not found')


# In[5]:


def cleanParquetPushes(f):
    try:
        df_parquet_repo = pd.read_parquet(f)
        df_parquet_repo['ordering'] = df_parquet_repo.groupby('push_id').cumcount()+1
        df_parquet_repo = df_parquet_repo[['push_id', 'repo_id', 'repo_name', 'actor_id','push_size', 'commit_groups',
                                           'commit sha', 'ordering', 'commit author name', 'commit author email',
                                               'committer name','commmitter email','commit message', 'commit additions',
                                           'commit deletions','commit changes total','commit files changed count',
                                           'commit file changes', 'commit time']]
    except:
        df_parquet_repo = pd.DataFrame()
        print(f)
    return df_parquet_repo


# In[6]:


files = glob.glob(f"data/github_commits/parquet/{folder}/*_push_*")
df_parquet_pushes_data = [cleanParquetPushes(f) for f in files]
df_parquet_pushes = pd.concat(df_parquet_pushes_data,ignore_index=True)
print("done reading data")

# In[7]:


df_parquet_pushes['commit_groups'] = df_parquet_pushes['commit_groups'].apply(lambda x: ast.literal_eval if type(x) == str else x)
df_parquet_pushes['commit parent'] = df_parquet_pushes['commit_groups'].apply(lambda x: x[0] if len(x)>0 else '')


# In[8]:


df_parquet_pushes.drop('commit_groups', axis = 1, inplace = True)


# In[9]:


df_push['created_at'] = pd.to_datetime(df_push['created_at'])


# In[10]:


df_push_commits = pd.merge(df_push, df_parquet_pushes, how = 'left',
                           on = ['repo_id', 'push_id', 'actor_id'])


# In[11]:


df_push_commits['commit time'] = pd.to_datetime(df_push_commits['commit time'],unit='s')


# In[12]:


df_push_commits['push_day'] = df_push_commits['created_at'].apply(lambda x: x.day)
df_push_commits['push_month'] = df_push_commits['created_at'].apply(lambda x: x.month)
df_push_commits['push_year'] = df_push_commits['created_at'].apply(lambda x: x.year)

df_push_commits['commit_day'] = df_push_commits['commit time'].apply(lambda x: x.day)
df_push_commits['commit_month'] = df_push_commits['commit time'].apply(lambda x: x.month)
df_push_commits['commit_year'] = df_push_commits['commit time'].apply(lambda x: x.year)


# In[13]:


df_push_commits = df_push_commits.rename({'push_size_x':'push_size'}, axis = 1).drop('push_size_y', axis = 1)


# In[14]:


df_push_commits['commit file changes'] = df_push_commits['commit file changes'].apply(
    lambda x: [] if type(x) == float or type(x) == type(None) else x)


# In[15]:


df_push_commits_s = df_push_commits#.sample(100000)


# In[16]:


null_commit_time = df_push_commits_s[df_push_commits_s['commit_year'].isnull()].index
df_push_commits_s.loc[null_commit_time, 
    ['commit_day', 'commit_month', 'commit_year']] = df_push_commits_s.loc[null_commit_time, ['push_day', 'push_month', 'push_year']]


# In[17]:


# In[18]:


df_push_commits_s['committer info'] = df_push_commits_s['committer name'] + " | " + df_push_commits_s['commmitter email']


# In[19]:


def getList(x):
    try:
        return [ele['file'] for ele in x]
    except:
        return [ele['file'] for sublst in x for ele in sublst]


# In[20]:


df_push_commits_s['commit file changes'] = df_push_commits_s['commit file changes'].apply(lambda x: x.decode() if type(x) == bytes else x)


# In[21]:


del df_push
del df_parquet_pushes
del df_push_commits


# In[ ]:


df_push_commits_s['commit file changes'] = df_push_commits_s['commit file changes'].apply(lambda x: ast.literal_eval(x) if type(x) == str else x)


# In[ ]:


# check to make sure each push is associated with one actor


# In[ ]:


def dropNAUnique(x):
    return x.dropna().unique().tolist()


# In[ ]:


df_push_commits_s['push_size_wt'] = df_push_commits_s['push_size'] / df_push_commits_s.groupby('push_id')['push_id'].transform('count')


# In[ ]:


def aggData(group_cols):
    
    df_results = df_push_commits_s.groupby(group_cols, sort=False, observed=True).agg(
        unique_push_actors=('actor_id', 'nunique'),
        unique_push_orgs=('org_id', 'nunique'),
        push_size=('push_size_wt', 'sum'),
        counted_commits=('push_head', 'count'),
        retrieved_commits=('commit sha', 'count'),
        unique_commit_authors=('commit author name', 'nunique'), 
        unique_commit_author_emails=('commit author email', 'nunique'),
        unique_committers=('committer name', 'nunique'),
        unique_committer_emails=('commmitter email', 'nunique'),
        commit_authors=('commit author name', dropNAUnique),
        committers=('committer info', dropNAUnique),
        LOC_added=('commit additions', 'sum'),
        avg_LOC_added=('commit additions', 'mean'),
        LOC_deleted=('commit deletions', 'sum'),
        avg_LOC_deleted=('commit deletions', 'mean'),
        files_changed=('commit files changed count', 'sum'),
        avg_files_changed=('commit files changed count', 'mean'),
        changed_files=('commit file changes', getList),
        uniq_changed_files=('commit file changes',  lambda x: len(getList(x)))
    )    
    return df_results


# In[ ]:
print("starting to export data")
df_push_commits_s.to_parquet(f'data/merged_data/{folder}/merged_commit_push.parquet')

# In[ ]:


"""%%time
df_push_commits_grouped_monthly = aggData(['repo_id', 'push_year', 'push_month'])
df_push_commits_grouped_monthly.to_csv('data/aggregated_data/aggregated_monthly_labor.csv', encoding='utf-8')"""


# In[ ]:


"""%%time
df_push_commit_time_grouped_monthly = aggData(['repo_id', 'commit_year', 'commit_month'])
df_push_commit_time_grouped_monthly.to_csv('data/aggregated_data/aggregated_monthly_labor_commit_time.csv', encoding='utf-8')"""


# In[ ]:


"""%%time
df_push_commits_grouped_daily = aggData(['repo_id', 'push_year', 'push_month', 'push_day'])
df_push_commits_grouped_daily.to_csv('data/aggregated_data/aggregated_daily_labor.csv', encoding='utf-8')"""


# In[ ]:


"""%%time
df_push_commit_time_grouped_daily = aggData(['repo_id', 'commit_year', 'commit_month', 'commit_day'])
df_push_commit_time_grouped_daily.to_csv('data/aggregated_data/aggregated_daily_labor_commit_time.csv', encoding='utf-8')"""


# In[ ]:


# In[ ]:




