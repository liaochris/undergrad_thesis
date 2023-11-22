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
from pandarallel import pandarallel
import glob
import json
import numpy as np
from itertools import chain
import dask.dataframe as da


# In[2]:


pandarallel.initialize(progress_bar=False)


# In[3]:


pd.set_option('display.max_columns', None)


# In[4]:


df_repo_info = pd.DataFrame()
df_actor_info = pd.DataFrame()
df_org_info = pd.DataFrame()


# In[5]:


df_push = pd.DataFrame()
for i in range(99):
    if i < 10:
        i = f"0{i}"
    df_push_i = pd.read_csv(f'data/github_clean/pushEvent0000000000{i}.csv', index_col = 0)
    df_push = pd.concat([df_push_i[['type', 'created_at', 'repo_id', 'actor_id', 'org_id', 'push_id',
                                    'push_size', 'push_size_distinct', 'push_before', 'push_head']],
                         df_push])
    df_repo_i = df_push_i[['repo_id', 'repo_name']].drop_duplicates()
    df_actor_i = df_push_i[['actor_id', 'actor_login', 'repo_id', 'org_id',]].drop_duplicates()
    df_org_i = df_push_i[['org_id', 'org_login']].drop_duplicates()

    df_repo_info = pd.concat([df_repo_info, df_repo_i]).drop_duplicates()
    df_actor_info = pd.concat([df_actor_info, df_actor_i]).drop_duplicates()
    df_org_info = pd.concat([df_org_info, df_org_i]).drop_duplicates()
print("df_push obtained")

# In[11]:


def cleanParquetPushes(f):
    try:
        df_parquet_repo = pd.read_parquet(f,engine='fastparquet')
        df_parquet_repo['ordering'] = df_parquet_repo.groupby('push_id').cumcount()+1
        df_parquet_repo = df_parquet_repo[['push_id', 'repo_id', 'actor_id','push_size', 'commit_groups',
                                           'commit sha', 'ordering', 'commit author name', 'commit author email',
                                               'committer name','commmitter email','commit message', 'commit additions',
                                           'commit deletions','commit changes total','commit files changed count',
                                           'commit file changes', 'commit time']]
    except:
        df_parquet_repo = pd.DataFrame()
        print(f)
    return df_parquet_repo


# In[ ]:


files = glob.glob("data/github_commits/parquet/*_push_*")
df_parquet_pushes_data = [cleanParquetPushes(f) for f in files]
df_parquet_pushes = pd.concat(df_parquet_pushes_data,ignore_index=True)
print("df_parquet_push obtained")

# In[19]:


df_parquet_pushes['commit_groups'] = df_parquet_pushes['commit_groups'].apply(lambda x: ast.literal_eval if type(x) == str else x)
df_parquet_pushes['commit parent'] = df_parquet_pushes['commit_groups'].apply(lambda x: x[0] if len(x)>0 else '')


# In[20]:


df_parquet_pushes.drop('commit_groups', axis = 1, inplace = True)


# In[21]:


df_push['created_at'] = pd.to_datetime(df_push['created_at'])


# In[22]:


df_push_commits = pd.merge(df_push, df_parquet_pushes, how = 'left',
                           on = ['repo_id', 'push_id', 'actor_id'])
print("df_push_commits made")

# In[23]:


df_push_commits['commit time'] = pd.to_datetime(df_push_commits['commit time'],unit='s')


# In[24]:


df_push_commits['push_day'] = df_push_commits['created_at'].parallel_apply(lambda x: x.day)
df_push_commits['push_month'] = df_push_commits['created_at'].parallel_apply(lambda x: x.month)
df_push_commits['push_year'] = df_push_commits['created_at'].parallel_apply(lambda x: x.year)

df_push_commits['commit_day'] = df_push_commits['commit time'].parallel_apply(lambda x: x.day)
df_push_commits['commit_month'] = df_push_commits['commit time'].parallel_apply(lambda x: x.month)
df_push_commits['commit_year'] = df_push_commits['commit time'].parallel_apply(lambda x: x.year)


# In[25]:


df_push_commits = df_push_commits.rename({'push_size_x':'push_size'}, axis = 1).drop('push_size_y', axis = 1)


# In[26]:


df_push_commits['commit file changes'] = df_push_commits['commit file changes'].apply(
    lambda x: [] if type(x) == float or type(x) == type(None) else x)


# In[27]:


df_push_commits_s = df_push_commits#.sample(100000)


# In[28]:


df_push_commits_s['commit file changes'] = df_push_commits_s['commit file changes'].apply(lambda x: ast.literal_eval if type(x) == str else x)


# In[29]:


def getList(x):
    try:
        return [ele['file'] for ele in x]
    except:
        return [ele['file'] for sublst in x for ele in sublst]


# In[30]:


df_push_commits_s['commit sha na'] = df_push_commits_s['commit sha'].isnull()
df_push_commits_s['commit author name na'] = df_push_commits_s['commit author name'].isnull()


# In[31]:


null_commit_time = df_push_commits_s[df_push_commits_s['commit_year'].isnull()].index
df_push_commits_s.loc[null_commit_time, 
    ['commit_day', 'commit_month', 'commit_year']] = df_push_commits_s.loc[null_commit_time, ['push_day', 'push_month', 'push_year']]

print("df_push_commits_s cleaned")
# In[ ]:


# check to make sure each push is associated with one actor


# In[36]:


def aggData(group_cols):
    df_results = df_push_commits_s.groupby(group_cols, sort=False, observed=True).agg(
        unique_push_actors=('actor_id', 'nunique'),
        unique_push_orgs=('org_id', 'nunique'),
        push_size=('push_size', 'sum'),
        counted_commits=('commit sha', 'count'),
        retrieved_commits=('commit sha', 'count'), # subtract by retrieved_commits_na_count after
        retrieved_commits_na_count=('commit sha na', 'sum'),
        unique_commit_authors=('commit author name', 'nunique'), # subract by unique_commit_authors_na
        unique_commit_authors_na=('commit author name na', 'max'),
        unique_commit_author_emails=('commit author email', 'nunique'),  # subract by unique_commit_authors_na
        unique_committers=('committer name', 'nunique'), # subract by unique_commit_authors_na 
        unique_committer_emails=('commmitter email', 'nunique'),  # subract by unique_commit_authors_na
        LOC_added=('commit additions', 'sum'),
        avg_LOC_added=('commit additions', 'mean'),
        LOC_deleted=('commit deletions', 'sum'),
        avg_LOC_changed=('commit deletions', 'mean'),
        files_changed=('commit files changed count', 'sum'),
        avg_files_changed=('commit files changed count', 'mean'),
        changed_files=('commit file changes', getList),
        uniq_changed_files=('commit file changes',  lambda x: len(getList(x)))
    )
    df_results['retrieved_commits'] = df_results['retrieved_commits'] - df_results['retrieved_commits_na_count']
    df_results['unique_commit_authors'] = df_results['unique_commit_authors'] - df_results['unique_commit_authors_na']
    df_results['unique_commit_author_emails'] = df_results['unique_commit_author_emails'] - df_results['unique_commit_authors_na']
    df_results['unique_committers'] = df_results['unique_committers'] - df_results['unique_commit_authors_na']
    df_results['unique_committer_emails'] = df_results['unique_committer_emails'] - df_results['unique_commit_authors_na']
    df_results.drop(['retrieved_commits_na_count', 'unique_commit_authors_na'], axis = 1, inplace = True)

    return df_results


# In[ ]:

print("Push Monthly")
df_push_commits_grouped_monthly = aggData(['repo_id', 'push_year', 'push_month'])
df_push_commits_grouped_monthly.to_csv('data/aggregated_data/aggregated_monthly_labor.csv')


# In[ ]:

print("Commit Monthly")
df_push_commit_time_grouped_monthly = aggData(['repo_id', 'commit_year', 'commit_month'])
df_push_commit_time_grouped_monthly.to_csv('data/aggregated_data/aggregated_monthly_labor_commit_time.csv')


# In[ ]:

print("Push Daily")
df_push_commits_grouped_daily = aggData(['repo_id', 'push_year', 'push_month', 'push_day'])
df_push_commits_grouped_daily.to_csv('data/aggregated_data/aggregated_daily_labor.csv')


# In[ ]:

print("Commit Daily")
df_push_commit_time_grouped_daily = aggData(['repo_id', 'commit_year', 'commit_month', 'commit_day'])
df_push_commit_time_grouped_daily.to_csv('data/aggregated_data/aggregated_daily_labor_commit_time.csv')


# In[ ]:

df_push_commit_time_grouped_daily.to_csv('data/merged_data/merged_push_commits.csv')

ddf = da.from_pandas(df_push_commits, chunksize=5000000)
save_dir = 'data/merged_data'
ddf.to_parquet(save_dir)

