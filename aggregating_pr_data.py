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
import numpy as np
from itertools import chain
import sys


# In[2]:


#pandarallel.initialize(progress_bar=False)


# In[3]:


pd.set_option('display.max_columns', None)


# In[4]:


df_actor_info = pd.DataFrame()


# In[6]:


folder = sys.argv[1]
print("reading pr data")
df_pr = pd.DataFrame()
file_count = np.array([int(ele.replace("prEventCommits000000000","").replace(".csv","")) for ele in os.listdir(f'data/github_clean/{folder}/') if 'prEventCommits000000000' in ele])

for i in range(max(file_count)+1):
    if int(i) < 10:
        i = f"0{i}"
    if int(i) < 100:
        i = f"0{i}"
    df_pr_i = pd.read_csv(f'data/github_clean/{folder}/prEventCommits000000000{i}.csv', index_col = 0)
    df_pr = pd.concat([df_pr_i[['type', 'created_at', 'repo_id', 'actor_id', 'org_id', 'pr_id',
                                'pr_number', 'pr_state', 'pr_locked', 'pr_merged_at','pr_closed_at','pr_updated_at',
                                'pr_commits', 'pr_additions','pr_deletions','pr_changed_files',
                                'pr_author_association', 'pr_assignees', 'pr_requested_reviewers', 'pr_requested_teams',
                                'pr_merged_by_login', 'pr_merged_by_id', 'pr_merged_by_type',
                                'pr_merged_by_site_admin', 'pr_label', 'commit_list',
                               ]],
                       df_pr])
    df_actor_i = df_pr_i[['actor_id', 'actor_login', 'repo_id', 'repo_name', 'org_id','org_login', 'created_at']].drop_duplicates()
    df_actor_info = pd.concat([df_actor_info, df_actor_i]).drop_duplicates()
    
df_actor_info = df_actor_info[df_actor_info['created_at'].apply(lambda x: x!="False")]
df_actor_info['created_at'] = pd.to_datetime(df_actor_info['created_at'])
df_actor_info = df_actor_info.groupby(['actor_id', 'actor_login', 'repo_id', 'repo_name', 'org_id','org_login',]).agg({'created_at':['min', 'max']})
df_actor_info = df_actor_info.reset_index()
df_actor_info.columns=['actor_id', 'actor_login', 'repo_id', 'repo_name', 'org_id','org_login','earliest_date','latest_date']
df_actor_info.to_csv(f'data/merged_data/{folder}/pr_actor.csv')


# In[7]:


df_pr = df_pr[~df_pr.index.isnull()]


# In[8]:


get_ipython().run_cell_magic('time', '', 'files = glob.glob(f"data/github_clean/{folder}/prReviewEvent0*")\ndf_pr_review_events = [pd.read_csv(f, index_col = 0) for f in files]\ndf_pr_review_events = pd.concat(df_pr_review_events,ignore_index=True)\ndf_pr_review_events = df_pr_review_events.drop_duplicates()')


# In[9]:


get_ipython().run_cell_magic('time', '', 'files = glob.glob(f"data/github_clean/{folder}/prReviewCommentEvent0*")\ndf_pr_review_comment_events = [pd.read_csv(f, index_col = 0) for f in files]\ndf_pr_review_comment_events = pd.concat(df_pr_review_comment_events,ignore_index=True)\ndf_pr_review_comment_events = df_pr_review_comment_events.drop_duplicates()')


# In[10]:


review_comments_add = df_pr_review_comment_events.copy()
review_comments_add.rename({'pr_review_comment_action':'pr_review_action', 'pr_review_comment_id':'pr_review_id',
                            'pr_review_comment_body': 'pr_review_body', 'pr_review_comment_commit_id':'pr_review_commit_id',
                            'pr_review_comment_author_association':'pr_review_author_association'},
                          axis = 1, inplace = True)
review_comments_add['pr_review_state'] = 'commented'
review_comments_add.drop(['pr_review_comment_site_admin'], axis = 1, inplace = True)


# In[11]:


get_ipython().run_cell_magic('time', '', "df_pr_all_reviews = pd.concat([df_pr_review_events, review_comments_add]).drop_duplicates().reset_index(drop = True)\ndf_pr_all_reviews['created_at'] = pd.to_datetime(df_pr_all_reviews['created_at'])")


# In[12]:


df_pr_all_reviews.sort_values('pr_review_body', inplace = True)
df_pr_all_reviews.drop_duplicates(subset = ['created_at', 'repo_id', 'actor_id', 'pr_review_id', 
                                            'pr_review_commit_id', 'pr_review_state'], keep = 'first', inplace = True)


# In[13]:


def cleanParquetPR(file):
    try:
        df = pd.read_parquet(file)
        return df
    except:
        print(file)
        return pd.DataFrame()


# In[14]:


get_ipython().run_cell_magic('time', '', 'print("reading pr parquet files")\nfiles = glob.glob(f"data/github_commits/parquet/{folder}/*_pr_*")\ndf_parquet_pr_data = [cleanParquetPR(f) for f in files]\ndf_parquet_pr = pd.concat(df_parquet_pr_data,ignore_index=True)')


# In[15]:


get_ipython().run_cell_magic('time', '', 'print("dropping duplicate parquet pr entries")\ndf_parquet_pr.sort_values(\'pr_state\', inplace = True)\ndf_parquet_pr.drop_duplicates(\n    subset = [\'pr_number\', \'repo_id\', \'repo_name\', \'actor_id\', \'actor_login\', \'org_id\', \'org_login\',\'commit sha\',\n              \'commit author name\', \'commit author email\', \'committer name\', \'commmitter email\', \'commit message\', \'commit additions\',\n              \'commit deletions\', \'commit changes total\', \'commit files changed count\', \'commit time\'], inplace = True)')


# In[16]:


df_parquet_pr['commit time'] = pd.to_datetime(df_parquet_pr['commit time'],unit='s')


# In[17]:


print("turning stuff into lists")
for col in ['pr_assignees', 'pr_requested_reviewers', 'pr_requested_teams', 'pr_label', 'commit_list']:
    print(col)
    df_pr[col] = df_pr[col].apply(lambda x: [] if type(x) == float or type(x) == type(None) or                                   (type(x) == str and x == "'float' object has no attribute 'split'") else x)
    df_pr[col] = df_pr[col].apply(lambda x: ast.literal_eval(x) if type(x) == str else x)


# In[18]:


print("various data cleaning commands")


# In[19]:


df_pr['pr_id'] = pd.to_numeric(df_pr['pr_id'])


# In[20]:


df_pr['valid_vals'] = df_pr.count(axis = 1)
df_pr['retrieved_commits'] = df_pr['commit_list'].apply(len)


# In[21]:


df_pr = df_pr.sort_values(['valid_vals', 'retrieved_commits', 'created_at'], ascending = False)


# In[22]:


df_pr['actor_id_state'] = df_pr['actor_id'].astype(str)+" | " 
df_pr['actor_id_state'] = df_pr['actor_id_state'] + df_pr['pr_state'].apply(lambda x: 'NAN STATE' if type(x) != str else x) + " | "
df_pr['actor_id_state'] = df_pr['actor_id_state'] + df_pr['org_id'].apply(lambda x: 'NAN ORG' if pd.isnull(x) else str(x))
df_pr['actor_id_state'] = df_pr['actor_id_state'] + " | " +  df_pr['pr_author_association'].apply(lambda x: 'NAN AUTHOR ASSOCIATION' if type(x) != str else x) 


df_parquet_pr['actor_id_state'] = df_parquet_pr['actor_id'].astype(str)+" | " +  df_parquet_pr['org_id'].apply(lambda x: 'NAN ORG' if pd.isnull(x) else str(x))  + " | "
df_parquet_pr['actor_id_state'] = df_parquet_pr['actor_id_state'] + df_parquet_pr['pr_state'].apply(lambda x: 'NAN STATE' if type(x) != str else x) 


# In[23]:


get_ipython().run_cell_magic('time', '', "df_pr['actor_id_list'] = df_pr['actor_id_state'].groupby(df_pr['pr_id']).transform(lambda x: [x.tolist()]*len(x))\ndf_pr['actor_id_list'] = df_pr['actor_id_list'].apply(np.unique)")


# In[24]:


df_pr_nodup = df_pr.drop_duplicates(subset = ['repo_id', 'pr_id'], keep = 'first')


# In[25]:


get_ipython().run_cell_magic('time', '', 'df_parquet_pr[\'pr_id_temp\'] = df_parquet_pr[\'repo_id\'].astype(str)+"_"+df_parquet_pr[\'pr_number\'].astype(str)\ndf_parquet_pr[\'actor_id_list\'] = df_parquet_pr[\'actor_id_state\'].groupby(df_parquet_pr[\'pr_id_temp\']).transform(lambda x: [x.tolist()]*len(x))\ndf_parquet_pr[\'actor_id_list\'] = df_parquet_pr[\'actor_id_list\'].apply(np.unique)')


# In[26]:


df_parquet_pr_nodup = df_parquet_pr.drop_duplicates(
    subset = ['repo_id', 'pr_id_temp', 'commit time', 'commit sha'], keep = 'first')


# In[27]:


df_parquet_pr_nodup['actor_id_list'] = df_parquet_pr_nodup['actor_id_list'].apply(lambda x: sorted(x))
df_pr_nodup['actor_id_list'] = df_pr_nodup['actor_id_list'].apply(lambda x: sorted(x))


# In[28]:


print(" created merged commit data")


# In[29]:


df_pr_commits = pd.merge(df_pr_nodup.drop(['type', 'actor_id', 'org_id', 'pr_state', 'pr_author_association', 'actor_id_state',
                                           'valid_vals'], axis =1), 
                         df_parquet_pr_nodup.drop(['actor_id_state','actor_id', 'actor_login', 'org_id', 'org_login'], axis = 1), 
                         on = ['repo_id', 'pr_number', ], 
                         how = 'left')


# In[30]:


print("clean merged commit data")


# In[31]:


df_pr_commits.rename({'actor_id_list_y':'commit_actor_id_list',
                      'actor_id_list_x':'pr_actor_id_list'}, axis= 1, inplace = True)


# In[32]:


df_pr_commits['created_at'] = pd.to_datetime(df_pr_commits['created_at'])
df_pr_commits['pr_merged_at'] = pd.to_datetime(df_pr_commits['pr_merged_at'].apply(lambda x: x if x != "[]" else np.nan))
df_pr_commits['pr_closed_at'] = pd.to_datetime(df_pr_commits['pr_closed_at'].apply(lambda x: x if x != "[]" else np.nan))
df_pr_commits['pr_updated_at'] = pd.to_datetime(df_pr_commits['pr_updated_at'].apply(lambda x: x if x != "[]" else np.nan))
df_pr_commits['commit time'] = pd.to_datetime(df_pr_commits['commit time'],unit='s')


# In[33]:


df_pr_commits['merge_day'] = df_pr_commits['pr_merged_at'].apply(lambda x: x.day)
df_pr_commits['merge_month'] = df_pr_commits['pr_merged_at'].apply(lambda x: x.month)
df_pr_commits['merge_year'] = df_pr_commits['pr_merged_at'].apply(lambda x: x.year)

df_pr_commits['closed_day'] = df_pr_commits['pr_closed_at'].apply(lambda x: x.day)
df_pr_commits['closed_month'] = df_pr_commits['pr_closed_at'].apply(lambda x: x.month)
df_pr_commits['closed_year'] = df_pr_commits['pr_closed_at'].apply(lambda x: x.year)

df_pr_commits['commit_day'] = df_pr_commits['commit time'].apply(lambda x: x.day)
df_pr_commits['commit_month'] = df_pr_commits['commit time'].apply(lambda x: x.month)
df_pr_commits['commit_year'] = df_pr_commits['commit time'].apply(lambda x: x.year)


# In[34]:


df_pr_commits['commit parent'] = df_pr_commits['commit_groups'].apply(lambda x: x[0] if type(x) == list and len(x)>0 else '')


# In[35]:


df_pr_commits.drop(['pr_id_temp', 'commit_groups', 'commit_list'], 
                   axis = 1, inplace = True)


# In[36]:


null_commit_time = df_pr_commits[df_pr_commits['commit_year'].isnull()].index
df_pr_commits.loc[null_commit_time, 
    ['commit_day', 'commit_month', 'commit_year']] = df_pr_commits.loc[null_commit_time, ['merge_day', 'merge_month', 'merge_year']]


# In[37]:
df_pr_commits['pr_changed_files'] = df_pr_commits['pr_changed_files'].apply(lambda x: x if x != "[]" else 0)

df_pr_commits['pr_commits_wt'] = df_pr_commits['pr_commits'] / df_pr_commits.groupby('pr_id')['pr_id'].transform('count')
df_pr_commits['pr_additions_wt'] = pd.to_numeric(df_pr_commits['pr_additions']) / df_pr_commits.groupby('pr_id')['pr_id'].transform('count')
df_pr_commits['pr_deletions_wt'] = pd.to_numeric(df_pr_commits['pr_deletions']) / df_pr_commits.groupby('pr_id')['pr_id'].transform('count')
df_pr_commits['pr_changed_files_wt'] = pd.to_numeric(df_pr_commits['pr_changed_files']) / df_pr_commits.groupby('pr_id')['pr_id'].transform('count')
df_pr_commits['retrieved_commits_wt'] = pd.to_numeric(df_pr_commits['retrieved_commits']) / df_pr_commits.groupby('pr_id')['pr_id'].transform('count')


# In[38]:


df_pr_commits['pr_assignees_list'] = df_pr_commits['pr_assignees'].apply(lambda x: [ele['id'] for ele in x] if len(x)>0 else [])
df_pr_commits['pr_requested_reviewers_list'] = df_pr_commits['pr_requested_reviewers'].apply(lambda x: [ele['id'] for ele in x] if len(x)>0 else [])
df_pr_commits['pr_requested_teams_list'] = df_pr_commits['pr_requested_teams'].apply(lambda x: [ele['id'] for ele in x] if len(x)>0 else [])


# In[39]:


df_pr_commits['closed_wt'] = (1-df_pr_commits['pr_closed_at'].isna()) / df_pr_commits.groupby('pr_id')['pr_id'].transform('count')
df_pr_commits['merged_wt'] = 1-df_pr_commits['pr_merged_at'].isna() / df_pr_commits.groupby('pr_id')['pr_id'].transform('count')


# In[40]:


df_pr_commits['pr_actors'] = df_pr_commits['pr_actor_id_list'].apply(lambda x: [ele.split("|")[0].strip() for ele in x])
df_pr_commits['pr_commit_actors'] = df_pr_commits['commit_actor_id_list'].apply(lambda x: [ele.split("|")[0].strip() for ele in x] if type(x) == list else [])
df_pr_commits['all_pr_actors'] = (df_pr_commits['pr_actors']+df_pr_commits['pr_commit_actors']).apply(lambda x: list(set(x)))


# In[41]:


df_pr_commits['pr_orgs'] = df_pr_commits['pr_actor_id_list'].apply(lambda x: [ele.split("|")[2].strip() for ele in x])
df_pr_commits['pr_commit_orgs'] = df_pr_commits['commit_actor_id_list'].apply(lambda x: [ele.split("|")[2].strip() for ele in x] if type(x) == list else [])
df_pr_commits['all_pr_orgs'] = (df_pr_commits['pr_orgs']+df_pr_commits['pr_commit_orgs']).apply(lambda x: list(set([ele for ele in x if ele != 'NAN ORG'])))


# In[42]:


df_pr_commits['commit file changes'] = df_pr_commits['commit file changes'].apply(lambda x: x.decode() if type(x) == bytes else x)


# In[ ]:


df_pr_commits['commit file changes'] = df_pr_commits['commit file changes'].apply(
    lambda x: [] if type(x) == float or type(x) == type(None) else x)
df_pr_commits['commit file changes'] = df_pr_commits['commit file changes'].apply(lambda x: ast.literal_eval(x) if type(x) == str else x)


# In[ ]:


df_pr_commits['committer info'] = df_pr_commits['committer name'] + " | " + df_pr_commits['commmitter email']


# In[ ]:


print("done cleaning df_pr_commits")


# In[ ]:


# function to turn list of lists into lists
def rollIntoOne(series):
    return len(series.apply(pd.Series).stack().reset_index(drop=True).unique())


# In[ ]:


def dropNAUnique(x):
    return x.dropna().unique().tolist()


# In[ ]:


def getList(x):
    try:
        return [ele['file'] for ele in x]
    except:
        return [ele['file'] for sublst in x for ele in sublst]


# In[ ]:


def aggData(df, group_cols):
    df_results = df.groupby(group_cols, sort=False, observed=True).agg(     
        pr_count=('pr_id', 'nunique'),
        unique_push_actors=('all_pr_actors', rollIntoOne),
        unique_push_orgs=('all_pr_orgs', rollIntoOne),
        claimed_commits=('pr_commits_wt', 'sum'),
        claimed_additions=('pr_additions_wt', 'sum'),
        claimed_deletions=('pr_deletions_wt', 'sum'),
        claimed_changed_files=('pr_changed_files_wt', 'sum'),
        closed_prs=('closed_wt', 'sum'),
        merged_prs=('merged_wt', 'sum'),
        merger_id_count=('pr_merged_by_id', 'nunique'),
        pr_labels=('pr_label', rollIntoOne),
        counted_commits=('retrieved_commits_wt', 'count'), #this is correct, excuse my naming
        retrieved_commits=('commit sha', 'count'),
        unique_commit_authors=('commit author name', 'nunique'), 
        unique_commit_author_emails=('commit author email', 'nunique'),
        unique_committers=('committer name', 'nunique'),
        unique_committer_emails=('commmitter email', 'nunique'),
        commit_authors=('commit author name', dropNAUnique),
        committers=('committer info', dropNAUnique),
        LOC_added=('commit additions', 'sum'),
        LOC_deleted=('commit deletions', 'sum'),
        files_changed=('commit files changed count', 'sum'),
        changed_files=('commit file changes', getList),
        uniq_changed_files=('commit file changes',  lambda x: len(getList(x)))
    )    
    return df_results


# In[ ]:


print("now exporting merged data")


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_pr_commits.to_csv(f'data/merged_data/{folder}/merged_commit_pr.csv', encoding='utf-8')")


# In[ ]:

"""%%time
print("merge date, monthly")
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
    df_pr_monthly = aggData(df_pr_commits, ['merge_month', 'merge_year', 'repo_id'])
    df_pr_monthly.to_csv('data/aggregated_data/aggregated_monthly_labor_pr.csv', encoding='utf-8')"""


# In[ ]:


"""%%time
print("commit date, monthly")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
    df_pr_monthly_commit = aggData(df_pr_commits, ['commit_month', 'commit_year', 'repo_id'])
    df_pr_monthly_commit.to_csv('data/aggregated_data/aggregated_monthly_labor_commit_pr.csv', encoding='utf-8')"""


# In[ ]:


"""%%time
print("merge date, daily")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    df_pr_commits_merged = df_pr_commits[~df_pr_commits['pr_merged_at'].isna()]
    df_pr_monthly_merged = aggData(df_pr_commits_merged, ['merge_month', 'merge_year', 'repo_id'])
    df_pr_monthly_merged.to_csv('data/aggregated_data/aggregated_monthly_labor_pr_merged_only.csv', encoding='utf-8')"""


# In[ ]:


"""%%time
print("merge date, daily")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
    df_pr_commits_merged = df_pr_commits[~df_pr_commits['pr_merged_at'].isna()]
    df_pr_monthly_commit_merged = aggData(df_pr_commits_merged, ['commit_month', 'commit_year', 'repo_id'])
    df_pr_monthly_commit_merged.to_csv('data/aggregated_data/aggregated_monthly_labor_commit_pr_merged_only.csv', encoding='utf-8')"""


# In[ ]:


"""prReview contains data about prReviews - link to examine 1) whose reviewing, 2) whether requested teams are reviewing,
                                                         3) how many reviews
prReviewCommentEvent contains statsitics about the type of discussion that''s going on about pr reviews, look at 
1) number of comments, 2) whose commenting"""


# In[ ]:





# In[ ]:




