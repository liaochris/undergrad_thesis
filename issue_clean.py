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
import warnings
warnings.filterwarnings("ignore")
username = "liaochris"
token = os.environ['token']
from pandarallel import pandarallel
pandarallel.initialize()

# In[194]:

if "partition" in sys.argv[1]:
    folder = "filtered_github_data"
    fname = sys.argv[1].replace(f'data/github_raw/{folder}/','').replace('.json','')
    ext = '.json'
elif "contributors" in sys.argv[1]:
    folder = "all_contributor_data"
    fname = sys.argv[1].replace(f'data/github_raw/{folder}/','').replace('.parquet','')
    ext = '.parquet'
else:
    folder = "github_data_pre_18"
    fname = sys.argv[1].replace(f'data/github_raw/{folder}/','').replace('.json','')
    ext = '.json'

# In[2]:
if ext == '.json':
    df_raw = pd.read_json(f'data/github_raw/{folder}/{fname}{ext}', lines=True)
if ext == '.parquet':
    try:
        df_raw = pd.read_parquet(f'data/github_raw/{folder}/{fname}{ext}', engine = 'pyarrow')
    except:
        sys.exit()

print(f"Starting to clean {folder}/{fname}{ext}")
# In[158]:

if 'partitions' not in fname:
    fname = "partitions"+fname.replace("github_data_pre18",'').replace("contributors","")
print(fname)

df = df_raw.copy()
tqdm.pandas()

# In[179]:



# In[160]:


# Unpack Data - Already Dictionary
for col in ['repo', 'actor', 'org']:
    expand_df = pd.DataFrame(df[col].values.tolist())
    expand_df.columns = [col+"_"+str(c_val) for c_val in expand_df.columns]
    df = pd.concat([df, expand_df], axis = 1)
    df.drop(col, axis = 1, inplace = True)


# In[161]:
# import repositories we want to filter on
package_repos = pd.read_csv('data/inputs/package_repos.csv', index_col = 0)
remove_repos = package_repos['github repo'].tolist()
# filter on repos
if "contributors" in sys.argv[1]:
    df = df[~df['repo_name'].isin(remove_repos)]

df['payload'] = df['payload'].apply(lambda x: json.loads(x) if type(x) != dict else x)
df['other'] = df['other'].apply(lambda x: json.loads(x) if type(x) != float and type(x) != dict and type(x) != type(None) else x)



df_pullrequest = df[df['type'] == 'PullRequestEvent']


# In[181]:


for col in ['id', 'node_id', 'title', 'state', 'locked', 'number', 'body', 'issue_url', 
            'merged_at', 'closed_at', 'updated_at', 'commits', 'additions', 'deletions', 'changed_files',
            'author_association', 'assignee', 'assignees', 'requested_reviewers', 'requested_teams', 'ref', 'action']:
    if col in ['author association', 'node_id']:
        df_pullrequest['pr_'+col] = df_pullrequest['payload'].apply(lambda x: x['pull_request'].get(col, ''))
    else:
        df_pullrequest['pr_'+col] = df_pullrequest['payload'].apply(lambda x: x['pull_request'].get(col, []))

df_pullrequest['pr_action'] = df_pullrequest['payload'].apply(lambda x: x['action'])
df_pullrequest['pr_merged_by_login'] = df_pullrequest['payload'].apply(
    lambda x: x['pull_request']['merged_by']['login'] 
    if 'merged_by' in x['pull_request'].keys() and x['pull_request']['merged_by'] != None else '')
df_pullrequest['pr_merged_by_id'] = df_pullrequest['payload'].apply(
    lambda x: x['pull_request']['merged_by']['id'] 
    if 'merged_by' in x['pull_request'].keys() and x['pull_request']['merged_by'] != None else '')
df_pullrequest['pr_merged_by_type'] = df_pullrequest['payload'].apply(
    lambda x: x['pull_request']['merged_by'].get('type', '')
    if 'merged_by' in x['pull_request'].keys() and x['pull_request']['merged_by'] != None else '')
df_pullrequest['pr_merged_by_site_admin'] = df_pullrequest['payload'].apply(
    lambda x: x['pull_request']['merged_by'].get('site_admin', '')
    if 'merged_by' in x['pull_request'].keys() and x['pull_request']['merged_by'] != None else '')

df_pullrequest['pr_label'] = df_pullrequest['payload'].apply(lambda x: [label['name'] for label in x['pull_request'].get('labels', [])])
df_pullrequest['pr_patch_url'] = df_pullrequest['payload'].apply(lambda x: x['pull_request'].get('patch_url', ''))
df_pullrequest['pr_commits_url'] = df_pullrequest['payload'].apply(lambda x: x['pull_request'].get('commits_url', []))


# In[182]:


df_pullrequest[['type', 'created_at', 'repo_id', 'repo_name', 'actor_id', 'actor_login', 'org_id', 'org_login',
                'pr_id', 'pr_node_id', 'pr_title', 'pr_state', 'pr_locked',
                'pr_number', 'pr_body', 'pr_issue_url', 'pr_merged_at', 'pr_action',
                'pr_closed_at', 'pr_updated_at', 'pr_commits', 'pr_additions', 'pr_deletions',
                'pr_changed_files', 'pr_author_association', 'pr_assignee', 'pr_assignees', 'pr_requested_reviewers', 
                'pr_requested_teams', 'pr_merged_by_login', 'pr_merged_by_id','pr_ref',
                'pr_merged_by_type', 'pr_merged_by_site_admin', 'pr_label', 'pr_patch_url',
                'pr_commits_url']].to_csv(
    f'data/github_clean/{folder}/prEvent{fname.split("partitions")[1]}.csv')




df_pullrequestreview = df[df['type'] == 'PullRequestReviewEvent']


# In[169]:


df_pullrequestreview['pr_review_action'] = df_pullrequestreview['payload'].apply(lambda x: x['action'])
df_pullrequestreview['pr_review_id'] = df_pullrequestreview['payload'].apply(lambda x: x['review']['id'])
df_pullrequestreview['pr_review_body'] = df_pullrequestreview['payload'].apply(lambda x: x['review']['body'])
df_pullrequestreview['pr_review_commit_id'] = df_pullrequestreview['payload'].apply(lambda x: x['review']['commit_id'])
df_pullrequestreview['pr_review_author_association'] = df_pullrequestreview['payload'].apply(lambda x: x['review']['author_association'])
df_pullrequestreview['pr_review_state'] = df_pullrequestreview['payload'].apply(lambda x: x['review']['state'])
df_pullrequestreview['pr_number'] = df_pullrequestreview['payload'].apply(lambda x: x['pull_request']['number'])

# In[170]:


for col in ['id', 'node_id', 'title', 'state', 'locked', 'number', 'body', 'issue_url', 
            'merged_at', 'closed_at', 'updated_at', 'commits', 'additions', 'deletions', 'changed_files',
            'author_association', 'assignee', 'assignees', 'requested_reviewers', 'requested_teams', 'ref']:
    if col in ['author association', 'node_id']:
        df_pullrequestreview['pr_'+col] = df_pullrequestreview['payload'].apply(lambda x: x['pull_request'].get(col, ''))
    else:
        df_pullrequestreview['pr_'+col] = df_pullrequestreview['payload'].apply(lambda x: x['pull_request'].get(col, []))

df_pullrequestreview['pr_user_id'] =  df_pullrequestreview['payload'].apply(lambda x: x['pull_request']['user']['id'] if 'user' in x['pull_request'] and 'id' in x['pull_request']['user'] else '')
df_pullrequestreview['pr_user_login'] = df_pullrequestreview['payload'].apply(lambda x: x['pull_request']['user']['login'] if 'user' in x['pull_request'] and 'login' in x['pull_request']['user'] else '')
df_pullrequestreview['pr_review_action'] = df_pullrequestreview['payload'].apply(lambda x: x['action'])
df_pullrequestreview['pr_merged_by_login'] = df_pullrequestreview['payload'].apply(
    lambda x: x['pull_request']['merged_by']['login'] 
    if 'merged_by' in x['pull_request'].keys() and x['pull_request']['merged_by'] != None else '')
df_pullrequestreview['pr_merged_by_id'] = df_pullrequestreview['payload'].apply(
    lambda x: x['pull_request']['merged_by']['id'] 
    if 'merged_by' in x['pull_request'].keys() and x['pull_request']['merged_by'] != None else '')
df_pullrequestreview['pr_review_author_association'] = df_pullrequestreview['payload'].apply(
    lambda x: x['pull_request']['author_association']
    if 'author_association' in x['pull_request'].keys() and x['pull_request']['author_association'] != None else '')



df_pullrequestreview[[
    'type', 'created_at', 'repo_id', 'repo_name', 'actor_id', 'actor_login', 'org_id', 'org_login',
    'pr_review_action', 'pr_review_id', 'pr_review_body', 'pr_review_commit_id', 'pr_review_author_association', 
    'pr_user_id', 'pr_user_login',
    'pr_review_state', 'pr_number', 'pr_review_author_association',
    'pr_id', 'pr_node_id', 'pr_title', 'pr_review_action', 'pr_state', 'pr_locked',
    'pr_number', 'pr_body', 'pr_issue_url', 'pr_merged_at',
    'pr_closed_at', 'pr_updated_at', 'pr_commits', 'pr_additions', 'pr_deletions',
    'pr_changed_files', 'pr_author_association', 'pr_assignee', 'pr_assignees', 'pr_requested_reviewers', 
    'pr_requested_teams', 'pr_merged_by_login', 'pr_merged_by_id','pr_ref',
]].to_csv(f'data/github_clean/{folder}/prReviewEvent{fname.split("partitions")[1]}.csv')




df_pullrequestreviewcomment = df[df['type'] == 'PullRequestReviewCommentEvent']


# In[172]:


df_pullrequestreviewcomment['pr_review_comment_action'] = df_pullrequestreviewcomment['payload'].apply(lambda x: x.get('action', ''))
df_pullrequestreviewcomment['pr_review_comment_id'] = df_pullrequestreviewcomment['payload'].apply(lambda x:
    x['comment'].get('pull_request_review_id',''))
df_pullrequestreviewcomment['pr_review_comment_commit_id'] = df_pullrequestreviewcomment['payload'].apply(lambda x: x['comment']['commit_id'])
df_pullrequestreviewcomment['pr_review_comment_site_admin'] = df_pullrequestreviewcomment['payload'].apply(lambda x: x['comment']['user'].get('site_admin',''))
df_pullrequestreviewcomment['pr_review_comment_body'] = df_pullrequestreviewcomment['payload'].apply(lambda x: x['comment']['body'])
df_pullrequestreviewcomment['pr_review_comment_author_association'] = df_pullrequestreviewcomment['payload'].apply(lambda x: 
    x['comment'].get('author_association','') if 'pull_request_url' in x['comment'] else np.nan)
df_pullrequestreviewcomment['pr_review_comment_reactions'] = df_pullrequestreviewcomment['payload'].apply(lambda x: x['comment'].get('reactions',{}))


for col in ['id', 'node_id', 'title', 'state', 'locked', 'number', 'body', 'issue_url', 
            'merged_at', 'closed_at', 'updated_at', 'commits', 'additions', 'deletions', 'changed_files',
            'author_association', 'assignees', 'requested_reviewers', 'requested_teams', 'ref']:
    if col in ['author association', 'node_id']:
        df_pullrequestreviewcomment['pr_'+col] = df_pullrequestreviewcomment['payload'].apply(lambda x: x['pull_request'].get(col, '') if \
                                                                                             'pull_request' in x else '')
    else:
        df_pullrequestreviewcomment['pr_'+col] = df_pullrequestreviewcomment['payload'].apply(lambda x: x['pull_request'].get(col, []) if \
                                                                                             'pull_request' in x else [])

df_pullrequestreviewcomment['pr_user_id'] =  df_pullrequestreviewcomment['payload'].apply(
    lambda x: x['pull_request']['user']['id'] if  'pull_request' in x and 'user' in x['pull_request'] and x['pull_request']['user'] != None and 'id' in x['pull_request']['user'] else '')
df_pullrequestreviewcomment['pr_user_login'] = df_pullrequestreviewcomment['payload'].apply(
    lambda x: x['pull_request']['user']['login'] if'pull_request' in x and 'user' in x['pull_request'] and x['pull_request']['user'] != None and 'login' in x['pull_request']['user'] else '')
df_pullrequestreviewcomment['pr_merged_by_login'] = df_pullrequestreviewcomment['payload'].apply(
    lambda x: x['pull_request']['merged_by']['login'] 
    if  'pull_request' in x and 'merged_by' in x['pull_request'].keys() and x['pull_request']['merged_by'] != None else '')
df_pullrequestreviewcomment['pr_merged_by_id'] = df_pullrequestreviewcomment['payload'].apply(
    lambda x: x['pull_request']['merged_by']['id'] 
    if 'pull_request' in x and 'merged_by' in x['pull_request'].keys() and x['pull_request']['merged_by'] != None else '')

# In[173]:


df_pullrequestreviewcomment[[
    'type', 'created_at', 'repo_id', 'repo_name', 'actor_id', 'actor_login', 'org_id', 'org_login',
    'pr_review_comment_action', 'pr_review_comment_id', 'pr_review_comment_commit_id', 'pr_review_comment_site_admin',
    'pr_review_comment_body', 'pr_review_comment_author_association', 'pr_number', 'pr_review_comment_reactions',
    'pr_id', 'pr_node_id', 'pr_title', 'pr_state', 'pr_locked',
    'pr_user_id', 'pr_user_login',
    'pr_number', 'pr_body', 'pr_issue_url', 'pr_merged_at',
    'pr_closed_at', 'pr_updated_at', 'pr_commits', 'pr_additions', 'pr_deletions',
    'pr_changed_files', 'pr_author_association', 'pr_assignees', 'pr_requested_reviewers', 
    'pr_requested_teams', 'pr_merged_by_login', 'pr_merged_by_id','pr_ref',]].to_csv(
    f'data/github_clean/{folder}/prReviewCommentEvent{fname.split("partitions")[1]}.csv')


