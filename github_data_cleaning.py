#!/usr/bin/env python
# coding: utf-8

# In[199]:


import pandas as pd
import urllib.request, json 
import ujson
from tqdm import tqdm
import requests
import os
import sys

# In[194]:


fname = sys.argv[1].replace('data/github_raw/filtered_github_data_large/','').replace('.json','')

username = "liaochris"
token = "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"
# In[2]:

df_raw = pd.read_json(f'data/github_raw/filtered_github_data_large/{fname}.json', lines=True)
print(f"Starting to clean filtered_github_data_large/{fname}.json")

# In[158]:


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


df['payload'] = df['payload'].apply(lambda x: json.loads(x))
df['other'] = df['other'].apply(lambda x: json.loads(x))


# In[10]:


df['type'].value_counts()


# ## ReleaseEvent

# In[154]:


df_release = df[df['type'] == 'ReleaseEvent']


# In[155]:


df_release['release_action'] = df_release['payload'].apply(lambda x: x['action'])
df_release['release_tag_name'] = df_release['payload'].apply(lambda x: x['release']['tag_name'])
df_release['release_name'] = df_release['payload'].apply(lambda x: x['release']['name'])
df_release['release_body'] = df_release['payload'].apply(lambda x: x['release']['body'])


# In[157]:


df_release[['type', 'created_at', 'repo_id', 'repo_name', 'actor_id', 'actor_login', 'org_id', 'org_login',
            'release_action', 'release_tag_name', 'release_name', 'release_body']].to_csv(
    f'data/github_clean/releaseEvent{fname.split("partitions")[1]}.csv')


# ## DeleteEvent

# In[147]:


df_delete = df[df['type'] == 'DeleteEvent']


# In[149]:


df_delete['event_type'] = df_delete['payload'].apply(lambda x: x['ref_type'])
df_delete['event_ref'] = df_delete['payload'].apply(lambda x: x['ref'])
df_delete['event_pusher_type'] = df_delete['payload'].apply(lambda x: x['pusher_type'])


# In[151]:


df_delete[['type', 'created_at', 'repo_id', 'repo_name', 'actor_id', 'actor_login', 'org_id', 'org_login',
           'event_type', 'event_ref', 'event_pusher_type']].to_csv(
    f'data/github_clean/deleteEvent{fname.split("partitions")[1]}.csv')


# ## CreateEvent

# In[134]:


df_create = df[df['type'] == 'CreateEvent']


# In[135]:


df_create['event_type'] = df_create['payload'].apply(lambda x: x['ref_type'])
df_create['event_ref'] = df_create['payload'].apply(lambda x: x['ref'])
df_create['repo_master_branch'] = df_create['payload'].apply(lambda x: x['master_branch'])
df_create['event_pusher_type'] = df_create['payload'].apply(lambda x: x['pusher_type'])


# In[140]:


df_create[['type', 'created_at', 'repo_id', 'repo_name', 'actor_id', 'actor_login', 'org_id', 'org_login',
           'event_type', 'event_ref', 'repo_master_branch', 'event_pusher_type']].to_csv(
    f'data/github_clean/createEvent{fname.split("partitions")[1]}.csv')


# ## Fork Event

# In[123]:


df_fork = df[df['type'] == 'ForkEvent']
df_fork[['type', 'created_at', 'repo_id', 'repo_name', 'actor_id', 'actor_login', 'org_id', 'org_login']].to_csv(
    f'data/github_clean/forkEvent{fname.split("partitions")[1]}.csv')


# # Pull Requests

# ## Pull Request Review Event

# In[87]:


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


df_pullrequestreview[[
    'type', 'created_at', 'repo_id', 'repo_name', 'actor_id', 'actor_login', 'org_id', 'org_login',
    'pr_review_action', 'pr_review_id', 'pr_review_body', 'pr_review_commit_id', 'pr_review_author_association', 
    'pr_review_state', 'pr_number'
]].to_csv(f'data/github_clean/prReviewEvent{fname.split("partitions")[1]}.csv')


# ## Pull Request Review Comment Event

# In[171]:


df_pullrequestreviewcomment = df[df['type'] == 'PullRequestReviewCommentEvent']
df_pullrequestreviewcomment['payload'] = df_pullrequestreviewcomment['payload'].apply(lambda x: {i:x[i] for i in x if i != 'pull_request'})


# In[172]:


df_pullrequestreviewcomment['pr_review_comment_action'] = df_pullrequestreviewcomment['payload'].apply(lambda x: x['action'])
df_pullrequestreviewcomment['pr_review_comment_id'] = df_pullrequestreviewcomment['payload'].apply(lambda x: x['comment']['pull_request_review_id'])
df_pullrequestreviewcomment['pr_review_comment_commit_id'] = df_pullrequestreviewcomment['payload'].apply(lambda x: x['comment']['commit_id'])
df_pullrequestreviewcomment['pr_review_comment_site_admin'] = df_pullrequestreviewcomment['payload'].apply(lambda x: x['comment']['user']['site_admin'])
df_pullrequestreviewcomment['pr_review_comment_body'] = df_pullrequestreviewcomment['payload'].apply(lambda x: x['comment']['body'])
df_pullrequestreviewcomment['pr_review_comment_author_association'] = df_pullrequestreviewcomment['payload'].apply(lambda x: x['comment']['author_association'])
df_pullrequestreviewcomment['pr_number'] = df_pullrequestreviewcomment['payload'].apply(lambda x: x['comment']['pull_request_url'].split("/")[-1])
df_pullrequestreviewcomment['pr_review_comment_reactions'] = df_pullrequestreviewcomment['payload'].apply(lambda x: x['comment'].get('reactions',{}))


# In[173]:


df_pullrequestreviewcomment[[
    'type', 'created_at', 'repo_id', 'repo_name', 'actor_id', 'actor_login', 'org_id', 'org_login',
    'pr_review_comment_action', 'pr_review_comment_id', 'pr_review_comment_commit_id', 'pr_review_comment_site_admin',
    'pr_review_comment_body', 'pr_review_comment_author_association', 'pr_number', 'pr_review_comment_reactions']].to_csv(
    f'data/github_clean/prReviewCommentEvent{fname.split("partitions")[1]}.csv')


# ## Pull Request Event

# In[175]:


df_pullrequest = df[df['type'] == 'PullRequestEvent']


# In[181]:


for col in ['id', 'node_id', 'title', 'state', 'locked', 'number', 'body', 'issue_url', 
            'merged_at', 'closed_at', 'updated_at', 'commits', 'additions', 'deletions', 'changed_files',
            'author_association', 'assignees', 'requested_reviewers', 'requested_teams']:
    df_pullrequest['pr_'+col] = df_pullrequest['payload'].apply(lambda x: x['pull_request'][col])

df_pullrequest['pr_merged_by_login'] = df_pullrequest['payload'].apply(
    lambda x: x['pull_request']['merged_by']['login'] if x['pull_request']['merged_by'] != None else '')
df_pullrequest['pr_merged_by_id'] = df_pullrequest['payload'].apply(
    lambda x: x['pull_request']['merged_by']['id'] if x['pull_request']['merged_by'] != None else '')
df_pullrequest['pr_merged_by_type'] = df_pullrequest['payload'].apply(
    lambda x: x['pull_request']['merged_by']['type'] if x['pull_request']['merged_by'] != None else '')
df_pullrequest['pr_merged_by_site_admin'] = df_pullrequest['payload'].apply(
    lambda x: x['pull_request']['merged_by']['site_admin'] if x['pull_request']['merged_by'] != None else '')

df_pullrequest['pr_label'] = df_pullrequest['payload'].apply(lambda x: [label['name'] for label in x['pull_request']['labels']])
df_pullrequest['pr_patch_url'] = df_pullrequest['payload'].apply(lambda x: x['pull_request']['patch_url'])
df_pullrequest['pr_commits_url'] = df_pullrequest['payload'].apply(lambda x: x['pull_request']['commits_url'])


# In[182]:


df_pullrequest[['type', 'created_at', 'repo_id', 'repo_name', 'actor_id', 'actor_login', 'org_id', 'org_login',
                'pr_id', 'pr_node_id', 'pr_title', 'pr_state', 'pr_locked',
                'pr_number', 'pr_body', 'pr_issue_url', 'pr_merged_at',
                'pr_closed_at', 'pr_updated_at', 'pr_commits', 'pr_additions', 'pr_deletions',
                'pr_changed_files', 'pr_author_association', 'pr_assignees', 'pr_requested_reviewers', 
                'pr_requested_teams', 'pr_merged_by_login', 'pr_merged_by_id',
                'pr_merged_by_type', 'pr_merged_by_site_admin', 'pr_label', 'pr_patch_url',
                'pr_commits_url']].to_csv(
    f'data/github_clean/prEvent{fname.split("partitions")[1]}.csv')


# ## Push Event

# In[62]:


df_push = df[df['type'] == 'PushEvent']


# In[63]:


df_push['push_id'] = df_push['payload'].apply(lambda x: x['push_id'])
df_push['push_size'] = df_push['payload'].apply(lambda x: x['size'])
df_push['push_size_distinct'] = df_push['payload'].apply(lambda x: x['distinct_size'])
df_push['push_before'] = df_push['payload'].apply(lambda x: x['before'])
df_push['push_head'] = df_push['payload'].apply(lambda x: x['head'])
df_push['commit_urls'] = df_push['payload'].apply(lambda x: [ele['url'] for ele in x['commits']])


# In[64]:


def getCommits(repo_info, before, head, original_urls):
    api_url = f"https://api.github.com/repos/{repo_info}/compare/{before}...{head}"
    try:
        with requests.get(api_url, auth=(username,token)) as url:
            data = url.json()
        commits = data['commits']
        return [c['url'] for c in commits]
    except:
        return original_urls


# In[65]:


min20_commits = df_push.query('push_size>20').index
df_push.loc[min20_commits, 'commit_urls'] = df_push.query('push_size>20').apply(lambda x: getCommits(x['repo_name'], x['push_before'], x['push_head'], x['commit_urls']), axis = 1)


# In[66]:


df_push[['type', 'created_at', 'repo_id', 'repo_name', 'actor_id', 'actor_login', 'org_id', 'org_login',
         'push_id', 'push_size', 'push_size_distinct', 'push_before', 'push_head', 'commit_urls']].to_csv(
    f'data/github_clean/pushEvent{fname.split("partitions")[1]}.csv')


# ## Watch Data

# In[48]:


df_watch = df[df['type'] == 'WatchEvent']
df_watch[['type', 'created_at', 'repo_id', 'repo_name', 'actor_id', 'actor_login', 'org_id', 'org_login']].to_csv(
    f'data/github_clean/watchEvent{fname.split("partitions")[1]}.csv')


# # Issues

# ## Issues Event

# In[70]:


df_issues = df[df['type'] == 'IssuesEvent']
df_issues['issue_action'] = df_issues['payload'].apply(lambda x: x['action'])
df_issues['issue_title'] = df_issues['payload'].apply(lambda x: x['issue']['title'])
df_issues['issue_labels'] = df_issues['payload'].apply(lambda x: x['issue']['labels'])
df_issues['issue_assignees'] = df_issues['payload'].apply(lambda x: x['issue']['assignees'])
df_issues['issue_count'] = df_issues['payload'].apply(lambda x: x['issue']['comments'])
df_issues['issue_body'] = df_issues['payload'].apply(lambda x: x['issue']['body'])
df_issues['issue_reactions'] = df_issues['payload'].apply(lambda x: x['issue'].get('reactions', {}))
df_issues['issue_reason'] = df_issues['payload'].apply(lambda x: x['issue'].get('state_reason', {}))


# In[71]:


df_issues[['type', 'created_at', 'repo_id', 'repo_name', 'actor_id', 'actor_login', 'org_id', 'org_login',
           'issue_action', 'issue_title', 'issue_labels', 'issue_assignees', 'issue_count',
           'issue_body', 'issue_reactions', 'issue_reason']].to_csv(
    f'data/github_clean/issuesEvent{fname.split("partitions")[1]}.csv')


# ## Issue Comment Event

# In[45]:


df_issuecomment = df[(df['type'] == 'IssueCommentEvent')]


# In[46]:


df_issuecomment['issue_id'] = df_issuecomment['payload'].apply(lambda x: x['comment']['issue_url'].split("/")[-1])
df_issuecomment['issue_comment_id'] = df_issuecomment['payload'].apply(lambda x: x['comment']['html_url'].split("-")[-1])
df_issuecomment['issue_comment_body'] = df_issuecomment['payload'].apply(lambda x: x['comment']['body'])
df_issuecomment['issue_comment_reactions'] = df_issuecomment['payload'].apply(lambda x: x['comment'].get('reactions', {}))
df_issuecomment['actor_repo_association'] = df_issuecomment['payload'].apply(lambda x: x['comment']['author_association'])
df_issuecomment['actor_type'] = df_issuecomment['payload'].apply(lambda x: x['comment']['user']['type'])
df_issuecomment['actor_site_admin_status'] = df_issuecomment['payload'].apply(lambda x: x['comment']['user']['site_admin'])


# In[47]:


df_issuecomment[['type', 'created_at', 'repo_id', 'repo_name', 'actor_id', 'actor_login', 'org_id', 'org_login',
                 'issue_id', 'issue_comment_id', 'issue_comment_body', 'issue_comment_reactions', 'actor_repo_association',
                 'actor_type', 'actor_site_admin_status']].to_csv(
    f'data/github_clean/issueCommentEvent{fname.split("partitions")[1]}.csv')

