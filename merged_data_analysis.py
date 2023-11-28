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
    pr_data[col] = pr_data[col].apply(lambda x: literal_eval(x)).apply(lambda x: x if len(x)== 0 else [ele['id'] for ele in x])
for col in ['pr_label', 'pr_actor_id_list', 'pr_assignees_list', 'pr_requested_reviewers_list','pr_requested_teams_list', 
            'pr_actors', 'pr_commit_actors', 'all_pr_actors', 'pr_orgs', 'pr_commit_orgs']:
    pr_data[col] = pr_data[col].apply(lambda x: literal_eval(x))

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
df_committers = df_committers[df_committers['name'] != "GitHub"]
df_committers = df_committers[df_committers['name'].apply(lambda x: "[bot]" not in x and '-bot' not in x)]
# known id
df_committers['actor_id'] = df_committers['email'].apply(
    lambda x: np.nan if "users.noreply.github.com" not in x or "+" not in x 
    else pd.to_numeric(x.split("+")[0].replace("\x96",""), errors = "coerce"))
df_committers['actor_login'] = df_committers['email'].apply(lambda x: x.split("@")[0].split("+")[-1] if "users.noreply.github.com" in x else np.nan)
# weird excxeption 
df_committers['actor_id'] = pd.to_numeric(df_committers['actor_id'].replace('palmtree5', 3577255).replace('user.email47079615', 47079615))
np.mean(df_committers['actor_id'].isnull())


# In[8]:


# same PR, different email, same name - use the non no-reply email
df_committers['pr_number_name'] =  df_committers['pr_number'].astype(str) + "_" + df_committers['name']
df_committers_dup = df_committers[['pr_number_name', 'actor_id']].dropna().drop_duplicates()
dict_committers_id = dict(zip(df_committers_dup['pr_number_name'], df_committers_dup['actor_id']))
df_committers['actor_id'] = df_committers.apply(lambda x: x['actor_id'] if not pd.isnull(x['actor_id']) else                                                 dict_committers_id.get(x['pr_number_name'], np.nan), axis = 1)
df_committers.drop(['pr_number_name'], axis = 1, inplace = True)
np.mean(df_committers['actor_id'].isnull())


# In[9]:


# same repo, different email, same name
df_committers['pr_number_repo'] =  df_committers['repo_name'].astype(str) + "_" + df_committers['name']
df_committers_dup = df_committers[['pr_number_repo', 'actor_id']].dropna().drop_duplicates()
dict_committers_repo_id = dict(zip(df_committers_dup['pr_number_repo'], df_committers_dup['actor_id']))
df_committers['actor_id'] = df_committers.apply(lambda x: x['actor_id'] if not pd.isnull(x['actor_id']) else                                                 dict_committers_repo_id.get(x['pr_number_repo'], np.nan), axis = 1)
df_committers.drop(['pr_number_repo'], axis = 1, inplace = True)
np.mean(df_committers['actor_id'].isnull())


# In[10]:


# name in commit is the same as actor login
df_pr_actor['actor_login_repo_id'] = df_pr_actor['actor_login'] + "_" + df_pr_actor['repo_id'].astype(str)
df_committers['name_repo_id'] = df_committers['name'] + '_' + df_committers['repo_id'].astype(str)
dict_actors_name_repo_id = dict(zip(df_pr_actor['actor_login_repo_id'], df_pr_actor['actor_id']))
df_committers['actor_id'] = df_committers.apply(lambda x: x['actor_id'] if not pd.isnull(x['actor_id']) else                                                 dict_actors_name_repo_id.get(x['name_repo_id'], np.nan), axis = 1)
df_committers.drop(['name_repo_id'], axis = 1, inplace = True)
np.mean(df_committers['actor_id'].isnull())


# In[11]:


df_committers['actor_id'] = pd.to_numeric(df_committers['actor_id'])


# In[12]:


df_email_actor = df_committers[['email', 'actor_id']].dropna().drop_duplicates()
dict_email_actor = df_email_actor.set_index('email').to_dict()['actor_id']
df_committers['actor_id'] = df_committers.apply(lambda x: dict_email_actor.get(x['email'], np.nan) if                                                 pd.isnull(x['actor_id']) else x['actor_id'], axis = 1)
np.mean(df_committers['actor_id'].isnull())


# In[15]:


with open('data/inputs/company_domain_match_list.yaml', 'r') as f:
    company_info = yaml.load(f, Loader=yaml.FullLoader)
df_company = pd.DataFrame(company_info)


# In[16]:


df_domains = df_company[['company', 'domains']].explode('domains').drop_duplicates().groupby('domains').agg({'company':list})
df_committers_uq = df_committers[['name', 'email', 'actor_id']].drop_duplicates()
df_committers_uq['company'] = df_committers_uq['email'].apply(lambda x: [df_domains.loc[email, 'company'] for email in df_domains.index if x.endswith(email) and not x.split(email)[0][-1].isdigit()                                                                          and not x.split(email)[0][-1].isalpha()])
df_committers_uq['company'] = df_committers_uq['company'].apply(lambda x: "|".join(list(set([ele[0] for ele in x]))))
df_committers_uq['company'] = df_committers_uq.apply(lambda x: '' if x['email'].endswith('users.noreply.github.com') else x['company'], axis = 1)


# In[17]:


# add regex stuff


# In[46]:


commit_data[commit_data['commit author email'].apply(lambda x: x == 'noreply@github.com' if not pd.isnull(x) else False)]


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
token = "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"


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
            return url['message']


# In[104]:


df_committers_uq['committer_info'] = df_committers_uq.apply(
    lambda x: getCommits(x['commit_repo'].split("_")[1], x['commit_repo'].split("_")[0],x['user_type']), axis = 1)


# In[ ]:


df_committers_uq.to_csv('data/merged_data/committers_info.csv')

