#!/usr/bin/env python
# coding: utf-8

# ## Data Cleaning Goals
# 1. I want to create data that will allow me to
#    
#    a. analyze the structure of hierarchy across GitHub Repositories (whose opening issues? whose commenting on issues? how many are there overall?)
#    
#    b. track the sequence of participation for each issue
#    
#    c. link PRs to issues
#    
#    d. collect covariates related to issues so I can measure "issue difficulty"
#    

# In[1]:


from IPython.display import display, HTML

def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n","<br>") ) )


# In[2]:


## Import Libraries and Data


# In[3]:



import glob
import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import ast
import numpy as np
from operator import itemgetter
import os
import pytz


# In[5]:




# In[6]:


pd.set_option('display.max_columns', None)


# In[7]:


def readDf(df,cols):
    try:
        cols = [col for col in cols if col in df.columns]
        return df[cols]
    except:
        return pd.DataFrame()


# In[8]:


def name_split(repo):
    return repo.split("/")[0]
def owner(bool):
    return 'owner' if bool else np.nan
def getLogin(x):
    return x['id']


# In[9]:

print("reading issue data")
get_ipython().run_cell_magic('time', '', "# Read data on  comments,\ncomcol = ['created_at', 'type', 'issue_number', 'repo_id', 'issue_author_association', 'repo_name', 'issue_user_login', 'actor_login',\n       'actor_id', 'latest_issue_assignee', 'latest_issue_assignees', 'org_id', 'org_login', 'issue_user_id']\nissue_com = glob.glob('data/github_clean/filtered_github_data/issueCo*')\nissue_com.extend(glob.glob('data/github_clean/github_data_pre_18/issueCo*'))\nissue_com.extend(glob.glob('data/github_clean/github_data_2324/issueCo*'))\ndf_issue_comments = pd.concat([readDf(pd.read_parquet(ele), comcol) for ele in issue_com]).reset_index(drop = True)\n\nissuecol = ['created_at', 'type', 'issue_number', 'repo_id', 'issue_author_association', 'repo_name', 'issue_user_login', 'actor_login',\n            'actor_id', 'issue_action', 'issue_assignee', 'issue_assignees', 'org_id', 'org_name', 'issue_user_id']\nissues = glob.glob('data/github_clean/filtered_github_data/issues*')\nissues.extend(glob.glob('data/github_clean/github_data_pre_18/issues*'))\nissues.extend(glob.glob('data/github_clean/github_data_2324/issues*'))\ndf_issue = pd.concat([readDf(pd.read_parquet(ele), issuecol) for ele in issues]).reset_index(drop = True)\n")


# ## Storing Data

# In[10]:


OriginalDachristaStatistics = pd.DataFrame()
DataDescriptives = pd.DataFrame()


# ## Data Cleaning

# In[11]:


# columns to rename
mod_columns = [ele for ele in df_issue_comments.columns if 'latest' in ele]
mod_dict = {ele : ele.replace('latest_', '') for ele in mod_columns}
df_issue_comments.rename(mod_dict, axis = 1, inplace = True)


# In[12]:


full_issue_data = pd.concat([df_issue,df_issue_comments])


# In[13]:


# clean data: remove entries with NA issue number values  
df_issue_clean = full_issue_data[~full_issue_data['issue_number'].isna()]
# clean data: add key variable
df_issue_clean['key'] = df_issue_clean['repo_id'].apply(str) + "_" + df_issue_clean['issue_number'].apply(str)


# In[14]:


df_issue_clean.reset_index(drop = True, inplace = True)


# In[15]:


df_issue_clean['created_at'] = pd.to_datetime(df_issue_clean['created_at'], errors='coerce', utc=True)


# In[16]:


# FIX people who aren't classified as owner
fix_ind = df_issue_clean[df_issue_clean.apply(lambda x: x['repo_name'].split("/")[0] == x['issue_user_login'], axis = 1)].index
df_issue_clean.loc[fix_ind, 'issue_author_association'] = 'OWNER'

fix_ind_actor = df_issue_clean[df_issue_clean.apply(lambda x: x['repo_name'].split("/")[0] == x['actor_login'], axis = 1)].index
df_issue_clean.loc[fix_ind_actor, 'actor_repo_association'] = 'OWNER'


# In[17]:


df_issue_clean['repo_org'] = df_issue_clean['repo_name'].apply(name_split)


# In[18]:


df_issue_clean['permissions'] = (df_issue_clean['repo_org'] == df_issue_clean['actor_login']).apply(owner)


# In[19]:


orgs = df_issue_clean[['org_id', 'org_login']].dropna().drop_duplicates()['org_login'].tolist()


# In[20]:


# back out organization
df_issue_clean['organization'] = df_issue_clean['repo_org'].isin(orgs)


# # Backing out Contributors
#    1. **Release:** creates tag
#    2. **PRs:**  
#       1. PR event `edited`, `closed`, `reopened`, `assigned`, `unassigned`, `review_requested`, `review_request_removed`, `labeled`, `unlabeled`, and `synchronize`.
#       2. PR Review event `created`
#       3. Someone else's PR Review Comment `edit` or `delete`
#       4. **merged by column**
#    3. **Issues:**
#       1. Issue event: `Reopen` issues closed by non-author, `close`, `assign`,`unassigned` `labeled` `unlabeled`  all general ones
#    4. **Commit Comments**: 
#       1. Check if any commit comments have `edited` tag??

# Issue
# - Reopened, and issue_author $\neq$ actor_login
# - Closed, and issue_author $\neq$ actor_login

# # Issues
# Identify contributors based off whetherthey closed/reopended issues and were not the author

# In[21]:


issue_contributors = df_issue_clean[
    (df_issue_clean['type'] == 'IssuesEvent') & \
    (df_issue_clean['issue_action'].isin(['closed', 'reopened'])) & \
    (df_issue_clean['issue_user_login'] != df_issue_clean['actor_login']) &
    (df_issue_clean['permissions'].isna())].index
df_issue_clean.loc[issue_contributors, 'permissions'] = 'triage'


# identify people who has privileges based assignee and assignees status

# In[22]:


# anyone who can assign other people to an issue

issue_assignee = df_issue_clean[['repo_id', 'issue_number', 'issue_assignee', 'created_at', 'organization']].sort_values('created_at').dropna().drop_duplicates(
    ['repo_id', 'issue_number', 'issue_assignee'])
issue_assignee['issue_assignee'] = issue_assignee['issue_assignee'].apply(ast.literal_eval)
issue_assignee['issue_assignee'] = issue_assignee['issue_assignee'].apply(getLogin)


# individuals who are the assignee have contributor or owner privileges

# In[23]:


issue_assignee['permissions'] = 'triage'


# In[24]:


df_issue_clean['issue_assignees'] = df_issue_clean['issue_assignees'].apply(lambda x: ast.literal_eval(x) if type(x) != list and x != None else x)
issue_assignees = df_issue_clean[['repo_id', 'issue_number', 'issue_assignees','organization']].explode('issue_assignees').dropna()
issue_assignees['issue_assignees'] = issue_assignees['issue_assignees'].apply(getLogin)
issue_assignees = issue_assignees.drop_duplicates().groupby(['repo_id', 'issue_number','organization']).agg({'issue_assignees':list}).reset_index()


# In[25]:


issue_commenters = pd.concat([df_issue_clean[(df_issue_clean['type'] == 'IssueCommentEvent')][['repo_id','issue_number','actor_id']].reset_index(drop = True),
                              df_issue_clean[['repo_id','issue_number','issue_user_id']].rename({'issue_user_id':'actor_id'}, axis = 1).reset_index(drop = True),
                             ]).drop_duplicates()
issue_commenters = issue_commenters.groupby(['repo_id','issue_number']).agg({'actor_id':list}).reset_index()


# In[26]:


# https://docs.github.com/en/issues/tracking-your-work-with-issues/assigning-issues-and-pull-requests-to-other-github-users#


# In[27]:


issue_assignees_not_commenter = pd.merge(issue_assignees, issue_commenters, how = 'left').dropna()
issue_assignees_not_commenter['issue_assignees'] = issue_assignees_not_commenter.apply(
    lambda x: [ele for ele in x['issue_assignees'] if ele not in x['actor_id']], axis = 1)
issue_assignees_not_commenter = issue_assignees_not_commenter.explode('issue_assignees').dropna().drop('actor_id', axis = 1)


# In[28]:


issue_assignees_date = df_issue_clean[['repo_id', 'issue_number', 'issue_assignees', 'created_at','organization']].explode('issue_assignees').dropna()
issue_assignees_date['issue_assignees'] = issue_assignees_date['issue_assignees'].apply(getLogin)
issue_assignees_date = issue_assignees_date.sort_values('created_at').drop_duplicates(['repo_id','issue_number','issue_assignees'])


# In[29]:


issue_assignees_not_commenter = pd.merge(issue_assignees_not_commenter, issue_assignees_date)


# In[30]:


issue_assignees_not_commenter['permissions'] = 'triage'


# In[31]:


issue_ranked = df_issue_clean[['repo_id', 'actor_id', 'created_at', 'permissions', 'organization']].dropna().sort_values(
    'created_at').drop_duplicates(
    ['repo_id','actor_id','permissions'])


# In[32]:


issue_ranked = pd.concat([
    issue_ranked,
    issue_assignees_not_commenter[['repo_id', 'issue_assignees', 'created_at', 'permissions', 'organization']].rename(
        {'issue_assignees':'actor_id'}, axis = 1),
    issue_assignee[['repo_id', 'issue_assignee', 'created_at', 'permissions', 'organization']].rename(
        {'issue_assignee':'actor_id'}, axis = 1)]).reset_index(drop = True)


# In[33]:


num_rank = {'owner': 1, 'write': 2, 'triage': 3}


# In[34]:


issue_ranked['derived_rank'] = issue_ranked['permissions'].apply(lambda x: num_rank[x])
issue_ranked = issue_ranked.sort_values(['created_at', 'derived_rank']).drop_duplicates(
    ['repo_id', 'actor_id', 'permissions']).reset_index(drop = True)


# In[35]:


issue_ranked['type'] = 'issue'


# In[36]:


issue_ranked.to_parquet('data/merged_data/imputed_ranks/issue_ranked.parquet')


# In[37]:


allIssueActors = df_issue_clean[['created_at','repo_id', 'actor_id', 'organization']].sort_values('created_at')
allIssueActors['actor_id'] = pd.to_numeric(allIssueActors['actor_id'], errors = 'coerce')
allIssueActors = allIssueActors[~allIssueActors['actor_id'].isna()]
allIssueActors = allIssueActors.drop_duplicates(['repo_id', 'actor_id', 'organization'])


# In[38]:


allIssueActors.to_parquet('data/merged_data/imputed_ranks/allIssueActors.parquet')


# ### PRs

# **PRs:**  
# 
# 1. PR event `edited`, `closed`, `reopened`, `assigned`, `unassigned`, `review_requested`, `review_request_removed`, `labeled`, `unlabeled`, and `synchronize`.
# 2. PR Review event `created`
# 3. Someone else's PR Review Comment `edit` or `delete`
# 4. **merged by column**

# In[39]:


get_ipython().run_cell_magic('time', '', "# Read data on pr's, pr review's, pr review comment's\nprcols = ['created_at', 'repo_id', 'repo_name', 'type', 'actor_id', 'actor_login', 'pr_number', 'pr_action',\n          'pr_assignee','pr_assignees', 'org_id', 'org_login', 'pr_requested_reviewers', 'pr_merged_by_login',\n          'pr_merged_by_id','pr_user_id']\nprData = glob.glob('data/github_clean/filtered_github_data/prEvent*.parquet')\nprData.extend(glob.glob('data/github_clean/github_data_pre_18/prEvent*.parquet'))\nprData.extend(glob.glob('data/github_clean/github_data_2324/prEvent*.parquet'))\ndf_prData = pd.concat([readDf(pd.read_parquet(ele), prcols) for ele in prData]).reset_index(drop = True)\n\nprreviewcols = ['created_at', 'repo_id', 'repo_name', 'type', 'actor_id', 'actor_login', 'pr_number', 'pr_review_action',\n                'pr_assignee','pr_assignees', 'org_id', 'org_login', 'pr_requested_reviewers', 'pr_merged_by_login',\n                'pr_merged_by_id','pr_user_id']\nprDataReview = glob.glob('data/github_clean/filtered_github_data/prReviewEvent*.parquet')\nprDataReview.extend(glob.glob('data/github_clean/github_data_pre_18/prReviewEvent*.parquet'))\nprDataReview.extend(glob.glob('data/github_clean/github_data_2324/prReviewEvent*.parquet'))\ndf_prDataReview = pd.concat([readDf(pd.read_parquet(ele), prreviewcols) for ele in prDataReview]).reset_index(drop = True)\n\nprreviewcommentcols = ['created_at', 'repo_id', 'repo_name', 'type', 'actor_id', 'actor_login', 'pr_number', 'pr_review_comment_action',\n                       'pr_assignee','pr_assignees', 'org_id', 'org_login', 'pr_requested_reviewers', 'pr_merged_by_login',\n                       'pr_merged_by_id','pr_user_id']\nprDataReviewComment = glob.glob('data/github_clean/filtered_github_data/prReviewCommentEvent*.parquet')\nprDataReviewComment.extend(glob.glob('data/github_clean/github_data_pre_18/prReviewCommentEvent*.parquet'))\nprDataReviewComment.extend(glob.glob('data/github_clean/github_data_2324/prReviewCommentEvent*.parquet'))\ndf_prDataReviewComment = pd.concat([readDf(pd.read_parquet(ele), prreviewcols) for ele in prDataReviewComment]).reset_index(drop = True)\n")


# In[40]:


prEventData = pd.concat([df_prData, df_prDataReview, df_prDataReviewComment])


# In[41]:


prEventData['repo_org'] = prEventData['repo_name'].apply(name_split)
prEventData['permissions'] = (prEventData['repo_org'] == prEventData['actor_login']).apply(owner)
orgs_pr = prEventData[['org_id', 'org_login']].dropna().drop_duplicates()['org_login'].tolist()
#orgs = pd.read_parquet('data/merged_data/imputed_ranks/org_login_id.parquet', index_col = 0)['org_login']
orgs.extend(orgs_pr)
prEventData['organization'] = prEventData['repo_org'].isin(orgs)


# In[42]:


prEventData['created_at'] = pd.to_datetime(prEventData['created_at'], errors = 'coerce')
prEventData['created_at'] = prEventData['created_at'].dt.tz_localize(None)


# In[43]:


prEventData.reset_index(drop = True, inplace = True)


# In[44]:


triage_action = prEventData[
    ((prEventData['pr_action'].isin(['closed', 'reopened'])) & (prEventData['type'] == 'PullRequestEvent')) |
    ((prEventData['pr_review_action'] == 'created') & (prEventData['type'] == 'PullRequestReviewEvent')) &
    (prEventData['permissions'].isna())].index
prEventData.loc[triage_action, 'permissions'] = 'triage'

write_action = prEventData[
    (prEventData['pr_action'].isin(['synchronize'])) &
    (prEventData['permissions'].isna())].index
prEventData.loc[write_action, 'permissions'] = 'write'


# In[45]:


# anyone who can assign other people to an pr
pr_assignee = prEventData[['repo_id', 'pr_number', 'pr_assignee', 'created_at' ,'organization']].sort_values('created_at').dropna().drop_duplicates(
    ['repo_id', 'pr_number', 'pr_assignee'])
pr_assignee['pr_assignee'] = pr_assignee['pr_assignee'].apply(ast.literal_eval)
pr_assignee = pr_assignee[pr_assignee['pr_assignee'].apply(lambda x: type(x) != list)]
pr_assignee['pr_assignee'] = pr_assignee['pr_assignee'].apply(getLogin)


# individuals who are the assignee have contributor or owner privileges

# In[46]:


pr_assignee['permissions'] = 'triage'


# In[47]:


prEventData['pr_assignees'] = prEventData['pr_assignees'].apply(lambda x: ast.literal_eval(x) if type(x) != list and type(x) != float and x != None else x)
pr_assignees = prEventData[['repo_id', 'pr_number', 'pr_assignees','organization']].explode('pr_assignees').dropna()
pr_assignees['pr_assignees'] = pr_assignees['pr_assignees'].apply(getLogin)
pr_assignees = pr_assignees.drop_duplicates().groupby(['repo_id', 'pr_number','organization']).agg({'pr_assignees':list}).reset_index()


# In[48]:


pr_commenters = pd.concat([prEventData[(prEventData['type'] == 'PullRequestReviewCommentEvent')][['repo_id','pr_number','actor_id']].reset_index(drop = True),
                           prEventData[['repo_id','pr_number','pr_user_id']].rename({'pr_user_id':'actor_id'}, axis = 1).reset_index(drop = True),
                          ]).drop_duplicates()
pr_commenters = pr_commenters.groupby(['repo_id','pr_number']).agg({'actor_id':list}).reset_index()


# In[49]:


# https://docs.github.com/en/issues/tracking-your-work-with-issues/assigning-issues-and-pull-requests-to-other-github-users#


# In[50]:


pr_assignees_not_commenter = pd.merge(pr_assignees, pr_commenters, how = 'left')
pr_assignees_not_commenter['pr_assignees'] = pr_assignees_not_commenter.apply(
    lambda x: x['pr_assignees'] if type(x['actor_id']) == float else [ele for ele in x['pr_assignees'] if ele not in x['actor_id']], axis = 1)
pr_assignees_not_commenter = pr_assignees_not_commenter.explode('pr_assignees').dropna().drop('actor_id', axis = 1)


# In[51]:


pr_assignees_date = prEventData[['repo_id', 'pr_number', 'pr_assignees', 'created_at']].explode('pr_assignees').dropna()
pr_assignees_date['pr_assignees'] = pr_assignees_date['pr_assignees'].apply(getLogin)
pr_assignees_date = pr_assignees_date.sort_values('created_at').drop_duplicates(['repo_id','pr_number','pr_assignees'])


# In[52]:


pr_assignees_not_commenter = pd.merge(pr_assignees_not_commenter, pr_assignees_date)


# In[53]:


pr_assignees_not_commenter['permissions'] = 'triage'


# In[54]:


prEventData['pr_requested_reviewers_length'] = prEventData['pr_requested_reviewers'].apply(
    lambda x: len(x) if not pd.isnull(x) else 0)
pr_data_index = prEventData[prEventData['pr_requested_reviewers_length']>2].index
prEventData.loc[pr_data_index, 'pr_requested_reviewers'] = prEventData.loc[pr_data_index].apply(
    lambda x: ast.literal_eval(x['pr_requested_reviewers']) if x['pr_requested_reviewers_length'] > 2 else np.nan, axis = 1)


# In[55]:


pr_reviewers = prEventData[prEventData['pr_requested_reviewers_length'].apply(lambda x: x>2)][[
    'created_at', 'repo_id', 'pr_requested_reviewers', 'organization']]
pr_reviewers['pr_requested_reviewers'] = pr_reviewers['pr_requested_reviewers'].apply(
    lambda x: [getLogin(ele) for ele in x])
pr_reviewers = pr_reviewers.explode("pr_requested_reviewers")
pr_reviewers = pr_reviewers.sort_values('created_at').drop_duplicates(['pr_requested_reviewers', 'repo_id'])
pr_reviewers['permissions'] = 'triage'
pr_reviewers.rename({'pr_requested_reviewrs': 'actor_id'}, axis = 1, inplace = True)


# In[56]:


prEventData.drop('pr_requested_reviewers_length', axis = 1, inplace = True)


# In[57]:


prEventData['repo_id'] = pd.to_numeric(prEventData['repo_id'], errors = 'coerce')
prEventData['actor_id'] = pd.to_numeric(prEventData['actor_id'], errors = 'coerce')
prEventData['pr_number'] = pd.to_numeric(prEventData['pr_number'], errors = 'coerce')
prEventData['org_id'] = pd.to_numeric(prEventData['org_id'], errors = 'coerce')
prEventData['pr_user_id'] = pd.to_numeric(prEventData['pr_user_id'], errors = 'coerce')
#prEventData = prEventData[(~prEventData['repo_id'].isna()) & (~prEventData['actor_id'].isna())
#     & (~prEventData['pr_number'].isna()) & (~prEventData['org_id'].isna()) & (~prEventData['pr_user_id'].isna())]


# In[58]:


prEventData['pr_requested_reviewers'] = prEventData['pr_requested_reviewers'].apply(lambda x: x if type(x) != str else np.nan)


# In[59]:


prEventData.to_parquet('data/merged_data/prEventData.parquet',)


# In[60]:


prEventData[~prEventData['pr_merged_by_id'].isna()]['type'].value_counts()


# In[61]:


pr_mergers = prEventData[~prEventData['pr_merged_by_id'].isna()][[
    'created_at', 'repo_id', 'pr_merged_by_id', 'organization']]
pr_mergers = pr_mergers.sort_values('created_at').drop_duplicates(['pr_merged_by_id', 'repo_id'])
pr_mergers['permissions'] = 'write'
pr_mergers.rename({'pr_merged_by_id': 'actor_id'}, axis = 1, inplace = True)


# In[62]:


pr_ranked = prEventData[['repo_id', 'actor_id', 'created_at', 'permissions', 'organization']].dropna().sort_values(
    'created_at').drop_duplicates(
    ['repo_id','actor_id','permissions'])


# In[63]:


pr_ranked = pd.concat([pr_ranked,
           pr_assignee.rename({'pr_assignee':'actor_id'}, axis = 1),
           pr_assignees_not_commenter.rename({'pr_assignees':'actor_id'}, axis = 1),
           pr_reviewers.rename({'pr_requested_reviewers':'actor_id'}, axis = 1),
           pr_mergers]).drop('pr_number', axis = 1).sort_values('created_at').drop_duplicates(
    ['repo_id', 'actor_id', 'organization', 'permissions'])


# In[64]:


pr_ranked['type'] = 'pr'


# In[69]:


pr_ranked['repo_id'] = pd.to_numeric(pr_ranked['repo_id'])


# In[70]:


pr_ranked.to_parquet('data/merged_data/imputed_ranks/pr_ranked.parquet')


# In[71]:


allPRActors = prEventData[['created_at','repo_id', 'actor_id', 'organization']].sort_values('created_at')
allPRActors['actor_id'] = pd.to_numeric(allPRActors['actor_id'], errors = 'coerce')
allPRActors = allPRActors[~allPRActors['actor_id'].isna()]
allPRActors = allPRActors.drop_duplicates(['repo_id', 'actor_id', 'organization'])


# In[72]:


allPRActors.to_parquet('data/merged_data/imputed_ranks/allPRActors.parquet')


# # pushes

# In[73]:


pushData = glob.glob('data/github_clean/filtered_github_data/pushEvent*')
pushData.extend(glob.glob('data/github_clean/github_data_pre_18/pushEvent*'))
pushData.extend(glob.glob('data/github_clean/github_data_2324/pushEvent*'))

pushEventData = pd.concat([pd.read_parquet(ele) for ele in pushData])[[
    'actor_login', 'actor_id', 'org_id', 'org_login', 'type', 'created_at', 'repo_id', 'repo_name', 'push_id', 'push_ref']]


# In[74]:


pushEventData['created_at'] = pd.to_datetime(pushEventData['created_at'])
pushEventData = pushEventData.reset_index(drop = True)


# In[75]:


pushEventData['repo_org'] = pushEventData['repo_name'].apply(name_split)
pushEventData['permissions'] = (pushEventData['repo_org'] == pushEventData['actor_login']).apply(owner)
orgs_push = pushEventData[['org_id', 'org_login']].dropna().drop_duplicates()['org_login'].tolist()
#orgs_pr = pd.read_csv('orgs_pr.csv', index_col = 0)['0'].tolist()
orgs.extend(orgs_push)
pushEventData['organization'] = pushEventData['repo_org'].isin(orgs)


# In[76]:


not_owner = pushEventData[pushEventData['permissions'].isna()].index
pushEventData.loc[not_owner, 'permissions'] = 'write'


# In[ ]:


committer_push = pd.read_parquet('data/merged_data/committers_info_push.parquet')
committer_pr = pd.read_parquet('data/merged_data/committers_info_pr.parquet')


committer_push.dropna(inplace = True)
committer_pr.dropna(inplace = True)
committer_push['committer_info'] = committer_push['committer_info'].apply(ast.literal_eval)
committer_pr['committer_info'] = committer_pr['committer_info'].apply(ast.literal_eval)
committer_push['email'] = committer_push['email'].apply(lambda x: x.lower())
committer_pr['email'] = committer_pr['email'].apply(lambda x: x.lower())

email_committer_info = pd.concat([committer_push[['email', 'actor_id']],
                                 committer_pr[['email', 'actor_id']]]).drop_duplicates().set_index('email').to_dict()['actor_id']

cols = ['repo_id', 'repo_name', 'actor_id', 'org_id', 'commit changes total','commit author name',
        'commit author email','committer name','commmitter email','commit files changed count','commit time',
       'commit additions','commit deletions','commit file changes', 'push_id']

commits_push_list = glob.glob('data/github_commits/parquet/filtered_github_data_large/*_push_*')
commits_push_list.extend(glob.glob('data/github_commits/parquet/github_data_pre_18/*_push_*'))
commits_push_list.extend(glob.glob('data/github_commits/parquet/github_data_2324/*_push_*'))

df_commits_push = pd.concat([pd.read_parquet(file) for file in commits_push_list])[cols]
df_commits_push['type'] = 'push commits'


# In[ ]:


df_commits = df_commits_push
df_commits['repo_org'] = df_commits['repo_name'].apply(lambda x: x.split("/")[0])
df_commits['organization'] = df_commits['repo_org'].isin(orgs)

df_commits.reset_index(drop = True, inplace = True)


df_commits = df_commits[['commit author email', 'commit time', 'repo_id', 'organization']]
df_commits['commit time'] = pd.to_datetime(df_commits['commit time'], unit = 's')

df_commits = df_commits[~df_commits['commit author email'].isna()]
df_commits = df_commits[~df_commits['repo_id'].isna()]
df_commits['commit_actor_id'] = df_commits['commit author email'].apply(lambda x: email_committer_info.get(x.lower(), np.nan))

df_commits = df_commits[~df_commits['commit_actor_id'].isna()]


df_commits['commit_actor_id'] = df_commits['commit author email'].apply(lambda x: email_committer_info.get(x.lower(), np.nan))

df_commits = df_commits[~df_commits['commit_actor_id'].isna()]


# In[ ]:


df_commits['permissions'] = 'write'
df_commits = df_commits.drop('commit author email', axis = 1).rename(
    {'commit time': 'created_at', 'commit_actor_id':'actor_id'}, axis = 1)


# In[ ]:


df_commits['created_at'] = df_commits['created_at'].dt.tz_localize(pytz.UTC)


# In[ ]:


df_commits_pushes = df_commits.sort_values('created_at').drop_duplicates(
    ['repo_id', 'actor_id', 'organization', 'permissions'])


# In[ ]:


push_ranked = pushEventData[['created_at','repo_id','organization','actor_id','permissions']].sort_values('created_at').drop_duplicates(
    ['repo_id', 'actor_id', 'organization', 'permissions'])


# In[ ]:


push_ranked = pd.concat([df_commits_pushes, push_ranked]).sort_values('created_at').drop_duplicates(
    ['repo_id', 'actor_id', 'organization', 'permissions'])


# In[ ]:


push_ranked['type'] = 'push'


# In[ ]:


push_ranked.to_parquet('data/merged_data/imputed_ranks/push_ranked.parquet')


# # combining everything

# In[ ]:


import pytz


# In[ ]:


#issue_ranked = pd.read_csv('data/merged_data/imputed_ranks/issue_ranked.csv', index_col = 0)
#push_ranked = pd.read_csv('data/merged_data/imputed_ranks/push_ranked.csv', index_col = 0)
#pr_ranked = pd.read_csv('data/merged_data/imputed_ranks/pr_ranked.csv', index_col = 0)


# In[ ]:


push_ranked['type'] = 'push'


# In[ ]:


all_ranked = pd.concat([push_ranked, issue_ranked, pr_ranked]).reset_index(drop = True)


# In[ ]:


all_ranked.drop('derived_rank', axis = 1, inplace = True)


# In[ ]:


all_ranked['created_at'] = all_ranked['created_at'].apply(lambda x: datetime(x.year, x.month, x.day, x.hour, x.minute, x.second, tzinfo = pytz.UTC))


# In[ ]:


all_ranked = all_ranked.sort_values('created_at').drop_duplicates(['repo_id', 'actor_id', 'organization', 'permissions'])


# In[ ]:


# eliminate collaborators who occur after owner
owner_date = all_ranked[all_ranked['permissions'] == 'owner'][['repo_id','actor_id','created_at']].rename({
    'created_at':'date_owner'}, axis = 1)
write_date = all_ranked[all_ranked['permissions'] == 'write'][['repo_id','actor_id','created_at']].rename({
    'created_at':'date_write'}, axis = 1)

all_ranked = pd.merge(all_ranked, owner_date, how = 'left')
all_ranked = pd.merge(all_ranked, write_date, how = 'left')


# In[ ]:


all_ranked = all_ranked.sort_values(['repo_id', 'actor_id','created_at'])


# In[ ]:


all_ranked_v1 = all_ranked[all_ranked.apply(
    lambda x: x['permissions'] == 'owner' or 
    (x['permissions'] == 'triage' and 
         (pd.isnull(x['date_owner']) or x['created_at']<x['date_owner']) and
         (pd.isnull(x['date_write']) or x['created_at']<x['date_write'])) or
    (x['permissions'] == 'write' and 
         (pd.isnull(x['date_owner']) or x['created_at']<x['date_owner'])), axis = 1)].drop(
    ['date_owner', 'date_write'], axis = 1)


# In[ ]:


all_ranked_v1.to_parquet('data/merged_data/imputed_ranks/all_ranked_v1.parquet')


# In[ ]:


all_ranked['corrected_permissions'] = all_ranked['permissions']


# In[ ]:


# cases where someone has triage permissions and then suddenly has write permissions 2 days or less later
# correct to just always having write permissions
triage_to_write = all_ranked[all_ranked.apply(
    lambda x: (x['permissions'] == 'triage' and 
              (x['date_write']-x['created_at']).days <= 2), axis = 1)].index
all_ranked.loc[triage_to_write, 'corrected_permissions'] = 'write'


# In[ ]:


# cases where someone has triage permissions and then suddenly has owner permissions 2 days or less later
# correct to just always having owner permissions
triage_to_owner = all_ranked[all_ranked.apply(
    lambda x: (x['permissions'] == 'triage' and 
              (x['date_owner']-x['created_at']).days <= 2), axis = 1)].index
all_ranked.loc[triage_to_owner, 'corrected_permissions'] = 'owner'


# In[ ]:


# cases where someone has write permissions and then suddenly has owner permissions 2 days or less later
# correct to just always having owner permissions
write_to_owner = all_ranked[all_ranked.apply(
    lambda x: (x['permissions'] == 'write' and 
              (x['date_owner']-x['created_at']).days <= 2), axis = 1)].index
all_ranked.loc[write_to_owner, 'corrected_permissions'] = 'owner'


# In[ ]:


all_ranked.sort_values('created_at').drop_duplicates(['repo_id', 'actor_id', 'organization','corrected_permissions'])[[
    'created_at', 'repo_id', 'actor_id', 'organization','corrected_permissions']]


# In[ ]:


all_ranked_v2 = all_ranked.sort_values('created_at').drop_duplicates(['repo_id', 'actor_id', 'organization','corrected_permissions'])[['created_at', 'repo_id', 'actor_id', 'organization','corrected_permissions']]


# In[ ]:


all_ranked_v2[['created_at', 'repo_id', 'actor_id', 'organization','corrected_permissions']].to_parquet(
    'data/merged_data/imputed_ranks/all_ranked_v2.parquet')


# # Other Data

# In[ ]:


allPushActors = push_ranked[['created_at','repo_id','actor_id','organization']]
allPushActors['created_at'] = pd.to_datetime(allPushActors['created_at'], errors = 'coerce', utc = True)
allIssueActors['created_at'] = pd.to_datetime(allIssueActors['created_at'], errors = 'coerce', utc = True) 
allPRActors['created_at'] = pd.to_datetime(allPRActors['created_at'], errors = 'coerce', utc = True)

# In[ ]:


allActors = pd.concat([allPushActors,allIssueActors,allPRActors]).sort_values('created_at').drop_duplicates(
    ['repo_id', 'actor_id', 'organization'])


# In[ ]:


allActors.to_parquet('data/merged_data/imputed_ranks/allActors.parquet')


# In[ ]:


pd.concat([prEventData[['repo_id', 'repo_name']].drop_duplicates(),
           pushEventData[['repo_id', 'repo_name']].drop_duplicates(),
           df_issue_clean[['repo_id', 'repo_name']].drop_duplicates()]).drop_duplicates().dropna().to_parquet(
    'data/merged_data/imputed_ranks/repo_login_id.parquet')


# In[ ]:




# In[ ]:


def cleanAssignee(df, col):
    df = df[col]
    df = df.dropna().apply(lambda x: ast.literal_eval(x) if x[0] == '{' else np.nan).dropna()
    df = pd.DataFrame(df)
    df['actor_id'] = df[col].apply(lambda x: x.get('id', np.nan))
    df['actor_login'] = df[col].apply(lambda x: x.get('login', np.nan))
    return df.dropna()


# In[ ]:


def cleanAssignees(df, col):
    df = df[col]
    df = df[df.apply(lambda x: type(x) == list and len(x)>0)]
    df = pd.DataFrame(df.explode().apply(
        lambda x: ast.literal_eval(x) if type(x) != dict else x))
    df['actor_id'] = df[col].apply(lambda x: x.get('id', np.nan))
    df['actor_login'] = df[col].apply(lambda x: x.get('login', np.nan))
    return df.dropna()


# In[ ]:


df_issue_clean_assignee = cleanAssignee(df_issue_clean, 'issue_assignee')
prEventData_assignee = cleanAssignee(prEventData, 'pr_assignee')


# In[ ]:


df_issue_clean_assignees = cleanAssignees(df_issue_clean, 'issue_assignees')
prEventData_assignees = cleanAssignees(prEventData, 'pr_assignees')
prEventData_requested_reviewers = cleanAssignees(prEventData, 'pr_requested_reviewers')


# In[ ]:


actor_login_id = pd.concat([prEventData[['actor_id', 'actor_login']].drop_duplicates(),
           pushEventData[['actor_id', 'actor_login']].drop_duplicates(),
           df_issue_clean[['actor_id', 'actor_login']].drop_duplicates(),
           prEventData[['pr_merged_by_id', 'pr_merged_by_login']].rename(
               {'pr_merged_by_id':'actor_id',
                'pr_merged_by_login': 'actor_login'}, axis = 1).drop_duplicates(),
           df_issue_clean[['issue_user_id','issue_user_login']].rename(
               {'issue_user_id':'actor_id',
                'issue_user_login': 'actor_login'}, axis = 1).drop_duplicates(),
           df_issue_clean_assignee[['actor_id','actor_login']].drop_duplicates(),
           prEventData_assignee[['actor_id','actor_login']].drop_duplicates(),
           df_issue_clean_assignees[['actor_id','actor_login']].drop_duplicates(),
           prEventData_assignees[['actor_id','actor_login']].drop_duplicates(),
           prEventData_requested_reviewers[['actor_id','actor_login']].drop_duplicates(),
          ]).drop_duplicates().dropna()


pd.concat([prEventData[['org_id', 'org_login']].drop_duplicates(),
           pushEventData[['org_id', 'org_login']].drop_duplicates(),
           df_issue_clean[['org_id', 'org_login']].drop_duplicates()]).drop_duplicates().dropna().to_parquet(
    'data/merged_data/imputed_ranks/org_login_id.parquet')


# In[ ]:

actor_login_id['actor_id'] = pd.to_numeric(actor_login_id['actor_id'])
actor_login_id.to_parquet('data/merged_data/imputed_ranks/actor_login_id.parquet')

