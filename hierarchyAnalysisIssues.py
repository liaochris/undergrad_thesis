#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import display, HTML

def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n","<br>") ) )


# In[ ]:


## Import Libraries and Data


# In[ ]:


import glob
import dask.dataframe as dd
import pandas as pd
from pandarallel import pandarallel
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import ast
import numpy as np
from operator import itemgetter
from stargazer.stargazer import Stargazer
import os
import datetime
from dateutil.rrule import rrule, MONTHLY, YEARLY, WEEKLY
from dateutil.relativedelta import relativedelta
from linearmodels.panel import PanelOLS
import multiprocessing
import statsmodels.formula.api as smf
import re
from itertools import product
import pytz


# In[ ]:


plt.style.use("seaborn")


# In[ ]:


pandarallel.initialize(progress_bar = False)


# In[ ]:


pd.set_option('display.max_columns', None)


# In[ ]:


def readDf(df,cols):
    cols = [col for col in cols if col in df.columns]
    return df[cols]


# In[ ]:


def readGlobDf(filename):
    df = pd.concat([pd.read_parquet(file, engine = 'pyarrow') for file in 
               glob.glob(filename.replace(".parquet","*.parquet"))])
    return df


# In[ ]:


def read_parquet(f, columns = None):
    try:
        if columns == None:
            return pd.read_parquet(f)
        else:
            return pd.read_parquet(f, columns = columns)
    except:
        return pd.DataFrame()


# # Data that I have

# ## Data that links actor, organization and repository ids to logins (names)

# In[ ]:


df_actor = readGlobDf('data/merged_data/imputed_ranks/actor_login_id.parquet')
df_org = readGlobDf('data/merged_data/imputed_ranks/org_login_id.parquet')
df_repo = readGlobDf('data/merged_data/imputed_ranks/repo_login_id.parquet')


# In[ ]:


df_repo['org_login'] = df_repo['repo_name'].apply(lambda x: x.split("/")[0])


# In[ ]:


org_dict = pd.merge(df_repo, df_org).drop_duplicates(['repo_id'])[['repo_id', 'org_id']].set_index('repo_id').to_dict()['org_id']


# ## Data on when all actors first appeared

# In[ ]:


df_actor_start = readGlobDf('data/merged_data/imputed_ranks/allActors.parquet')


# ## Data on imputed ranks

# In[ ]:


all_ranked_v1 = readGlobDf('data/merged_data/imputed_ranks/all_ranked_v1.parquet')
all_ranked_v1['created_at'] = pd.to_datetime(all_ranked_v1['created_at'])

all_ranked_v1 = pd.merge(
    all_ranked_v1.drop('org_id', axis = 1), 
    df_actor_start[['repo_id','org_id']].sort_values('repo_id').drop_duplicates('repo_id'), how = 'left')


# In[ ]:


all_ranked_v2 = readGlobDf('data/merged_data/imputed_ranks/all_ranked_v2.parquet')
all_ranked_v2['created_at'] = pd.to_datetime(all_ranked_v2['created_at'])

all_ranked_v2 = pd.merge(
    all_ranked_v2.drop('org_id', axis = 1), 
    df_actor_start[['repo_id','org_id']].sort_values('repo_id').drop_duplicates('repo_id'), how = 'left')


# ## Data on whose committed to a repo, and aggregated data describing their email addresses

# In[ ]:


committer_push = readGlobDf('data/merged_data/committers_info_push.parquet')
committer_pr = readGlobDf('data/merged_data/committers_info_pr.parquet')

# note that the below is an underestimate of the # of corporate contributors (because some people use their personal emails)
committer_info = readGlobDf('data/merged_data/committer_detailed_info.parquet')


# ## Data on Issues and PRs

# In[ ]:


issueEventData = readGlobDf('data/merged_data/issue_data.parquet')
prEventData = readGlobDf('data/merged_data/prEventData.parquet')


# In[ ]:


linked_issues = pd.read_parquet('data/merged_data/linked_issues.parquet')
issueEventData = pd.merge(issueEventData, linked_issues[['repo_id','potential_issues','pr_number']],
                          left_on = ['repo_id','issue_number'], 
                          right_on = ['repo_id','potential_issues'], how = 'left')
pr_authors = pd.concat([prEventData[prEventData['pr_action'] == 'opened'][['repo_id','pr_number','actor_id']].rename({'actor_id':'pr_user_id'}, axis = 1),
                        prEventData[['repo_id','pr_number','pr_user_id']].dropna()]).drop_duplicates()
issueEventData = pd.merge(issueEventData, pr_authors, how = 'left')


# In[ ]:


prEventData = pd.merge(prEventData, linked_issues[['repo_id','potential_issues','pr_number']],
                       on = ['repo_id','pr_number'], how = 'left')
issue_authors = issueEventData[['repo_id','issue_number','issue_user_id']].dropna().drop_duplicates()
prEventData = pd.merge(prEventData, issue_authors,
                       left_on = ['repo_id','potential_issues'],
                       right_on = ['repo_id','issue_number'], how = 'left')


# # Analysis

# In[ ]:


org_true = all_ranked_v2[all_ranked_v2['org_id'].isna()].index
all_ranked_v2.loc[org_true, 'corrected_permissions'] = all_ranked_v2.loc[org_true, 'corrected_permissions'].apply(
    lambda x: 'collaborator' if x != 'owner' else x)


# In[ ]:


import pytz


# In[ ]:


print('\nOrganization',
      all_ranked_v2[~all_ranked_v2['org_id'].isna()].sort_values(
           'created_at', ascending = False).drop_duplicates(
          ['repo_id', 'actor_id', 'org_id'])['corrected_permissions'].value_counts(normalize = True).round(2))
print('\nNot Organization',
      all_ranked_v2[all_ranked_v2['org_id'].isna()].sort_values(
          'created_at', ascending = False).drop_duplicates(
          ['repo_id', 'actor_id', 'org_id'])['corrected_permissions'].value_counts(normalize = True).round(2))


# ## Analysis - think about TASKS

# In[ ]:


issueEventData['created_at']= pd.to_datetime(issueEventData['created_at'], format='mixed', errors = 'coerce')
issueEventData['created_at'] = issueEventData['created_at'].dt.tz_localize(pytz.UTC)


# In[ ]:


df_comments = issueEventData[(issueEventData['type'] == 'IssueCommentEvent')]
df_closed = issueEventData[issueEventData['issue_action'] == 'closed']


# In[ ]:


all_ranked_v2['permissions_date'] = all_ranked_v2.apply(lambda x: [x['corrected_permissions'], x['created_at']], axis = 1)


# In[ ]:


permissions_dict = all_ranked_v2[(~all_ranked_v2['org_id'].isna()) & (all_ranked_v2['corrected_permissions'] != 'owner')].groupby(
    ['actor_id', 'repo_id']).agg({'permissions_date':lambda x: sorted(list(x), key = lambda y: y[1])}).to_dict()['permissions_date']


# In[ ]:


def addPermissions(lst, date):
    if type(lst) == float:
        return np.nan
    if len(lst) == 1:
        return lst[0][0]
    else:
        lst = sorted(lst, key = lambda x: x[1], reverse = True)
        for i in range(len(lst)):
            if lst[i][1].tz_localize(None)>=date.tz_localize(None):
                return lst[i][0]
        return lst[-1][0]


# In[ ]:


df_comments['repo_id'] = df_comments['repo_id'].apply(lambda x: org_dict.get(x, f"REPO_{x}"))
df_closed['repo_id'] = df_closed['repo_id'].apply(lambda x: org_dict.get(x, f"REPO_{x}"))
df_comments['permissions'] = df_comments.apply(lambda x: addPermissions(
    permissions_dict.get((x['actor_id'], x['repo_id']), np.nan), x['created_at']), axis = 1)
df_closed['permissions'] = df_closed.apply(lambda x: addPermissions(permissions_dict.get((x['actor_id'], x['repo_id']), np.nan), x['created_at']), axis = 1)


# In[ ]:


df_commenting_permissions = df_comments[~df_comments['permissions'].isna()]
df_commenting_permissions['organization'] = True
df_commenting_permissions['corrected_permissions'] = df_commenting_permissions['permissions']

df_closed_permissions = df_closed[~df_closed['permissions'].isna()]
df_closed_permissions['organization'] = True
df_closed_permissions['corrected_permissions'] = df_closed_permissions['permissions']


# ## Commits

# In[ ]:


committer_push.dropna(inplace = True)
committer_pr.dropna(inplace = True)


# In[ ]:


committer_push['committer_info'] = committer_push['committer_info'].apply(lambda x: ast.literal_eval if type(x) != list else x)
committer_pr['committer_info'] = committer_pr['committer_info'].apply(lambda x: ast.literal_eval if type(x) != list else x)


# In[ ]:


committer_push['email'] = committer_push['email'].apply(lambda x: x.lower())
committer_pr['email'] = committer_pr['email'].apply(lambda x: x.lower())


# In[ ]:


email_committer_info = pd.concat([committer_push[['email', 'actor_id']],
                                 committer_pr[['email', 'actor_id']]]).drop_duplicates().set_index('email').to_dict()['actor_id']


# In[ ]:


def readCommits(file, usecols):
    return pd.read_parquet(file, usecols = usecols)


# In[ ]:


cols = ['repo_id', 'repo_name', 'actor_id', 'org_id', 'commit changes total','commit author name',
        'commit author email','committer name','commmitter email','commit files changed count','commit time',
       'commit additions','commit deletions','commit file changes']
cols_pr = cols.copy()
cols.extend(['push_id',])
cols_pr.extend(['pr_number'])


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_commits_pr = pd.concat([pd.read_parquet(file, engine = 'pyarrow') for file in \n                           glob.glob('data/github_commits/parquet/filtered_github_data/*_pr_*')])\ndf_commits_pr['type'] = 'pr commits'\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_commits_push = pd.concat([pd.read_parquet(file, engine = 'pyarrow') for file in \n                           glob.glob('data/github_commits/parquet/filtered_github_data/*_push_*')])\ndf_commits_push['type'] = 'push commits'\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_commits_pr_pre = pd.concat([pd.read_parquet(file, engine = 'pyarrow') for file in \n                           glob.glob('data/github_commits/parquet/github_data_pre_18/*_pr_*')])\ndf_commits_pr_pre['type']  = 'pr commits'\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_commits_push_pre = pd.concat([pd.read_parquet(file, engine = 'pyarrow') for file in \n                           glob.glob('data/github_commits/parquet/github_data_pre_18/*_push_*')])\ndf_commits_push_pre['type']  = 'push commits'\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_commits_pr_post = pd.concat([read_parquet(file) for file in \n                           glob.glob('data/github_commits/parquet/github_data_2324/*_pr_*')])\ndf_commits_pr_post['type']  = 'pr commits'\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_commits_push_post = pd.concat([read_parquet(file) for file in \n                           glob.glob('data/github_commits/parquet/github_data_2324/*_push_*')])\ndf_commits_push_post['type']  = 'push commits'\n")


# In[ ]:


df_commits = pd.concat([df_commits_pr, df_commits_push, df_commits_pr_pre, df_commits_push_pre,
                        df_commits_pr_post, df_commits_push_post])


# In[ ]:


df_commits.reset_index(drop = True, inplace = True)


# In[ ]:


pushData = glob.glob('data/github_clean/filtered_github_data/pushEvent*')
pushData.extend(glob.glob('data/github_clean/github_data_pre_18/pushEvent*'))
pushData.extend(glob.glob('data/github_clean/github_data_2324/pushEvent*'))

pushEventData = pd.concat([pd.read_parquet(ele) for ele in pushData if '.parquet' in ele]) [[
    'actor_login', 'actor_id', 'org_id', 'org_login', 'type', 'created_at', 'repo_id', 'repo_name', 'push_id', 'push_ref']]
df_commits = pd.merge(df_commits, pushEventData[['push_id', 'push_ref']].drop_duplicates(), how = 'left')


# In[ ]:


df_commits['commit time'] = pd.to_datetime(df_commits['commit time'], unit = 's')


# In[ ]:


df_commits = df_commits[~df_commits['commit author email'].isna()]
df_commits = df_commits[~df_commits['repo_id'].isna()]


# In[ ]:


df_commits['commit_actor_id'] = df_commits['commit author email'].apply(lambda x: email_committer_info.get(x.lower(), np.nan))


# In[ ]:


df_commits = df_commits[~df_commits['commit_actor_id'].isna()]


# In[ ]:


df_commits['commit day'] = df_commits['commit time'].apply(lambda x: datetime.datetime(x.year, x.month, x.day))


# In[ ]:


earliest_commit = df_commits.groupby(['commit_actor_id', 'repo_id'])['commit time'].min()
earliest_commit = earliest_commit.reset_index()


# In[ ]:


earliest_commit['commit time'] = earliest_commit['commit time'].dt.tz_localize(pytz.UTC)


# In[ ]:


daily_commit_info = df_commits.groupby(['commit_actor_id', 'repo_id', 'commit day']).agg({'commit changes total':['sum', 'count']})


# # Thinking More Critically About Tasks

# In[ ]:


df_actor_start['permissions'] = 'read'


# In[ ]:


df_actor_all = pd.concat([df_actor_start, all_ranked_v1]).reset_index(drop = True)


# In[ ]:


df_actor_all = pd.merge(df_actor_all,
         all_ranked_v1.drop(['type'], axis = 1).rename({'created_at':'promoted_date', 'permissions':'corrected_permissions'}, axis = 1),
         how = 'left', on = ['repo_id', 'actor_id'])


# In[ ]:


df_actor_all['org_id'] = df_actor_all.apply(lambda x: x['org_id_x'] if not pd.isnull(x['org_id_x']) 
                                            else x['org_id_y'] if not pd.isnull(x['org_id_y']) else np.nan, axis = 1)


# In[ ]:


df_actor_all = df_actor_all[~df_actor_all['created_at'].isna()]


# In[ ]:


df_actor_all['created_at']= pd.to_datetime(df_actor_all['created_at'], format='mixed', errors = 'coerce')
#df_actor_all['created_at'] = df_actor_all['created_at'].apply(lambda x: x.astimezone(pytz.UTC))


# In[ ]:


# assume that if someone jumps from having read to more than read privileges in 7 days, it's bc they always had those privileges
read_to_privileged = df_actor_all[df_actor_all.apply(
    lambda x: (x['permissions'] == 'read' and 
              (x['promoted_date']-x['created_at']).days <= 7 and not pd.isnull(x['promoted_date'])), axis = 1)].index
df_actor_all.loc[read_to_privileged, 'permissions'] = df_actor_all.loc[read_to_privileged, 'corrected_permissions']


# In[ ]:


# assume that if someone jumps from having read to more than read privileges in 7 days, it's bc they always had those privileges
read_to_privileged = df_actor_all[df_actor_all.apply(
    lambda x: (x['permissions'] == 'read' and 
              (x['promoted_date']-x['created_at']).days <= 7 and not pd.isnull(x['promoted_date'])), axis = 1)].index
df_actor_all.loc[read_to_privileged, 'permissions'] = df_actor_all.loc[read_to_privileged, 'corrected_permissions']


# In[ ]:


df_actor_all = df_actor_all.sort_values('created_at').drop_duplicates(['repo_id', 'actor_id', 'permissions']).drop(
    ['type','promoted_date','corrected_permissions'], axis = 1)


# In[ ]:


perm_dict = {'read': 4, 'triage': 3, 'write': 2, 'owner': 1}
df_actor_all['permissions_ranked'] = df_actor_all['permissions'].apply(lambda x: perm_dict[x])


# In[ ]:


df_actor_all = df_actor_all.sort_values(['created_at', 'permissions']).drop_duplicates(
    ['created_at', 'repo_id', 'actor_id'])


# In[ ]:


df_actor_all.drop(['org_id_x','org_id_y'], axis = 1, inplace = True)


# # Analysis

# In[ ]:


top_python = pd.read_csv('data/inputs/top_python_stars.csv').rename({'id':'repo_id'}, axis = 1)
top_python_grouped = top_python.groupby('repo_id')['watch_count'].sum().reset_index()


# In[ ]:


#python_opensource = pd.read_table('data/inputs/README-Python.md', sep="|", header=0, index_col=0, skipinitialspace=True, on_bad_lines='warn').dropna(axis=1, how='all').iloc[1:]   
#python_opensource['repo_name'] = python_opensource['NAME/PLACE'].apply(lambda x: "/".join(x.split("(")[1].replace(")","").split("/")[-3:-1]))


# In[ ]:


df_actor_all = df_actor_all.reset_index(drop = True)


# In[ ]:


python_opensource = pd.merge(top_python_grouped.sort_values('watch_count').tail(1000), df_repo)
repo_ids = python_opensource['repo_id'].tolist()
free_copilot_inds = df_actor_all[(df_actor_all['created_at']<datetime.datetime(2022, 6, 23, tzinfo = pytz.UTC)) & 
         (df_actor_all['permissions'].isin(['owner', 'write'])) & (df_actor_all['repo_id'].isin(repo_ids))].index
df_actor_all.loc[free_copilot_inds, 'Free_Copilot'] = True
free_cp = df_actor_all[df_actor_all['Free_Copilot'] == True]['actor_id'].tolist()
df_actor_all['Free_Copilot'] = df_actor_all['actor_id'].isin(free_cp)


# In[ ]:


bots = pd.to_numeric(df_actor[df_actor['actor_login'].apply(lambda x: '[bot]' in x)]['actor_id']).unique()
df_actor_all = df_actor_all[~df_actor_all['actor_id'].isin(bots)]


# ## Issue Templates

# In[ ]:


import json
file_names = [".github/ISSUE_TEMPLATE.md","issue_template.md","ISSUE_TEMPLATE.md",".github/issue_template.md",".github/ISSUE_TEMPLATE","ISSUE_TEMPLATE","issue_template", ".github/issue_template"]


# In[ ]:
df_commits['commit file changes'] = df_commits['commit file changes'].astype(str)
df_commits_issues = df_commits[df_commits['commit file changes'].apply(lambda x: any([ele in x for ele in file_names]))]
df_commits_issues['commit file changes'] = df_commits_issues['commit file changes'].apply(lambda x: json.loads(x) if x != None else x)
df_commits_issues['changed files'] = df_commits_issues['commit file changes'].apply(lambda x: [ele['file'] for ele in x] if x != None else x)

df_commits_issues.groupby(['repo_id'])['commit day'].min()

# In[ ]:

# In[ ]:





# In[ ]:





# In[ ]:


break


# In[ ]:





# ## Population Data

# In[ ]:


special_actors = df_actor_all[df_actor_all['permissions'].isin(['owner','write'])]['actor_id'].unique().tolist()
population = df_actor_all
population['Free_Copilot'] = population['Free_Copilot'].astype(int)
population['key'] = population['repo_id'].apply(lambda x: str(int(x))) + "_" + population['actor_id'].apply(lambda x: str(int(x)))
population['top_repos'] = pd.to_numeric(population['repo_id'].isin(repo_ids))


# In[ ]:


watch_parquet = glob.glob('data/github_clean/filtered_github_data/watchEvent*.parquet')
watch_parquet.extend(glob.glob('data/github_clean/github_data_pre_18/watchEvent*.parquet'))
watch_parquet.extend(glob.glob('data/github_clean/github_data_2324/watchEvent*.parquet'))
df_watch_parquet = pd.concat([read_parquet(ele, ['created_at', 'repo_id']) for ele in watch_parquet])
df_watch_parquet['created_at'] = pd.to_datetime(df_watch_parquet['created_at'], format='mixed')
df_watch = df_watch_parquet[df_watch_parquet['created_at']<datetime.datetime(2022, 6, 23, tzinfo = pytz.UTC)]

fork_parquet = glob.glob('data/github_clean/filtered_github_data/forkEvent*.parquet')
fork_parquet.extend(glob.glob('data/github_clean/github_data_pre_18/forkEvent*.parquet'))
fork_parquet.extend(glob.glob('data/github_clean/github_data_2324/forkEvent*.parquet'))
df_fork_parquet = pd.concat([read_parquet(ele, ['created_at', 'repo_id']) for ele in fork_parquet])
df_fork_parquet['created_at'] = pd.to_datetime(df_fork_parquet['created_at'])
df_fork = df_fork_parquet


# In[ ]:


df_starsforks = pd.concat([df_watch.groupby('repo_id').count(), df_fork.groupby('repo_id').count()], axis = 1)
df_starsforks.columns = ['stars_copilotrelease', 'forks_copilotrelease']
df_starsforks = df_starsforks.reset_index()


# In[ ]:


individuals_contact = committer_info[['actor_id', 'emails', 'names']].explode(['actor_id']).explode(['emails']).explode(['names']).dropna()
individuals_contact['actor_id'] = pd.to_numeric(individuals_contact['actor_id'])
individuals_contact = individuals_contact.groupby('actor_id').agg({'emails':list, 'names': list}).reset_index()
individuals_contact['emails'] = individuals_contact['emails'].apply(lambda x: list(set(x)))
individuals_contact['names'] = individuals_contact['names'].apply(lambda x: list(set(x)))
individuals_contact['Actual Name'] =individuals_contact['names'].apply(
    lambda x: [ele for ele in x if (len(ele.split(" "))>=2 and 
               all([word[0].isupper() if len(word)>0 else False for word in ele.split(" ")]))] if type(x) == list else x)
individuals_contact['Actual Name'] = individuals_contact['Actual Name'].apply(lambda x: x if len(x)>0 and type(x) == list else np.nan)


# In[ ]:


population = pd.merge(population, df_starsforks, how = 'left')
population = pd.merge(population, individuals_contact, how = 'left')
population = pd.merge(population, df_repo[['repo_id','repo_name']], how = 'left')


# In[ ]:


population.to_parquet('data/intermediaries/population_copilot.parquet')


# In[ ]:


break


# ## Forming the Dataset

# In[ ]:


issueEventData['created_at_month'] = issueEventData['created_at'].apply(
    lambda x: datetime.datetime(x.year, x.month, 1, tzinfo = pytz.UTC))
prEventData['created_at_month'] = prEventData['created_at'].apply(
    lambda x: datetime.datetime(x.year, x.month, 1, tzinfo = pytz.UTC) if type(x.year) != float else np.nan)


# In[ ]:


df_commits['created_at_month'] = df_commits['commit time'].apply(
    lambda x: datetime.datetime(x.year, x.month, 1, tzinfo = pytz.UTC) if type(x.year) != float else np.nan)


# In[ ]:


# opened issues
df_active = pd.concat([
    # issues opened
            issueEventData[issueEventData['issue_action'] == 'opened'].groupby(
                ['actor_id','repo_id','created_at_month'])['issue_action'].count().rename('issues_opened'),
    # how many of the issues that were opened were associated with a PR that same person opened a PR for
           issueEventData[issueEventData.apply(lambda x: x['issue_action'] == 'opened' and x['pr_user_id'] == x['actor_id'], axis = 1)].groupby(
               ['actor_id','repo_id','created_at_month'])['issue_action'].count().rename('issues_opened_pr'),
    # issue comments
           issueEventData[issueEventData['type'] == 'IssueCommentEvent'].groupby(
               ['actor_id','repo_id','created_at_month'])['type'].count().rename('issue_comments'),
    # issue comments that required a PR
           issueEventData[issueEventData.apply(lambda x: not pd.isnull(x['potential_issues']) and x['type'] == 'IssueCommentEvent', axis = 1)].groupby(
               ['actor_id','repo_id','created_at_month'])['type'].count().rename('issue_comments_pr'),
    # of unique issues that this person commented on
           issueEventData[issueEventData['type'] == 'IssueCommentEvent'][['actor_id','repo_id','issue_number','created_at_month']].drop_duplicates(
               ['actor_id','repo_id','issue_number','created_at_month']).groupby(
               ['actor_id','repo_id','created_at_month'])['created_at_month'].count().rename('unique_issue_comments'),
    # of unique issues that this person commented on that required a PR
           issueEventData[issueEventData.apply(
               lambda x: not pd.isnull(x['potential_issues']) and x['type'] == 'IssueCommentEvent', axis = 1)][['actor_id','repo_id','issue_number','created_at_month']].drop_duplicates().groupby(
               ['actor_id','repo_id','created_at_month'])['created_at_month'].count().rename('unique_issue_comments_pr'),
    # how many PRs were opened that month
            prEventData[prEventData['pr_action'] == 'opened'][['actor_id','repo_id','pr_number','created_at_month']].drop_duplicates().groupby(
                ['actor_id','repo_id','created_at_month'])['created_at_month'].count().rename('prs_opened'),
    # how many of the PRs that were opened were associated with an issue
            prEventData[(prEventData['actor_id'] == prEventData['issue_user_id']) &
                        (prEventData['pr_action'] == 'opened')][['actor_id','repo_id','pr_number','created_at_month']].drop_duplicates().groupby(
                ['actor_id','repo_id','created_at_month'])['created_at_month'].count().rename('prs_opened_issue'),
          ], axis = 1).reset_index()


# In[ ]:


df_active = df_active.reset_index()


# In[ ]:


df_all_dates = df_active.groupby(['actor_id','repo_id'])['created_at_month'].min().rename('earliest_appearance').reset_index()
df_all_dates['created_at_month'] = df_all_dates['earliest_appearance'].apply(
    lambda x: pd.date_range(x, datetime.datetime(2024, 2, 1, tzinfo = pytz.UTC), freq='MS'))
df_all_dates = df_all_dates.explode('created_at_month').drop('earliest_appearance', axis = 1)


# In[ ]:


df_active['earliest_date'] = df_active.groupby(['actor_id','repo_id'])['created_at_month'].transform('min')
df_active['latest_date'] = df_active.groupby(['actor_id','repo_id'])['created_at_month'].transform('max')


# In[ ]:


# generate months in between activity
df_all_dates = pd.merge(df_all_dates, df_active[['actor_id','repo_id','earliest_date','latest_date']].drop_duplicates())
df_all_dates_trim = df_all_dates[(df_all_dates['created_at_month'] >= df_all_dates['earliest_date']) &
    (df_all_dates['created_at_month']<=df_all_dates['latest_date'])]


# In[ ]:


# months
df_active_full = pd.merge(df_all_dates_trim.drop(['earliest_date','latest_date'], axis = 1), df_active, how = 'left')
df_active_full = df_active_full.fillna(0)


# In[ ]:


# merge in population rank data
# read shapiro paper about event studies


# In[ ]:


df_active_full_pop = pd.merge(df_active_full, population, how = 'left')
df_active_full_pop['rank_count'] = df_active_full_pop.groupby(['actor_id','repo_id','created_at_month'])['actor_id'].transform('count')
df_active_full_pop = df_active_full_pop[
    (df_active_full_pop['created_at_month']+datetime.timedelta(days=31)>=df_active_full_pop['created_at']) |
    (df_active_full_pop['rank_count'] == 1)].sort_values(
    ['actor_id','repo_id','created_at_month','created_at'])
df_active_full_pop = df_active_full_pop.drop_duplicates(['actor_id','repo_id','created_at_month'])


# ## Controls

# In[ ]:


#df_fork
#df_watch


# ## Reference ONLY

# In[ ]:


df_fork['created_at'] = pd.to_datetime(df_fork['created_at'], utc = True)
df_watch['created_at'] = pd.to_datetime(df_watch['created_at'], utc = True)


# In[ ]:


df_fork['created_at_month_year'] = df_fork['created_at'].apply(
    lambda x: datetime.datetime(x.year, x.month, 28, tzinfo = pytz.UTC))
df_watch['created_at_month_year'] = df_watch['created_at'].apply(
    lambda x: datetime.datetime(x.year, x.month, 28, tzinfo = pytz.UTC))
df_watch_fork = pd.concat([df_watch.groupby(['created_at_month_year', 'repo_id']).count(),
                           df_fork.groupby(['created_at_month_year', 'repo_id']).count()], axis = 1)
df_watch_fork.columns = ['stars', 'forks']
df_watch_fork = pd.concat([df_watch_fork.fillna(0).reset_index().sort_values(
    ['repo_id', 'created_at_month_year']),
               df_watch_fork.fillna(0).reset_index().sort_values(
                   ['repo_id', 'created_at_month_year']).groupby(
                   ['repo_id'])[['stars','forks']].transform('cumsum')], axis = 1)
df_watch_fork.columns = ['created_at_month_year', 'repo_id', 'gained stars', 'gained forks', 'cumulative stars', 'cumulative forks']


# In[ ]:


df_repo_issue = issueEventData[issueEventData['issue_action'].isin(['opened', None])].sort_values(
    'created_at').drop_duplicates(['repo_id','issue_number'])
df_repo_issue['created_at_month_year'] = df_repo_issue['created_at'].apply(
    lambda x: datetime.datetime(x.year, x.month, 28, tzinfo = pytz.UTC))
df_repo_pr = prEventData[prEventData['pr_action'].isin(['opened', None])].sort_values(
    'created_at').drop_duplicates(['repo_id','pr_number'])
df_repo_pr = df_repo_pr[~df_repo_pr['created_at'].isna()]
df_repo_pr['created_at_month_year'] = df_repo_pr['created_at'].apply(
    lambda x: datetime.datetime(x.year, x.month, 28, tzinfo = pytz.UTC))
df_issue_pr = pd.concat([df_repo_issue.groupby(['created_at_month_year', 'repo_id'])['type'].count(),
                           df_repo_pr.groupby(['created_at_month_year', 'repo_id'])['type'].count()], axis = 1)
df_issue_pr.columns = ['opened issues', 'opened prs']
df_issue_pr = pd.concat([df_issue_pr.fillna(0).reset_index().sort_values(
    ['repo_id', 'created_at_month_year']),
               df_issue_pr.fillna(0).reset_index().sort_values(
                   ['repo_id', 'created_at_month_year']).groupby(
                   ['repo_id'])[['opened issues','opened prs']].transform('cumsum')], axis = 1)
df_issue_pr.columns = ['created_at_month_year', 'repo_id', 'opened issues','opened prs', 'cumulative opened issues','cumulative opened prs']
                         


# In[ ]:


all_ranked_v1['created_at_month_year'] = all_ranked_v1['created_at'].apply(
    lambda x: datetime.datetime(x.year, x.month, 28, tzinfo = pytz.UTC))

df_ranks = pd.concat([all_ranked_v1[all_ranked_v1['permissions']=='triage'].groupby(
    ['created_at_month_year','repo_id'])['organization'].count(),
                         all_ranked_v1[all_ranked_v1['permissions'].isin(['triage','write','owner'])].groupby(
    ['created_at_month_year','repo_id'])['organization'].count(),], axis = 1)
df_ranks.columns = ['triagers', 'writers']
df_ranks_cum = pd.concat([df_ranks.fillna(0).reset_index().sort_values(
    ['repo_id', 'created_at_month_year']),
               df_ranks.fillna(0).reset_index().sort_values(
                   ['repo_id', 'created_at_month_year']).groupby(
                   ['repo_id'])[['triagers','writers']].transform('cumsum')], axis = 1)
df_ranks_cum.columns = ['created_at_month_year', 'repo_id', 'added triagers','added writers', 'cumulative triagers','cumulative writers']


# In[ ]:


df_repo_controls = pd.merge(pd.merge(df_watch_fork, df_issue_pr, how = 'left'), df_ranks_cum, how = 'left').fillna(0)


# In[ ]:


df_commits['created_at_month_year'] = df_commits['commit day'].apply(
    lambda x: datetime.datetime(x.year, x.month, 28, tzinfo = pytz.UTC))


# In[ ]:


df_contributing_count = df_commits[['repo_id', 'commit_actor_id', 'created_at_month_year']].drop_duplicates().groupby(
    ['commit_actor_id','created_at_month_year']).count()
df_contributing_count = df_contributing_count.reset_index()
df_contributing_count.rename({'repo_id': 'commit_projects'}, axis = 1, inplace = True)
df_commits_user = df_commits[['commit_actor_id', 'created_at_month_year']].value_counts()


# In[ ]:


df_repo_issue_user = issueEventData[issueEventData['issue_action'].isin(['opened', None])].sort_values(
    'created_at').drop_duplicates(['actor_id','issue_number'])
df_repo_issue_user['created_at_month_year'] = df_repo_issue['created_at'].apply(
    lambda x: datetime.datetime(x.year, x.month, 28, tzinfo = pytz.UTC))
df_repo_pr_user = prEventData[prEventData['pr_action'].isin(['opened', None])].sort_values(
    'created_at').drop_duplicates(['actor_id','pr_number'])
df_repo_pr_user = df_repo_pr[~df_repo_pr['created_at'].isna()]
df_repo_pr_user['created_at_month_year'] = df_repo_pr['created_at'].apply(
    lambda x: datetime.datetime(x.year, x.month, 28, tzinfo = pytz.UTC))
df_issue_pr_user = pd.concat([df_repo_issue_user.groupby(['created_at_month_year', 'actor_id'])['type'].count(),
                           df_repo_pr_user.groupby(['created_at_month_year', 'actor_id'])['type'].count()], axis = 1)
df_issue_pr_user.columns = ['opened issues', 'opened prs']
df_issue_pr_user = pd.concat([df_issue_pr_user.fillna(0).reset_index().sort_values(
    ['actor_id', 'created_at_month_year']),
               df_issue_pr_user.fillna(0).reset_index().sort_values(
                   ['actor_id', 'created_at_month_year']).groupby(
                   ['actor_id'])[['opened issues','opened prs']].transform('cumsum')], axis = 1)
df_issue_pr_user.columns = ['created_at_month_year', 'actor_id', 'user opened issues','user opened prs', 'cumulative user opened issues','cumulative user opened prs']
                         


# In[ ]:


df_ranks_user = pd.concat([all_ranked_v1[all_ranked_v1['permissions']=='triage'].groupby(
    ['created_at_month_year','actor_id'])['organization'].count(),
                         all_ranked_v1[all_ranked_v1['permissions'].isin(['triage','write','owner'])].groupby(
    ['created_at_month_year','actor_id'])['organization'].count(),], axis = 1)
df_ranks_user.columns = ['triagers', 'writers']
df_ranks_user_cum = pd.concat([df_ranks_user.fillna(0).reset_index().sort_values(
    ['actor_id', 'created_at_month_year']),
               df_ranks_user.fillna(0).reset_index().sort_values(
                   ['actor_id', 'created_at_month_year']).groupby(
                   ['actor_id'])[['triagers','writers']].transform('cumsum')], axis = 1)
df_ranks_user_cum.columns = ['created_at_month_year', 'actor_id', 'user added triagers','user added writers', 'user cumulative triagers','user cumulative writers']


# In[ ]:


contributed_projects_forks_watch = pd.merge(df_commits[['repo_id', 'commit_actor_id', 'created_at_month_year']].drop_duplicates(),
                                            df_watch_fork).groupby(['commit_actor_id','created_at_month_year'])[
    ['gained stars','gained forks','cumulative stars','cumulative forks']].sum().reset_index()
contributed_projects_forks_watch.columns = [
    'commit_actor_id', 'created_at_month_year','user gained stars','user gained forks','user cumulative stars','user cumulative forks']


# In[ ]:


df_contributor_controls = pd.concat([df_commits_user.reset_index().set_index(['commit_actor_id','created_at_month_year']).rename(
    {'count':'commit count'}, axis = 1),
           df_contributing_count.set_index(['commit_actor_id','created_at_month_year']),
           df_issue_pr_user.set_index(['actor_id','created_at_month_year']),
           df_ranks_user_cum.set_index(['actor_id','created_at_month_year']),
           #contributed_projects_forks_watch.set_index(['commit_actor_id','created_at_month_year'])
                                    ], axis = 1)
df_contributor_controls = df_contributor_controls.reset_index().rename({'level_0':'commit_actor_id'}, axis = 1)


# In[ ]:


df_repo_controls
df_contributor_controls


# ## DiD - Commits

# In[ ]:


pr_status_info = prEventData[['created_at', 'pr_number', 'repo_id', 'pr_action']].sort_values(
    'created_at', ascending = True).dropna().drop_duplicates(['pr_number', 'repo_id', 'pr_action'])


# In[ ]:


df_commits['key'] = df_commits['repo_id'].apply(lambda x: str(int(x))) + "_" + df_commits['commit_actor_id'].apply(lambda x: str(int(x)))


df_copilot_commits = df_commits[df_commits['key'].isin(population['key'].tolist())]
df_copilot_commits['created_at_month_year'] = df_copilot_commits['commit time'].apply(
    lambda x: datetime.datetime(x.year, x.month, 28, tzinfo = pytz.UTC))
df_copilot_commits = df_copilot_commits[
    (df_copilot_commits['created_at_month_year']>= datetime.datetime(2021, 6, 23, tzinfo = pytz.UTC))]

pr_status_dict = pr_status_info[['pr_number','repo_id','pr_action']].set_index(['pr_number','repo_id']).to_dict()['pr_action']
df_copilot_commits['pr_status'] = df_copilot_commits.apply(lambda x: pr_status_dict.get((x['pr_number'], x['repo_id']), np.nan), axis = 1)

df_copilot_commits = pd.merge(df_copilot_commits, prEventData[['repo_id', 'pr_number','pr_merged_by_id']].dropna(), how = 'left')
df_copilot_commits['pr_merged'] = (~df_copilot_commits['pr_merged_by_id'].isna()).astype(int)
df_copilot_commits['push_ref'] = (df_copilot_commits['push_ref'].isin(['refs/heads/master', 'refs/heads/main'])).astype(int)
df_copilot_commits['commit changes total pr merged'] = df_copilot_commits.apply(
    lambda x: x['commit changes total']*x['pr_merged'], axis = 1)
df_copilot_commits['commit changes total push main'] = df_copilot_commits.apply(
    lambda x: x['commit changes total']*x['push_ref'], axis = 1)
df_copilot_commits['commit changes added pr merged'] = df_copilot_commits.apply(
    lambda x: x['commit additions']*x['pr_merged'], axis = 1)
df_copilot_commits['commit changes added push main'] = df_copilot_commits.apply(
    lambda x: x['commit additions']*x['push_ref'], axis = 1)
df_copilot_commits['commit changes deleted pr merged'] = df_copilot_commits.apply(
    lambda x: x['commit deletions']*x['pr_merged'], axis = 1)
df_copilot_commits['commit changes deleted push main'] = df_copilot_commits.apply(
    lambda x: x['commit deletions']*x['push_ref'], axis = 1)


# In[ ]:


df_copilot_commits_grouped =  df_copilot_commits.groupby(['commit_actor_id', 'created_at_month_year', 'repo_id', 'type']).agg(
    {'commit changes total':['sum', 'count'],
     'commit additions': 'sum',
     'commit deletions': 'sum',
     'commit files changed count': 'sum',
     'pr_merged':'sum','push_ref':'sum',
     'commit changes total pr merged': 'sum',
     'commit changes total push main': 'sum',
     'commit changes added pr merged': 'sum',
     'commit changes added push main': 'sum',
     'commit changes deleted pr merged': 'sum',
     'commit changes deleted push main': 'sum'}, ).reset_index()



# In[ ]:


df_copilot_commits_grouped.columns =  ['commit_actor_id', 'created_at_month_year', 'repo_id','type',
     'commit_changes_sum', 'commit_count', 'commit_additions_sum', 'commit_deletions_sum',
     'commit_files_changed_sum', 'pr_merged_sum', 'push_main_sum','commit_changes_pr_merged_sum',
     'commit_changes_push_main_sum', 'commit_additions_pr_merged_sum', 'commit_additions_push_main_sum',
     'commit_deletions_pr_merged_sum', 'commit_deletions_push_main_sum',]
df_copilot_commits_grouped = df_copilot_commits_grouped.pivot(
    index = ['commit_actor_id', 'created_at_month_year','repo_id'],
    columns =  'type', values = ['commit_changes_sum', 'commit_count', 'commit_additions_sum', 'commit_deletions_sum',
                                 'commit_files_changed_sum','pr_merged_sum', 'push_main_sum','commit_changes_pr_merged_sum',
                                'commit_changes_push_main_sum', 'commit_additions_pr_merged_sum', 'commit_additions_push_main_sum',
                                 'commit_deletions_pr_merged_sum', 'commit_deletions_push_main_sum',]).reset_index()




# In[ ]:


df_copilot_commits_grouped = df_copilot_commits_grouped.drop(
    [('pr_merged_sum','push commits'), ('push_main_sum', 'pr commits'), ('commit_changes_push_main_sum','pr commits'),
     ('commit_changes_pr_merged_sum','push commits'), ('commit_additions_push_main_sum','pr commits'),
     ('commit_additions_pr_merged_sum','push commits'), ('commit_deletions_push_main_sum','pr commits'),
     ('commit_deletions_pr_merged_sum','push commits')], axis = 1)


# In[ ]:


df_copilot_commits_grouped.columns =  ['commit_actor_id', 'created_at_month_year', 'repo_id' ,'commit_changes_sum_pr','commit_changes_sum_push', 
     'commit_count_pr','commit_count_push', 'commit_additions_sum_pr','commit_additions_sum_push', 
     'commit_deletions_sum_pr','commit_deletions_sum_push', 'commit_files_changed_sum_pr',
     'commit_files_changed_sum_push', 'pr_merged_sum', 'push_main_sum','commit_changes_pr_merged_sum',
     'commit_changes_push_main_sum', 'commit_additions_pr_merged_sum', 'commit_additions_push_main_sum',
     'commit_deletions_pr_merged_sum', 'commit_deletions_push_main_sum']





# In[ ]:


df_copilot_commits_grouped = pd.merge(
    df_copilot_commits_grouped,
    population[['actor_id', 'repo_id', 'Free_Copilot','greater_1000_stars','greater_500_stars']].rename({'actor_id':'commit_actor_id', 'Free_Copilot':'treatment'}, axis = 1).drop_duplicates())
#df_copilot_commits_grouped = pd.merge(df_copilot_commits_grouped, df_repo_controls)
#df_copilot_commits_grouped = pd.merge(df_copilot_commits_grouped, df_contributor_controls)


# In[ ]:


df_copilot_commits_grouped.fillna(0).to_csv('results/data/df_copilot_commits_grouped.csv')


# In[ ]:


issueEventData['issue_pull_request_number'] = issueEventData['issue_pull_request'].apply(lambda x: ast.literal_eval(x)['html_url'] if not pd.isnull(x) else x)


# In[ ]:


issueEventData['issue_pull_request_number'] = pd.to_numeric(issueEventData['issue_pull_request_number'].apply(lambda x: x.split("/")[-1] if not pd.isnull(x) else x))


# In[ ]:


prEventData_na = prEventData[(~prEventData['repo_id'].isna()) & 
    (prEventData.apply(lambda x: not pd.isnull(x['pr_user_id']) or x['pr_action'] == 'opened', axis= 1))]
prEventData_na['key'] = prEventData_na.apply(
    lambda x: str(int(x['repo_id']))+"_"+
    str(int(x['pr_user_id'])) if x['pr_action'] != 'opened' else str(int(x['actor_id'])), axis = 1)

prEventData_copilot = prEventData_na[prEventData_na['key'].isin(population['key'].tolist())]
prEventData_copilot['created_at_month_year'] = prEventData_copilot['created_at'].apply(
    lambda x: datetime.datetime(x.year, x.month, 28, tzinfo = pytz.UTC))

linked_prs = issueEventData[['repo_id','issue_pull_request_number']].dropna().drop_duplicates()
linked_prs['linked_pr'] = 1
prEventData_copilot = pd.merge(prEventData_copilot, linked_prs.rename({'issue_pull_request_number':'pr_number'}, axis = 1), how = 'left')
prEventData_copilot['linked_pr'] = prEventData_copilot['linked_pr'].fillna(0)

prEventData_copilot = pd.merge(prEventData_copilot.drop('pr_merged_by_id', axis = 1), 
                                   prEventData[['repo_id', 'pr_number','pr_merged_by_id']].drop_duplicates(), how = 'left')
prEventData_copilot['merged_pr'] = prEventData_copilot['pr_merged_by_id'].apply(lambda x: not pd.isnull(x)).astype(int)


# In[ ]:


pr_commit_data = df_commits[['repo_id', 'created_at_month_year', 'commit changes total', 'commit additions','commit deletions']].groupby(
    ['repo_id', 'created_at_month_year']).sum().reset_index()


# In[ ]:


prEventData_copilot = pd.merge(prEventData_copilot,pr_commit_data,how = 'left')


# In[ ]:


prEventData_copilot['commit_changes_merged_prs'] = prEventData_copilot.apply(
    lambda x: x['commit changes total'] * (1-int(pd.isnull(x['pr_merged_by_id']))), axis = 1)
prEventData_copilot['commit_additions_merged_prs'] = prEventData_copilot.apply(
    lambda x: x['commit additions'] * (1-int(pd.isnull(x['pr_merged_by_id']))), axis = 1)
prEventData_copilot['commit_deletions_merged_prs'] = prEventData_copilot.apply(
    lambda x: x['commit deletions'] * (1-int(pd.isnull(x['pr_merged_by_id']))), axis = 1)


# In[ ]:


df_copilot_prs_grouped =  prEventData_copilot[['pr_user_id', 'repo_id', 'pr_number', 'created_at', 
                                               'created_at_month_year','linked_pr','merged_pr',
                                              'commit changes total', 'commit additions','commit deletions',
                                              'commit_changes_merged_prs','commit_additions_merged_prs','commit_deletions_merged_prs']].sort_values(
    'created_at').drop_duplicates(
    ['pr_user_id', 'repo_id', 'pr_number',]).groupby(
    ['pr_user_id', 'created_at_month_year', 'repo_id']).agg(
    {'pr_number':['count'], 'linked_pr':['sum'], 'merged_pr':'sum', 
    'commit changes total':'sum', 'commit additions':'sum','commit deletions':'sum',
     'commit_changes_merged_prs':'sum', 'commit_additions_merged_prs':'sum','commit_deletions_merged_prs':'sum'}).reset_index()
df_copilot_prs_grouped.columns = ['pr_user_id','created_at_month_year','repo_id','pr_number_count','linked_pr_sum','merged_pr_sum',
                                 'pr_commit_changes','pr_commit_additions','pr_commit_deletions',
                                 'merged_pr_commit_changes','merged_pr_commit_additions','merged_pr_commit_deletions']


# In[ ]:


df_copilot_prs_grouped = pd.merge(df_copilot_prs_grouped, 
         population[['actor_id', 'repo_id',  'Free_Copilot','greater_1000_stars','greater_500_stars']].rename({'actor_id':'pr_user_id', 'Free_Copilot':'treatment'}, axis = 1).drop_duplicates())
#df_copilot_prs_grouped = pd.merge(df_copilot_prs_grouped, df_repo_controls)
#df_copilot_prs_grouped = pd.merge(df_copilot_prs_grouped, df_contributor_controls.rename({'commit_actor_id':'pr_user_id'}, axis = 1))


# In[ ]:


df_copilot_prs_grouped.fillna(0).to_csv('results/data/df_copilot_prs_grouped.csv')


# In[ ]:


df_commits['push_ref_sum'] = df_commits['push_ref'].apply(lambda x: x in ['refs/heads/master', 'refs/heads/main'])
df_commits['push_ref_changes'] = pd.to_numeric(df_commits['push_ref_sum'] * df_commits['commit changes total'])
df_commits['push_ref_additions'] = pd.to_numeric(df_commits['push_ref_sum'] * df_commits['commit additions'])
df_commits['push_ref_deletions'] = pd.to_numeric(df_commits['push_ref_sum'] * df_commits['commit deletions'])

df_push = df_commits[
    ['push_id', 'commit_actor_id', 'commit time', 'repo_id', 'push_ref_sum','push_ref_changes','push_ref_additions','push_ref_deletions',
    'commit changes total','commit additions', 'commit deletions']].dropna().sort_values('commit time').drop_duplicates(
    ['push_id', 'commit_actor_id', 'repo_id']).groupby(
    ['commit_actor_id','repo_id', 'commit time']).agg(
    {'push_id':'count', 'push_ref_sum':'sum','push_ref_changes':'sum', 'push_ref_additions':'sum', 'push_ref_deletions':'sum',
     'commit changes total':'sum', 'commit additions':'sum','commit deletions':'sum', }).reset_index()


# In[ ]:


df_push['key'] = df_push.apply(
    lambda x: str(int(x['repo_id']))+"_"+str(int(x['commit_actor_id'])), axis = 1)

df_push = df_push[df_push['key'].isin(population['key'].tolist())]
df_push['created_at_month_year'] = df_push['commit time'].apply(
    lambda x: datetime.datetime(x.year, x.month, 28, tzinfo = pytz.UTC))
df_push.rename({'push_id':'push_count', 'push_ref_sum':'push_main_count'}, axis = 1, inplace = True)


# In[ ]:


df_copilot_push_grouped = pd.merge(
    df_push.groupby(['created_at_month_year', 'commit_actor_id', 'repo_id'])[
    ['push_count','push_main_count','commit changes total','commit additions','commit deletions',
     'push_ref_changes','push_ref_additions','push_ref_deletions',]].sum().reset_index(),
    population[['actor_id',  'repo_id', 'Free_Copilot','greater_1000_stars','greater_500_stars']].rename(
        {'actor_id':'commit_actor_id', 'Free_Copilot':'treatment'}, axis = 1).drop_duplicates())
#df_copilot_push_grouped = pd.merge(df_copilot_push_grouped, df_repo_controls)
#df_copilot_push_grouped = pd.merge(df_copilot_push_grouped, df_contributor_controls)


# In[ ]:


df_copilot_push_grouped.fillna(0).to_csv('results/data/df_copilot_push_grouped.csv')


# In[ ]:


prEventDataMerged = prEventData[~prEventData['pr_merged_by_login'].isna()][['repo_id', 'pr_number']].drop_duplicates()


# In[ ]:


pr_commits_earliest = df_commits.groupby(['repo_id', 'pr_number']).agg({'commit time':['min', 'count']})


# In[ ]:


df_pr_lengths = pd.merge(prEventData,prEventDataMerged).groupby(['repo_id', 'actor_id','pr_number', 'pr_action'])['created_at'].min().reset_index()


# In[ ]:


df_pr_lengths['created_at'] = df_pr_lengths['created_at'].astype(str)
df_pr_lengths_wide = df_pr_lengths.pivot(index = ['repo_id', 'actor_id', 'pr_number'], columns = 'pr_action',values = 'created_at')
df_pr_lengths_wide = df_pr_lengths_wide[df_pr_lengths_wide.columns[:2]].dropna().reset_index()
df_pr_lengths_wide.columns = ['repo_id', 'actor_id', 'pr_number', 'closed', 'opened']
df_pr_lengths_wide['closed'] = pd.to_datetime(df_pr_lengths_wide['closed'])
df_pr_lengths_wide['opened']  = pd.to_datetime(df_pr_lengths_wide['opened'])
df_commits_pr_grouped = df_commits.groupby(['pr_number', 'repo_id']).agg(
    {'commit time':['count', 'min', 'max'], 'commit changes total':'sum', }).reset_index().drop_duplicates()
df_commits_pr_grouped.columns = ['pr_number', 'repo_id', 'commits', 'earliest_date', 'latest_date', 'commit changes']

df_pr_lengths_wide = pd.merge(df_pr_lengths_wide, df_commits_pr_grouped)


# In[ ]:


pr_commits_earliest = pr_commits_earliest.reset_index()
pr_commits_earliest.columns = ['repo_id', 'pr_number', 'earliest_commit_time', 'commit_count']


# In[ ]:


df_pr_lengths_wide = pd.merge(df_pr_lengths_wide,pr_commits_earliest)
df_pr_lengths_wide['hours'] = (df_pr_lengths_wide['closed']-df_pr_lengths_wide['opened']).apply(lambda x: x.total_seconds()/3600)
df_pr_lengths_wide['opened'] = pd.to_datetime(df_pr_lengths_wide['opened'], utc = True)
df_pr_lengths_wide['earliest_commit_time'] = pd.to_datetime(df_pr_lengths_wide['earliest_commit_time'], utc = True)
df_pr_lengths_wide['created_at_month_year'] = np.minimum(df_pr_lengths_wide['opened'],df_pr_lengths_wide['earliest_commit_time']).apply(
    lambda x: datetime.datetime(x.year, x.month, 28, tzinfo = pytz.UTC))


# In[ ]:


df_pr_lengths_wide['hours_per_commit'] = df_pr_lengths_wide['hours']/df_pr_lengths_wide['commits']
df_pr_lengths_wide['hours_per_commit_change'] = df_pr_lengths_wide['hours']/df_pr_lengths_wide['commit changes']


# In[ ]:


df_pr_lengths_wide['code_hours'] = (df_pr_lengths_wide['latest_date']-df_pr_lengths_wide['earliest_date']).apply(lambda x: x.total_seconds()/3600)
df_pr_lengths_wide['code_hours_per_commit'] = df_pr_lengths_wide['code_hours']/df_pr_lengths_wide['commits']
df_pr_lengths_wide['code_hours_per_commit_change'] = df_pr_lengths_wide['code_hours'] /df_pr_lengths_wide['commit changes']


# In[ ]:


df_pr_lengths_agg = df_pr_lengths_wide.groupby(['repo_id','actor_id','created_at_month_year']).agg(
    {'hours':'mean','hours_per_commit':'mean','hours_per_commit_change':'mean',
     'commit_count':'sum', 'commit changes':'sum',
    'code_hours':'mean','code_hours_per_commit':'mean', 'code_hours_per_commit_change':'mean',}).reset_index()


# In[ ]:


df_copilot_pr_lengths_grouped = pd.merge(
    df_pr_lengths_agg, 
    population[['actor_id',  'repo_id', 'Free_Copilot','greater_1000_stars','greater_500_stars']].rename({'Free_Copilot':'treatment'}, axis = 1).drop_duplicates())
#df_copilot_pr_lengths_grouped = pd.merge(df_copilot_pr_lengths_grouped, df_repo_controls)
#df_copilot_pr_lengths_grouped = pd.merge(df_copilot_pr_lengths_grouped, df_contributor_controls.rename({'commit_actor_id':'actor_id'}, axis = 1))


# In[ ]:


df_copilot_pr_lengths_grouped.fillna(0).to_csv('results/data/df_copilot_pr_lengths_grouped.csv')


# In[ ]:


pushEventData['created_at'] = pd.to_datetime(pushEventData['created_at'], utc = True)


# In[ ]:


df_push_lengths = pd.merge(df_commits[df_commits['type'] == 'push commits'][['repo_id', 'push_id','commit time', 'commit changes total', 'commit files changed count']].rename({'commit time':'earliest_date'},axis= 1),
                           pushEventData[['push_id','repo_id', 'actor_id','created_at', 'push_ref']]).groupby(
    ['repo_id', 'actor_id','push_id', 'push_ref']).agg(
    {'created_at': 'min', 'earliest_date':['min','max'], 'commit changes total':'sum','commit files changed count':'count'}).reset_index()

df_push_lengths.columns = [
    'repo_id', 'actor_id','push_id', 'push_ref', 'created_at','earliest_date','latest_date','commit changes total','commit files changed count']
df_push_lengths['created_at'] = pd.to_datetime(df_push_lengths['created_at'], utc = True)
df_push_lengths['earliest_date'] = df_push_lengths['earliest_date'].dt.tz_localize(pytz.UTC)
df_push_lengths = df_push_lengths[(df_push_lengths['created_at']-df_push_lengths['earliest_date']).dt.days>0]


# In[ ]:


df_push_lengths['hours'] = (df_push_lengths['created_at']-df_push_lengths['earliest_date']).dt.total_seconds()/3600
df_push_lengths['hours_per_commit'] = (df_push_lengths['hours'])/df_push_lengths['commit files changed count']
df_push_lengths['hours_per_commit_change'] = (df_push_lengths['hours'])/df_push_lengths['commit changes total']


# In[ ]:


df_copilot_push_lengths = pd.merge(df_push_lengths, 
                                   population[['actor_id', 'Free_Copilot']].rename({'Free_Copilot':'treatment'}, axis = 1).drop_duplicates())
df_copilot_push_lengths['created_at_month_year'] = df_copilot_push_lengths['earliest_date'].apply(
    lambda x: datetime.datetime(x.year, x.month, 28, tzinfo = pytz.UTC))


# In[ ]:


df_copilot_push_lengths_grouped = df_copilot_push_lengths.groupby(
    ['repo_id','actor_id', 'created_at_month_year'])[['hours','hours_per_commit','hours_per_commit_change',]].mean().reset_index()
df_copilot_push_lengths_grouped = pd.merge(df_copilot_push_lengths_grouped, 
         population[['actor_id', 'repo_id',  'Free_Copilot','greater_1000_stars','greater_500_stars']].rename({'Free_Copilot':'treatment'}, axis = 1).drop_duplicates())
#df_copilot_push_lengths_grouped = pd.merge(df_copilot_push_lengths_grouped, df_repo_controls)
#df_copilot_push_lengths_grouped = pd.merge(df_copilot_push_lengths_grouped, df_contributor_controls.rename({'commit_actor_id':'actor_id'}, axis = 1))
df_copilot_push_lengths_grouped.fillna(0).to_csv('results/data/df_copilot_push_lengths_grouped.csv')


# In[ ]:


df_copilot_push_lengths_grouped_main = df_copilot_push_lengths[df_copilot_push_lengths['push_ref'].isin(['refs/heads/master','refs/heads/main'])].groupby(
    ['repo_id','actor_id', 'created_at_month_year'])[['hours','hours_per_commit','hours_per_commit_change',]].mean().reset_index()
df_copilot_push_lengths_grouped_main = pd.merge(df_copilot_push_lengths_grouped_main, 
         population[['actor_id', 'repo_id',  'Free_Copilot','greater_1000_stars','greater_500_stars']].rename({'Free_Copilot':'treatment'}, axis = 1).drop_duplicates())
#df_copilot_push_lengths_grouped = pd.merge(df_copilot_push_lengths_grouped, df_repo_controls)
#df_copilot_push_lengths_grouped = pd.merge(df_copilot_push_lengths_grouped, df_contributor_controls.rename({'commit_actor_id':'actor_id'}, axis = 1))
df_copilot_push_lengths_grouped_main.fillna(0).to_csv('results/data/df_copilot_push_lengths_main_grouped.csv')



# In[ ]:


df_copilot_push_lengths_grouped_notmain = df_copilot_push_lengths[~df_copilot_push_lengths['push_ref'].isin(['refs/heads/master','refs/heads/main'])].groupby(
    ['repo_id','actor_id', 'created_at_month_year'])[['hours','hours_per_commit','hours_per_commit_change',]].mean().reset_index()
df_copilot_push_lengths_grouped_notmain = pd.merge(df_copilot_push_lengths_grouped_notmain, 
         population[['actor_id',  'repo_id', 'Free_Copilot','greater_1000_stars','greater_500_stars']].rename({'Free_Copilot':'treatment'}, axis = 1).drop_duplicates())
#df_copilot_push_lengths_grouped = pd.merge(df_copilot_push_lengths_grouped, df_repo_controls)
#df_copilot_push_lengths_grouped = pd.merge(df_copilot_push_lengths_grouped, df_contributor_controls.rename({'commit_actor_id':'actor_id'}, axis = 1))
df_copilot_push_lengths_grouped_notmain.fillna(0).to_csv('results/data/df_copilot_push_lengths_notmain_grouped.csv')



# In[ ]:


df_all_data = pd.concat([df_copilot_commits_grouped.fillna(0).rename({'commit_actor_id':'actor_id'}, axis = 1).set_index(
    ['actor_id','created_at_month_year','repo_id']).drop([
               'treatment','greater_1000_stars','greater_500_stars'], axis = 1),
           df_copilot_push_lengths_grouped.fillna(0).rename(
               {'hours':'avg_hours_per_push', 'hours_per_commit':'avg_hours_per_push_commit', 
                'hours_per_commit_change':'avg_hours_per_push_commit_change'}, axis = 1).set_index(
               ['actor_id','created_at_month_year','repo_id']).drop([
               'treatment','greater_1000_stars','greater_500_stars'], axis = 1),
           df_copilot_pr_lengths_grouped.fillna(0).set_index(['actor_id','created_at_month_year','repo_id']).rename(
               {'hours':'avg_hours_per_pr', 'hours_per_commit':'avg_hours_per_pr_commit', 
                'hours_per_commit_change':'avg_hours_per_pr_commit_change',
               'commit_count':'pr_commit_count', 'commit changes': 'pr_length_commit_changes',
               'code_hours':'pr_code_hours','code_hours_per_commit':'code_hours_per_pr_commit',
               'code_hours_per_commit_change':'code_hours_per_pr_commit_change', }, axis = 1).drop([
               'treatment','greater_1000_stars','greater_500_stars'], axis = 1),
           df_copilot_push_grouped.fillna(0).rename({'commit_actor_id':'actor_id'}, axis = 1).set_index(
               ['actor_id','created_at_month_year','repo_id']).rename(
               {'commit changes total':'push_commit_changes_total','commit additions':'push_commit_additions',
                'commit deletions':'push_commit_deletions', 'push_ref_changes':'push_main_commit_changes',
               'push_ref_additions':'push_main_commit_additions', 'push_ref_deletions':'push_main_commit_deletions'}, axis = 1).drop([
               'treatment','greater_1000_stars','greater_500_stars'], axis = 1),
           df_copilot_prs_grouped.fillna(0).rename({'pr_user_id':'actor_id'}, axis = 1).set_index(
               ['actor_id','created_at_month_year','repo_id']).rename({
               'pr_number_count':'pr_count', 'linked_pr_sum':'issue_pr_count', 'merged_pr_sum':'merged_pr_count'}, axis = 1).drop([
               'treatment','greater_1000_stars','greater_500_stars'], axis = 1),
          ], axis = 1)


# In[ ]:


df_all_data.rename({'pr_merged_sum':'commit_count_pr_merged',
         'push_main_sum': 'commit_count_push_main'}, axis = 1, inplace = True)


# In[ ]:


agg_dict = {}
for col in df_all_data.columns[3:]:
    agg_dict[col] = 'sum'
agg_dict['repo_id'] = 'count'


# In[ ]:


df_all_data = df_all_data.reset_index()
df_all_data.fillna('NaN')
df_all_data['active_repos'] = df_all_data.groupby(['actor_id','created_at_month_year'])['repo_id'].transform('count')
df_all_data = df_all_data.groupby(['actor_id','created_at_month_year','active_repos','repo_id'], dropna = False).sum(min_count=1).reset_index()


# In[ ]:


df_all_data_final = pd.merge(
    df_all_data,#[(~df_all_data['commit_count_pr'].isna())],
    population[['actor_id','repo_id','Free_Copilot']].drop_duplicates().rename(
             {'Free_Copilot':'treatment'}, axis = 1))


# In[ ]:


df_fork_total = df_fork[['created_at_month_year','repo_id']].value_counts().reset_index().sort_values(
    'created_at_month_year').rename({'count':'fork_count'}, axis = 1)
df_fork_total['cumulative_forks'] = df_fork_total.groupby('repo_id').transform('cumcount')


# In[ ]:


df_watch_total = df_watch[['created_at_month_year','repo_id']].value_counts().reset_index().sort_values(
    'created_at_month_year').rename({'count':'watch_count'}, axis = 1)
df_watch_total['cumulative_watches'] = df_watch_total.groupby('repo_id').transform('cumcount')


# In[ ]:


df_all_data_final['month'] = df_all_data_final['created_at_month_year'].dt.month
df_all_data_final['appearances'] = df_all_data_final.sort_values('created_at_month_year').groupby(
    ['actor_id','repo_id']).transform('cumcount')
df_all_data_final['coworkers'] = df_all_data_final.groupby(['repo_id','created_at_month_year'])['month'].transform('count')


# In[ ]:


# how many issues did they open
# how many issues did they comment on?
# how many issues is the project dealing with
issueEventData['created_at_month_year'] = issueEventData['created_at'].apply(lambda x: datetime.datetime(x.year, x.month, 28, 0 ,0, 0, tzinfo = pytz.UTC))
repo_issue_opened = issueEventData[issueEventData['issue_action'] == 'opened'].groupby(
    ['repo_id', 'created_at_month_year'])['created_at'].count().rename('repo_issues_opened').reset_index()
repo_issue_closed = issueEventData[issueEventData['issue_action'] == 'closed'].groupby(
    ['repo_id', 'created_at_month_year'])['created_at'].count().rename('repo_issues_closed').reset_index()
user_issue_handling = issueEventData[issueEventData['type'] == 'IssueEvent'].groupby(
    ['actor_id','repo_id', 'created_at_month_year'])['created_at'].count().rename('issues_managed').reset_index()
user_issue_comments = issueEventData[issueEventData['type'] == 'IssueCommentEvent'].groupby(
    ['actor_id','repo_id', 'created_at_month_year'])['created_at'].count().rename('comments_made').reset_index()



# In[ ]:


df_all_data_final = pd.merge(pd.merge(df_all_data_final, df_fork_total, how = 'left'), df_watch_total, how = 'left')


# In[ ]:


df_all_data_final = pd.merge(pd.merge(pd.merge(pd.merge(df_all_data_final, 
                                               repo_issue_opened, how = 'left'), user_issue_handling, how = 'left'),
                             user_issue_comments, how = 'left'),
                             repo_issue_closed, how = 'left')


# In[ ]:


df_all_data_final['repos_2000'] = df_all_data_final['repo_id'].isin(repos2000)
df_all_data_final['repos_1000'] = df_all_data_final['repo_id'].isin(repos1000)
df_all_data_final['repos_500'] = df_all_data_final['repo_id'].isin(repos500)


# In[ ]:


df_all_dates = df_all_data_final.groupby(['actor_id','repo_id'])['created_at_month_year'].min().rename('earliest_appearance').reset_index()
df_all_dates['created_at_month_year'] = df_all_dates['earliest_appearance'].apply(
    lambda x: pd.date_range(x - pd.DateOffset(days=27), 
                            datetime.datetime(2024, 2, 28, tzinfo = pytz.UTC), freq='MS') + pd.DateOffset(days=27))
df_all_dates = df_all_dates.explode('created_at_month_year').drop('earliest_appearance', axis = 1)


# In[ ]:


df_all_data_final['earliest_date'] = df_all_data_final.groupby(['actor_id','repo_id'])['created_at_month_year'].transform('min')
df_all_data_final['latest_date'] = df_all_data_final.groupby(['actor_id','repo_id'])['created_at_month_year'].transform('max')


# In[ ]:


df_all_dates.set_index(['actor_id','repo_id','created_at_month_year'], inplace = True)
df_all_data_final.set_index(['actor_id','repo_id','created_at_month_year'], inplace = True)


# In[ ]:


df_all_data_2[(df_all_data_2['repos_2000'] == True) & (df_all_data_2['treatment'] == 1)]


# In[ ]:


python_opensource[python_opensource['repo_id'] == 25437858]


# In[ ]:


df_actor_all[df_actor_all['actor_id'] == 71]


# In[ ]:


df_repo[df_repo['repo_id'] == 25437858.0]


# In[ ]:





# In[ ]:


df_all_data = pd.concat([df_all_dates,df_all_data_final], axis = 1).reset_index()
df_all_data.sort_values(['actor_id','repo_id','created_at_month_year'], inplace = True)
df_all_data[['earliest_date','latest_date']] = df_all_data[['earliest_date','latest_date']].fillna(method = 'ffill')
df_all_data = df_all_data[df_all_data['created_at_month_year']>=df_all_data['earliest_date']]
df_all_data = df_all_data[df_all_data['created_at_month_year']<=df_all_data['latest_date']]
df_all_data['created_at_year'] = df_all_data['created_at_month_year'].apply(lambda x: x.year)
df_all_data = df_all_data[df_all_data['created_at_year']<2025]
df_all_data[['repos_2000','repos_1000','repos_500']] = df_all_data[['repos_2000','repos_1000','repos_500']].fillna(method = 'ffill')
df_all_data_2 = df_all_data.fillna(0)


# In[ ]:


df_all_data_2.drop('df_all_dates', axis = 1, inplace = True)


# In[ ]:


df_all_data_2.rename({'pr_merged_sum':'commit_count_pr_merged',
         'push_main_sum': 'commit_count_push_main'}, axis = 1, inplace = True)


# In[ ]:


df_all_data_2.to_parquet('results/data/df_final.parquet')


# In[ ]:





# In[ ]:




