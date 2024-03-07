#!/usr/bin/env python
# coding: utf-8

# In[322]:
import sys
sys.executable


# In[381]:
import ast
import pandas as pd
import numpy as np
from pygit2 import Object, Repository, GIT_SORT_TIME
from pygit2 import init_repository, Patch
from colorama import Fore
from tqdm import tqdm
import swifter
from pandarallel import pandarallel
import subprocess
import warnings
from joblib import Parallel, delayed
import os
import multiprocessing
import time
import random


# In[388]:
def createCommitGroup(push_before, push_head, commit_urls, push_size):
    try:
        if push_size == 0:
            return [[]]
        elif push_size == 1:
            return [[push_before, push_head]]
        else:
            avail_commits = len(commit_urls)
            lst_result = [[push_before, commit_urls[0].split("/")[-1]]]
            for i in range(avail_commits-1):
                lst_result.append([commit_urls[i].split("/")[-1], commit_urls[i+1].split("/")[-1]])
            return lst_result
    except:
        return [[]]
        
# In[390]:
def returnCommitStats(x):
    try:
        if len(x) < 2:
            return []
        commit_parent_sha = x[0]
        commit_head_sha = x[1]
        commit_parent = repo.get(commit_parent_sha)
        commit_head = repo.get(commit_head_sha)
        if type(commit_parent) != type(None) and type(commit_head) != type(None):
            diff = repo.diff(commit_parent, commit_head, context_lines=0, interhunk_lines=0)
            commit_sha = commit_head_sha
            commit_author_name = commit_head.author.name
            commit_author_email = commit_head.author.email
            committer_author_name = commit_head.committer.name
            committer_author_email = commit_head.committer.email
            commit_message = commit_head.message
            commit_additions = diff.stats.insertions
            commit_deletions = diff.stats.deletions
            commit_changes_total = commit_additions + commit_deletions
            commit_files_changed_count = diff.stats.files_changed
            commit_time = commit_head.commit_time
            commit_file_changes = []
            for obj in diff:
                if type(obj) == Patch:
                    additions = 0
                    deletions = 0
                    for hunk in obj.hunks:
                      for line in hunk.lines:
                        # The new_lineno represents the new location of the line after the patch. If it's -1, the line has been deleted.
                        if line.new_lineno == -1: 
                            deletions += 1
                        # Similarly, if a line did not previously have a place in the file, it's been added fresh. 
                        if line.old_lineno == -1: 
                            additions += 1
                    commit_file_changes.append({'file':obj.delta.new_file.path,
                                                'additions': additions,
                                                'deletions': deletions,
                                                'total': additions + deletions})
            return [commit_sha, commit_author_name, commit_author_email, committer_author_name, committer_author_email,
                    commit_message, commit_additions, commit_deletions, commit_changes_total, commit_files_changed_count,
                    commit_file_changes, commit_time]
        return []
    except:
        return []
        
def cleanCommitData(library, repo_loc):
    # In[386]:
    df_library = df_push[df_push['repo_name'] == library]
    df_library.loc[:,'commit_urls'] = df_library['commit_urls'].apply(lambda x: ast.literal_eval(x) if type(x) != list and type(x) != type(pd.Series()) and type(x) != np.ndarray else x)
    commit_urls = df_library['commit_urls'].explode().tolist()
    # In[387]:
    global repo
    repo = Repository(repo_loc)
    sum = 0
    for commit in commit_urls:
        if type(commit) != float:
            sha = commit.split("/")[-1]
            commit_a = repo.get(sha)
            if type(commit_a) != type(None):
                sum += 1
    print(f"{sum} out of {len(commit_urls)} commit SHAs can be found ({100*round(sum/len(commit_urls), 3)}%) for {library}")     
    # In[389]:
    df_library['commit_groups'] = \
        df_library.apply(lambda x: createCommitGroup(x['push_before'], x['push_head'], x['commit_urls'], x['push_size']) , axis = 1)
    df_commit_groups = df_library[['push_id', 'repo_id', 'repo_name', 'actor_id', 'actor_login', 
                                   'org_id', 'org_login', 'push_size', 'push_before', 'push_head',
                                   'commit_groups']].explode('commit_groups')
    commit_data = df_commit_groups['commit_groups'].parallel_apply(lambda x: returnCommitStats(x))
    # In[ ]:
    df_commit = pd.DataFrame(commit_data.tolist(),
                            columns = ['commit sha', 'commit author name', 'commit author email', 'committer name',
                                       'commmitter email', 'commit message', 'commit additions', 'commit deletions',
                                       'commit changes total', 'commit files changed count', 'commit file changes',
                                       'commit time'])
    # In[ ]:
    df_commit_final = pd.concat([df_commit_groups.reset_index(drop = True), df_commit], axis = 1)
    return df_commit_final


def getCommitData(library):
    # download repo
    lib_p2 = library.split("/")[1]
    lib_ren = library.replace("/","___")
    if f'commits_push_{lib_ren}.parquet' not in os.listdir(f'data/github_commits/parquet/{folder}'):
        try:
            print(f"Starting {library}")
            start = time.time()
            if lib_ren not in os.listdir("repos"):
                subprocess.Popen(["git", "clone", f"git@github.com:{library}.git", f"{lib_ren}"], cwd = "repos").communicate()
            print(f"Finished cloning {library}")
            df_lib = cleanCommitData(library, f"repos/{lib_ren}")
            df_lib.to_parquet(f'data/github_commits/parquet/{folder}/commits_push_{lib_ren}.parquet',
                              engine='fastparquet')
            end = time.time()
            print(f"{library} completed in {start - end}")
            return "success"
        except Exception as e:
            print(e)
        try:
            subprocess.Popen(["rm", "-rf", f"{lib_ren}"], cwd = "repos").communicate()
        except:
            print(f"could not delete repo {lib_ren}")
    return 'already done'


if __name__ == '__main__':   
    
    # In[382]:
    pandarallel.initialize(progress_bar=True)
    warnings.filterwarnings("ignore")
    
    folder = sys.argv[1]
    # In[385]:
    # import all push data
    df_push = pd.DataFrame()
    commit_urls = []
    for val in np.arange(0, 500, 1):
        if int(val) < 10:
            val = f"0{val}"
        if int(val) < 100:
            val = f"0{val}"
        try:
            df_part = pd.read_parquet(f'data/github_clean/{folder}/pushEvent000000000{val}.parquet')
            df_part['partition'] = val
            df_push = pd.concat([df_push, df_part])
        except:
            print(f'data/github_clean/{folder}/pushEvent000000000{val}.csv not found')
    
    repos_fail = []
    repos = df_push['repo_name'].unique().tolist()
    random.shuffle(repos)
    results = []
    for r in repos:
        result = getCommitData(r)
        print(r, result, folder)
        if result == "failure":
            repos_fail.append(result)
        results.append(result)
        
    print("Done!")
    
    
