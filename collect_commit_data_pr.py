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
def createCommitGroup(commit_list, parent_commit):
    try:
        if type(commit_list) != list and type(commit_list) != type(pd.Series()) and type(commit_list) != np.ndarray:
            commit_list = ast.literal_eval(commit_list)
        if len(commit_list) == 0:
            return [[]]
        elif len(commit_list) == 1:
            return [[parent_commit, commit_list[0]]]
        else:
            avail_commits = len(commit_list)
            lst_result = [[parent_commit, commit_list[0]]]
            for i in range(avail_commits-1):
                lst_result.append([commit_list[i], commit_list[i+1]])
            return lst_result
    except:
        return [[]]



def getHead(commit_list, pull_number, repo_loc):
    try:
        if type(commit_list) != list and type(commit_list) != type(pd.Series())  and type(commit_list) != np.ndarray:
            commit = ast.literal_eval(commit_list)[0]
        else:
            commit = commit_list[0]
        pull_fetch = subprocess.Popen(["git","fetch", "origin", f"pull/{pull_number}/head"], cwd = f"{repo_loc}",
                                      shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).wait()
        result = subprocess.run(["git","show", f"{commit}^"], cwd = f"{repo_loc}", capture_output = True, text = True).stdout[7:47]
        return result
    except Exception as e:
        return []


# In[390]:
def returnCommitStats(x):
    try:
        if len(x) < 2:
            return []
        if x[0] == [] or x[1] == []:
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
        
def cleanCommitData(library, repo_loc, partition, num_partitions = 20):
    # In[386]:
    df_library = df_pr[df_pr['repo_name'] == library]
    if partition < num_partitions:
        df_library = df_library.head(partition * int(df_library.shape[0]/num_partitions)).tail(int(df_library.shape[0]/num_partitions))
    else:
        df_library = df_library.tail(df_library.shape[0] - (num_partitions - 1) * int(df_library.shape[0]/num_partitions))
    # In[387]:
    global repo
    repo = Repository(repo_loc)
    df_library['parent_commit'] = df_library.parallel_apply(lambda x: getHead(x['commit_list'], x['pr_number'], repo_loc), axis = 1)
    print(f"finished getting parent commits for {library}")
    df_library['commit_groups'] = \
        df_library.parallel_apply(lambda x: createCommitGroup(x['commit_list'], x['parent_commit']), axis = 1)
    df_commit_groups = df_library[['pr_number', 'repo_id', 'repo_name', 'actor_id', 'actor_login', 
                                   'org_id', 'org_login', 'pr_state', 'commit_groups']].explode('commit_groups')
    df_commit_groups = df_commit_groups[df_commit_groups['commit_groups'].apply(lambda x: len(x)>0)]
    commit_data = df_commit_groups['commit_groups'].parallel_apply(lambda x: returnCommitStats(x))
    # In[ ]:
    df_commit = pd.DataFrame(commit_data.tolist(),
                            columns = ['commit sha', 'commit author name', 'commit author email', 'committer name',
                                       'commmitter email', 'commit message', 'commit additions', 'commit deletions',
                                       'commit changes total', 'commit files changed count', 'commit file changes', 
                                       'commit time'])
    # In[ ]:
    df_commit_final = pd.concat([df_commit_groups.reset_index(drop = True), df_commit], axis = 1)
    for col in ['pr_number', 'repo_id', 'actor_id']:
        df_commit_final[col] = pd.to_numeric(df_commit_final[col])
        
    return df_commit_final


def getCommitData(library, partition, num_partitions, folder):
    # download repo
    lib_p2 = library.split("/")[1]
    lib_ren = library.replace("/","___")
    if f'commits_pr_{lib_ren}.parquet' not in os.listdir(f'data/github_commits/parquet/{folder}'):
        try:
            print(f"Starting {library}")
            start = time.time()
            if lib_ren not in os.listdir("repos2"):
                subprocess.Popen(["git", "clone", f"git@github.com:{library}.git", f"{lib_ren}"], cwd = "repos2").communicate()
            else:
                return 
            print(f"Finished cloning {library}")
            df_lib = cleanCommitData(library, f"repos2/{lib_ren}", partition, num_partitions)

            if partition == 1 and num_partitions == 1:
                df_lib.to_parquet(f'data/github_commits/parquet/{folder}/commits_pr_{lib_ren}.parquet',
                                  engine='fastparquet')
            else:
                df_lib.to_parquet(f'data/github_commits/parquet/{folder}/commits_pr_{lib_ren}_p{partition}.parquet',
                                  engine='fastparquet')
            end = time.time()
            subprocess.Popen(["rm", "-rf", f"{lib_ren}"], cwd = "repos2").communicate()
            print(f"{library} completed in {start - end}")
            return "success"
        except Exception as e:
            return f"failure, {str(e)}"
    return 'success'



if __name__ == '__main__':   
    
    # In[382]:
    pandarallel.initialize(progress_bar=True)
    warnings.filterwarnings("ignore")
    folder = sys.argv[1]

    # In[385]:
    # import all pull request data
    df_pr = pd.DataFrame()
    commit_urls = []
    for val in np.arange(0, 500, 1):
        if int(val) < 10:
            val = f"0{val}"
        if int(val) < 100:
            val = f"0{val}"
        try:
            df_part = pd.read_csv(f'data/github_clean/{folder}/prEventCommits000000000{val}.csv', index_col = 0)    
            df_part['partition'] = val
            df_pr = pd.concat([df_pr, df_part])
        except:
            print(f'data/github_clean/{folder}/prEventCommits000000000{val}.csv not found')
    

    repos = df_pr['repo_name'].unique().tolist()
    repos = [ele for ele in repos if "/" in ele]
    random.shuffle(repos)
    results = []
    #repos = ['lablup/backend.ai']
    #repos = ['Azure/azure-sdk-for-python']

    for r in repos:
        if r not in ["ansible/ansible", "apache/airflow", "apache/spark", "pandas-dev/pandas", "pytorch/pytorch"]:
            result = getCommitData(r,1,1, folder)
            #for i in np.arange(1, 21, 1):
            #    result = getCommitData(r,i, 20, folder)
            #    print(r, result)
            
            results.append(result)
            
    print("Done!")
        
        
