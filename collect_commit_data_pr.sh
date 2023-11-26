#! /usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

#python collect_commit_data_pr.py filtered_github_data_large
#rm -rf repos2
#mkdir repos2

python collect_commit_data_pr.py github_data_pre_18
rm -rf repos2
mkdir repos2