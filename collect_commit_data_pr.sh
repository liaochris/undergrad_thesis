#! /usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

: '
mkdir -p repos2
mkdir -p data/github_commits/parquet/filtered_github_data
python collect_commit_data_pr.py filtered_github_data
rm -rf repos2

mkdir -p repos2
mkdir -p data/github_commits/parquet/github_data_pre_18
python collect_commit_data_pr.py github_data_pre_18
rm -rf repos2

'
mkdir -p repos2
mkdir -p data/github_commits/parquet/github_data_2324
python collect_commit_data_pr.py github_data_2324