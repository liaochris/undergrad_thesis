#! /usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail


mkdir -p data/github_commits/parquet/filtered_github_data
mkdir -p repos
python3 collect_commit_data.py filtered_github_data
rm -rf repos/*


mkdir -p data/github_commits/parquet/github_data_pre_18
mkdir -p repos
python3 collect_commit_data.py github_data_pre_18
rm -rf repos/*

mkdir -p data/github_commits/parquet/github_data_2324
mkdir -p repos
python3 collect_commit_data.py github_data_2324
rm -rf repos/*

