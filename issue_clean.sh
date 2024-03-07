#! /usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

: ' 
mkdir -p data/github_clean/filtered_github_data
# cleans post-2018 data
FILES="$(ls data/github_raw/filtered_github_data/ | grep "partitions" | grep -v ".gstmp")"
for file in $FILES
do
    /usr/bin/env python issue_clean.py "data/github_raw/filtered_github_data/$file"
    echo "$file has been cleaned"
    #rm -rf "data/github_raw/filtered_github_data_large/$file"
done

mkdir -p data/github_clean/github_data_pre_18
# cleans pre-2018 data
FILES="$(ls data/github_raw/github_data_pre_18 | grep "github_data_pre" | grep -v ".gstmp" | shuf)"
for file in $FILES
do
    /usr/bin/env python issue_clean.py "data/github_raw/github_data_pre_18/$file"
    echo "$file has been cleaned"
    #rm -rf "data/github_raw/github_data_pre_18/$file"
done
'

mkdir -p data/github_clean/github_data_2324
# cleans post 2023 data
FILES="$(ls data/github_raw/github_data_2324 | grep "Event" | grep -v ".gstmp" | shuf)"
for file in $FILES
do
    if [[ $file = *'push'* ]]; then
        /usr/bin/env python new_issue_clean.py "data/github_raw/github_data_2324/$file"
    else
        cp "data/github_raw/github_data_2324/$file" "data/github_clean/github_data_2324/$file"
    fi
    echo "$file has been cleaned"
    #rm -rf "data/github_raw/github_data_2324/$file"
done