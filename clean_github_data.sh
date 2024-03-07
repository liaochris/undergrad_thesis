#! /usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

# cleans post-2018 data
FILES="$(ls data/github_raw/filtered_github_data/ | grep "partitions" | grep -v ".gstmp")"
for file in $FILES
do
    /usr/bin/env python github_data_cleaning.py "data/github_raw/filtered_github_data/$file"
    echo "$file has been cleaned"
    #rm -rf "data/github_raw/filtered_github_data_large/$file"
done

: '
# cleans pre-2018 data
FILES="$(ls data/github_raw/github_data_pre_18 | grep "github_data_pre" | grep -v ".gstmp" | shuf)"
for file in $FILES
do
    /usr/bin/env python github_data_cleaning.py "data/github_raw/github_data_pre_18/$file"
    echo "$file has been cleaned"
    #rm -rf "data/github_raw/github_data_pre_18/$file"
    sleep 60
done
'