#! /usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

FILES="$(ls data/github_raw/github_data_pre_18 | grep "github_data_pre" | grep -v ".gstmp" | shuf)"


for file in $FILES
do
    /usr/bin/env python github_data_cleaning.py "data/github_raw/github_data_pre_18/$file"
    echo "$file has been cleaned"
    rm -rf "data/github_raw/github_data_pre_18/$file"
    sleep 60
done