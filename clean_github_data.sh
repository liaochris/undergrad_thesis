#! /usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

FILES="$(ls data/github_raw/filtered_github_data_large/ | grep "partitions" | grep -v ".gstmp")"

for file in $FILES
do
    /usr/bin/env python github_data_cleaning.py "data/github_raw/filtered_github_data_large/$file"
    echo "$file has been cleaned"
    rm -rf "data/github_raw/filtered_github_data_large/$file"
    sleep 60
done