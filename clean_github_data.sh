#! /usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

# cleans post-2018 data
FILES="$(ls data/github_raw/filtered_github_data_large/ | grep "partitions" | grep -v ".gstmp")"
for file in $FILES
do
    /usr/bin/env python github_data_cleaning.py "data/github_raw/filtered_github_data_large/$file"
    echo "$file has been cleaned"
    #rm -rf "data/github_raw/filtered_github_data_large/$file"
    sleep 60
done

# cleans pre-2018 data
FILES="$(ls data/github_raw/github_data_pre_18 | grep "github_data_pre" | grep -v ".gstmp" | shuf)"
for file in $FILES
do
    /usr/bin/env python github_data_cleaning.py "data/github_raw/github_data_pre_18/$file"
    echo "$file has been cleaned"
    #rm -rf "data/github_raw/github_data_pre_18/$file"
    sleep 60
done


# set parallelize
# initialize a semaphore with a given number of tokens
open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}

# run the given command asynchronously and pop/push tokens
run_with_lock(){
    local x
    # this read waits until there is something to read
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    # push the return code of the command to the semaphore
    printf '%.3d' $? >&3
    )&
}

cpus=4
open_sem $cpus
# cleans all data from all other users in non-python libraries
FILES="$(ls data/github_raw/all_contributor_data | grep "contributors" | grep -v ".gstmp" | shuf)"
for file in $FILES
do
    run_with_lock /usr/bin/env python github_data_cleaning.py "data/github_raw/all_contributor_data/$file"
    #rm -rf "data/github_raw/all_contributor_data/$file"
done
