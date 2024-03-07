#! /usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES


#for i in `seq 0 1 1999`
#do
#    time python obtain_commit_data_pr.py $i filtered_github_data
#done

#for i in `seq 0 1 406`
#do
#    time python obtain_commit_data_pr.py $i github_data_pre_18
#done

for i in `seq $1 1 $2`
do
    time python obtain_commit_data_pr.py $i github_data_2324
done
