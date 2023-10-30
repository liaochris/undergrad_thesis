#! /usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# download jsons in
for i in `seq $1 $2 99`
do
    time /usr/bin/env python3.11 obtain_commit_data_pr.py $i
done