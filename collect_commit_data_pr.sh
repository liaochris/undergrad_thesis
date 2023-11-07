#! /usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

python3 collect_commit_data_pr.py

