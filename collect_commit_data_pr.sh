#! /usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

/usr/bin/env python3.11 collect_commit_data_pr.py
rm -rf repos/*