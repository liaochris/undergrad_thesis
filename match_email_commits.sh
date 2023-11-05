#! /usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

# download jsons in
/usr/bin/env python3.11 match_email_commits.py