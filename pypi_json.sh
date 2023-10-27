#! /usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

# input either packages_filtered.csv orp ackages.csv
PACKAGES="$(cat $1 | cut -d"," -f2)"

# make directory if nonexistent
if [[ ! -d "data/pip_json" ]]; then
    mkdir "data/pip_json"
fi

# download jsons in
for p in $PACKAGES; do 
    curl -s "https://pypi.org/pypi/$p/json" >> "data/pip_json/$p.json"
done