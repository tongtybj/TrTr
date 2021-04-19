#!/bin/bash

YEAR="${1:-2018}"
CHALLENGE="${2:-main}"

python "download_unzip.py" "$YEAR" "$CHALLENGE"

# create symbolic link
cd ..
ln -sfb "VOT/dataset/${YEAR}" "VOT${YEAR}"
