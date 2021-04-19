#!/bin/bash

dl="dataset/dl"
data="dataset/test"

mkdir -p "${dl}"
(
    cd "${dl}"
    baseurl="http://cvlab.hanyang.ac.kr/tracker_benchmark"
    wget "$baseurl/datasets.html"
    cat datasets.html | grep '\.zip' | sed -e 's/\.zip".*/.zip/' | sed -e s'/.*"//' >files.txt
    cat files.txt | xargs -n 1 -P 8 -I {} wget -c "$baseurl/{}"
)

mkdir -p "${data}"
dl="$( cd "${dl}" && pwd )"
(
    cd "${data}"
    ls "${dl}"/*.zip | xargs -t -n 1 unzip -q -o
)
