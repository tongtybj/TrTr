#!/bin/bash

dl="dataset/dl"
data="dataset/test"

mkdir -p "${dl}"
(
    cd "${dl}"
    wget -c http://ci2cv.net/nfs/Get_NFS.sh || exit 1
    bash Get_NFS.sh
)

mkdir -p "${data}"
dl="$( cd "${dl}" && pwd )"
(
    cd "${data}"
    ls "${dl}"/*.zip | xargs -t -n 1 unzip -q -o
)
