# Prepare for VOT benchmark

## Create symbolic link for dataset  (Option, but strongly recommended)

````shell
ln -s $PWD/dataset ./dataset
````
**note**: `$PWD/dataset` is the directory to store dataset. Suppose you have sufficient space under `$PWD/dataset`

## Download and unzip dataset

````shell
./install.sh       # 2018 (default)
./install.sh 2019  # 2019
./install.sh 2020  # 2020
````

