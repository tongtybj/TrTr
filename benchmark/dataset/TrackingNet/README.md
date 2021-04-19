# Prepare for VOT benchmark

## Download test dataset

Please **manually** download the test dataset `TEST.zip` from [here](https://drive.google.com/drive/u/0/folders/1gJOR-r-jPFFFCzKKlMOW80WFtuaMiaf6)
**note**: file size is too big, cannot use `gdown`

## Create directory for dataset

````shell
ln -sfb $PWD/dataset ./dataset
````
**note**: `$PWD/dataset` is the directoty contain TEST.zip

## Unzip dataset

````shell
python unzip.py
````

