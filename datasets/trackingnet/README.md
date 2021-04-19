# Preprocessing TrackingNet

### Download dataset

Please download dataset from https://drive.google.com/drive/folders/1gJOR-r-jPFFFCzKKlMOW80WFtuaMiaf6 (all chunks)
**note**: the dataset size is too large to use console comand (e.g., gdown) to downlaod. Currently, manual downloading from web browser is the only way to get the dataset.

````shell
ln -sfb $PWD/trackingnet ./dataset
````

### Unzip the files
````shell
python unzip.py
````


### Crop & Generate data info (10 min)

````shell
python curate.py
````
