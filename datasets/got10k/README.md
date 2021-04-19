# Preprocessing GOT-10k

### Download dataset

- download dataset from [here](http://got-10k.aitestunion.com/downloads_dataset/full_data)
- unzip under `$PWD/got10k`

### Create symbolic link

````shell
cd $PWD/trtr/datasets/got10k
ln -sfb $PWD/got10k ./dataset
````

### Crop & Generate data info (10 min)

````shell
python curate.py
````
