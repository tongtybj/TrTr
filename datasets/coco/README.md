# Preprocessing COCO

### Download raw images and annotations

```shell
cd $PWD/CoCo
wget http://images.CoCodataset.org/zips/train2017.zip
wget http://images.CoCodataset.org/zips/val2017.zip
wget http://images.CoCodataset.org/annotations/annotations_trainval2017.zip

unzip ./train2017.zip
unzip ./val2017.zip
unzip ./annotations_trainval2017.zip
```
**note**: `$PWD/CoCo` is a directory to store dataset. Suppose you have sufficient space under `$PWD/CoCo`

### Create symbolic link

```shell
cd $PWD/trtr/datasets/coco
ln -sfb $PWD/CoCo ./dataset
````

### Crop & Generate data info (10 min)

```shell
python curate.py # [option: num_threads]
```
