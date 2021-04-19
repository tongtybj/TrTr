# Preprocessing VID (Object detection from video)

Large Scale Visual Recognition Challenge 2015 (ILSVRC2015)

### Download dataset (86GB)

```shell
cd $PWD/ILSVRC2015
wget http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz
tar -xzvf ./ILSVRC2015_VID.tar.gz
```
**note**: `$PWD/ILSVRC2015` is a directory to store dataset. Suppose you have sufficient space under `$PWD/ILSVRC2015`

### Create symbolic link

```shell
cd $PWD/trtr/datasets/vid
ln -s $PWD/ILSVRC2015 ./dataset
```

### Crop & Generate data info (~ 60 min)

```shell
python curate_vid.py
```
**note**: you can add option like: `python curate_vid.py 24` 
