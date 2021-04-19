# Preprocessing LaSOT

### Download dataset

- download dataset (all chunks) from [here](https://drive.google.com/file/d/1O2DLxPP8M4Pn4-XCttCJUW3A29tDIeNa/view)
- unzip under `$PWD/lasot`, which gnerates `$PWD/lasot/LaSOTBenchmark`
**note**: you can also download a single subset from [here](https://drive.google.com/drive/folders/1v09JELSXM_v7u3dF7akuqqkVG8T1EK2_)

### Create symbolic link

```shell
mv $PWD/lasot/LaSOTBenchmark $PWD/lasot/train
cd $PWD/trtr/datasets/lasot
ln -sfb $PWD/lasot ./dataset
```

### Crop & Generate data info (10 min)

````shell
python curate.py
````
