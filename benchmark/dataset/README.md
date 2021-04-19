# Testing dataset directory
## Benchmarks
-  [VOT](http://www.votchallenge.net/)
-  [OTB](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html)
-  [UAV123](https://ivul.kaust.edu.sa/Pages/Dataset-UAV123.aspx)
-  [NFS](http://ci2cv.net/nfs/index.html)
-  [LaSOT](https://cis.temple.edu/lasot/)
-  [TrackingNet (Evaluation on Server)](https://tracking-net.org)
-  [GOT-10k (Evaluation on Server)](http://got-10k.aitestunion.com)

## Download Dataset

#### VOT

Please `cd VOT`, and follow [README.md](./VOT/README.md).

#### OTB (OTB100)

Please `cd OTB`, and follow [README.md](./OTB/README.md).

#### UAV (UAV123)

Please `cd UAV`, and follow [README.md](./UAV/README.md).

#### NFS (NFS30)

Please `cd NFS`, and follow [README.md](./NFS/README.md).

#### TrackingNet

- Please `cd TrackingNet`, and follow [README.md](./TrackingNet/README.md).
- Please submit the tracking result as `.zip` file to the organization. Please follow [the official instruction](https://github.com/SilvioGiancola/TrackingNet-devkit)

#### LaSOT
- Please **manually** download the zipped test dataset from [here](https://drive.google.com/file/d/1EpeWYN4Li7eTvzTYg-B917S7RWNbwzHv/view)
- Please **manually** unzip the dataset
- Please **manually** create a symbolic link: `ln -s ${unzipped_dir} ./benchmark/dataset/LaSOT`
- Please submit the tracking result as `.zip` file to the organization. Please follow [the official instruction](http://got-10k.aitestunion.com/submit_instructions)

## Directory Strcture

After downloading all datasets, the directory under `./benchmark/dataset`should have the below format

- ./LaSOT (symbolic link)
- ./NFS (directory)
    - ./dataset (symbolic link)   
- ./OTB (directory)
    - ./dataset (symbolic link)
- ./TrackingNet (directory)
    - ./dataset (symbolic link)   
- ./UAV (directory)   
    - ./dataset (symbolic link)   
- ./VOT (directory)   
    - ./dataset (symbolic link)   
- ./VOT2018 (symbolic link)  
- ./VOT2019 (symbolic link)


   
