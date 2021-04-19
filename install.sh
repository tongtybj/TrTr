#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install.sh conda_install_path environment_name cudatoolkit_version(option)"
    exit 0
fi

conda_install_path=$1
conda_env_name=$2
cudatoolkit=${3:-10.1}

source $conda_install_path/etc/profile.d/conda.sh
echo "****************** Creating conda environment ${conda_env_name} python=3.8 ******************"
conda create -y --name $conda_env_name python=3.8

echo ""
echo ""
echo "****************** Activating conda environment ${conda_env_name} ******************"
conda activate $conda_env_name

echo ""
echo ""
echo "****************** Checkout github submodule  ******************"
git submodule update --init --recursive

echo ""
echo ""
echo "****************** Installing pytorch==1.7.0 with cudatoolkit  ******************"
conda install -y pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=$cudatoolkit -c pytorch

echo ""
echo ""
echo "****************** Installing matplotlib ******************"
conda install -y matplotlib

echo ""
echo ""
echo "****************** Installing tqdm ******************"
conda install -y tqdm

echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip install visdom

echo ""
echo ""
echo "****************** Installing gdown ******************"
pip install gdown

echo ""
echo ""
echo "****************** Installing cython ******************"
conda install -y cython

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip install pycocotools

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
pip install jpeg4py

echo ""
echo ""
echo "****************** Installing colorama ******************"
pip install colorama

echo ""
echo ""
echo "****************** Installing jsonargparse ******************"
pip install jsonargparse jsonschema

echo ""
echo ""
echo "****************** Downloading networks ******************"
mkdir networks

echo ""
echo ""
echo "****************** TrTr ResNet-50 Network ******************"
gdown https://drive.google.com/uc\?id\=1WSX3_QhwL8eqjqLSfLmB8iVMyqYpmOp6 -O networks/trtr_resnet50.pth

echo ""
echo ""
echo "****************** build toolkit for polygon computation **************"
python setup.py build_ext --inplace


echo ""
echo ""
echo "****************** Installation complete! ******************"
