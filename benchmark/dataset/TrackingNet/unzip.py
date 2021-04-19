from os.path import join, isdir
from os import listdir, mkdir, makedirs
import sys
import cv2
import numpy as np
import glob
from concurrent import futures
import zipfile
import os

base_path = "./dataset"


if __name__ == '__main__':


    subset_dir = join(base_path, 'TEST.zip')
    subset_name = subset_dir.split('/')[-1].split('.')[0]
    unzip_path = join(base_path, subset_name)
    if not isdir(unzip_path): mkdir(unzip_path)

    with zipfile.ZipFile(subset_dir, 'r') as zf:
        zf.extractall(unzip_path)

    save_base_path = join(base_path, 'TEST', 'videos')
    if not isdir(save_base_path): mkdir(save_base_path)

    zip_videos = sorted(glob.glob(join(base_path, 'TEST', 'zips', '*.zip')))

    print('{} has {} zipped videos'.format(subset_dir, len(zip_videos)))

    for zip_video in zip_videos:
        video_name = zip_video.split('/')[-1].split('.')[0]
        save_path = join(save_base_path, video_name)
        if not isdir(save_path): mkdir(save_path)
        with zipfile.ZipFile(zip_video, 'r') as zf:
            zf.extractall(save_path)

        os.remove(zip_video)

    os.rmdir(join(base_path, 'TEST', 'zips'))
