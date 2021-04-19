from os.path import join, isdir
from os import listdir, mkdir, makedirs
import sys
import cv2
import numpy as np
import glob
from concurrent import futures
import zipfile
import os


sys.path.append('..')
import utils

base_path = "./dataset"

SUBSET_PREFIX = "TRAIN_"

debug = True

def unzip_subsets():

    subset_dirs = sorted(glob.glob(join(base_path, SUBSET_PREFIX + '*zip')))

    print(subset_dirs)
    if debug:
        subset_dirs = subset_dirs[:1]

    for subset_dir in subset_dirs:
        print('unzip {}'.format(subset_dir))
        subset_name = subset_dir.split('/')[-1].split('.')[0]
        unzip_path = join(base_path, subset_name)
        if not isdir(unzip_path): mkdir(unzip_path)

        with zipfile.ZipFile(subset_dir, 'r') as zf:
            zf.extractall(unzip_path)

def unzip_videos():

    subset_dirs = sorted(glob.glob(join(base_path, SUBSET_PREFIX + '*') + "[!zip]"))

    print(subset_dirs)
    for subset_dir in subset_dirs:
        save_base_path = join(subset_dir, 'videos')
        if not isdir(save_base_path): mkdir(save_base_path)

        zip_videos = sorted(glob.glob(join(subset_dir, 'zips', '*.zip')))

        print('{} has {} zipped videos'.format(subset_dir, len(zip_videos)))

        if debug:
            zip_videos= zip_videos[:10]

        for zip_video in zip_videos:
            #print(zip_video)
            video_name = zip_video.split('/')[-1].split('.')[0]
            save_path = join(save_base_path, video_name)
            if not isdir(save_path): mkdir(save_path)
            with zipfile.ZipFile(zip_video, 'r') as zf:
                zf.extractall(save_path)

            if not debug:
                os.remove(zip_video)

        if not debug:
            os.rmdir(subset_dir)


if __name__ == '__main__':

    unzip_subsets()
    unzip_videos()
