import os
import json

from tqdm import tqdm
from glob import glob

from .dataset import Dataset
from .video import Video
import numpy as np

class UAVVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        img_names: image names
        gt_rect: groundtruth rectangle

    """
    def __init__(self, name, root, gt_rects, img_names, absent, load_img=False):
        super(UAVVideo, self).__init__(name, root, name, gt_rects[0], img_names, gt_rects, None, load_img)
        self.absent = np.array(absent, np.int8)

class UAVDataset(Dataset):
    """
    Args:
        name: dataset name, should be UAV (only suppot UAV123)
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False, single_video=None):
        super(UAVDataset, self).__init__(name, dataset_root)

        # load videos
        dataset_dir = os.path.join(dataset_root, 'dataset', 'UAV123')
        anno_files = sorted(glob(os.path.join(dataset_dir, 'anno', 'UAV123', '*.txt')))
        assert len(anno_files) == 123
        video_names = [x.split('/')[-1].split('.')[0] for x in anno_files]

        pbar = tqdm(video_names, desc='loading '+name, ncols=100)

        # load configuration from .m file
        with open(os.path.join(dataset_dir, 'configSeqs.m'), 'r') as f:
            video_raw_config = f.readlines()

        for i, e in enumerate(video_raw_config):
            if e.startswith('seqUAV123'):
                video_raw_config = video_raw_config[i:i+123]
                break

        video_config = {}
        for raw_conf in video_raw_config:
            conf = raw_conf.split(",")

            video_config[conf[1].strip('\'')] = [conf[3].split("\\")[-2], int(conf[5]), int(conf[7])] # video, start, end

        self.videos = {}

        for idx, video in enumerate(pbar):

            if single_video and single_video != video:
                continue

            video_dir = os.path.join(dataset_dir, 'data_seq', 'UAV123', video_config[video][0])
            if not os.path.isdir(video_dir):
                continue

            img_names = sorted(glob(os.path.join(video_dir, '*.jpg')), key=lambda x:int(os.path.basename(x).split('.')[0]))
            img_names = img_names[video_config[video][1]-1: video_config[video][2]]

            with open(anno_files[idx], 'r') as f:
                gt_rects = [list(map(float, x.strip().split(','))) for x in f.readlines()]

            #workaround
            img_names = img_names[1:]
            gt_rects = gt_rects[1:]

            if  video in ['car17', 'person23']: # first annotation is not good
                gt_rects[0] = gt_rects[1]

            absent = [1 if np.isnan(np.array(rect)).any() else 0 for rect in gt_rects]

            pbar.set_postfix_str(video)
            self.videos[video] = UAVVideo(video, dataset_root, gt_rects, img_names, absent)

        # set attr
        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
