import json
import os
import numpy as np

from tqdm import tqdm
from glob import glob

from .dataset import Dataset
from .video import Video

class TrackingNetVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        img_names: image names
        gt_rect: groundtruth rectangle
    """
    def __init__(self, name, root, gt_rects, img_names, load_img=False):
        super(TrackingNetVideo, self).__init__(name, root, name, gt_rects[0], img_names, gt_rects, None, load_img)


class TrackingNetDataset(Dataset):
    """
    Args:
        name:  dataset name, should be "TrackingNet"
        dataset_root, dataset root dir
    """
    def __init__(self, name, dataset_root, load_img=False, single_video=None):
        super(TrackingNetDataset, self).__init__(name, dataset_root)

        # load videos
        videos_dir = os.path.join(dataset_root, 'dataset', 'TEST', 'videos')
        video_names = os.listdir(videos_dir)

        pbar = tqdm(video_names, desc='loading '+name, ncols=100)

        if os.path.isdir(os.path.join(dataset_root, 'dataset', 'TEST', 'ground_truth')):
            self.has_ground_truth = True
        else:
            self.has_ground_truth = False

        self.videos = {}
        for video in pbar:

            if single_video and single_video != video:
                continue

            video_dir = os.path.join(videos_dir, video)
            if not os.path.isdir(video_dir):
                continue

            pbar.set_postfix_str(video)

            img_names = sorted(glob(os.path.join(video_dir, '*.jpg')), key=lambda x:int(os.path.basename(x).split('.')[0]))

            gt_rects = [[0,0,0,0]] * len(img_names)
            with open(os.path.join(dataset_root, 'dataset', 'TEST', 'anno',  video + '.txt'), 'r') as f:
                gt_rects[0] = list(map(float, f.readline().strip().split(',')))

            if self.has_ground_truth:
                init_rect = gt_rects[0]
                with open(os.path.join(dataset_root, 'dataset', 'TEST', 'ground_truth', video +'.txt'), 'r') as f:
                    gt_rects = [list(map(float, x.strip().split(','))) for x in f.read().splitlines()]
                assert gt_rects[0] == init_rect

            self.videos[video] = TrackingNetVideo(video, dataset_root, gt_rects, img_names)

        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
