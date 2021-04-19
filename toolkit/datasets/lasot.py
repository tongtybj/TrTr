import os
import json
import numpy as np

from tqdm import tqdm
from glob import glob

from .dataset import Dataset
from .video import Video

class LaSOTVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        img_names: image names
        gt_rect: groundtruth rectangle
        absent: abnset attribute of video
    """
    def __init__(self, name, root, gt_rects, img_names, absent, load_img=False):
        super(LaSOTVideo, self).__init__(name, root, name, gt_rects[0], img_names, gt_rects, None, load_img)
        self.absent = np.array(absent, np.int8)

    def load_tracker(self, path, tracker_names=None, store=True):
        """
        Args:
            path(str): path to result
            tracker_name(list): name of tracker
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path)
                    if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            traj_file = os.path.join(path, name, self.name+'.txt')
            if os.path.exists(traj_file):
                with open(traj_file, 'r') as f :
                    pred_traj = [list(map(float, x.strip().split(',')))
                            for x in f.readlines()]
            else:
                print("File not exists: ", traj_file)

            if store:
                self.pred_trajs[name] = pred_traj
            else:
                return pred_traj
        self.tracker_names = list(self.pred_trajs.keys())


class LaSOTDataset(Dataset):
    """
    Args:
        name: dataset name, should be 'LaSOT'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False, single_video=None):
        super(LaSOTDataset, self).__init__(name, dataset_root)

        # load videos
        with open(os.path.join(dataset_root, 'testing_set.txt'), 'r') as f:
            video_names = f.read().splitlines()

        pbar = tqdm(video_names, desc='loading '+name, ncols=100)

        self.videos = {}
        for video in pbar:

            if single_video and single_video != video:
                continue

            if not os.path.isdir(os.path.join(dataset_root, video)):
                continue

            pbar.set_postfix_str(video)

            with open(os.path.join(dataset_root, video, 'groundtruth.txt'), 'r') as f:
                gt_rects = [list(map(float, x.strip().split(','))) for x in f.readlines()]
            img_names = [os.path.join(video, 'img', os.path.basename(x)) for x in sorted(glob(os.path.join(dataset_root, video, 'img', '*.jpg')), key=lambda x:int(os.path.basename(x).split('.')[0]))]

            absent = [0] * len(img_names)
            f_name = os.path.join(dataset_root, video, 'out_of_view.txt')
            if os.path.exists(f_name):
                with open(f_name, 'r') as f:
                    absent = [int(v) for v in f.read().split(",")]

            f_name = os.path.join(dataset_root, video, 'full_occlusion.txt')
            if os.path.exists(f_name):
                with open(f_name, 'r') as f:
                    for idx, v in enumerate(f.read().split(",")):
                        if int(v) == 1:
                            absent[idx] = 1

            self.videos[video] = LaSOTVideo(video, dataset_root, gt_rects, img_names, absent)

        # set attr
        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())


