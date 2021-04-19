import json
import os
import numpy as np

from tqdm import tqdm
from glob import glob

from .dataset import Dataset
from .video import Video


class NFSVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        img_names: image names
        gt_rect: groundtruth rectangle
    """
    def __init__(self, name, root, gt_rects, img_names, absent, load_img=False):
        super(NFSVideo, self).__init__(name, root, name, gt_rects[0], img_names, gt_rects, None, load_img)
        self.absent = np.array(absent, np.int8)
        # Note: no different between with/withou absent flag when calculate OPE benchmark (check evaluation/ope_benchmark.py)

class NFSDataset(Dataset):
    """
    Args:
        name:  dataset name, should be "NFS" (only support  "NFS30")
        dataset_root, dataset root dir
    """
    def __init__(self, name, dataset_root, load_img=False, single_video=None):
        super(NFSDataset, self).__init__(name, dataset_root)

        # load videos
        mode = '30' # only support NFS30
        dataset_dir =  os.path.join(dataset_root, 'dataset', 'test')
        video_names = [x for x in  os.listdir(dataset_dir) if '__MACOSX' not in x]

        pbar = tqdm(video_names, desc='loading '+name, ncols=100)

        self.videos = {}
        for video in pbar:

            if single_video and single_video != video:
                continue

            if not os.path.isdir(os.path.join(dataset_dir, video)):
                continue

            img_names = sorted(glob(os.path.join(dataset_dir, video, mode, video, '*.jpg')), key=lambda x:int(os.path.basename(x).split('.')[0]))
            with open(os.path.join(dataset_dir, video, mode, video+'.txt'), 'r') as f:
                configs = [list(map(float, x.strip().split(' ')[1:7])) for x in f.readlines()]
                gt_rects = [[config[0], config[1], config[2]-config[0], config[3]-config[1]] for idx, config in enumerate(configs) if idx % 8 == 0]
                absent = [int(config[5]) for idx, config in enumerate(configs) if idx % 8 == 0 ]

                if video == 'dog_2':
                    img_names = sorted(glob(os.path.join(dataset_dir, video, '240', video, '*.jpg')), key=lambda x:int(os.path.basename(x).split('.')[0]))
                    img_names = [img for idx, img in enumerate(img_names) if idx % 8 == 0]


            # TODO: pingpong_2 has wrong ground truth annotation, but do not inflence OPE??

            for idx, ab in enumerate(absent):
                if ab == 1:
                    gt_rects[idx]  = [0, 0, 0, 0]

            if len(img_names) > len(gt_rects):
                img_names = img_names[0:len(gt_rects)]

            pbar.set_postfix_str(video)
            self.videos[video] = NFSVideo(video, dataset_root, gt_rects, img_names, absent)

        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
