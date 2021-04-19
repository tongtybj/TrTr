import json
import os
import re
import numpy as np

from PIL import Image
from tqdm import tqdm
from glob import glob

from .dataset import Dataset
from .video import Video
import re

class OTBVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        img_names: image names
        gt_rect: groundtruth rectangle
    """
    def __init__(self, name, root, gt_rects, img_names, load_img=False):
        super(OTBVideo, self).__init__(name, root, name, gt_rects[0], img_names, gt_rects, None, load_img)

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
            if not os.path.exists(traj_file):
                raise ValueError("{}".format(self.name, traj_file))
            assert os.path.exists(traj_file)
            with open(traj_file, 'r') as f :
                pred_traj = [list(map(float, re.split('[,\t]', x.strip())))
                        for x in f.readlines()]
                if len(pred_traj) != len(self.gt_traj):
                    print(name, len(pred_traj), len(self.gt_traj), self.name)
                if store:
                    self.pred_trajs[name] = pred_traj
                else:
                    return pred_traj
        self.tracker_names = list(self.pred_trajs.keys())



class OTBDataset(Dataset):
    """
    Args:
        name: dataset name, should be 'OTB100', 'CVPR13', 'OTB50'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False, single_video=None):
        super(OTBDataset, self).__init__(name, dataset_root)

        dataset_dir =  os.path.join(dataset_root, 'dataset', 'test')
        video_names = [x for x in  os.listdir(dataset_dir) if '__MACOSX' not in x]


        # load videos
        pbar = tqdm(video_names, desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:

            if not os.path.isdir(os.path.join(dataset_dir, video)):
                continue

            anno_files = glob(os.path.join(dataset_dir, video, 'groundtruth_rect.*'))
            # need to copy Jogging to Jogging-1 and Jogging-2, and copy Skating2 to Skating2-1 and Skating2-2 or using softlink)
            for idx, anno_file in enumerate(anno_files):

                video_name = video
                if len(anno_files) > 1:
                    if not video == 'Human4':
                        video_name = video_name + '-' + str(idx+1)


                if single_video and single_video != video_name:
                    continue

                img_names = sorted(glob(os.path.join(dataset_dir, video, 'img', '*.jpg')), key=lambda x:int(os.path.basename(x).split('.')[0]))


                with open(anno_file, 'r') as f:
                    gt_rects = [list(map(float, re.split('[,\t ]', x.strip()))) for x in f.readlines()]



                if len(img_names) > len(gt_rects):
                    if video_name == 'David':
                        img_names = img_names[len(img_names) - len(gt_rects):]
                    else:
                        img_names = img_names[0:len(gt_rects)]
                if video_name == 'Tiger1':
                    img_names = img_names[5:]
                    gt_rects = gt_rects[5:]

                if len(gt_rects) == 0: # corner case of Human4
                    continue

                pbar.set_postfix_str(video_name)
                self.videos[video_name] = OTBVideo(video_name, dataset_root, gt_rects, img_names)

        # set attr
        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
