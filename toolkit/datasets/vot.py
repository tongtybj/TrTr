import os
import cv2
import json
import numpy as np

from glob import glob
from tqdm import tqdm
from PIL import Image

from .dataset import Dataset
from .video import Video

class VOTVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        img_names: image names
        gt_rects: groundtruth rectangle
        camera_motion: camera motion tag
        illum_change: illum change tag
        motion_change: motion change tag
        size_change: size change
        occlusion: occlusion
    """
    def __init__(self, name, root, img_names, gt_rects,
            camera_motion, illum_change, motion_change, size_change, occlusion, load_img=False):
        super(VOTVideo, self).__init__(name, root, name,
                gt_rects[0], img_names, gt_rects, None, load_img)
        self.tags= {'all': [1] * len(gt_rects)}
        self.tags['camera_motion'] = camera_motion
        self.tags['illum_change'] = illum_change
        self.tags['motion_change'] = motion_change
        self.tags['size_change'] = size_change
        self.tags['occlusion'] = occlusion

        # empty tag
        all_tag = [v for k, v in self.tags.items() if len(v) > 0 ]
        self.tags['empty'] = np.all(1 - np.array(all_tag), axis=1).astype(np.int32).tolist()
        # self.tags['empty'] = np.all(1 - np.array(list(self.tags.values())),
        #         axis=1).astype(np.int32).tolist()

        self.tag_names = list(self.tags.keys())
        if not load_img:
            #print("load image")
            img_name = os.path.join(root, self.img_names[0])
            img = np.array(Image.open(img_name), np.uint8)
            self.width = img.shape[1]
            self.height = img.shape[0]

    def select_tag(self, tag, start=0, end=0):
        if tag == 'empty':
            return self.tags[tag]
        return self.tags[tag][start:end]

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
            traj_files = glob(os.path.join(path, name, 'baseline', self.name, '*0*.txt'))

            pred_traj = []
            for traj_file in traj_files:
                with open(traj_file, 'r') as f:
                    traj = [list(map(float, x.strip().split(',')))
                            for x in f.readlines()]
                    pred_traj.append(traj)
            if store:
                self.pred_trajs[name] = pred_traj
            else:
                return pred_traj

class VOTDataset(Dataset):
    """
    Args:
        name: dataset name, should be 'VOT2018', 'VOT2016', 'VOT2019'
        dataset_root: dataset root
        load_img: wether to load all imgs
        single_video: a sinlge video from dataset
    """
    def __init__(self, name, dataset_root, load_img=False, single_video=None):
        super(VOTDataset, self).__init__(name, dataset_root)

        # load videos
        video_names = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]

        pbar = tqdm(video_names, desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:

            if single_video and single_video != video:
                continue

            if not os.path.isdir(os.path.join(dataset_root, video)):
                continue

            pbar.set_postfix_str(video)
            with open(os.path.join(dataset_root, video, 'groundtruth.txt'), 'r') as f:
                if name == 'VOT2020':
                    gt_rects = [list(map(float, x[1:].strip().split(',')))[0:4] for x in f.readlines()]
                else:
                    gt_rects = [list(map(float, x.strip().split(','))) for x in f.readlines()]

            img_names = [os.path.join(video, 'color', x)  for x in sorted(os.listdir(os.path.join(dataset_root, video, 'color')), key=lambda x:int(os.path.basename(x).split('.')[0]))]

            try:
                with open(os.path.join(dataset_root, video, 'camera_motion.tag'), 'r') as f:
                    camera_motion = [int(v) for v in f.read().splitlines()]
                    camera_motion += [0] * (len(img_names) - len(camera_motion))
            except:
                camera_motion = [0] * len(img_names)
            try:
                with open(os.path.join(dataset_root, video, 'illum_change.tag'), 'r') as f:
                    illum_change = [int(v) for v in f.read().splitlines()]
                    illum_change += [0] * (len(img_names) - len(illum_change))
            except:
                illum_change = [0] * len(img_names)
            try:
                with open(os.path.join(dataset_root, video, 'motion_change.tag'), 'r') as f:
                    motion_change = [int(v) for v in f.read().splitlines()]
                    motion_change += [0] * (len(img_names) - len(motion_change))
            except:
                motion_change = [0] * len(img_names)
            try:
                with open(os.path.join(dataset_root, video, 'size_change.tag'), 'r') as f:
                    size_change = [int(v) for v in f.read().splitlines()]
                    size_change += [0] * (len(img_names) - len(size_change))
            except:
                size_change = [0] * len(img_names)
            try:
                with open(os.path.join(dataset_root, video, 'occlusion.tag'), 'r') as f:
                    occlusion = [int(v) for v in f.read().splitlines()]
                    occlusion += [0] * (len(img_names) - len(occlusion))
            except:
                occlusion = [0] * len(img_names)

            self.videos[video] = VOTVideo(video, dataset_root,img_names, gt_rects, camera_motion,
                                          illum_change, motion_change, size_change, occlusion, load_img=load_img)

        self.tags = ['all', 'camera_motion', 'illum_change', 'motion_change',
                     'size_change', 'occlusion', 'empty']
