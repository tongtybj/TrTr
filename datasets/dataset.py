from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from jsonargparse import ArgumentParser, ActionParser
from importlib import import_module
import mimetypes
import logging
import sys
import os
from os.path import join

import pathlib
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms as T
import math

try:
    from .augmentation import Augmentation, corner2center, center2corner, Center
    from .utils import gaussian_radius, draw_umich_gaussian
except ImportError as e:
    if "attempted relative import with no known parent package" == str(e):
         # for test.py
        from augmentation import Augmentation, corner2center, center2corner, Center
        from utils import gaussian_radius, draw_umich_gaussian

logger = logging.getLogger("global")


class SubDataset(object):
    def __init__(self, path, ann_file, frame_range, num_use, start_idx):

        self.path = path
        self.ann_file = ann_file
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        logger.info("loading " + self.path)


        ext = str(self.ann_file).split(".")[-1]
        read_type = 'rb' if ext == "pickle" else 'r'
        try:
            module = import_module(ext)

            with open(self.ann_file, read_type) as f:

                meta_data = module.load(f)
                meta_data = self._filter_zero(meta_data)
        except:
            raise TypeError("can not load module {}, currently can only support json and pickle".format(ext))

        for video in list(meta_data.keys()):
            for track in meta_data[video]["tracks"]:
                frames = meta_data[video]["tracks"][track]
                # print("frames for {}: {}".format(video, frames.keys()))
                frames = list(map(int,
                              filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
                meta_data[video]["tracks"][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video]["tracks"][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]["tracks"]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        #print("size of meta_data for {} is {}".format(ann_file, len(meta_data)))
        self.labels = meta_data
        self.num = len(self.labels)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())
        logger.info("{} loaded".format(self.path))
        self.path_format = '{}.{}.{}.jpg'
        self.pick = self.shuffle()

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            new_tracks['frame_size'] = tracks['frame_size']
            new_tracks['tracks'] = {}
            for trk, frames in tracks['tracks'].items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks['tracks'][trk] = new_frames
            if len(new_tracks['tracks']) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.path, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))

        pick = []
        # assume self.num_use > self.num
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]


    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)
        image_path = os.path.join(self.path, video,
                                  self.path_format.format(frame, track, 'x'))
        image_anno = self.labels[video]["tracks"][track][frame]
        image_size = self.labels[video]["frame_size"]
        return image_path, image_anno, image_size

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video["tracks"].keys())) # random choose a track id from a video
        track_info = video["tracks"][track]

        frames = track_info['frames']
        template_frame = np.random.randint(0, len(frames))
        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames)-1) + 1
        search_range = frames[left:right]
        template_frame = frames[template_frame]
        search_frame = np.random.choice(search_range)
        return self.get_image_anno(video_name, track, template_frame), \
            self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video["tracks"].keys()))
        track_info = video["tracks"][track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


class TrkDataset(Dataset):
    def __init__(self,
                 image_set, dataset_paths,
                 dataset_video_frame_ranges, dataset_num_uses,
                 template_shift, template_scale, template_color,
                 search_shift, search_scale, search_blur, search_color,
                 exempler_size = 127, search_size = 255,
                 negative_rate = 0.5, model_stride = 1):
        super(TrkDataset, self).__init__()

        self.exempler_size = exempler_size
        self.search_size = search_size
        self.negative_rate = negative_rate

        self.output_size = (self.search_size + model_stride - 1) // model_stride

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0

        for path, video_frame_range, num_use in zip(dataset_paths, dataset_video_frame_ranges, dataset_num_uses):

            assert Path(path).exists(), f'provided VID path {path} does not exist'

            ANN_PATHS = {
                "train": join(path, 'train.pickle'),
                "val":   join(path, 'val.pickle')
            }

            sub_dataset = SubDataset(
                path, ANN_PATHS[image_set],
                int(video_frame_range),
                int(num_use),
                start)
            start += sub_dataset.num
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)


        print("size of {} dataset: {}".format(image_set, self.num))
        self.pick = self.shuffle()

        # data augmentation
        self.template_aug = Augmentation(
            template_shift,
            template_scale,
            0,
            0,
            template_color)
        self.search_aug = Augmentation(
            search_shift,
            search_scale,
            search_blur,
            0,
            search_color)

        # input image scaling for the backbone pretrained with ImageNet
        # https://pytorch.org/docs/stable/torchvision/models.html
        self.image_normalize = T.Compose([
            T.ToTensor(), # Converts a numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def normalize(self, image, bbox = None):
        image = self.image_normalize(image)

        if not bbox:
            return image
        else:
            h, w = image.shape[-2:]
            bbox = torch.as_tensor(corner2center(bbox), dtype=torch.float32)
            bbox = bbox / torch.tensor([w, h, w, h], dtype=torch.float32)
            return image, bbox

    def shuffle(self):
        pick = []
        for sub_dataset in self.all_dataset:
            pick += sub_dataset.pick

        assert len(pick) == self.num
        logger.info("dataset length {}".format(self.num))
        return pick

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape, raw_img_size):
        imh, imw = image.shape[:2]
        assert imh == imw
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = self.exempler_size
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        s_x = imh / scale_z
        bb_center = [(shape[2]+shape[0])/2., (shape[3]+shape[1])/2.]
        mask = [0, 0, imw, imh]
        if bb_center[0] < s_x/2:
            mask[0] = (s_x/2 - bb_center[0]) * scale_z
        if bb_center[1] < s_x/2:
            mask[1] = (s_x/2 - bb_center[1]) * scale_z
        if bb_center[0] + s_x/2 > raw_img_size[0]:
            mask[2] = mask[2] - (bb_center[0] + s_x/2 - raw_img_size[0]) * scale_z
        if bb_center[1] + s_x/2 > raw_img_size[1]:
            mask[3] = mask[3] - (bb_center[1] + s_x/2 - raw_img_size[1]) * scale_z


        w = w*scale_z
        h = h*scale_z
        cx, cy = imw/2., imh/2. # TODO: original is imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))

        #print("raw shape: {}, raw bbox : {}, raw mask: {}".format(shape, bbox, mask))

        return bbox, mask

    def __len__(self):
        return self.num

    def __getitem__(self, index):

        index = self.pick[index]
        dataset, index = self._find_dataset(index)

        # gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()

        # negative sample
        neg = self.negative_rate and self.negative_rate > np.random.random()

        if neg:
            template = dataset.get_random_target(index)
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)

        # get image
        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])


        # get bounding box
        template_box, template_mask  = self._get_bbox(template_image, template[1], template[2])
        search_box, search_mask = self._get_bbox(search_image, search[1], search[2])

        #print("search_box: {}, search_mask: {}".format(search_box, search_mask))

        # augmentation
        template, _ , template_mask = self.template_aug(template_image,
                                                        template_box,
                                                        template_mask,
                                                        self.exempler_size)

        search, input_bbox, search_mask = self.search_aug(search_image,
                                                          search_box,
                                                          search_mask,
                                                          self.search_size)
        #print("after aug search_box: {}, search_mask: {}".format(search_box, search_mask))

        hm = np.zeros((self.output_size, self.output_size), dtype=np.float32)


        valid = not neg
        scale = float(self.output_size) / float(self.search_size)
        output_bbox = [input_bbox.x1 * scale, input_bbox.y1 * scale, input_bbox.x2 * scale, input_bbox.y2 * scale]
        radius = gaussian_radius((math.ceil(output_bbox[3] - output_bbox[1]), math.ceil(output_bbox[2] - output_bbox[0])))
        radius = max(0, int(radius))
        ct = np.array([(output_bbox[0] + output_bbox[2]) / 2, (output_bbox[1] + output_bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        reg = torch.tensor(ct - ct_int, dtype=torch.float32) # range is [0, 1)
        wh = torch.tensor([input_bbox[2] - input_bbox[0], input_bbox[3] - input_bbox[1]], dtype=torch.float32) / float(self.search_size) # normalized
        ind = ct_int[1] * self.output_size + ct_int[0]
        if valid:
            draw_umich_gaussian(hm, ct_int, radius)
        hm = torch.tensor(hm, dtype=torch.float32) # range is [0, 1)


        # print("index: {}, gaussian radius: {}, ct: {}, reg: {}, wh: {}, valid: {}".format(index, radius, ct, reg, wh, valid)) # debug

        """
        print("dataset idx: {}".format(index))
        print("aug search image for {}: {}".format(index, search))
        temp_search = search.astype(np.uint8).copy()
        bbox_int = np.round(bbox).astype(np.uint16)
        cv2.rectangle(temp_search, (bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3]), (0,255,0))

        cv2.imshow('auged search_image', temp_search)
        k = cv2.waitKey(0)
        """

        # normalize
        # we have to change to type from float to uint8 for torchvision.transforms.ToTensor
        # https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToTensor
        template = self.normalize(np.round(template).astype(np.uint8))
        search = self.normalize(np.round(search).astype(np.uint8))
        template_mask = torch.as_tensor(template_mask)
        search_mask =  torch.as_tensor(search_mask)

        target_dict =  {'hm': hm, 'reg': reg, 'wh': wh, 'ind': torch.as_tensor(ind), 'valid': torch.as_tensor(valid), 'bbox_debug': torch.as_tensor(input_bbox)}

        return template, search, template_mask, search_mask, target_dict

def get_args_parser():
    parser = ArgumentParser(prog = 'dataset')

    parser.add_argument('--paths', default=[], nargs='+',
                        help = 'the paths to datasets')
    parser.add_argument('--num_uses', default=[-1], nargs='+',
                        help = 'the number of images to used for training in each dataset')
    parser.add_argument('--eval_num_uses', default=[], nargs='+',
                        help = 'the number of images to used for evaluation in each dataset')
    parser.add_argument('--video_frame_ranges', default=[100], nargs='+',
                        help = 'the frames interval in each dataset')

    parser.add_argument('--template_aug_shift', default=4, type=int)
    parser.add_argument('--template_aug_scale', default=0.05, type=float)
    parser.add_argument('--template_aug_color', default=1.0, type=float) # Pysot is 1.0
    parser.add_argument('--search_aug_shift', default=64, type=int)
    parser.add_argument('--search_aug_scale', default=0.18, type=float)
    parser.add_argument('--search_aug_blur', default=0.2, type=float)  # Pysot is 0.2
    parser.add_argument('--search_aug_color', default=1.0, type=float)  # Pysot is 1.0
    parser.add_argument('--exempler_size', default=127, type=int)
    parser.add_argument('--search_size', default=255, type=int)
    parser.add_argument('--negative_aug_rate', default=0.2, type=float)

    return parser

def build(image_set, args, model_stride):

    assert len(args.paths) == len(args.video_frame_ranges) == len(args.num_uses)

    if image_set == 'val':
        if len(args.num_uses) == len(args.eval_num_uses):
            args.num_uses = args.eval_num_uses
        else:
            args.num_uses = [-1] * len(args.num_uses)

    dataset = TrkDataset(image_set,
                         args.paths,
                         args.video_frame_ranges,
                         args.num_uses,
                         args.template_aug_shift,
                         args.template_aug_scale,
                         args.template_aug_color,
                         args.search_aug_shift,
                         args.search_aug_scale,
                         args.search_aug_blur,
                         args.search_aug_color,
                         args.exempler_size,
                         args.search_size,
                         args.negative_aug_rate,
                         model_stride)
    return dataset
