'''
usage:

$ python test.py --paths ./yt_bb/dataset/Curation ./vid/dataset/Curation/ --video_frame_ranges 100 3 --num_uses 20 20
'''

from jsonargparse import ArgumentParser
import datetime
import json
import random
import time
import sys
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from dataset import build as build_dataset
from dataset import get_args_parser as dataset_args_parser

sys.path.append('..')
import util.misc as utils
from util.box_ops import  box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

import cv2
import torchvision.transforms as T

def get_args_parser():
    parser = dataset_args_parser()

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epoch', default=1, type=int)

    parser.add_argument('--num_workers', default=2, type=int)

    parser.add_argument('--model_stride', default=8, type=int) # debug

    return parser


def main(args):

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_train = build_dataset(image_set='train', args=args, model_stride = args.model_stride)
    dataset_val = build_dataset(image_set='val', args=args, model_stride = args.model_stride)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    for epoch in range(args.epoch):

        print("the len of dataset_train: {}".format(len(dataset_train)))

        for i, obj in enumerate(data_loader_train):
            print("{} iterator has {} batches".format(i, len(obj[2])))
            targets = [{k: v.to(device) for k, v in t.items()} for t in obj[4]]

            template_nested_samples = utils.nested_tensor_from_tensor_list(obj[0], obj[2])
            search_nested_samples = utils.nested_tensor_from_tensor_list(obj[1], obj[3])
            template_samples = template_nested_samples.tensors.to(device) # use several time to load to gpu
            search_samples = search_nested_samples.tensors.to(device)  # use several time to load to gpu
            template_masks = template_nested_samples.mask.to(device) # use several time to load to gpu
            search_masks = search_nested_samples.mask.to(device) # use several time to load to gpu


            # print(targets) # debug
            image_revert = T.Compose([
                T.Normalize(
                    -np.array(dataset_train.image_normalize.transforms[1].mean)
                    / np.array(dataset_train.image_normalize.transforms[1].std),
                    1 / np.array(dataset_train.image_normalize.transforms[1].std)),
                T.Normalize(0, [1.0 / 255, 1.0 / 255, 1.0 / 255])
            ])

            for batch_i in range(len(obj[2])):

                revert_search_image = image_revert(search_samples[batch_i])
                search_image = revert_search_image.to('cpu').detach().numpy().copy()
                search_image = (np.round(search_image.transpose(1,2,0))).astype(np.uint8).copy()

                revert_template_image = image_revert(template_samples[batch_i])
                template_image = revert_template_image.to('cpu').detach().numpy().copy()
                template_image = (np.round(template_image.transpose(1,2,0))).astype(np.uint8).copy()

                output_size = targets[batch_i]["hm"].shape[0]
                sesarch_size = search_image.shape[0]

                ind = targets[batch_i]["ind"].item()
                ct = torch.as_tensor([ind % output_size, ind // output_size], dtype=torch.float32)

                revert_ct = (ct + targets[batch_i]["reg"].to('cpu')) * float(sesarch_size) / float(output_size)
                revert_wh = targets[batch_i]["wh"].to('cpu') * float(sesarch_size)
                revert_bbox = torch.round(box_cxcywh_to_xyxy(torch.cat([revert_ct, revert_wh]))).int()
                # draw the reverted bbox from the target info
                cv2.rectangle(search_image, (revert_bbox[0], revert_bbox[1]), (revert_bbox[2], revert_bbox[3]), (0,0,255), 2)
                # draw the orginal ground truth bbox
                orig_bbox = torch.round(targets[batch_i]["bbox_debug"].to('cpu')).int()
                cv2.rectangle(search_image, (orig_bbox[0], orig_bbox[1]), (orig_bbox[2], orig_bbox[3]), (0,255,0))
                revert_ct  = np.round(revert_ct.numpy()).astype(np.int)
                cv2.circle(search_image, (revert_ct[0], revert_ct[1]), 2, (0,255,0), -1) # no need

                # draw the mask
                template_mask = torch.round(obj[2][batch_i].to('cpu')).int()
                cv2.rectangle(template_image, (template_mask[0], template_mask[1]), (template_mask[2], template_mask[3]), (0,255,0))

                search_mask = torch.round(obj[3][batch_i].to('cpu')).int()
                cv2.rectangle(search_image, (search_mask[0], search_mask[1]), (search_mask[2], search_mask[3]), (0,255,0))

                #print("search_mask_float : {}, orig_bbox_float: {}".format(targets[batch_i]["search_mask"].to('cpu'), targets[batch_i]["bbox_debug"].to('cpu')))
                #print("search_mask: {}, orig_bbox: {}".format(search_mask, orig_bbox))

                heatmap = (torch.round(targets[batch_i]["hm"] * 255)).to('cpu').detach().numpy().astype(np.uint8)
                # print("center of gaussian peak: {}".format(np.unravel_index(np.argmax(heatmap), heatmap.shape)))
                # mask the heatmap to the original image
                heatmap_resize = cv2.resize(heatmap, search_image.shape[1::-1])
                # print("heatmap peal: {}, revert_wh: {}".format(np.unravel_index(np.argmax(heatmap_resize), heatmap_resize.shape), revert_ct))
                heatmap_color = np.stack([heatmap_resize, np.zeros(search_image.shape[1::-1], dtype=np.uint8), heatmap_resize], -1)
                search_image = np.round(0.4 * heatmap_color + 0.6 * search_image.copy()).astype(np.uint8)

                cv2.imshow('heatmap', heatmap)
                cv2.imshow('search_image', search_image)
                cv2.imshow('template_image', template_image)

                search_image2 = deepcopy(search_image)
                search_mask_img = search_masks[batch_i].to('cpu').detach().numpy()
                search_image2[np.repeat(search_mask_img[:, :, np.newaxis], 3, axis=2)] = 0
                cv2.imshow('search_image_mask', search_image2)

                template_image2 = deepcopy(template_image)
                template_mask_img = template_masks[batch_i].to('cpu').detach().numpy()
                template_image2[np.repeat(template_mask_img[:, :, np.newaxis], 3, axis=2)] = 0
                cv2.imshow('template_image_mask', template_image2)

                k = cv2.waitKey(0)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
