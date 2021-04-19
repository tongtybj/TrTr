from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import sys
import cv2
import copy
import torch
import numpy as np
from glob import glob

sys.path.append('..')
from toolkit.datasets import DatasetFactory

# (B,G,R)
COLOR = ((0, 0, 255), # red
         (0, 255, 0), # green
         (255, 0, 0), # blue
         (0, 255, 255)) # yellow

def main(args):

    # create dataset
    if not args.dataset_path:
        args.dataset_path = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(args.dataset_path, 'dataset', args.dataset)

    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False,
                                            single_video=args.video)

    result_paths = [os.path.join(args.tracker_path, args.dataset, model_name) for model_name in args.model_names]


    for v_idx, video in enumerate(dataset):
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue

        pred_trajs = []
        for (model_name, result_path) in zip(args.model_names, result_paths):
            result_file = os.path.join(result_path, video.name + '.txt')
            if args.dataset == 'GOT-10k':
                result_file = os.path.join(result_path, video.name, video.name + '_001.txt')
            if args.dataset in ['VOT2018', 'VOT2019']:
                result_file = os.path.join(result_path, 'baseline', video.name, video.name + '_001.txt')
            if not os.path.exists(result_file):
                print("cannot find tracking result of {} for No.{} {}".format(model_name, v_idx+1, video.name))
                continue

            with open(result_file, 'r') as f:
                pred_traj = [list(map(lambda v: round(float(v)), x.strip().split(','))) for x in f.read().splitlines()]

            if len(pred_traj) != len(video):
                print("the size of tracking result is wrong: {} vs {}".format(len(pred_traj),len(video)))
                continue

            pred_trajs.append(pred_traj)


        pred_trajs = list(zip(*pred_trajs))

        for idx, ((img, gt_bbox), bboxes) in enumerate(zip(video, pred_trajs)):

            if args.draw_gt:
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (255, 255, 0), 3)
            for bbox_id, bbox in enumerate(bboxes):
                if len(bbox) == 4:
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), COLOR[bbox_id], 4)
            cv2.putText(img, "#{:03d}".format(idx+1), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2) # (B,G,R)

            wait_time = args.play_speed
            if idx == len(video) -1:
                msg = "enter 'm' to mark video, other keys for next"
                cv2.putText(img, msg, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                wait_time = 0

            cv2.imshow("video", img)

            k = cv2.waitKey(wait_time)

            if k == 27:         # wait for ESC key to exit
                exit()
            if k == ord("s"):
                break
            if k == ord("m"):
                # mark this video
                save_file = os.path.join(result_path, 'marked_videos.txt')
                print("mark video{} {}".format(v_idx + 1, video.name))
                with open(save_file, 'a') as f:
                    f.write('{}, {}'.format(v_idx + 1, video.name)+'\n')

            sys.stderr.write("inference on vidoe{} {}:  {} / {}\r".format(v_idx + 1, video.name,  idx+1, len(video)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser('view tracking result', add_help=False)
    parser.add_argument('--dataset_path', default="", type=str, help='path of datasets')
    parser.add_argument('--tracker_path', default="", type=str, help='path of tracker result')
    parser.add_argument('--dataset', type=str, help='the benchmark', default="VOT2018")
    parser.add_argument('--draw_gt', action='store_true')
    parser.add_argument('--model_names', default=[], nargs='+')
    parser.add_argument('--video', default='', type=str, help='eval one special video')
    parser.add_argument('--play_speed', default=30, type=int, help='video play speed')
    args = parser.parse_args()

    main(args)
