from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import copy
from glob import glob
from jsonargparse import ArgumentParser, ActionParser, ActionConfigFile
import numpy as np
import os
import sys
import torch

sys.path.append('..')
from util.box_ops import get_axis_aligned_bbox
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str

from external_tracker import build_external_tracker
from models.tracker import build_tracker as build_baseline_tracker
from models.hybrid_tracker import build_tracker as build_online_tracker
from models.hybrid_tracker import get_args_parser as tracker_args_parser


def get_args_parser():
    parser = ArgumentParser(prog='benchmark dataset inference')

    # tracking
    parser.add_argument('--use_baseline_tracker', action='store_true',
                        help='whether use baseline(offline) tracker')
    parser.add_argument('--external_tracker', type=str, default='',
                        choices=('', 'atom', 'dimp', 'prdimp'),
                        help='if not empty, the external tracker will be used')

    parser.add_argument('--dataset_path', type=str, default='',
                        help='path of datasets')
    parser.add_argument('--dataset', type=str, default='VOT2018',
                        choices=('VOT2018', 'VOT2019', 'VOT2020', 'OTB', 'UAV', 'NFS', 'TrackingNet', 'LaSOT', 'GOT-10k'),
                        help='the name of benchmark')
    parser.add_argument('--video', type=str, default='',
                        help='eval one special video')
    parser.add_argument('--vis', action='store_true',
                        help='whether visualzie result')
    parser.add_argument('--debug_vis', action='store_true',
                        help='whether visualize the debug result')
    parser.add_argument('--model_name', type=str, default='trtr',
                        help='the name of tracker')

    parser.add_argument('--result_path', type=str, default='results',
                        help='the path to store trackingresults')

    parser.add_argument('--save_image_num_per_video', type=int, default=1,
                        help='save the tracking result as image, please choose value larger than 1, or 0 for saving every frame')

    parser.add_argument('--repetition', default=1, type=int)
    parser.add_argument('--min_lost_rate_for_repeat', default=0.1, type=float) # change for different benchmark

    parser.add_argument('--tracker', action=ActionParser(parser=tracker_args_parser()))

    # yaml config file for all parameters
    parser.add_argument('--cfg_file', action=ActionConfigFile)

    return parser

def main(args, tracker):

    # create dataset
    if not args.dataset_path:
        args.dataset_path = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(args.dataset_path, 'dataset', args.dataset)

    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False,
                                            single_video=args.video)

    model_name = args.model_name

    if args.debug_vis:
        args.vis = True

    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019', 'VOT2020']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue

            video_total_lost = 0
            for cnt in range(args.repetition):
                frame_counter = 0
                lost_number = 0
                toc = 0
                init_toc = 0
                valid_frames = 0
                pred_bboxes = []

                template_image = None
                search_image = None
                raw_heatmap = None
                post_heatmap = None

                for idx, (img, gt_bbox) in enumerate(video):
                    if len(gt_bbox) == 4:
                        gt_bbox = [gt_bbox[0], gt_bbox[1],
                                   gt_bbox[0], gt_bbox[1]+gt_bbox[3],
                                   gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3],
                                   gt_bbox[0]+gt_bbox[2], gt_bbox[1]]
                    tic = cv2.getTickCount()
                    if idx == frame_counter:
                        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                        gt_bbox_ = [cx - w/2, cy - h/2, w, h]
                        tracker.init(img, gt_bbox_)
                        init_toc += cv2.getTickCount() - tic
                        pred_bbox = gt_bbox_
                        pred_bboxes.append(1)

                    elif idx > frame_counter:
                        outputs = tracker.track(img)
                        pred_bbox = outputs['bbox']
                        pred_bbox = [pred_bbox[0], pred_bbox[1],
                                     pred_bbox[2] - pred_bbox[0],
                                     pred_bbox[3] - pred_bbox[1]]

                        valid_frames += 1
                        toc += cv2.getTickCount() - tic

                        overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))

                        if overlap > 0:
                            # not lost
                            pred_bboxes.append(pred_bbox)
                        else:
                            # lost object
                            pred_bboxes.append(2)
                            frame_counter = idx + 5 # skip 5 frames
                            lost_number += 1

                            if args.vis and args.debug_vis:

                                cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 2))], True, (0, 255, 0), 3)

                                bbox = list(map(int, pred_bbox))
                                cv2.rectangle(img, (bbox[0], bbox[1]),
                                              (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                cv2.putText(img, 'lost', (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                cv2.imshow(video.name, img)

                                for key, value in outputs.items():
                                    if isinstance(value, np.ndarray):
                                        if len(value.shape) == 3 or len(value.shape) == 2:
                                            cv2.imshow(key, value)

                                k = cv2.waitKey(0)
                                if k == 27:         # wait for ESC key to exit
                                    sys.exit()

                    else:
                        pred_bboxes.append(0)
                    if idx == 0:
                        if args.vis:
                            cv2.destroyAllWindows()
                    if args.vis and idx > frame_counter:
                        cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 2))], True, (0, 255, 0), 3)

                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                        cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow(video.name, img)

                        if args.debug_vis:

                            for key, value in outputs.items():
                                if isinstance(value, np.ndarray):
                                    if len(value.shape) == 3 or len(value.shape) == 2:
                                        cv2.imshow(key, value)

                            k = cv2.waitKey(0)
                            if k == 27:         # wait for ESC key to exit
                                break
                        else:
                            k = cv2.waitKey(1)
                            if k == 27:         # wait for ESC key to exit
                                break

                    sys.stderr.write("inference on {}:  {} / {}\r".format(video.name, idx+1, len(video)))

                toc /= cv2.getTickFrequency()
                init_toc /= cv2.getTickFrequency()
                # save results
                video_path = os.path.join(args.result_path, args.dataset, model_name,
                        'baseline', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_{:03d}.txt'.format(video.name, cnt+1))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        if isinstance(x, int):
                            f.write("{:d}\n".format(x))
                        else:
                            f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
                log = '({:3d}) Video ({:2d}): {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                        v_idx+1, cnt+1, video.name, init_toc + toc, valid_frames / toc, lost_number)
                print(log)
                with open(os.path.join(args.result_path, args.dataset, model_name, 'log.txt'), 'a') as f:
                    f.write(log + '\n')
                video_total_lost += lost_number
            total_lost += video_total_lost
            if args.repetition > 1:
                log = '({:3d}) Video: {:12s} Avg Lost: {:.3f}'.format(v_idx+1, video.name, video_total_lost/args.repetition)
                print(log)
                with open(os.path.join(args.result_path, args.dataset, model_name, 'log.txt'), 'a') as f:
                    f.write(log + '\n')

        log = "{:s} total (avg) lost: {:.3f}".format(model_name, total_lost/args.repetition)
        print(log)
        with open(os.path.join(args.result_path, args.dataset, model_name, 'log.txt'), 'a') as f:
            f.write(log + '\n')
    else:
        # OPE tracking

        find_best = True

        if not dataset.has_ground_truth:
            find_best = False

        # if repeat 3 times for GOT-10k, use the official benchmark mode (no find best)
        if args.dataset == 'GOT-10k':
            if args.repetition == 3:
                find_best = False

        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue

            best_pred_bboxes = []
            min_lost_number = 1e6
            for cnt in range(args.repetition):
                toc = 0
                init_toc = 0
                pred_bboxes = []
                track_times = []
                template_image = None
                search_image = None
                raw_heatmap = None
                post_heatmap = None
                lost_number = 0

                if find_best and min_lost_number < args.min_lost_rate_for_repeat * len(video):
                    print("Abolish reset of trails ({}~) becuase the min lost number is small enough: {} / {}".format(cnt+1 , min_lost_number, args.min_lost_rate_for_repeat * len(video)))
                    break

                save_image_offset = 0
                if args.save_image_num_per_video > 1:
                    save_image_offset = len(video) // (args.save_image_num_per_video - 1)
                if args.save_image_num_per_video == 0:
                    save_image_offset = 1

                for idx, (img, gt_bbox) in enumerate(video):
                    tic = cv2.getTickCount()
                    if idx == 0:
                        outputs = tracker.init(img, gt_bbox)
                        init_toc += cv2.getTickCount() - tic
                        pred_bbox = gt_bbox
                        pred_bboxes.append(pred_bbox)
                    else:
                        outputs = tracker.track(img)
                        toc += cv2.getTickCount() - tic
                        pred_bbox_ = outputs['bbox']
                        pred_bbox = [pred_bbox_[0], pred_bbox_[1],
                                     pred_bbox_[2] - pred_bbox_[0],
                                     pred_bbox_[3] - pred_bbox_[1]]
                        pred_bboxes.append(pred_bbox)

                    track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())

                    gt_bbox_int = list(map(lambda x: int(x) if not np.isnan(x) else 0, gt_bbox))
                    pred_bbox_int = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox_int[0], gt_bbox_int[1]),
                                  (gt_bbox_int[0]+gt_bbox_int[2], gt_bbox_int[1]+gt_bbox_int[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox_int[0], pred_bbox_int[1]),
                                  (pred_bbox_int[0]+pred_bbox_int[2], pred_bbox_int[1]+pred_bbox_int[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    if save_image_offset > 0:

                        image_path = os.path.join(args.result_path, args.dataset, model_name, 'images', video.name)
                        if not os.path.isdir(image_path):
                            os.makedirs(image_path)

                        if idx % save_image_offset == 0:
                            imagename = os.path.join(image_path,  'image{:03d}.jpg'.format(idx))
                            cv2.imwrite(imagename,img)


                    if idx == 0:
                        if args.vis:
                            cv2.destroyAllWindows()
                            if args.debug_vis and isinstance(outputs, dict):
                                for key, value in outputs.items():
                                    if isinstance(value, np.ndarray):
                                        if len(value.shape) == 3 or len(value.shape) == 2:
                                            cv2.imshow(key, value)
                    else:
                        if not gt_bbox == [0,0,0,0] and not np.isnan(np.array(gt_bbox)).any():
                            if pred_bbox[0] + pred_bbox[2] < gt_bbox[0] or pred_bbox[0] > gt_bbox[0] + gt_bbox[2] or pred_bbox[1] + pred_bbox[3] < gt_bbox[1] or pred_bbox[1] > gt_bbox[1] + gt_bbox[3]:
                                lost_number += 1

                        if find_best and lost_number > min_lost_number:
                            break


                        if args.vis or args.debug_vis:
                            cv2.imshow(video.name, img)

                            if args.debug_vis:

                                for key, value in outputs.items():
                                    if isinstance(value, np.ndarray):
                                        if len(value.shape) == 3 or len(value.shape) == 2:
                                            cv2.imshow(key, value)

                                k = cv2.waitKey(0)
                                if k == 27:         # wait for ESC key to exit
                                    min_lost_number = 1e6 # this allows to  try args.repetition times for debug
                                    lost_number = 1e6 # this allows to  try args.repetition times for debug
                                    break
                            else:
                                k = cv2.waitKey(1)
                                if k == 27:         # wait for ESC key to exit
                                    min_lost_number = 1e6 # this allows to  try args.repetition times for debug
                                    lost_number = 1e6 # this allows to  try args.repetition times for debug
                                    break

                    sys.stderr.write("inference on {}:  {} / {}\r".format(video.name, idx+1, len(video)))

                if find_best and lost_number > min_lost_number:
                    print('Stop No.{} trial becuase the lost number already exceed the min lost number: {} > {} '.format(cnt+1, lost_number, min_lost_number))
                    continue

                if lost_number == 1e6:
                    continue

                if lost_number < min_lost_number:
                    min_lost_number = lost_number

                toc /= cv2.getTickFrequency()
                init_toc /= cv2.getTickFrequency()
                # save results
                if 'GOT-10k' == args.dataset:
                    video_path = os.path.join(args.result_path, args.dataset, model_name, video.name)
                    if not os.path.isdir(video_path):
                        os.makedirs(video_path)
                    id = cnt + 1
                    if find_best:
                        id = 1
                    result_path = os.path.join(video_path, '{}_{:03d}.txt'.format(video.name, id))
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([vot_float2str("%.4f", i) for i in x ])+'\n')
                    result_path = os.path.join(video_path,
                            '{}_time.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in track_times:
                            f.write("{:.6f}\n".format(x))
                else:
                    model_path = os.path.join(args.result_path, args.dataset, model_name)
                    if not os.path.isdir(model_path):
                        os.makedirs(model_path)
                    result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')

                log = '({:3d}) Video: {:12s} Trail: {:2d}  Time: {:5.1f}s Speed: {:3.1f}fps Lost: {:d}/{:d}'.format(
                    v_idx+1, video.name, cnt+1, init_toc + toc, idx / toc, lost_number, len(video))
                print(log)
                with open(os.path.join(args.result_path, args.dataset, model_name, 'log.txt'), 'a') as f:
                    f.write(log + '\n')



if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    # create tracker
    if args.use_baseline_tracker:
        tracker = build_baseline_tracker(args.tracker)
    elif args.external_tracker:
        tracker = build_external_tracker(args)
    else:
        tracker = build_online_tracker(args.tracker)

    main(args, tracker)
