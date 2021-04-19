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
from toolkit.utils.region import vot_float2str

from models.tracker import build_tracker as build_baseline_tracker
from models.hybrid_tracker import build_tracker as build_online_tracker

def get_args_parser():
    parser = argparse.ArgumentParser('benchmark dataset inference', add_help=False)

    # Model parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # * Backbone
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--resnet_dilation', action='store_false',
                        help="If true (default), we replace stride with dilation in resnet blocks") #default is true
    parser.add_argument('--return_interm_layers', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=1, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer") # switch by eval() / train()
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--return_layers', default=[], nargs='+')
    parser.add_argument('--dcf_layers', default=[], nargs='+')
    parser.add_argument('--weighted', action='store_true',
                        help="the weighted for the multiple input embedding for transformer")
    parser.add_argument('--transformer_mask', action='store_false',
                        help="mask for transformer") # workaround to enable transformer mask for default
    parser.add_argument('--multi_frame', action='store_true',
                        help="use multi frame for encoder (template images)")
    parser.add_argument('--repetition', default=1, type=int)
    parser.add_argument('--min_lost_rate_for_repeat', default=0.1, type=float) # change for different benchmark

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--loss_mask', action='store_true',
                        help="mask for heamtmap loss")


    # * Loss coefficients
    parser.add_argument('--reg_loss_coef', default=1, type=float,
                        help="weight (coeffficient) about bbox offset reggresion loss")
    parser.add_argument('--wh_loss_coef', default=1, type=float,
                        help="weight (coeffficient) about bbox width/height loss")

    # tracking
    parser.add_argument('--checkpoint', default="", type=str)
    parser.add_argument('--exemplar_size', default=127, type=int)
    parser.add_argument('--search_size', default=255, type=int)
    parser.add_argument('--context_amount', default=0.5, type=float)
    parser.add_argument('--use_baseline_tracker', action='store_true')

    # * hyper-parameter for tracking
    parser.add_argument('--score_threshold', default=0.05, type=float,
                        help='the lower score threshold to recognize a target (score_target > threshold) ')
    parser.add_argument('--window_steps', default=3, type=int,
                        help='the pyramid factor to gradually reduce the widow effect')
    parser.add_argument('--window_factor', default=0.4, type=float,
                        help='the factor of the hanning window for heatmap post process')
    parser.add_argument('--tracking_size_penalty_k', default=0.04, type=float,
                        help='the factor to penalize the change of size')
    parser.add_argument('--tracking_size_lpf', default=0.8, type=float,
                        help='the factor of the lpf for size tracking')
    parser.add_argument('--dcf_rate', default=0.8, type=float,
                        help='the weight for integrate dcf and trtr for heatmap ')
    parser.add_argument('--dcf_sample_memory_size', default=250, type=int,
                        help='the size of the trainining sample for DCF ')

    parser.add_argument('--dcf_size', default=0, type=int,
                        help='the size for feature for dcf')
    parser.add_argument('--boundary_recovery', action='store_true',
                        help='whether use boundary recovery')
    parser.add_argument('--hard_negative_recovery', action='store_true',
                        help='whether use hard negative recovery')
    parser.add_argument('--lost_target_recovery', action='store_true',
                        help='whether use lost target recovery')
    parser.add_argument('--lost_target_margin', default=0.3, type=float)
    parser.add_argument('--translation_threshold', default=0.03, type=float)
    parser.add_argument('--lost_target_cnt_threshold', default=60, type=int)
    parser.add_argument('--lost_target_score_threshold', default=0.5, type=float)

    parser.add_argument('--dataset_path', default="", type=str, help='path of datasets')
    parser.add_argument('--dataset', type=str, help='the benchmark', default="VOT2018")
    parser.add_argument('--video', default='', type=str, help='eval one special video')
    parser.add_argument('--model_name', default='ground_truth', type=str)
    parser.add_argument('--video_list', default='', type=str)
    parser.add_argument('--result_path', default='ground_truth', type=str)

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

    videos = []
    if args.video_list != '':
        if not os.path.exists(args.video_list):
            print("cannot find marked video lsit file: {}. Please use '$ python view_result.py' to create a proper marked video list".format(args.video_list))
            exit()
        with open(args.video_list, 'r') as f:
            videos = [x.strip().split(',')[-1].replace(' ', '') for x in f.read().splitlines()]

    if args.video != '':
        videos = [args.video]

    for v_idx, video in enumerate(dataset):

        if video.name not in videos:
            continue

        while True:
            ret = run_tracker(args, video)
            if ret > 0:
                break

def run_tracker(args, video):

    pred_bboxes = []
    for idx, (img, gt_bbox) in enumerate(video):

        if idx == 0:
            outputs = tracker.init(img, gt_bbox)
            pred_bbox = gt_bbox
        else:
            outputs = tracker.track(img)
            pred_bbox_ = outputs['bbox']
            pred_bbox = [pred_bbox_[0], pred_bbox_[1],
                         pred_bbox_[2] - pred_bbox_[0],
                         pred_bbox_[3] - pred_bbox_[1]]

        pred_bbox_int = list(map(int, pred_bbox))
        cv2.rectangle(img, (pred_bbox_int[0], pred_bbox_int[1]),
                      (pred_bbox_int[0]+pred_bbox_int[2], pred_bbox_int[1]+pred_bbox_int[3]), (0, 255, 255), 3)
        cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        msg = "'l': try again; 'r': reset tracker; 's': skip this video; 'ESC': halt process"
        cv2.putText(img, msg, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if idx == 0:
            cv2.destroyAllWindows()
        else:
            cv2.imshow('video', img)

            k = cv2.waitKey(0)
            if k == 27:         # wait for ESC key to exit
                exit()
            elif k == ord('l'):
                print('Start from first frame again')
                return 0
            elif k == ord('s'):
                break
            elif k == ord('r'):
                print('Reset tracker with a new initial bounding box. Please select new init bbox from GUI')
                init_rect = cv2.selectROI('video', img, False, False)
                outputs = tracker.init(img, init_rect)
                pred_bbox = init_rect

        pred_bboxes.append(pred_bbox)

        sys.stderr.write("inference on {}:  {} / {}\r".format(video.name, idx+1, len(video)))

    if len(pred_bboxes) < len(video):
        return 2

    model_name = args.model_name
    if 'GOT-10k' == args.dataset:
        video_path = os.path.join(args.result_path, args.dataset, model_name, video.name)
        if not os.path.isdir(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([vot_float2str("%.4f", i) for i in x ])+'\n')
    else:
        model_path = os.path.join(args.result_path, args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        result_path = os.path.join(model_path, '{}.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')

    return 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Create ground truth from tracker', parents=[get_args_parser()])
    args = parser.parse_args()

    # create tracker
    if len(args.return_layers) == 0:
        args.return_layers = ['layer3']
    if len(args.dcf_layers) == 0:
        args.dcf_layers = ['layer2', 'layer3']

    if args.use_baseline_tracker:
        tracker = build_baseline_tracker(args)
    else:
        tracker = build_online_tracker(args)

    main(args, tracker)
