from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
from glob import glob
from jsonargparse import ArgumentParser, ActionParser
import numpy as np
import os
import sys
import torch

from models.tracker import build_tracker as build_baseline_tracker
from models.hybrid_tracker import build_tracker as build_online_tracker
from models.hybrid_tracker import get_args_parser as tracker_args_parser

def get_args_parser():
    parser = ArgumentParser(prog='demo')

    parser.add_argument('--use_baseline_tracker', action='store_true',
                        help='whether use baseline(offline) tracker')
    parser.add_argument('--video_name', default='', type=str,
                        help='empty to use webcam, otherwise *.mp4, *.avi, *jpg, *JPEG, or *.png are allowed')
    parser.add_argument('--debug', action='store_true',
                        help='whether visualize the debug result')

    parser.add_argument('--tracker', action=ActionParser(parser=tracker_args_parser()))

    return parser

def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        ext = os.listdir(video_name)[0].split(".")[-1]
        assert ext == "jpg" or ext == "JPEG" or ext == "png"

        images = glob(os.path.join(video_name, '*.' + ext))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main(args, tracker):


    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'

    template_image = None
    for frame in get_frames(args.video_name):

        if first_frame:
            try:
                cv2.namedWindow("Select Roi",1)
                cv2.putText(frame, "select bounding box using cursor,", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, "and press 'Enter' to start", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                init_rect = cv2.selectROI("Select Roi", frame, False, False)
            except:
                exit()

            tracker.init(frame, init_rect)
            first_frame = False
            continue


        output = tracker.track(frame)

        bbox = np.round(output["bbox"]).astype(np.uint16)

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      (0, 255, 0), 3)
        cv2.imshow("result", frame)

        wait = 1
        if args.debug:
            wait = 0
            for key, value in output.items():
                if isinstance(value, np.ndarray):
                    if len(value.shape) == 3 or len(value.shape) == 2:
                        cv2.imshow(key, value)

        k = cv2.waitKey(wait)

        if k == 27:         # wait for ESC key to exit
            sys.exit()


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    print(args)

    # create tracker
    if args.use_baseline_tracker:
        tracker = build_baseline_tracker(args.tracker)
    else:
        tracker = build_online_tracker(args.tracker)

    main(args, tracker)

