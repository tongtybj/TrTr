from os.path import join, isdir
from os import listdir, mkdir, makedirs
import sys
import cv2
import numpy as np
import glob
import cv2
import os


sys.path.append('..')
import utils

base_path = "./dataset/train"

def show_videos():

    with open(join(base_path, 'list.txt')) as f:
        videos = f.readlines()

    for video in videos:

        video_dir = join(base_path, video.replace("\n", ""))
        print(video_dir)
        frames = sorted(glob.glob(join(video_dir, '*.jpg')))

        with open(join(video_dir, 'groundtruth.txt')) as f:
            ann = f.readlines()

        with open(join(video_dir, 'absence.label')) as f:
            absence = f.readlines()

        assert len(absence) == len(ann)

        for id, frame in enumerate(frames):
            im = cv2.imread(frame)

            bbox = [float(s) for s in  ann[id].split(',')]

            pt1 = (int(bbox[0]), int(bbox[1]))
            pt2 = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))

            wait = 5
            if (bbox[2] == 0 and bbox[3] == 0) or int(absence[id]) == 1:
                print("absence: {}".format(int(absence[id])))
                cv2.putText(im,'absence',(10, int(im.shape[1] * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,0,0),2,cv2.LINE_AA)
                wait = -1
            else:
                cv2.rectangle(im, pt1, pt2, (0,255,0), 3)

            cv2.imshow('img', im)

            k = cv2.waitKey(wait)
            if k == 27:         # wait for ESC key to exit
                sys.exit()



if __name__ == '__main__':

    show_videos()
