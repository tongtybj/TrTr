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

    subsets = os.listdir(base_path)

    for subset in subsets:
        subset_dir = join(base_path, subset)
        videos = os.listdir(subset_dir)

        for video in videos:

            video_dir = join(subset_dir, video)
            #print(video_dir)
            frames = sorted(glob.glob(join(video_dir, 'img', '*.jpg')))

            with open(join(video_dir, 'groundtruth.txt')) as f:
                ann = f.readlines()

            with open(join(video_dir, 'out_of_view.txt')) as f:
                absence_list = f.readlines()[0]
            absence = [int(ab) for ab in absence_list.split(',')]

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
