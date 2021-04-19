from os.path import join, isdir
from os import listdir, mkdir, makedirs
import sys
import cv2
import numpy as np
import os
import pickle
import json
import glob
from concurrent import futures
import xml.etree.ElementTree as ET

sys.path.append('..')
import utils

debug = False

base_path = "./dataset"
train_base_path = "./dataset/train"
val_base_path = "./dataset/val"
save_base_path = join(base_path, 'Curation')


def crop_video(subdir, subset, video):
    video_dir = join(subdir, video)

    video_crop_base_path = join(save_base_path, subset, video)
    if not isdir(video_crop_base_path): makedirs(video_crop_base_path)

    if debug:
        print("crop video: {}".format(video))
    frames = sorted(glob.glob(join(video_dir, '*.jpg')))

    with open(join(video_dir, 'groundtruth.txt')) as f:
        ann = f.readlines()

    with open(join(video_dir, 'absence.label')) as f:
        absense = f.readlines()


    for id, frame in enumerate(frames):
        im = cv2.imread(frame)
        avg_chans = np.mean(im, axis=(0, 1))

        bbox = [float(s) for s in  ann[id].split(',')]

        if (bbox[2] == 0 and bbox[3] == 0) or int(absense[id]) == 1:
            continue

        trackid = 0
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

        filename = frame.split('/')[-1].split('.')[0]
        z, x = utils.crop_image(im, bbox, padding=avg_chans)
        cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(int(filename), trackid)), x)


def image_curation(num_threads=24):
    subsets = ['train', 'val']

    for subset in subsets:

        subdir = join(base_path, subset)

        with open(join(base_path, subset, 'list.txt')) as f:
            videos = f.readlines()

        if debug:
          videos = videos[:20]
        n_videos = len(videos)

        """
        for video in videos:
            crop_video(subdir, subset, video.replace("\n", ""))
        """

        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_video, subdir, subset, video.replace("\n", "")) for video in videos]
            for i, f in enumerate(futures.as_completed(fs)):
                utils.print_progress(i, n_videos, prefix=subset, suffix='Done ', barLength=40)


def save_config():
    snippets = dict()
    instance_size = utils.get_instance_size()

    subsets = ['train', 'val']

    for subset in subsets:
        subdir = join(base_path, subset)

        with open(join(base_path, subset, 'list.txt')) as f:
            videos = f.readlines()

        for vi, video in enumerate(videos):

            video = video.replace("\n", "")
            if debug and vi == 20:
              break

            video_dir = join(subdir, video)
            print('subset: {} video id: {:04d} / {:04d}'.format(subdir, vi, len(videos)))
            frames = sorted(glob.glob(join(video_dir, '*.jpg')))

            with open(join(video_dir, 'groundtruth.txt')) as f:
                ann = f.readlines()

            with open(join(video_dir, 'absence.label')) as f:
                absense = f.readlines()

            frame_sz = None
            snippets[join(subset, video)] = dict()
            snippets[join(subset, video)]['tracks'] = dict()
            snippet = dict()
            for id, frame in enumerate(frames):
                if id == 0:
                    im = cv2.imread(frame)
                    frame_sz = [im.shape[1], im.shape[0]]
                    snippets[join(subset, video)]['frame_size'] = frame_sz

                bbox = [float(s) for s in  ann[id].split(',')]
                if (bbox[2] == 0 and bbox[3] == 0) or int(absense[id]) == 1:
                    continue

                bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

                if bbox[0] < 0:
                    bbox[0] = 0
                if bbox[1] < 0:
                    bbox[1] = 0

                filename = frame.split('/')[-1].split('.')[0]
                snippet['{:06d}'.format(int(filename))] = bbox
            snippets[join(subset, video)]['tracks']['00'] = snippet

    train = {k:v for (k,v) in snippets.items() if 'train' in k}
    val = {k:v for (k,v) in snippets.items() if 'val' in k}

    json.dump(train, open(join(save_base_path,'train.json'), 'w'), indent=4, sort_keys=True)
    json.dump(val, open(join(save_base_path,'val.json'), 'w'), indent=4, sort_keys=True)

    with open(join(save_base_path, 'train.pickle'), 'wb') as f:
        pickle.dump(train, f)

    with open(join(save_base_path, 'val.pickle'), 'wb') as f:
        pickle.dump(val, f)


if __name__ == '__main__':

    if not isdir(save_base_path): mkdir(save_base_path)

    print("crop the images for training")


    if len(sys.argv) == 2:
        image_curation(sys.argv[1])
    else:
        image_curation()

    print("save the configuration for the curation data")

    save_config()
