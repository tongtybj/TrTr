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
save_base_path = join(base_path, 'Curation')
TRAIN_SUBSET_PREFIX = "TRAIN_"
# TEST_SUBSET_PREFIX = "TEST" # no test dataset for val


def crop_video(subdir, subset, video):
    video_crop_base_path = join(save_base_path, subset, video)
    if not isdir(video_crop_base_path): makedirs(video_crop_base_path)

    #print("crop video: {}".format(video))
    frames = glob.glob(join(subdir, 'videos', video, '*.jpg'))
    def index_keys(text):
        return int(text.split('/')[-1].split('.')[0])
    frames = sorted(frames, key=index_keys)

    with open(join(subdir, 'anno', video + '.txt')) as f:
        ann = f.readlines()

    for id, frame in enumerate(frames):

        im = cv2.imread(frame)
        avg_chans = np.mean(im, axis=(0, 1))
        bbox_str = [float(s) for s in  ann[id].split(',')]

        trackid = 0
        bbox = [bbox_str[0], bbox_str[1], bbox_str[0]+bbox_str[2], bbox_str[1]+bbox_str[3]]

        filename = frame.split('/')[-1].split('.')[0]
        z, x = utils.crop_image(im, bbox, padding=avg_chans)
        cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(int(filename), trackid)), x)


def image_curation(num_threads=24):

    subset_dirs = sorted(glob.glob(join(base_path, TRAIN_SUBSET_PREFIX + '*') + "[!zip]"))
    #subset_dirs.append(join(base_path, TEST_SUBSET_PREFIX))

    for subdir  in subset_dirs:

        videos = sorted(os.listdir((join(subdir, 'videos'))))

        if debug:
          videos = videos[:1]
        n_videos = len(videos)

        subset = subdir.split('/')[-1]

        #for video in videos:
        #    crop_video(subdir, subset, video)

        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_video, subdir, subset, video) for video in videos]
            for i, f in enumerate(futures.as_completed(fs)):
                utils.print_progress(i, n_videos, prefix=subset, suffix='Done ', barLength=40)

def save_config():
    snippets = dict()
    instance_size = utils.get_instance_size()

    subset_dirs = sorted(glob.glob(join(base_path, TRAIN_SUBSET_PREFIX + '*') + "[!zip]"))

    for subdir in subset_dirs:
        subset = subdir.split('/')[-1]

        videos = sorted(os.listdir((join(subdir, 'videos'))))
        for vi, video in enumerate(videos):

            print('subset: {} video id: {:04d} / {:04d}'.format(subdir, vi, len(videos)))

            frames = glob.glob(join(subdir, 'videos', video, '*.jpg'))
            def index_keys(text):
                return int(text.split('/')[-1].split('.')[0])
            frames = sorted(frames, key=index_keys)

            with open(join(subdir, 'anno', video + '.txt')) as f:
                ann = f.readlines()

            frame_sz = None
            snippets[join(subset, video)] = dict()
            snippets[join(subset, video)]['tracks'] = dict()
            snippet = dict()
            for id, frame in enumerate(frames):
                if id == 0:
                    im = cv2.imread(frame)
                    frame_sz = [im.shape[1], im.shape[0]]
                    snippets[join(subset, video)]['frame_size'] = frame_sz

                bbox_str = [float(s) for s in  ann[id].split(',')]
                bbox = [bbox_str[0], bbox_str[1], bbox_str[0]+bbox_str[2], bbox_str[1]+bbox_str[3]]

                if bbox[0] < 0:
                    bbox[0] = 0
                if bbox[1] < 0:
                    bbox[1] = 0

                filename = frame.split('/')[-1].split('.')[0]
                snippet['{:06d}'.format(int(filename))] = bbox
            snippets[join(subset, video)]['tracks']['00'] = snippet

            if debug:
              break

    train = snippets
    val = {k:v for i, (k,v) in enumerate(snippets.items()) if i < 100}

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
