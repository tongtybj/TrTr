from os.path import join, isdir
from os import mkdir, makedirs
import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET
from concurrent import futures
import time
import sys
from os.path import join
import json
import pickle


sys.path.append('..')
import utils

debug = False

def crop_img(xml, sub_set_crop_path):
    xmltree = ET.parse(xml)
    objects = xmltree.findall('object')

    frame_crop_base_path = join(sub_set_crop_path, xml.split('/')[-1].split('.')[0])
    if not isdir(frame_crop_base_path): makedirs(frame_crop_base_path)

    img_path = xml.replace('xml', 'JPEG').replace('Annotations', 'Data')

    im = cv2.imread(img_path)
    avg_chans = np.mean(im, axis=(0, 1))

    for id, object_iter in enumerate(objects):
        bndbox = object_iter.find('bndbox')
        bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]

        z, x = utils.crop_image(im, bbox, padding=avg_chans)
        cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(0, id)), x)


def image_curation(num_threads=24):
    VID_base_path = './dataset'
    crop_path = './{}/Curation'.format(VID_base_path)
    if not isdir(crop_path): mkdir(crop_path)
    ann_base_path = join(VID_base_path, 'Annotations/DET/')
    subdir_map = {'train/ILSVRC2013_train': 'a',
                  'train/ILSVRC2014_train_0000': 'b',
                  'train/ILSVRC2014_train_0001': 'c',
                  'train/ILSVRC2014_train_0002': 'd',
                  'train/ILSVRC2014_train_0003': 'e',
                  'train/ILSVRC2014_train_0004': 'f',
                  'train/ILSVRC2014_train_0005': 'g',
                  'train/ILSVRC2014_train_0006': 'h',
                  'val': 'i'}

    dataset = dict()

    for subdir, sub_set  in subdir_map.items():

        sub_set_base_path = join(ann_base_path, subdir)

        if 'a' == sub_set:
            xmls = sorted(glob.glob(join(sub_set_base_path, '*', '*.xml')))
        else:
            xmls = sorted(glob.glob(join(sub_set_base_path, '*.xml')))

        if debug: # only test few imgs (e.g. 2) for each subset
            xmls = xmls[:2]

        n_imgs = len(xmls)
        sub_set_crop_path = join(crop_path, sub_set)

        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_img, xml, sub_set_crop_path) for xml in xmls]
            for i, f in enumerate(futures.as_completed(fs)):
                utils.print_progress(i, n_imgs, prefix=sub_set, suffix='Done ', barLength=80)

        # curate config
        for f, xml in enumerate(xmls):
            print('subset: {} frame id: {:08d} / {:08d}'.format(sub_set, f, n_imgs))
            xmltree = ET.parse(xml)
            objects = xmltree.findall('object')
            image_size = xmltree.find('size')
            image_width = int(image_size.find('width').text)
            image_height = int(image_size.find('height').text)

            video = join(sub_set, xml.split('/')[-1].split('.')[0])

            for id, object_iter in enumerate(objects):
                bndbox = object_iter.find('bndbox')
                bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                        int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
                frame = '%06d' % (0)
                obj = '%02d' % (id)
                if video not in dataset:
                    dataset[video] = dict()
                    dataset[video]['frame_size'] = [image_width, image_height]
                    dataset[video]['tracks'] = dict()

                if obj not in dataset[video]:
                    dataset[video]['tracks'][obj] = {}
                dataset[video]['tracks'][obj][frame] = bbox

    train = {k:v for (k,v) in dataset.items() if 'i/' not in k}
    val = {k:v for (k,v) in dataset.items() if 'i/' in k}

    json.dump(train, open('{}/train.json'.format(crop_path), 'w'), indent=4, sort_keys=True)
    json.dump(val, open('{}/val.json'.format(crop_path), 'w'), indent=4, sort_keys=True)
    with open('{}/train.pickle'.format(crop_path), 'wb') as f:
        pickle.dump(train, f)
    with open('{}/val.pickle'.format(crop_path), 'wb') as f:
        pickle.dump(val, f)


if __name__ == '__main__':
    since = time.time()

    if len(sys.argv) == 2:
        image_curation(int(sys.argv[1]))
    else:
        image_curation()

    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
