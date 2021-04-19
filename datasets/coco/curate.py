from pycocotools.coco import COCO
import cv2
import numpy as np
from os.path import join, isdir
from os import mkdir, makedirs
from concurrent import futures
import sys
import time
from os.path import join
import json
import pickle

sys.path.append('..')
import utils

debug = False

def crop_img(img, anns, set_crop_base_path, set_img_base_path):
    frame_crop_base_path = join(set_crop_base_path, img['file_name'].split('/')[-1].split('.')[0])
    if not isdir(frame_crop_base_path): makedirs(frame_crop_base_path)

    im = cv2.imread('{}/{}'.format(set_img_base_path, img['file_name']))
    avg_chans = np.mean(im, axis=(0, 1))
    for trackid, ann in enumerate(anns):
        rect = ann['bbox']
        bbox = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]
        if rect[2] <= 0 or rect[3] <=0:
            continue
        z, x = utils.crop_image(im, bbox, padding=avg_chans)
        #cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.z.jpg'.format(0, trackid)), z)
        cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(0, trackid)), x)


def image_curation(num_threads=12, year = 2017):
    data_dir = './dataset'
    crop_path = './{}/Curation'.format(data_dir)
    if not isdir(crop_path): mkdir(crop_path)

    for data_type in ['val', 'train']:
        data_type_year = data_type + str(year)
        set_crop_base_path = join(crop_path, data_type)
        set_img_base_path = join(data_dir, data_type_year)

        annFile = '{}/annotations/instances_{}.json'.format(data_dir,data_type_year)
        coco = COCO(annFile)

        if debug: # only test few images (e.g., 10):
            imgs = {}
            for i, (k,v) in enumerate(coco.imgs.items()):
                if i > 10:
                    coco.imgs = imgs
                    break

                imgs[k] = v


        n_imgs = len(coco.imgs)

        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_img, coco.loadImgs(id)[0],
                                  coco.loadAnns(coco.getAnnIds(imgIds=id, iscrowd=None)),
                                  set_crop_base_path, set_img_base_path) for id in coco.imgs]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                utils.print_progress(i, n_imgs, prefix=data_type_year, suffix='Done ', barLength=40)

        dataset = dict()
        for n, img_id in enumerate(coco.imgs):
            print('subset: {} image id: {:04d} / {:04d}'.format(data_type, n, n_imgs))
            img = coco.loadImgs(img_id)[0]
            annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = coco.loadAnns(annIds)
            img_crop_base_path = join(data_type, img['file_name'].split('/')[-1].split('.')[0])

            if len(anns) > 0:
                dataset[img_crop_base_path] = dict()
                dataset[img_crop_base_path]['frame_size'] = [img['width'], img['height']]
                dataset[img_crop_base_path]['tracks'] = dict()

            for trackid, ann in enumerate(anns):
                rect = ann['bbox']
                c = ann['category_id']
                bbox = [rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]
                if rect[2] <= 0 or rect[3] <= 0:  # lead nan error in cls.
                    continue
                dataset[img_crop_base_path]['tracks']['{:02d}'.format(trackid)] = {'000000': bbox}

        print('save json (dataset), please wait 20 seconds~')
        json.dump(dataset, open('{}/{}.json'.format(crop_path, data_type), 'w'), indent=4, sort_keys=True)

        with open('{}/{}.pickle'.format(crop_path, data_type), 'wb') as f:
            pickle.dump(dataset, f)


    print('done')


if __name__ == '__main__':
    since = time.time()
    if len(sys.argv) == 2:
        image_curation(int(sys.argv[1]))
    else:
        image_curation()

    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
