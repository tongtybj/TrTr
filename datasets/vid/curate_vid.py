from os.path import join, isdir
from os import listdir, mkdir, makedirs
import sys
import cv2
import numpy as np
import pickle
import json
import glob
from concurrent import futures
import xml.etree.ElementTree as ET

sys.path.append('..')
import utils

debug = False

VID_base_path = "./ILSVRC2015"
save_base_path = join(VID_base_path, 'Curation')
ann_base_path = join(VID_base_path, 'Annotations/VID')
img_base_path = join(VID_base_path, 'Data/VID')

subdir_map = {'train/ILSVRC2015_VID_train_0000': 'a',
              'train/ILSVRC2015_VID_train_0001': 'b',
              'train/ILSVRC2015_VID_train_0002': 'c',
              'train/ILSVRC2015_VID_train_0003': 'd',
              'val': 'e'}


def crop_video(subdir, sub_set, video):
    video_crop_base_path = join(save_base_path, sub_set, video)
    if not isdir(video_crop_base_path): makedirs(video_crop_base_path)

    xmls = sorted(glob.glob(join(ann_base_path, subdir, video, '*.xml')))

    for xml in xmls:

        xmltree = ET.parse(xml)
        objects = xmltree.findall('object')
        objs = []
        filename = xmltree.findall('filename')[0].text

        im = cv2.imread(xml.replace('xml', 'JPEG').replace('Annotations', 'Data'))
        avg_chans = np.mean(im, axis=(0, 1))
        for object_iter in objects:

            trackid = int(object_iter.find('trackid').text)
            bndbox = object_iter.find('bndbox')

            bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]

            z, x = utils.crop_image(im, bbox, padding=avg_chans)
            # cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.z.jpg'.format(int(filename), trackid)), z)
            cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(int(filename), trackid)), x)


def image_curation(num_threads=24):

    for subdir, sub_set  in subdir_map.items():

        sub_set_base_path = join(ann_base_path, subdir)

        videos = sorted(listdir(sub_set_base_path))

        if debug:
          videos = videos[:1]
        n_videos = len(videos)

        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_video, subdir, sub_set, video) for video in videos]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                utils.print_progress(i, n_videos, prefix=sub_set, suffix='Done ', barLength=40)



def save_config():
    snippets = dict()

    instance_size = utils.get_instance_size()

    for subdir, sub_set  in subdir_map.items():
        subdir_base_path = join(ann_base_path, subdir)
        videos = sorted(listdir(subdir_base_path))
        for vi, video in enumerate(videos):
            print('subset: {} video id: {:04d} / {:04d}'.format(subdir, vi, len(videos)))

            frames = []
            xmls = sorted(glob.glob(join(subdir_base_path, video, '*.xml')))

            id_set = []
            id_frames = [[]] * 60  # at most 60 objects

            for num, xml in enumerate(xmls):
                f = dict()
                xmltree = ET.parse(xml)
                size = xmltree.findall('size')[0]
                frame_sz = [int(it.text) for it in size]
                objects = xmltree.findall('object')
                objs = []
                for object_iter in objects:
                    trackid = int(object_iter.find('trackid').text)

                    name = (object_iter.find('name')).text
                    bndbox = object_iter.find('bndbox')
                    occluded = int(object_iter.find('occluded').text)
                    o = dict()
                    o['c'] = name
                    o['bbox'] = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                                 int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]

                    s_z, scale_z = utils.siamfc_like_scale(o['bbox'])
                    s_x = instance_size / scale_z
                    #print("{}, {}, {}, s_x: {}".format(video, xml, trackid, s_x))
                    bb_center = [(o['bbox'][2]+o['bbox'][0])/2., (o['bbox'][3]+o['bbox'][1])/2.]

                    mask = [0, 0, instance_size-1, instance_size-1]
                    if bb_center[0] < s_x/2:
                      mask[0] = (s_x/2 - bb_center[0]) * scale_z
                    if bb_center[1] < s_x/2:
                      mask[1] = (s_x/2 - bb_center[1]) * scale_z
                    if bb_center[0] + s_x/2 > frame_sz[0]:
                      mask[2] = mask[2] - (bb_center[0] + s_x/2 - frame_sz[0]) * scale_z
                    if bb_center[1] + s_x/2 > frame_sz[1]:
                      mask[3] = mask[3] - (bb_center[1] + s_x/2 - frame_sz[1]) * scale_z

                    o['mask'] = mask

                    o['trackid'] = trackid
                    o['occ'] = occluded
                    objs.append(o)

                    if trackid not in id_set:
                        id_set.append(trackid)
                        id_frames[trackid] = []
                    id_frames[trackid].append(num)

                f['frame_sz'] = frame_sz
                f['img_path'] = xml.split('/')[-1].replace('xml', 'JPEG')
                f['objs'] = objs
                frames.append(f)

            if len(id_set) > 0:
                snippets[join(sub_set, video)] = dict()
                snippets[join(sub_set, video)]['frame_size'] = frames[0]['frame_sz']
                snippets[join(sub_set, video)]['tracks'] = dict()

            for selected in id_set:
                frame_ids = sorted(id_frames[selected])
                sequences = np.split(frame_ids, np.array(np.where(np.diff(frame_ids) > 1)[0]) + 1)
                sequences = [s for s in sequences if len(s) > 1]  # remove isolated frame.
                for seq in sequences:
                    snippet = dict()
                    for frame_id in seq:
                        frame = frames[frame_id]
                        for obj in frame['objs']:
                            if obj['trackid'] == selected:
                                o = obj
                                continue
                        snippet[frame['img_path'].split('.')[0]] = o['bbox']
                    snippets[join(sub_set, video)]['tracks']['{:02d}'.format(selected)] = snippet

            if debug:
              break

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
