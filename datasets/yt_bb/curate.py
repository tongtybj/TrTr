import os
from os.path import join, isdir
from concurrent import futures
import argparse
import json
import pickle
import sys
import cv2
import numpy as np
from download import parse_annotations

sys.path.append('../')
import utils
from yt_utils import parse_annotations, d_sets

# The data sets to be downloaded
YY_TB_base_path = "./youtube_bb"
save_base_path = join(YY_TB_base_path, 'Curation')

debug = False

# Download and cut a clip to size
def crop_video(vid):

    d_set_dir = vid.clips[0].d_set_dir

    for clip in vid.clips:
        name = clip.name
        yt_id = clip.yt_id
        class_id = clip.class_id
        obj_id = clip.obj_id
        d_set_dir = clip.d_set_dir

        for index in range(len(clip.timestamps)):
            presence = clip.presences[index]
            timestamp = clip.timestamps[index]
            bbox = clip.bboxes[index]

            class_dir = join(d_set_dir, str(clip.class_id))
            frame_path = class_dir+ '/'  + yt_id +  '_' + str(timestamp) + \
                         '_' + str(class_id) + '_' + str(obj_id) + '.jpg'
            # Verify that the video has been downloaded. Skip otherwise
            #print(frame_path)
            if not os.path.exists(frame_path):
                break

            if presence == 'absent':
                continue
            
            image = cv2.imread(frame_path)

            avg_chans = np.mean(image, axis=(0, 1))

            # Uncomment lines below to print bounding boxes on downloaded images
            h, w = image.shape[:2]
            x1 = bbox[0] * w
            x2 = bbox[1] * w
            y1 = bbox[2] * h
            y2 = bbox[3] * h

            if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0 or y2 < y1 or x2 < x1:
                continue

            
            # Make the class directory if it doesn't exist yet
            # video_crop_base_path = join(save_base_path, d_set_dir.split("/")[-1], str(class_id), yt_id) # with dataset type
            video_crop_base_path = join(save_base_path, str(class_id), yt_id)

            if not isdir(video_crop_base_path): os.makedirs(video_crop_base_path)

            # Save the extracted image
            bbox = np.round(np.array([x1, y1, x2, y2])).astype(np.uint16)

            if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
                continue

            z, x = utils.crop_image(image, bbox, padding=avg_chans)
            #cv2.imwrite(join(crop_class_dir, '{:06d}.{:02d}.z.jpg'.format(int(timestamp)/1000, int(obj_id))), z)
            cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(int(timestamp/1000), int(obj_id))), x)


# Parse the annotation csv file and schedule downloads and cuts
def image_curation(num_threads, vid_start, data_type):
    """Crop the entire youtube-bb data set into `crop_path`.
    """
    # For each of the two datasets
    d_set = d_sets[data_type]

    js = {}

    annotations,clips,vids = parse_annotations(d_set, YY_TB_base_path)

    # Crop in parallel threads giving
    print("Cropping images on {} videos for {}".format(len(vids) - vid_start, d_set))

    if debug: # only test with one video:
        vids = vids[:10]

    print(d_set + ': start vid from ' + str(vid_start))
    total_len = len(vids)
    segment_vids = vids[vid_start:]

    # debug
    #for vid in segment_vids:
    #    crop_video(vid)
    #raise
    
    
    with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        fs = [executor.submit(crop_video, vid) for vid in segment_vids]
        for i, f in enumerate(futures.as_completed(fs)):
            sys.stderr.write("Curate video: {} / {} \r".format(i + vid_start, total_len))

    
    for i, vid in enumerate(vids):

        sys.stderr.write("dumping video {} / {} in {} \r".format(i, total_len, d_set))
        image_w = 0
        image_h = 0

        for clip in vid.clips:

            name = clip.name
            yt_id = clip.yt_id
            class_id = clip.class_id
            obj_id = clip.obj_id
            d_set_dir = clip.d_set_dir

            for index in range(len(clip.timestamps)):

                presence = clip.presences[index]
                timestamp = clip.timestamps[index]
                x1, x2, y1, y2 = clip.bboxes[index]

                if presence == 'absent':
                    continue

                if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0 or y2 < y1 or x2 < x1:
                    continue

                frame = '%06d' % (int(timestamp) / 1000)
                obj = '%02d' % (int(obj_id))
                # video = join(d_set, str(class_id) , yt_id) # with dataset type
                video = join(str(class_id) , yt_id)
                video_crop_base_path = join(save_base_path, video)

                video_file = join(video_crop_base_path, '{}.{}.x.jpg'.format(frame, obj))
                # print(video_file)
                if not os.path.exists(video_file):
                    continue

                if image_w == 0 or image_h == 0:
                    # TODO: any faster way to know the image shape
                    class_dir = join(d_set_dir, str(clip.class_id))
                    frame_path = class_dir+ '/'  + yt_id +  '_' + str(timestamp) + \
                                 '_' + str(class_id) + '_' + str(obj_id) + '.jpg'
                    assert os.path.exists(frame_path)
                    image = cv2.imread(frame_path)
                    image_h, image_w = image.shape[:2]

                x1 = x1 * image_w
                x2 = x2 * image_w
                y1 = y1 * image_h
                y2 = y2 * image_h

                if video not in js:
                    js[video] = {}
                    js[video]["frame_size"] = [image_w, image_h]
                    js[video]["tracks"] = {}
                if obj not in js[video]["tracks"]:
                    js[video]["tracks"][obj] = {}
                js[video]["tracks"][obj][frame] = [x1, y1, x2, y2]


    if 'yt_bb_detection_train' == d_set:
        json.dump(js, open(join(save_base_path, 'train.json'), 'w'), indent=4, sort_keys=True)
        pickle.dump(js, open(join(save_base_path, 'train.pickle'), 'wb'))
    else:
        json.dump(js, open(join(save_base_path, 'val.json'), 'w'), indent=4, sort_keys=True)
        pickle.dump(js, open(join(save_base_path, 'val.pickle'), 'wb'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Parse args download youtube bb', add_help=False)
    parser.add_argument('--data_type', default=0, type=int) # 0: train, 1: validation
    parser.add_argument('--num_threads', default=1, type=int)
    parser.add_argument('--vid_start', default=0, type=int)
  
    args = parser.parse_args()

    image_curation(args.num_threads, args.vid_start, args.data_type)

