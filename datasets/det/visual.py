from os.path import join
from os import listdir
import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET

visual = True
color_bar = np.random.randint(0, 255, (90, 3))

VID_base_path = './dataset'
ann_base_path = join(VID_base_path, 'Annotations/DET/train/')
img_base_path = join(VID_base_path, 'Data/DET/train/')
subdirs = ['ILSVRC2013_train',
           'ILSVRC2014_train_0000',
           'ILSVRC2014_train_0001',
           'ILSVRC2014_train_0002',
           'ILSVRC2014_train_0003',
           'ILSVRC2014_train_0004',
           'ILSVRC2014_train_0005',
           'ILSVRC2014_train_0006']

for subdir in subdirs:
    subdir_base_path = join(ann_base_path, subdir)
    class_names = sorted(listdir(subdir_base_path))
    for vi, class_name in enumerate(class_names):
        print('subset: {} video id: {:04d} / {:04d}'.format(subdir, vi, len(class_names)))

        class_base_path = join(subdir_base_path, class_name)
        xmls = sorted(glob.glob(join(class_base_path, '*.xml')))
        for xml in xmls:
            f = dict()
            xmltree = ET.parse(xml)
            size = xmltree.findall('size')[0]
            frame_sz = [int(it.text) for it in size]
            objects = xmltree.findall('object')
            # if visual:
            img_path = xml.replace('xml', 'JPEG').replace('Annotations', 'Data')
            im = cv2.imread(img_path)
            for object_iter in objects:
                bndbox = object_iter.find('bndbox')
                bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                        int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
                if visual:
                    pt1 = (int(bbox[0]), int(bbox[1]))
                    pt2 = (int(bbox[2]), int(bbox[3]))
                    cv2.rectangle(im, pt1, pt2, color_bar[vi].tolist(), 3)
            if visual:
                cv2.imshow('img', im)
                k = cv2.waitKey(200)
                if k == 27:         # wait for ESC key to exit
                    sys.exit()

