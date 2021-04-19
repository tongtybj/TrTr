from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import zipfile
import urllib.request

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('year', type=str, default='2018', help='e.g., 2018, 2019, 2020')
    parser.add_argument('challenge', type=str, default='main', help='e.g., main, shortterm')
    args = parser.parse_args()

    if args.year == '2020' and args.challenge == 'main':
        args.challenge == 'shortterm'


    root_path = 'dataset'
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    vot_dir = os.path.join(root_path, args.year)
    if not os.path.exists(vot_dir):
        os.makedirs(vot_dir)

    base_url= "http://data.votchallenge.net/vot{}/{}".format(args.year, args.challenge)
    desc_file = os.path.join(vot_dir, 'description.json')
    with urllib.request.urlopen(os.path.join(base_url, os.path.basename(desc_file))) as web_file:
        data = web_file.read()
        with open(desc_file, mode='wb') as local_file:
            local_file.write(data)
            print("finish download {}".format(desc_file))

    with open(desc_file, 'r') as f:
        dataset = json.load(f)

    for idx, seq in enumerate(dataset['sequences']):

        anno_zip = os.path.join(vot_dir, seq['annotations']['url'])
        with urllib.request.urlopen(os.path.join(base_url, os.path.basename(anno_zip))) as web_file:
            data = web_file.read()
            with open(anno_zip, mode='wb') as local_file:
                local_file.write(data)

        print("{}/{} finish download {} ".format(idx+1, len(dataset['sequences']), anno_zip))

        color_zip = os.path.join(vot_dir, seq['name'] + '_images.zip')
        with urllib.request.urlopen(os.path.join(base_url, seq['channels']['color']['url'])) as web_file:
            data = web_file.read()
            with open(color_zip, mode='wb') as local_file:
                local_file.write(data)

        print("{}/{} finish download {}".format(idx+1, len(dataset['sequences']), color_zip))

        video_dir = os.path.join(vot_dir, seq['name'])
        with zipfile.ZipFile(anno_zip, 'r') as zf:
            zf.extractall(video_dir)

        with zipfile.ZipFile(color_zip, 'r') as zf:
            zf.extractall(os.path.join(video_dir, 'color'))

        os.remove(anno_zip)
        os.remove(color_zip)

if __name__ == '__main__':
    main()
