import torch
import os
import sys
import importlib

sys.path.append(os.path.join(os.path.dirname(__file__), '../../external_module/pytracking'))

class TrackerWrapper:

    def __init__(self, args):

        if args.external_tracker == "atom":
            param_name = 'default_vot'
        if args.external_tracker == "dimp":
            param_name = 'dimp50_vot18'
        if args.external_tracker == "prdimp":
            args.external_tracker = "dimp"
            param_name = 'prdimp50_vot18'

        param_module = importlib.import_module('pytracking.parameter.{}.{}'.format(args.external_tracker, param_name))
        params = param_module.parameters()

        tracker_module = importlib.import_module('pytracking.tracker.{}'.format(args.external_tracker))
        tracker_class = tracker_module.get_tracker_class()
        self.tracker = tracker_class(params)

    def init(self, img, bbox):
        info = {'init_bbox': [bbox[0], bbox[1], bbox[2], bbox[3]]}
        self.tracker.initialize(img, info)

    def track(self, img):
        out =  self.tracker.track(img)
        bbox_wh = out['target_bbox']
        bbox = [bbox_wh[0], bbox_wh[1], bbox_wh[0] + bbox_wh[2], bbox_wh[1] + bbox_wh[3]]

        # TODO, add more return values
        return {
            'bbox': bbox,
            'score': 1.0
        }

