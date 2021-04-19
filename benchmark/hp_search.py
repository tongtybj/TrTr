from jsonargparse import ArgumentParser, ActionParser
import os
import time
import datetime
import test as benchmark
import itertools
from models.tracker import build_tracker as build_baseline_tracker
from models.hybrid_tracker import build_tracker as build_online_tracker

def get_args_parser(hpnames):

    parser = benchmark.get_args_parser()

    for name in hpnames:
        parser.add_argument('--' + name, default=[], nargs='+')

    parser.add_argument('--separate_mode', action='store_true',
                        help="best score mode if separate_mode is false")

    parser.add_argument('--save_path', default='hp_search', type=str)

    return parser

def main(args, hpnames):

    # separate mode or best score mode
    repetition = 1
    if args.separate_mode:
        repetition = args.repetition
        args.repetition  = 1 # separate model (non best score model)

    # TODO: record model configuration in checkpoint
    if len(args.tracker.model.backbone.return_layers) == 0:
        args.tracker.model.backbone.return_layers = ['layer3']

    if len(args.tracker.dcf.layers) == 0:
        args.tracker.dcf.layers = ['layer2', 'layer3']

    # create dir for result
    layers_info = 'trtr_layer'
    for layer in args.tracker.model.backbone.return_layers:
        layers_info += '_' + layer[-1]
    if args.use_baseline_tracker:
        layers_info += '_baseline'
    else:
        layers_info += '_dcf_layer'
        for layer in args.tracker.dcf.layers:
            layers_info += '_' + layer[-1]


    if args.separate_mode:
        args.save_path += '_separate'
    args.result_path = os.path.join(args.save_path,
                                    os.path.splitext(os.path.basename(args.tracker.checkpoint))[0],
                                    layers_info)
    dataset_path = os.path.join(args.result_path, args.dataset)
    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)

    hparams = dict()
    for name in hpnames:

        val = args
        for idx,  n in enumerate(name.split('.')):

            if idx == len(name.split('.')) - 1:
                default_value = getattr(val, n[:-1])

            val = getattr(val, n)


        if len(val) == 0:
            hparams[name] = [default_value]
        else:
            hparams[name] = [type(default_value)(v) for v in val]
    hparams['runs'] = list(range(1, repetition + 1))

    tracker_num = len(list(itertools.product(*hparams.values())))

    for tracker_id, hparam_set in enumerate(itertools.product(*hparams.values())):
        t = time.time()
        print("start {}/{} tracker test".format(tracker_id + 1, tracker_num))
        model_name = ''
        if args.use_baseline_tracker:
            model_name = 'baseline_'
        for idx, (name, val) in enumerate(zip(hparams.keys(), hparam_set)):

            args_temp = args
            for str_id,  n in enumerate(name[:-1].split('.')):
                if str_id == len(name.split('.')) - 1:
                    setattr(args_temp, n, val)

                args_temp = getattr(args_temp, n)


            if args.use_baseline_tracker and 'dcf' in name:
                continue

            model_name += name[:-1].replace('tracker.', '').replace('postprocess.', '').replace('.', '_') + "_" + str(val).replace('.', 'p')
            if idx < len(hparam_set) - 1:
                model_name += '_'

        if not args.use_baseline_tracker:
            model_name += '_false_positive' # workaround to distinguish with old model name

        if args.tracker.model.transformer_mask:
            model_name += '_with_transformer_mask'

        #print(model_name)
        model_dir = os.path.join(dataset_path, model_name)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        args.model_name = model_name
        with open(os.path.join(model_dir, 'log.txt'), 'a') as f:
            f.write('parameters: \n')
            f.write('{}'.format(vars(args)) + '\n\n')

        # create tracker
        if args.use_baseline_tracker:
            tracker = build_baseline_tracker(args.tracker)
        else:
            tracker = build_online_tracker(args.tracker)

        # start test with benchmark
        benchmark.main(args, tracker)

        du = round(time.time() - t)
        print("finish {}/{} tracker test, take {}, rest {} ".format(tracker_id + 1, tracker_num, datetime.timedelta(seconds = du), datetime.timedelta(seconds = du * (tracker_num - tracker_id - 1))))

if __name__ == '__main__':

    hpnames = ['tracker.search_sizes', 'tracker.postprocess.window_factors', 'tracker.postprocess.tracking_size_lpfs', 'tracker.dcf.sizes', 'tracker.dcf.rates', 'tracker.dcf.sample_memory_sizes'] # please add hyperparameter here

    parser = get_args_parser(hpnames)
    args = parser.parse_args()

    #print(args)

    main(args, hpnames)

