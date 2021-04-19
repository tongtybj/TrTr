import datetime
from jsonargparse import ArgumentParser, ActionParser, ActionConfigFile
import json
import random
import time
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from datasets.dataset import build as build_dataset
from datasets.dataset import get_args_parser as dataset_args_parser
from engine import evaluate, train_one_epoch
from models import build_model
from benchmark import eval as benchmark_eval
from benchmark import test as benchmark_test
from models.trtr import get_args_parser as trtr_args_parser

# for test
from models.tracker import build_tracker
from models.tracker import get_args_parser as tracker_args_parser


def get_args_parser():
    parser = ArgumentParser('training')

    # training
    parser.add_argument('--device', default='cuda',
                        help='device to use for inference')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate for network excluding backbone')
    parser.add_argument('--lr_backbone', default=1e-5, type=float,
                        help='learning rate for backbone, 0 to freeze backbone')

    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=6, type=int)
    parser.add_argument('--lr_gamma', default=0.5, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')

    # dataset
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--model_save_step', default=50, type=int,
                        help='step to save model')
    parser.add_argument('--benchmark_test_step', default=1, type=int,
                        help='step to test benchmark')
    parser.add_argument('--benchmark_start_epoch', default=0, type=int,
                        help='epoch to start benchmark')

    # Dataset
    parser.add_argument('--dataset', action=ActionParser(parser=dataset_args_parser()))

    # TrTr
    parser.add_argument('--model', action=ActionParser(parser=trtr_args_parser()))

    # yaml config file for all parameters
    parser.add_argument('--cfg_file', action=ActionConfigFile)

    return parser


def main(args):
    utils.init_distributed_mode(args)

    print("args: {}".format(args))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # special process to control whether freeze backbone
    args.model.train_backbone = args.lr_backbone > 0

    model, criterion, postprocessors = build_model(args.model)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma = args.lr_gamma)

    dataset_train = build_dataset(image_set='train', args=args.dataset, model_stride = model_without_ddp.backbone.stride)
    dataset_val = build_dataset(image_set='val', args=args.dataset, model_stride = model_without_ddp.backbone.stride)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1


    benchmark_test_parser = benchmark_test.get_args_parser()
    benchmark_test_args = benchmark_test_parser.get_defaults()
    benchmark_test_args.tracker.model = args.model # overwrite the parameters about network model
    benchmark_test_args.result_path = Path(os.path.join(args.output_dir, 'benchmark'))
    benchmark_test_args.dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'benchmark')

    benchmark_eval_parser = benchmark_eval.get_args_parser()
    benchmark_eval_args = benchmark_eval_parser.get_defaults()
    benchmark_eval_args.tracker_path = benchmark_test_args.result_path
    best_eao = 0
    best_ar = [0, 10] # accuracy & robustness


    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        # training
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every args.model_save_step epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.model_save_step == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

            # hack: only inference model
            utils.save_on_master({'model': model_without_ddp.state_dict()}, output_dir / 'checkpoint_only_inference.pth')

        # evalute
        val_stats = evaluate(model, criterion, postprocessors, data_loader_val, device, args.output_dir)

        log_stats = {'epoch': epoch,
                     **{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # evualute with benchmark
        if utils.is_main_process():
            if (epoch + 1) % args.benchmark_test_step == 0 and epoch > args.benchmark_start_epoch:

                tracker = build_tracker(benchmark_test_args.tracker, model = model_without_ddp, postprocessors = postprocessors)
                benchmark_test_args.model_name = "epoch" + str(epoch)
                benchmark_start_time = time.time()
                benchmark_test.main(benchmark_test_args, tracker)
                benchmark_time = time.time() - benchmark_start_time

                benchmark_eval_args.model_name = "epoch" + str(epoch)
                benchmark_eval_args.tracker_prefix = "epoch" + str(epoch)
                eval_results = benchmark_eval.main(benchmark_eval_args)
                eval_result = list(eval_results.values())[0]

                if benchmark_test_args.dataset in ['VOT2018', 'VOT2019']:
                    if args.output_dir:
                        with (output_dir / str("benchmark_" +  benchmark_test_args.dataset + ".txt")).open("a") as f:
                            f.write("epoch: " + str(epoch) + ", best EAO: " + str(best_eao) + ", " + json.dumps(eval_result) +  "\n")

                    if best_eao < eval_result['EAO']:

                        best_eao = eval_result['EAO']

                        if args.output_dir:
                            best_eao_int = int(best_eao*1000)

                            # record: only inference model
                            utils.save_on_master({'model': model_without_ddp.state_dict()}, output_dir / f'checkpoint{epoch:04}_best_eao_{best_eao_int:03}_only_inference.pth')

                    if best_ar[0] < eval_result['accuracy'] and best_ar[1] > eval_result['robustness']:

                        best_ar[0] = eval_result['accuracy']
                        best_ar[1] = eval_result['robustness']

                        if args.output_dir:
                            best_accuracy_int = int(best_ar[0]*1000)
                            best_robustness_int = int(best_ar[1]*1000)

                            # record: only inference model
                            utils.save_on_master({'model': model_without_ddp.state_dict()}, output_dir / f'checkpoint{epoch:04}_best_ar_{best_accuracy_int:03}_{best_robustness_int:03}_only_inference.pth')

                print("benchmark time: {}".format(benchmark_time))

        if args.distributed:
            torch.distributed.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
