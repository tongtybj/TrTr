"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for template_samples, search_samples, template_masks, search_masks, targets in metric_logger.log_every(data_loader, print_freq, header):

        #print("search mask: {}".format(search_masks))

        template_nested_samples = utils.nested_tensor_from_tensor_list(template_samples, template_masks).to(device)
        search_nested_samples = utils.nested_tensor_from_tensor_list(search_samples, search_masks).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(search_nested_samples, template_nested_samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged training stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    for template_samples, search_samples, template_masks, search_masks, targets in metric_logger.log_every(data_loader, 10, header):

        template_nested_samples = utils.nested_tensor_from_tensor_list(template_samples, template_masks).to(device)
        search_nested_samples = utils.nested_tensor_from_tensor_list(search_samples, search_masks).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(search_nested_samples, template_nested_samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged validation stats:", metric_logger)

    # do post processing if necessary for further testing
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats
