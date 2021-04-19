"""
Backbone modules.
"""
from collections import OrderedDict
from jsonargparse import ArgumentParser

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from util.misc import NestedTensor, is_main_process


def get_args_parser():
    parser = ArgumentParser(prog='backbone')

    parser.add_argument('--model', type=str, default='resnet50',
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', type=bool, default=True,
                        help="whether replace stride with dilation in ResNet blocks.")
    parser.add_argument('--return_layers', default=[], nargs='+',
                        help="layers to return from the backbone. Can be multiple layers to do layer-wise aggregation")

    return parser

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone_name: str, backbone: nn.Module, train_backbone: bool, return_layers: List):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        self.return_layers = return_layers
        return_layer_map = {}
        for idx, layer in enumerate(range(2, int(return_layers[-1][-1]) + 1)):
            return_layer_map['layer' + str(layer)] = str(layer)
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layer_map)
        self.name = backbone_name

        self.num_channels_list = []
        for layer in return_layers:
            if layer == "layer2":
                if backbone_name == "resnet18":
                    self.num_channels_list.append(128)
                if backbone_name == "resnet50":
                    self.num_channels_list.append(512)
            elif layer == "layer3":
                if backbone_name == "resnet18":
                    self.num_channels_list.append(256)
                if backbone_name == "resnet50":
                    self.num_channels_list.append(1024)
            elif layer == "layer4":
                if backbone_name == "resnet18":
                    self.num_channels_list.append(512)
                if backbone_name == "resnet50":
                    self.num_channels_list.append(2048)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():

            if 'layer' + name not in self.return_layers:
                continue

            #print(name, ", ", x.shape)
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]

            # TODO: workaround to avoid NaN of attention calculation because of a full "True" mask
            invalid_indices = (torch.logical_not(mask).sum(dim=[1,2]) == 0).nonzero().squeeze(-1)
            if(len(invalid_indices)):
                #print("workaround to avoid NaN for {}".format(invalid_indices))
                mask[invalid_indices] = torch.zeros(x.shape[-2:], dtype=torch.bool, device=mask.device)

            out[name] = NestedTensor(x, mask)
        return out, xs


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_layers: List,
                 dilation: bool):
        if dilation:
            dilation = [False, True, True] # workaround to achieve stride of 8
        else:
            dilation = [False, False, False]

        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=dilation,
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        super().__init__(name, backbone, train_backbone, return_layers)

        final_layer = int(return_layers[-1][-1])
        self.dilation = dilation
        self.stride = 4
        for layer in range(final_layer - 1):
            if not dilation[layer]:
                self.stride = self.stride * 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

        self.num_channels_list = []
        self.stride = backbone.stride
        self.dilation = backbone.dilation

    def forward(self, tensor_list: NestedTensor, multi_frame = False):

        xs, extra_out = self[0](tensor_list) # extract feature from search image (embedding)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x, multi_frame).to(x.tensors.dtype))

            # print("backbone {}: shape: {}".format(name, x.tensors.shape))

        return out, pos, extra_out

def build_backbone(args, position_embedding, train = False):

    if len(args.return_layers) == 0:
        if 'resnet' in args.model:
            args.return_layers = ['layer3']

    backbone = Backbone(args.model, train, args.return_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels_list = backbone.num_channels_list
    return model
