"""
TrTr model and criterion classes.
"""

import copy
from jsonargparse import ArgumentParser, ActionParser

import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       reg_l1_loss, neg_loss,
                       get_world_size, is_dist_avail_and_initialized)

from .position_encoding import build_position_encoding
from .backbone import build_backbone
from .backbone import get_args_parser as backbone_args_parser
from .transformer import build_transformer
from .transformer import get_args_parser as transformer_args_parser

import time
import numpy as np

class TRTR(nn.Module):
    """ This is the TRTR module that performs target tracking """
    def __init__(self, backbone, transformer, aux_loss=False, transformer_mask=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.transformer = nn.ModuleList([copy.deepcopy(transformer) for i in backbone.num_channels_list]) # multiple for layer-wise backbone output

        hidden_dim = transformer.d_model

        # heatmap and bbox
        # TODO: try to use MLP or Conv2d_3x3 before fully-connected like CenterNet
        backbone_layer_num = len(backbone.num_channels_list)
        self.reg_embed = MLP(backbone_layer_num * hidden_dim, backbone_layer_num * hidden_dim, 2, 3)
        self.wh_embed = MLP(backbone_layer_num * hidden_dim, backbone_layer_num * hidden_dim, 2, 3)
        self.heatmap_embed = nn.Linear(backbone_layer_num * hidden_dim, 1)
        self.heatmap_embed.bias.data.fill_(-2.19)

        self.input_projs = nn.ModuleList([nn.Conv2d(num_channels, hidden_dim, kernel_size=1) for num_channels in backbone.num_channels_list])

        self.backbone = backbone
        self.aux_loss = aux_loss

        self.template_src_projs = []
        self.template_mask = None
        self.template_pos = None
        self.memory = []

        self.transformer_mask = transformer_mask

    def forward(self, search_samples: NestedTensor, template_samples: NestedTensor = None):
        """ template_samples is a NestedTensor for template image:
               - samples.tensor: batched images, of shape [batch_size x 3 x H_template x W_template]
               - samples.mask: a binary mask of shape [batch_size x H_template x W_template], containing 1 on padded pixels
            search_samples is also a NestedTensor for searching image:
               - samples.tensor: batched images, of shape [batch_size x 3 x H_search x W_search]
               - samples.mask: a binary mask of shape [batch_size x H_search x W_search], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_heatmap": The heatmap of the target bbox, of shape= [batch_size x (H_search x W_search) x 1]
               - "pred_dense_reg": The regression of bbox for all query (i.e. all pixels), of shape= [batch_size x (H_search x W_search) x 2]
                                   The regression reg O = [ p/stride - p_tilde], where p and p_tilde are the corrdinates
                                   in input and output, respectively.
               - "pred_dense_wh": The size of bbox for all query (i.e. all pixels), of shape= [batch_size x (H_search x W_search) x 2]
                                  The height and width values are normalized in [0, 1],
                                  relative to the size of each individual image (disregarding possible padding).
                                  See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        if template_samples is not None:
            assert isinstance(template_samples, NestedTensor)
        assert isinstance(search_samples, NestedTensor)


        template_features = None
        template_pos = None
        if template_samples is not None:

            multi_frame = False
            if len(template_samples.tensors) > 1 and len(search_samples.tensors) == 1:
                # print("do multiple frame mode for backbone")
                multi_frame = True

            template_features, self.template_pos, _ = self.backbone(template_samples, multi_frame = multi_frame)

            self.template_mask = None
            if self.transformer_mask:
                self.template_mask = template_features[-1].mask


            self.template_src_projs = []
            for input_proj, template_feature in zip(self.input_projs, template_features):
                self.template_src_projs.append(input_proj(template_feature.tensors))

            self.memory = []
            # print("backbone template mask: {}".format(self.template_mask))

        start = time.time()
        search_features, search_pos, all_features  = self.backbone(search_samples)
        # print("search image feature extraction: {}".format(time.time() - start))
        search_mask = None
        if self.transformer_mask:
            search_mask = search_features[-1].mask

        #torch.set_printoptions(profile="full")
        #print("backbone search mask: {}".format(search_mask))

        search_src_projs = []
        for input_proj, search_feature in zip(self.input_projs, search_features):
            search_src_projs.append(input_proj(search_feature.tensors))


        hs_list = []
        for i, (template_src_proj, search_src_proj, transformer) in enumerate(zip(self.template_src_projs, search_src_projs, self.transformer)):
            if template_samples is not None:
                hs, memory = transformer(template_src_proj, self.template_mask, self.template_pos[-1], search_src_proj, search_mask, search_pos[-1])
                self.memory.append(memory)
            else:
                hs = transformer(template_src_proj, self.template_mask, self.template_pos[-1], search_src_proj, search_mask, search_pos[-1], self.memory[i])[0]

            hs_list.append(hs)

        concat_hs = torch.cat(hs_list, -1)

        hs_reg = self.reg_embed(concat_hs)
        hs_wh =  self.wh_embed(concat_hs)
        hs_hm = self.heatmap_embed(concat_hs)

        outputs_heatmap = hs_hm  # we have different sigmoid process for training and inference, so we do not get sigmoid here.

        # TODO: whether can you sigmoid() for the offset regression,
        # YoLo V3 uses sigmoid: https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
        outputs_bbox_reg = hs_reg.sigmoid()
        outputs_bbox_wh = hs_wh.sigmoid()

        search_mask = search_features[-1].mask.flatten(1).unsqueeze(-1) # [bn, output_hegiht *  output_width, 1]

        out = {'pred_heatmap': outputs_heatmap[-1], 'pred_bbox_reg': outputs_bbox_reg[-1], 'pred_bbox_wh': outputs_bbox_wh[-1], 'search_mask': search_mask, 'all_features': all_features}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_heatmap, outputs_bbox_reg, outputs_bbox_wh, search_mask)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_bbox_reg, outputs_bbox_wh, search_mask):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_heatmap': a, 'pred_bbox_reg': b, 'pred_bbox_wh': c, 'search_mask': search_mask}
                for a, b, c  in zip(outputs_class[:-1], outputs_bbox_reg[:-1], outputs_bbox_wh[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for TRTR.
    """
    def __init__(self, weight_dict, loss_mask=False):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = [self.loss_heatmap, self.loss_bbox] # workaround for heatmap based boundbox
        self.loss_mask = loss_mask

    def loss_heatmap(self, outputs, targets, num_boxes):
        """ Focal loss for the heatmap
        targets dicts must contain the key "heatmap" containing a tensor of dim [batch_size]
        """
        assert 'pred_heatmap' in outputs and 'search_mask' in outputs
        src_heatmap = torch.clamp(outputs['pred_heatmap'].sigmoid(), min=1e-4, max=1-1e-4) # [bN, output_height *  output_width, 1], clamp for focal loss
        target_heatmap = torch.stack([t['hm'] for t in targets]) # [bn, output_hegiht, output_width]
        target_heatmap = target_heatmap.flatten(1).unsqueeze(-1)  # [bn, output_hegiht *  output_width, 1]

        if self.loss_mask:
            mask = torch.logical_not(outputs['search_mask'])
        else:
            mask = None

        loss_hm = neg_loss(src_heatmap, target_heatmap, mask)

        losses = {'loss_hm': loss_hm}

        return losses

    def loss_bbox(self, outputs, targets, num_boxes):
        """Compute the losses related to the bounding boxes regression, the L1 regression loss
           targets dicts must contain the key "boxes" containing a tensor of dim [batch_size, 2]
           The target boxes regression are expected in format (gt_x / stride - floor(gt_x / stride), gt_x / stride - floor(gt_x / stride)).
           The target boxes width/height are expected in format (w, h), which is normalized by the input size.
           TODO: check the effect with/ without the GIoU loss
        """
        assert 'pred_bbox_reg' in outputs and 'pred_bbox_wh' in outputs

        all_src_boxes_reg = outputs['pred_bbox_reg'] # [bN, output_hegiht * output_width, 2]
        all_target_boxes_reg = torch.stack([t['reg'] for t in targets]) # [bn, 2]
        all_src_boxes_wh = outputs['pred_bbox_wh'] # [bN, output_hegiht * output_width, 2]
        all_target_boxes_wh = torch.stack([t['wh'] for t in targets]) # [bn, 2]
        all_target_boxes_ind = torch.as_tensor([t['ind'].item() for t in targets], device = all_src_boxes_reg.device) # [bn]

        # print("all_target_boxes_reg: {}, all_src_boxes_reg: {}".format(all_target_boxes_reg.shape, all_src_boxes_reg.shape))
        # print("all_target_boxes_wh: {}, all_src_boxes_wh: {}".format(all_target_boxes_wh.shape, all_src_boxes_wh.shape))
        # print("all_target_boxes_ind: {}".format(all_target_boxes_ind))


        # only calculate the loss for bbox has the object
        mask = [id for id, t in enumerate(targets) if t['valid'].item() == 1] # only extract the index with object
        src_boxes_reg = all_src_boxes_reg[mask]
        target_boxes_reg = all_target_boxes_reg[mask]
        src_boxes_wh = all_src_boxes_wh[mask]
        target_boxes_wh = all_target_boxes_wh[mask]
        target_boxes_ind = all_target_boxes_ind[mask]

        # print("mask: {}".format(mask))
        # print("target_boxes_reg: {}, src_boxes_reg: {}".format(target_boxes_reg.shape, src_boxes_reg.shape))
        # print("target_boxes_wh: {}, src_boxes_wh: {}".format(target_boxes_wh.shape, src_boxes_wh.shape))
        # print("target_boxes_ind: {}".format(target_boxes_ind))

        #loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox_reg = reg_l1_loss(src_boxes_reg, target_boxes_ind, target_boxes_reg)
        loss_bbox_wh = reg_l1_loss(src_boxes_wh, target_boxes_ind, target_boxes_wh)

        losses = {}
        losses['loss_bbox_reg'] = loss_bbox_reg.sum() / num_boxes
        losses['loss_bbox_wh'] = loss_bbox_wh.sum() / num_boxes

        return losses

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # TODO: this is a reserved function fro a negative sample training to improve the robustness like DasiamRPN
        num_boxes = sum(t['valid'].item() for t in targets)
        # print("num of valid boxes: {}".format(num_boxes)) # debug
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(loss(outputs, targets, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    l_dict = loss(aux_outputs, targets, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

class PostProcess(nn.Module):
    def __init__(self):

        super().__init__()

    @torch.no_grad()
    def forward(self, outputs):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
        """

        # do post sigmoid process
        heatmap = outputs['pred_heatmap'].sigmoid().squeeze(-1)
        # mask the heatmap
        heatmap.masked_fill_(outputs['search_mask'].squeeze(-1), 0.0) # TODO check the validity

        out = {'pred_heatmap': heatmap, 'pred_bbox_reg': outputs['pred_bbox_reg'], 'pred_bbox_wh': outputs['pred_bbox_wh']}

        return out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def get_args_parser():
    parser = ArgumentParser(prog = 'trtr')

    parser.add_argument('--device', default='cuda',
                        help='device to use for inference')

    # Modle Parameters
    parser.add_argument('--transformer_mask', type=bool, default=False,
                        help="whether masking padding area to zero in attention mechanism")

    # Loss
    parser.add_argument('--aux_loss', type=bool, default=True,
                        help="whether use auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--loss_mask', type=bool, default=False,
                        help="whether use mask for heamtmap loss")

    parser.add_argument('--reg_loss_coef', type=float, default=1,
                        help="weight (coeffficient) about bbox offset reggresion loss")
    parser.add_argument('--wh_loss_coef', type=float, default=1,
                        help="weight (coeffficient) about bbox width/height loss")


    # Backbone
    parser.add_argument('--backbone', action=ActionParser(parser=backbone_args_parser()))
    # Position Embedding
    parser.add_argument('--position_embedding', type=str, default='sine', choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    # Transformer
    parser.add_argument('--transformer', action=ActionParser(parser=transformer_args_parser()))

    return parser


def build(args):
    device = torch.device(args.device)

    position_embedding = build_position_encoding(args)

    if hasattr(args, 'train_backbone'):
        train_backbone = args.train_backbone
    else:
        train_backbone = False

    backbone = build_backbone(args.backbone, position_embedding, train = train_backbone)
    transformer = build_transformer(args.transformer)

    print("start build model")
    model = TRTR(
        backbone,
        transformer,
        aux_loss=args.aux_loss,
        transformer_mask = args.transformer_mask,
    )

    weight_dict = {'loss_hm': 1, 'loss_bbox_reg': args.reg_loss_coef, 'loss_bbox_wh': args.wh_loss_coef}

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.transformer.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(weight_dict=weight_dict, loss_mask = args.loss_mask)
    criterion.to(device)

    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors
