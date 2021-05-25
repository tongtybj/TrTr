"""
TRTR Transformer class.

Copy from DETR, whish has following modification compared to original transformer (torch.nn.Transformer):
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""

import copy
from typing import Optional, List
from jsonargparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch import nn, Tensor

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, template_src, template_mask, template_pos_embed, search_src, search_mask, search_pos_embed, memory = None):
        """
        template_src: [batch_size x hidden_dim x H_template x W_template]
        template_mask: [batch_size x H_template x W_template]
        template_pos_embed: [batch_size x hidden_dim x H_template x W_template]

        search_src: [batch_size x hidden_dim x H_search x W_search]
        search_mask: [batch_size x H_search x W_search]
        search_pos_embed: [batch_size x hidden_dim x H_search x W_search]
        """

        if len(template_src) > 1 and len(search_src) == 1:
            # print("do multiple frame mode ")
            template_src = template_src.flatten(2) # flatten: bNxCxHxW to bNxCxHW
            template_src = torch.cat(torch.split(template_src,1), -1) # concat: bNxCxHW to 1xCxbNHW
            template_src = template_src.permute(2, 0, 1) # permute 1xCxbNHW to bNHWx1xC for encoder in transformer

            template_pos_embed = template_pos_embed.flatten(2) # flatten: bNxCxHxW to bNxCxHW
            template_pos_embed = torch.cat(torch.split(template_pos_embed,1), -1) # concat: bNxCxHW to 1xCxbNHW
            template_pos_embed = template_pos_embed.permute(2, 0, 1) # permute 1xCxbNHW to bNHWx1xC for encoder in transformer

            if template_mask is not None:
                template_mask = template_mask.flatten(1) # flatten: bNxHxW to bNxHW
                template_mask = torch.cat(torch.split(template_mask,1), -1) # concat: bNxHW to 1xbNHW

        else:
            # flatten and permute bNxCxHxW to HWxbNxC for encoder in transformer
            template_src = template_src.flatten(2).permute(2, 0, 1)
            template_pos_embed = template_pos_embed.flatten(2).permute(2, 0, 1)
            if template_mask is not None:
                template_mask = template_mask.flatten(1)


        # encoding the template embedding with positional embbeding
        if memory is None:
            memory = self.encoder(template_src, src_key_padding_mask=template_mask, pos=template_pos_embed)

        # flatten and permute bNxCxHxW to HWxbNxC for decoder in transformer
        search_src = search_src.flatten(2).permute(2, 0, 1) # tgt
        search_pos_embed = search_pos_embed.flatten(2).permute(2, 0, 1)
        if template_mask is not None:
            search_mask = search_mask.flatten(1)

        hs = self.decoder(search_src, memory,
                          memory_key_padding_mask=template_mask,
                          tgt_key_padding_mask=search_mask,
                          encoder_pos=template_pos_embed,
                          decoder_pos=search_pos_embed)

        return hs.transpose(1, 2), memory


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                encoder_pos: Optional[Tensor] = None,
                decoder_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           encoder_pos=encoder_pos,
                           decoder_pos=decoder_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2, attn_weight_map = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                               key_padding_mask=src_key_padding_mask)
        #print("encoder: self attn_weight_map: {}".format(attn_weight_map))
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     encoder_pos: Optional[Tensor] = None,
                     decoder_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, decoder_pos)
        tgt2, attn_weight_map = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                               key_padding_mask=tgt_key_padding_mask)
        #print("decoder: self attn_weight_map: {}".format(attn_weight_map))
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn_weight_map = self.multihead_attn(query=self.with_pos_embed(tgt, decoder_pos),
                                                          key=self.with_pos_embed(memory, encoder_pos),
                                                          value=memory, attn_mask=memory_mask,
                                                          key_padding_mask=memory_key_padding_mask)
        #print("decoder: multihead attn_weight_map: {}".format(attn_weight_map))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    encoder_pos: Optional[Tensor] = None,
                    decoder_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, decoder_pos)
        tgt2, attn_weight_map = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                               key_padding_mask=tgt_key_padding_mask)
        #print("decoder: self attn_weight_map: {}".format(attn_weight_map))
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, attn_weight_map = self.multihead_attn(query=self.with_pos_embed(tgt2, decoder_pos),
                                                    key=self.with_pos_embed(memory, encoder_pos),
                                                    value=memory, attn_mask=memory_mask,
                                                    key_padding_mask=memory_key_padding_mask)
        #print("decoder: multihead attn_weight_map: {}".format(attn_weight_map))
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                encoder_pos: Optional[Tensor] = None,
                decoder_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask,
                                    encoder_pos, decoder_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask,
                                 encoder_pos, decoder_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def get_args_parser():
    parser = ArgumentParser(prog='transformer')

    parser.add_argument('--enc_layers', type=int, default=1,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', type=int, default=1,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--nheads', type=int, default=8,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout', type=float, default=0.1,
                        help="Dropout applied in the transformer")
    parser.add_argument('--pre_norm', type=bool, default=False,
                        help="whether do layer normzalize before attention mechansim")

    return parser


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
