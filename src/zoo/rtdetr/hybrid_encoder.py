"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation
from ...core import register


__all__ = ['HybridEncoder']


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        """
        Note: stride can be int or tuple, e.g. (2, 1) to downsample only H.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=3,
        expansion=1.0,
        bias=None,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None, return_attn: bool = False):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)

        q = k = self.with_pos_embed(src, pos_embed)
        attn_out, attn_weights = self.self_attn(
            q,
            k,
            value=src,
            attn_mask=src_mask,
            need_weights=return_attn,
            average_attn_weights=False,
        )

        src = residual + self.dropout1(attn_out)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)

        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)

        if return_attn:
            return src, attn_weights
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None, return_attn: bool = False):
        output = src
        attn_list = []

        for layer in self.layers:
            if return_attn:
                output, attn_weights = layer(
                    output,
                    src_mask=src_mask,
                    pos_embed=pos_embed,
                    return_attn=True,
                )
                attn_list.append(attn_weights)
            else:
                output = layer(
                    output,
                    src_mask=src_mask,
                    pos_embed=pos_embed,
                    return_attn=False,
                )

        if self.norm is not None:
            output = self.norm(output)

        if return_attn:
            return output, attn_list
        return output


@register()
class HybridEncoder(nn.Module):
    """
    TF-adapted HybridEncoder that can export encoder attention for loss-side regularization.

    The encoder itself only exposes intermediate attention and 2D spatial metadata.
    Structured low-rank losses are computed later in criterion, not inside this module.
    """

    __share__ = ['eval_spatial_size']

    def __init__(
        self,
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act='gelu',
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act='silu',
        eval_spatial_size=None,
        version='v2',
        boundary_enabled=True,
        use_boundary_gate=True,
        boundary_head_dim=64,
        boundary_smooth_kernel=5,
        boundary_gate_gamma=0.5,
        detach_boundary_gate=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # Keep these compatibility attributes so existing configs still load.
        self.boundary_enabled = boundary_enabled
        self.use_boundary_gate = use_boundary_gate
        self.boundary_head_dim = boundary_head_dim
        self.boundary_smooth_kernel = boundary_smooth_kernel
        self.boundary_gate_gamma = boundary_gate_gamma
        self.detach_boundary_gate = detach_boundary_gate

        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if version == 'v1':
                proj = nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                )
            elif version == 'v2':
                proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim)),
                ]))
            else:
                raise AttributeError()
            self.input_proj.append(proj)

        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act,
        )
        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 3, (2, 1), act=act))
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature,
                )
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w = torch.arange(int(w), dtype=torch.float32)
        # Match NCHW flatten(2): rows (H) vary slower and columns (W) vary faster.
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'

        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    @staticmethod
    def _align_time_dim(a: torch.Tensor, b: torch.Tensor):
        if a.shape[-1] == b.shape[-1]:
            return a, b
        width = min(a.shape[-1], b.shape[-1])
        return a[..., :width], b[..., :width]

    def forward(
        self,
        feats,
        return_encoder_aux: bool = False,
        compute_rank_loss: bool = False,
        rank_loss_type: str = 'local',
        rank_on: str = 'time',
        rank_window_size: int = 9,
        rank_keep_k: int = 1,
        rank_sample_stride: int = 1,
        head_reduce: str = 'mean',
        return_attn_maps: bool = False,
        apply_boundary_gate: bool = None,
    ):
        del rank_loss_type, rank_on, rank_window_size, rank_keep_k, rank_sample_stride, head_reduce, apply_boundary_gate

        assert len(feats) == len(self.in_channels), f"Expect {len(self.in_channels)} feats, got {len(feats)}"

        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        need_encoder_aux = return_encoder_aux or compute_rank_loss or return_attn_maps
        encoder_aux = {
            'enc_attn_list': [],
            'enc_spatial_shapes': [],
            'enc_level_indices': [],
            'enc_token_counts': [],
            'records': [],
        }

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                pos_embed = self.build_2d_sincos_position_embedding(
                    w, h, self.hidden_dim, self.pe_temperature
                ).to(src_flatten.device)

                if need_encoder_aux:
                    memory, attn_list = self.encoder[i](
                        src_flatten,
                        pos_embed=pos_embed,
                        return_attn=True,
                    )
                    encoder_aux['enc_attn_list'].append(attn_list)
                    encoder_aux['enc_spatial_shapes'].append((h, w))
                    encoder_aux['enc_level_indices'].append(enc_ind)
                    encoder_aux['enc_token_counts'].append(h * w)
                    encoder_aux['records'].append({
                        'enc_ind': enc_ind,
                        'hw': (h, w),
                        'attn_list': attn_list,
                    })
                else:
                    memory = self.encoder[i](src_flatten, pos_embed=pos_embed, return_attn=False)

                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]

            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high

            target_h = feat_low.shape[-2]
            target_w = feat_high.shape[-1]
            upsample_feat = F.interpolate(feat_high, size=(target_h, target_w), mode='nearest')
            upsample_feat, feat_low = self._align_time_dim(upsample_feat, feat_low)

            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.concat([upsample_feat, feat_low], dim=1)
            )
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]

            downsample_feat = self.downsample_convs[idx](feat_low)
            downsample_feat, feat_high = self._align_time_dim(downsample_feat, feat_high)

            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        if need_encoder_aux:
            return outs, encoder_aux
        return outs
