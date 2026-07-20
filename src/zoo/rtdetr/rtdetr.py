"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn

from ...core import register


__all__ = ['RTDETR']


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder']

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        encoder_return_aux: bool = False,
        encoder_compute_rank_loss_in_eval: bool = False,
        encoder_rank_loss_type: str = 'local',
        encoder_rank_on: str = 'time',
        encoder_rank_window_size: int = 9,
        encoder_rank_keep_k: int = 1,
        encoder_rank_sample_stride: int = 1,
        encoder_head_reduce: str = 'mean',
        encoder_return_attn_maps: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

        # These attributes are kept so existing configs still deserialize cleanly.
        self.encoder_return_aux = encoder_return_aux
        self.encoder_compute_rank_loss_in_eval = encoder_compute_rank_loss_in_eval
        self.encoder_rank_loss_type = encoder_rank_loss_type
        self.encoder_rank_on = encoder_rank_on
        self.encoder_rank_window_size = encoder_rank_window_size
        self.encoder_rank_keep_k = encoder_rank_keep_k
        self.encoder_rank_sample_stride = encoder_rank_sample_stride
        self.encoder_head_reduce = encoder_head_reduce
        self.encoder_return_attn_maps = encoder_return_attn_maps

    def forward(self, x, targets=None):
        feats = self.backbone(x)

        need_encoder_aux = (
            self.encoder_return_aux
            or self.encoder_return_attn_maps
            or ((not self.training) and self.encoder_compute_rank_loss_in_eval)
        )

        encoder_out = self.encoder(
            feats,
            return_encoder_aux=need_encoder_aux,
            compute_rank_loss=need_encoder_aux,
            rank_loss_type=self.encoder_rank_loss_type,
            rank_on=self.encoder_rank_on,
            rank_window_size=self.encoder_rank_window_size,
            rank_keep_k=self.encoder_rank_keep_k,
            rank_sample_stride=self.encoder_rank_sample_stride,
            head_reduce=self.encoder_head_reduce,
            return_attn_maps=self.encoder_return_attn_maps,
        )

        if isinstance(encoder_out, tuple):
            feats, encoder_aux = encoder_out
        else:
            feats, encoder_aux = encoder_out, None

        outputs = self.decoder(feats, targets=targets, encoder_aux=encoder_aux)
        if encoder_aux is not None and isinstance(outputs, dict):
            outputs['encoder_aux'] = encoder_aux
            outputs['enc_attn_list'] = encoder_aux.get('enc_attn_list', [])
            outputs['enc_spatial_shapes'] = encoder_aux.get('enc_spatial_shapes', [])
            outputs['enc_level_indices'] = encoder_aux.get('enc_level_indices', [])
        return outputs

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
