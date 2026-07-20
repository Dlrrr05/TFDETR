"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

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
    ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def _parse_encoder_output(self, encoder_out):
        """
        Robustly parse encoder outputs.

        Supported forms:
            1) feats
            2) (feats, tf_aux)

        Returns:
            feats: features for decoder
            tf_aux: optional disentanglement auxiliary outputs
        """
        tf_aux = None

        if isinstance(encoder_out, tuple):
            if len(encoder_out) == 2:
                feats, tf_aux = encoder_out
            else:
                raise ValueError(
                    f'Unexpected encoder tuple length: {len(encoder_out)}. '
                    f'Expect encoder to return feats or (feats, tf_aux).'
                )
        else:
            feats = encoder_out

        return feats, tf_aux

    def forward(self, x, targets=None):
        # 1) backbone
        feats = self.backbone(x)

        # 2) encoder
        encoder_out = self.encoder(feats)
        feats, tf_aux = self._parse_encoder_output(encoder_out)

        # 3) fallback: if encoder caches tf_aux internally, fetch it
        if tf_aux is None and hasattr(self.encoder, 'get_last_tf_aux'):
            tf_aux = self.encoder.get_last_tf_aux()

        # 4) decoder
        outputs = self.decoder(feats, targets)

        # 5) attach TF auxiliary features for criterion / visualization
        # decoder in RT-DETR normally returns a dict
        if tf_aux is not None and isinstance(outputs, dict):
            outputs['tf_aux'] = tf_aux

        return outputs

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self