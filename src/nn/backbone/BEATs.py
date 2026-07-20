# --------------------------------------------------------
# BEATs: Audio Pre-Training with Acoustic Tokenizers (https://arxiv.org/abs/2212.09058)
# Github source: https://github.com/microsoft/unilm/tree/master/beats
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import LayerNorm
import torchaudio.compliance.kaldi as ta_kaldi

from .backbone import TransformerEncoder

logger = logging.getLogger(__name__)


class BEATsConfig:
    def __init__(self, cfg=None):
        self.input_patch_size: int = -1
        self.embed_dim: int = 512
        self.conv_bias: bool = False

        self.encoder_layers: int = 12
        self.encoder_embed_dim: int = 768
        self.encoder_ffn_embed_dim: int = 3072
        self.encoder_attention_heads: int = 12
        self.activation_fn: str = "gelu"

        self.layer_wise_gradient_decay_ratio: float = 1.0
        self.layer_norm_first: bool = False
        self.deep_norm: bool = False

        self.dropout: float = 0.1
        self.attention_dropout: float = 0.1
        self.activation_dropout: float = 0.0
        self.encoder_layerdrop: float = 0.0
        self.dropout_input: float = 0.0

        self.conv_pos: int = 128
        self.conv_pos_groups: int = 16

        self.relative_position_embedding: bool = False
        self.num_buckets: int = 320
        self.max_distance: int = 1280
        self.gru_rel_pos: bool = False

        self.finetuned_model: bool = False
        self.predictor_dropout: float = 0.1
        self.predictor_class: int = 527

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


class BEATs(nn.Module):
    def __init__(self, cfg: BEATsConfig) -> None:
        super().__init__()
        logger.info(f"BEATs Config: {cfg.__dict__}")

        self.cfg = cfg

        self.embed = cfg.embed_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.input_patch_size = cfg.input_patch_size
        self.patch_embedding = nn.Conv2d(
            1,
            self.embed,
            kernel_size=self.input_patch_size,
            stride=self.input_patch_size,
            bias=cfg.conv_bias
        )

        self.dropout_input = nn.Dropout(cfg.dropout_input)

        assert not cfg.deep_norm or not cfg.layer_norm_first
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        if cfg.finetuned_model:
            self.predictor_dropout = nn.Dropout(cfg.predictor_dropout)
            self.predictor = nn.Linear(cfg.encoder_embed_dim, cfg.predictor_class)
        else:
            self.predictor = None

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def preprocess(
        self,
        source: torch.Tensor,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
    ) -> torch.Tensor:
        """
        source: [B, T]
        return: [B, num_frames, 128]
        """
        fbanks = []
        for waveform in source:
            waveform = waveform.unsqueeze(0) * 2 ** 15
            fbank = ta_kaldi.fbank(
                waveform,
                num_mel_bins=128,
                sample_frequency=16000,
                frame_length=25,
                frame_shift=10
            )
            fbanks.append(fbank)

        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - fbank_mean) / (2 * fbank_std)
        return fbank

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            source: [B, T]
            padding_mask: [B, T] bool or None
            return_all_hiddens: whether to return intermediate transformer layer outputs

        Returns:
            if predictor is None:
                - default:
                    x, padding_mask
                - if return_all_hiddens:
                    {
                        "x": x,
                        "padding_mask": padding_mask,
                        "layer_results": layer_results,
                        "hidden_states": hidden_states,
                    }

            if predictor is not None:
                - default:
                    lprobs, padding_mask
                - if return_all_hiddens:
                    {
                        "x": x,
                        "padding_mask": padding_mask,
                        "layer_results": layer_results,
                        "hidden_states": hidden_states,
                        "logits": logits,
                        "lprobs": lprobs,
                    }
        """
        fbank = self.preprocess(source, fbank_mean=fbank_mean, fbank_std=fbank_std)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(fbank, padding_mask)

        fbank = fbank.unsqueeze(1)                    # [B, 1, F, T]
        features = self.patch_embedding(fbank)        # [B, C, F', T']
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)           # [B, L, C]
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        x = self.dropout_input(features)

        # x: [B, L, D]
        # layer_results: implementation-dependent, usually list of per-layer outputs
        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
        )

        # normalize intermediate hidden states to List[[B, L, D]]
        hidden_states = None
        if return_all_hiddens:
            hidden_states = []
            if layer_results is not None:
                for layer_out in layer_results:
                    # common cases:
                    # 1) tensor [L, B, D]
                    # 2) tuple(tensor, attn, ...)
                    if isinstance(layer_out, (tuple, list)):
                        layer_out = layer_out[0]

                    if torch.is_tensor(layer_out):
                        if layer_out.dim() != 3:
                            continue

                        # convert [L, B, D] -> [B, L, D]
                        if layer_out.shape[0] > layer_out.shape[1]:
                            layer_out = layer_out.transpose(0, 1).contiguous()

                        hidden_states.append(layer_out)

        if self.predictor is not None:
            x_drop = self.predictor_dropout(x)
            logits = self.predictor(x_drop)

            if padding_mask is not None and padding_mask.any():
                logits = logits.clone()
                logits[padding_mask] = 0
                logits = logits.sum(dim=1)
                logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits)
            else:
                logits = logits.mean(dim=1)

            lprobs = torch.sigmoid(logits)

            if return_all_hiddens:
                return {
                    "x": x,
                    "padding_mask": padding_mask,
                    "layer_results": layer_results,
                    "hidden_states": hidden_states,
                    "logits": logits,
                    "lprobs": lprobs,
                }

            return lprobs, padding_mask

        if return_all_hiddens:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "layer_results": layer_results,
                "hidden_states": hidden_states,
            }
        
        print("==== DEBUG BEATs.extract_features ====")
        print("x type:", type(x))
        if torch.is_tensor(x):
            print("x shape:", x.shape)

        print("layer_results type:", type(layer_results))
        if isinstance(layer_results, (list, tuple)):
            print("len(layer_results):", len(layer_results))
            for i, layer_out in enumerate(layer_results):
                print(f"[layer_results {i}] type:", type(layer_out))

                tmp = layer_out
                depth = 0
                while isinstance(tmp, (tuple, list)) and len(tmp) > 0 and depth < 5:
                    print(f"    unpack level {depth}: type={type(tmp)}, len={len(tmp)}")
                    tmp = tmp[0]
                    depth += 1

                if torch.is_tensor(tmp):
                    print(f"    final tensor shape: {tmp.shape}")
                else:
                    print(f"    final object type after unpack: {type(tmp)}")
        else:
            print("layer_results is not list/tuple")
        print("==== END DEBUG ====")
        return x, padding_mask

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
        return_all_hiddens: bool = False,
    ):
        return self.extract_features(
            source=source,
            padding_mask=padding_mask,
            fbank_mean=fbank_mean,
            fbank_std=fbank_std,
            return_all_hiddens=return_all_hiddens,
        )
