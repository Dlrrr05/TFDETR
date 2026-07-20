"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
BEATs backbone with intermediate-layer 2D lifting for RT-DETR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from .common import get_activation, FrozenBatchNorm2d
from ...core import register

# 按你项目里的实际路径改
from .BEATs import BEATs, BEATsConfig


__all__ = ['BEATsBackbone2D']


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if padding is None:
            padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)

        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=padding,
            bias=bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class LayerTo2D(nn.Module):
    """
    将某一层 Transformer hidden states:
        [B, T, D]
    转成 2D feature map:
        [B, C, F, T]  (或者经过时间下采样后 [B, C, F, T'])
    """
    def __init__(
        self,
        in_dim,
        out_dim=256,
        map_size=(8, 32),
        stride=(1, 1),
        act='relu',
        refine_depth=2
    ):
        super().__init__()

        freq_bins, inner_dim = map_size
        self.freq_bins = freq_bins
        self.inner_dim = inner_dim
        self.out_dim = out_dim

        self.token_expand = nn.Linear(in_dim, freq_bins * inner_dim)

        if isinstance(stride, int):
            stride = (stride, stride)

        blocks = []
        ch_in = inner_dim
        for i in range(refine_depth):
            block_stride = stride if i == 0 else (1, 1)
            ch_out = out_dim
            blocks.append(
                ConvNormLayer(
                    ch_in,
                    ch_out,
                    kernel_size=3,
                    stride=block_stride,
                    padding=1,
                    act=act
                )
            )
            ch_in = ch_out

        self.refine = nn.Sequential(*blocks)

    def forward(self, x):
        B, T, D = x.shape

        x = self.token_expand(x)
        x = x.view(B, T, self.freq_bins, self.inner_dim)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.refine(x)
        return x


@register()
class BEATsBackbone2D(nn.Module):
    """
    从 BEATs 的中间层抽取特征:
        layer 3, 7, 11 (按人类习惯从1开始数)
    然后分别转成 2D feature maps:
        out_3, out_7, out_11

    输入:
        x: [B, T] 或 [B, 1, T]

    输出:
        outs: list[Tensor]
            每个元素形状为 [B, C, F, T_i]
    """

    def __init__(
        self,
        checkpoint,
        return_idx=[0, 1, 2],
        hidden_dim=256,
        layers=(3, 7, 11),
        map_sizes=((8, 32), (8, 32), (8, 32)),
        out_strides=(1, 2, 4),
        time_strides=(1, 1, 1),
        act='relu',
        input_mode='waveform',
        channel_reduction='mean',
        freeze_beats=True,
        freeze_at=-1,
        freeze_norm=True,
    ):
        super().__init__()

        assert checkpoint is not None, 'BEATs checkpoint path must be provided.'
        assert len(layers) == 3, 'This version expects 3 transformer layers, e.g. (3, 7, 11).'
        assert len(map_sizes) == 3
        assert len(out_strides) == 3
        assert len(time_strides) == 3

        self.return_idx = return_idx
        self.layers_1based = tuple(layers)
        self.layers_0based = tuple([l - 1 for l in layers])
        self.input_mode = input_mode
        self.channel_reduction = channel_reduction

        self.beats, beats_dim = self._build_beats(checkpoint)
        self._hook_features = OrderedDict()
        self._hooks = []
        self._register_layer_hooks()

        self.stage_convs = nn.ModuleList([
            LayerTo2D(
                in_dim=beats_dim,
                out_dim=hidden_dim,
                map_size=map_sizes[0],
                stride=(out_strides[0], time_strides[0]),
                act=act,
                refine_depth=2,
            ),
            LayerTo2D(
                in_dim=beats_dim,
                out_dim=hidden_dim,
                map_size=map_sizes[1],
                stride=(out_strides[1], time_strides[1]),
                act=act,
                refine_depth=2,
            ),
            LayerTo2D(
                in_dim=beats_dim,
                out_dim=hidden_dim,
                map_size=map_sizes[2],
                stride=(out_strides[2], time_strides[2]),
                act=act,
                refine_depth=2,
            ),
        ])

        _out_channels = [hidden_dim, hidden_dim, hidden_dim]
        _out_strides = list(out_strides)

        self.out_channels = [_out_channels[i] for i in return_idx]
        self.out_strides = [_out_strides[i] for i in return_idx]

        if freeze_beats:
            self._freeze_parameters(self.beats)

        if freeze_at >= 0:
            for i in range(min(freeze_at + 1, len(self.stage_convs))):
                self._freeze_parameters(self.stage_convs[i])

        if freeze_norm:
            self._freeze_norm(self)

    def _build_beats(self, checkpoint):
        state = torch.load(checkpoint, map_location='cpu')

        if 'cfg' not in state:
            raise ValueError('Cannot find "cfg" in checkpoint.')
        if 'model' not in state:
            raise ValueError('Cannot find "model" in checkpoint.')

        cfg = BEATsConfig(state['cfg'])
        model = BEATs(cfg)

        model_state = {}
        for k, v in state['model'].items():
            nk = k.replace('module.', '')
            model_state[nk] = v

        msg = model.load_state_dict(model_state, strict=False)
        print(f'Load BEATs state_dict | missing={len(msg.missing_keys)} | unexpected={len(msg.unexpected_keys)}')

        if not hasattr(cfg, 'encoder_embed_dim'):
            raise ValueError('BEATs config has no attribute "encoder_embed_dim".')

        beats_dim = cfg.encoder_embed_dim
        return model, beats_dim

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _prepare_wav(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]

        if x.dim() == 2:
            return x
        elif x.dim() == 3:
            if x.shape[1] == 1:
                return x[:, 0, :]
            elif x.shape[2] == 1:
                return x[:, :, 0]
            else:
                raise ValueError(f'Unsupported 3D input shape: {x.shape}')
        elif x.dim() == 4:
            if x.shape[1] == 1 and x.shape[2] == 1:
                return x[:, 0, 0, :]
            else:
                raise ValueError(f'Unsupported 4D input shape: {x.shape}')
        else:
            raise ValueError(f'Unsupported input shape: {x.shape}')

    def _prepare_fbank(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]

        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f'Unsupported spectrogram input shape: {x.shape}')

        if x.shape[1] == 1:
            spec = x
        elif self.channel_reduction == 'mean':
            spec = x.mean(dim=1, keepdim=True)
        elif self.channel_reduction == 'first':
            spec = x[:, :1]
        else:
            raise ValueError(f'Unsupported channel_reduction: {self.channel_reduction}')

        return spec[:, 0]

    def _normalize_hook_output(self, output):
        """
        将 hook 抓到的 layer output 统一整理为 [B, T, D]
        常见可能是:
            - [T, B, D]
            - [B, T, D]
            - tuple(tensor, ...)
        """
        if isinstance(output, tuple):
            output = output[0]

        if not torch.is_tensor(output):
            raise RuntimeError(f'Unsupported hook output type: {type(output)}')

        if output.dim() != 3:
            raise RuntimeError(f'Expected 3D hidden state, got {output.shape}')

        # 常见 fairseq 风格: [T, B, D]
        # 经验判断：若第0维远大于 batch，且第1维较小，通常是 [T, B, D]
        if output.shape[0] > output.shape[1]:
            output = output.transpose(0, 1).contiguous()

        # 现在应为 [B, T, D]
        return output

    def _register_layer_hooks(self):
        """
        直接在 encoder.layers 上挂 hook，抓第3/7/11层输出
        对应 0-based index: 2/6/10
        """
        if not hasattr(self.beats, 'encoder'):
            raise ValueError('BEATs model has no attribute "encoder".')
        if not hasattr(self.beats.encoder, 'layers'):
            raise ValueError('BEATs encoder has no attribute "layers".')

        total_layers = len(self.beats.encoder.layers)
        for l in self.layers_0based:
            if l < 0 or l >= total_layers:
                raise ValueError(f'Requested layer {l+1}, but BEATs has only {total_layers} layers.')

        def make_hook(layer_idx_0based):
            def hook(module, input, output):
                self._hook_features[layer_idx_0based] = self._normalize_hook_output(output)
            return hook

        for l in self.layers_0based:
            h = self.beats.encoder.layers[l].register_forward_hook(make_hook(l))
            self._hooks.append(h)

    def _forward_beats(self, wav, padding_mask=None):
        self._hook_features.clear()

        if hasattr(self.beats, 'extract_features'):
            _ = self.beats.extract_features(wav, padding_mask=padding_mask)
        else:
            _ = self.beats(wav, padding_mask=padding_mask)

        # 检查 hook 是否成功捕获
        missing = [l for l in self.layers_0based if l not in self._hook_features]
        if len(missing) > 0:
            miss_layers = [m + 1 for m in missing]
            raise RuntimeError(
                f'Failed to capture hidden states for layers {miss_layers}. '
                f'Please check the internal structure of your BEATs implementation.'
            )

        return [
            self._hook_features[self.layers_0based[0]],
            self._hook_features[self.layers_0based[1]],
            self._hook_features[self.layers_0based[2]],
        ]

    def _forward_beats_from_fbank(self, fbank, padding_mask=None):
        self._hook_features.clear()

        if padding_mask is not None:
            padding_mask = self.beats.forward_padding_mask(fbank, padding_mask)

        features = self.beats.patch_embedding(fbank.unsqueeze(1))
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.beats.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.beats.forward_padding_mask(features, padding_mask)

        if self.beats.post_extract_proj is not None:
            features = self.beats.post_extract_proj(features)

        x = self.beats.dropout_input(features)
        _ = self.beats.encoder(x, padding_mask=padding_mask)

        missing = [l for l in self.layers_0based if l not in self._hook_features]
        if len(missing) > 0:
            miss_layers = [m + 1 for m in missing]
            raise RuntimeError(
                f'Failed to capture hidden states for layers {miss_layers}. '
                f'Please check the internal structure of your BEATs implementation.'
            )

        return [
            self._hook_features[self.layers_0based[0]],
            self._hook_features[self.layers_0based[1]],
            self._hook_features[self.layers_0based[2]],
        ]

    def forward(self, x, padding_mask=None):
        if self.input_mode == 'waveform':
            wav = self._prepare_wav(x)
            hidden_states = self._forward_beats(wav, padding_mask=padding_mask)
        elif self.input_mode in ('fbank', 'spectrogram', 'image'):
            fbank = self._prepare_fbank(x)
            hidden_states = self._forward_beats_from_fbank(fbank, padding_mask=padding_mask)
        else:
            raise ValueError(f'Unsupported input_mode: {self.input_mode}')

        outs = []
        for idx, (h, proj) in enumerate(zip(hidden_states, self.stage_convs)):
            feat2d = proj(h)
            if idx in self.return_idx:
                outs.append(feat2d)

        return outs
