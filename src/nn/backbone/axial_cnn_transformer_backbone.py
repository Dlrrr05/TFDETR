"""Axial CNN-Transformer backbone for spectrogram-like inputs."""

import os
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import freeze_batch_norm2d, get_activation
from ...core import register


__all__ = ['AxialCNNTransformerBackbone']


class DropPath(nn.Module):
    """Stochastic depth per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ChannelLayerNorm2d(nn.Module):
    """LayerNorm over channel dimension for NCHW tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ConvNormAct2d(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        act: str = 'silu',
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        act: str = 'silu',
    ):
        super().__init__()
        self.dw = ConvNormAct2d(ch_in, ch_in, kernel_size=kernel_size, stride=stride, groups=ch_in, act=act)
        self.pw = ConvNormAct2d(ch_in, ch_out, kernel_size=1, stride=1, groups=1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class SEBlock2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.gate(scale)
        return x * scale


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int = None,
        act: str = 'silu',
        use_se: bool = True,
        drop_path: float = 0.0,
    ):
        super().__init__()
        hidden_channels = hidden_channels or channels
        self.conv1 = ConvNormAct2d(channels, hidden_channels, kernel_size=1, act=act)
        self.conv2 = DepthwiseSeparableConv2d(hidden_channels, channels, kernel_size=3, stride=1, act=act)
        self.se = SEBlock2d(channels) if use_se else nn.Identity()
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.se(y)
        return x + self.drop_path(y)


class ConvStem(nn.Module):
    def __init__(self, in_channels: int, stem_dim: int = 64, act: str = 'silu'):
        super().__init__()
        mid_dim = max(stem_dim // 2, 16)
        self.conv1 = ConvNormAct2d(in_channels, mid_dim, kernel_size=3, stride=2, act=act)
        self.conv2 = ConvNormAct2d(mid_dim, stem_dim, kernel_size=3, stride=2, act=act)
        self.conv3 = ConvNormAct2d(stem_dim, stem_dim, kernel_size=3, stride=1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class CNNStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 2,
        act: str = 'silu',
        use_se: bool = True,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.proj = ConvNormAct2d(in_channels, out_channels, kernel_size=1, stride=1, act=act)
        self.blocks = nn.Sequential(*[
            ResidualConvBlock(
                out_channels,
                hidden_channels=out_channels,
                act=act,
                use_se=use_se,
                drop_path=drop_path,
            )
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.blocks(x)


class StageTransition(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act: str = 'silu', use_se: bool = False):
        super().__init__()
        self.down = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, stride=2, act=act)
        self.se = SEBlock2d(out_channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.down(x))


class AxialRelPosBias(nn.Module):
    """Learnable 1D relative position bias for a single axis."""

    def __init__(self, num_heads: int, max_positions: int):
        super().__init__()
        self.num_heads = num_heads
        self.max_positions = int(max_positions)
        self.bias = nn.Parameter(torch.zeros(2 * self.max_positions - 1, num_heads))
        nn.init.trunc_normal_(self.bias, std=0.02)

    def _resized_bias(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if length <= self.max_positions:
            return self.bias.to(device=device, dtype=dtype)

        resized = F.interpolate(
            self.bias.transpose(0, 1).unsqueeze(0),
            size=2 * length - 1,
            mode='linear',
            align_corners=True,
        )
        return resized.squeeze(0).transpose(0, 1).to(device=device, dtype=dtype)

    def forward(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        bias_table = self._resized_bias(length, device=device, dtype=dtype)
        offset = (bias_table.shape[0] - 1) // 2
        positions = torch.arange(length, device=device)
        rel = positions[:, None] - positions[None, :] + offset
        return bias_table[rel].permute(2, 0, 1).contiguous()


class AxialAttentionBase(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_positions: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads.'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.rel_pos_bias = AxialRelPosBias(num_heads, max_positions=max_positions)

    def _reshape_to_axis_tokens(self, x: torch.Tensor):
        raise NotImplementedError

    def _restore_from_axis_tokens(self, x: torch.Tensor, shape_meta):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, shape_meta = self._reshape_to_axis_tokens(x)
        batch_tokens, seq_len, _ = tokens.shape

        qkv = self.qkv(tokens).reshape(batch_tokens, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))
        attn = attn + self.rel_pos_bias(seq_len, device=tokens.device, dtype=tokens.dtype).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_tokens, seq_len, self.dim)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return self._restore_from_axis_tokens(out, shape_meta)


class FrequencyAxialAttention(AxialAttentionBase):
    """Attend over the frequency axis for each time position."""

    def _reshape_to_axis_tokens(self, x: torch.Tensor):
        b, c, f, t = x.shape
        tokens = x.permute(0, 3, 2, 1).reshape(b * t, f, c)
        return tokens, (b, c, f, t)

    def _restore_from_axis_tokens(self, x: torch.Tensor, shape_meta):
        b, c, f, t = shape_meta
        x = x.reshape(b, t, f, c).permute(0, 3, 2, 1)
        return x.contiguous()


class TemporalAxialAttention(AxialAttentionBase):
    """Attend over the time axis for each frequency position."""

    def _reshape_to_axis_tokens(self, x: torch.Tensor):
        b, c, f, t = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(b * f, t, c)
        return tokens, (b, c, f, t)

    def _restore_from_axis_tokens(self, x: torch.Tensor, shape_meta):
        b, c, f, t = shape_meta
        x = x.reshape(b, f, t, c).permute(0, 3, 1, 2)
        return x.contiguous()


class AxialFFN(nn.Module):
    def __init__(self, dim: int, ff_mult: int = 4, dropout: float = 0.0, act: str = 'silu'):
        super().__init__()
        hidden = dim * ff_mult
        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = get_activation(act)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class AxialConvModule(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        dropout: float = 0.0,
        act: str = 'silu',
        use_se: bool = False,
    ):
        super().__init__()
        hidden = dim * 2
        self.pw1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.dw = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=dim,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(dim)
        self.act = get_activation(act)
        self.se = SEBlock2d(dim) if use_se else nn.Identity()
        self.pw2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw1(x)
        x = F.glu(x, dim=1)
        x = self.dw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pw2(x)
        x = self.drop(x)
        return x


class FrequencyAxialBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_mult: int = 4,
        conv_kernel_size: int = 7,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        drop_path: float = 0.0,
        act: str = 'silu',
        max_positions: int = 256,
        use_se: bool = False,
    ):
        super().__init__()
        self.norm_ffn1 = ChannelLayerNorm2d(dim)
        self.ffn1 = AxialFFN(dim, ff_mult=ff_mult, dropout=proj_dropout, act=act)
        self.norm_attn = ChannelLayerNorm2d(dim)
        self.attn = FrequencyAxialAttention(
            dim=dim,
            num_heads=num_heads,
            max_positions=max_positions,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.conv = AxialConvModule(
            dim=dim,
            kernel_size=conv_kernel_size,
            dropout=proj_dropout,
            act=act,
            use_se=use_se,
        )
        self.norm_ffn2 = ChannelLayerNorm2d(dim)
        self.ffn2 = AxialFFN(dim, ff_mult=ff_mult, dropout=proj_dropout, act=act)
        self.out_norm = ChannelLayerNorm2d(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(0.5 * self.ffn1(self.norm_ffn1(x)))
        x = x + self.drop_path(self.attn(self.norm_attn(x)))
        x = x + self.drop_path(self.conv(x))
        x = x + self.drop_path(0.5 * self.ffn2(self.norm_ffn2(x)))
        return self.out_norm(x)


class TemporalAxialBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_mult: int = 4,
        conv_kernel_size: int = 7,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        drop_path: float = 0.0,
        act: str = 'silu',
        max_positions: int = 1024,
        use_se: bool = False,
    ):
        super().__init__()
        self.norm_ffn1 = ChannelLayerNorm2d(dim)
        self.ffn1 = AxialFFN(dim, ff_mult=ff_mult, dropout=proj_dropout, act=act)
        self.norm_attn = ChannelLayerNorm2d(dim)
        self.attn = TemporalAxialAttention(
            dim=dim,
            num_heads=num_heads,
            max_positions=max_positions,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.conv = AxialConvModule(
            dim=dim,
            kernel_size=conv_kernel_size,
            dropout=proj_dropout,
            act=act,
            use_se=use_se,
        )
        self.norm_ffn2 = ChannelLayerNorm2d(dim)
        self.ffn2 = AxialFFN(dim, ff_mult=ff_mult, dropout=proj_dropout, act=act)
        self.out_norm = ChannelLayerNorm2d(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(0.5 * self.ffn1(self.norm_ffn1(x)))
        x = x + self.drop_path(self.attn(self.norm_attn(x)))
        x = x + self.drop_path(self.conv(x))
        x = x + self.drop_path(0.5 * self.ffn2(self.norm_ffn2(x)))
        return self.out_norm(x)


class FrequencyAxialStage(nn.Module):
    def __init__(self, blocks: Iterable[nn.Module]):
        super().__init__()
        self.blocks = nn.ModuleList(list(blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class TemporalAxialStage(nn.Module):
    def __init__(self, blocks: Iterable[nn.Module]):
        super().__init__()
        self.blocks = nn.ModuleList(list(blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class CrossStageFusion(nn.Module):
    def __init__(self, channels: int, use_gated_fusion: bool = False):
        super().__init__()
        self.use_gated_fusion = use_gated_fusion
        if use_gated_fusion:
            self.gate = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )

    def forward(self, main: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if not self.use_gated_fusion:
            return main + skip

        gate = self.gate(torch.cat([main, skip], dim=1))
        return gate * main + (1.0 - gate) * skip


@register()
class AxialCNNTransformerBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        stem_dim: int = 64,
        stage_dims: Sequence[int] = (128, 256, 512),
        depth_cnn: int = 2,
        depth_freq: int = 2,
        depth_time: int = 2,
        num_heads: Sequence[int] = (4, 8),
        ff_mult: int = 4,
        conv_kernel_size: int = 7,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        drop_path: float = 0.05,
        use_se: bool = True,
        use_stage_fusion: bool = True,
        use_gated_fusion: bool = False,
        act: str = 'silu',
        return_idx: Sequence[int] = (0, 1, 2),
        freeze_at: int = -1,
        freeze_norm: bool = False,
        pretrained=False,
        max_freq_positions: int = 256,
        max_time_positions: int = 1024,
    ):
        super().__init__()
        assert len(stage_dims) == 3, 'stage_dims must contain three output channel values.'
        assert len(num_heads) == 2, 'num_heads must contain [freq_heads, time_heads].'

        c1, c2, c3 = [int(v) for v in stage_dims]
        self.return_idx = list(return_idx)
        self._all_out_channels = [c1, c2, c3]
        self._all_out_strides = [4, 8, 16]
        self.out_channels = [self._all_out_channels[i] for i in self.return_idx]
        self.out_strides = [self._all_out_strides[i] for i in self.return_idx]

        self.stem = ConvStem(in_channels=in_channels, stem_dim=stem_dim, act=act)
        self.cnn_stage = CNNStage(
            in_channels=stem_dim,
            out_channels=c1,
            depth=depth_cnn,
            act=act,
            use_se=use_se,
            drop_path=0.0,
        )

        self.transition1 = StageTransition(c1, c2, act=act, use_se=False)
        self.transition2 = StageTransition(c2, c3, act=act, use_se=False)

        dpr = torch.linspace(0.0, float(drop_path), steps=max(depth_freq + depth_time, 1)).tolist()
        freq_dpr = dpr[:depth_freq]
        time_dpr = dpr[depth_freq:depth_freq + depth_time]

        self.freq_stage = FrequencyAxialStage([
            FrequencyAxialBlock(
                dim=c2,
                num_heads=int(num_heads[0]),
                ff_mult=ff_mult,
                conv_kernel_size=conv_kernel_size,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                drop_path=freq_dpr[i] if i < len(freq_dpr) else 0.0,
                act=act,
                max_positions=max_freq_positions,
                use_se=use_se,
            )
            for i in range(depth_freq)
        ])
        self.time_stage = TemporalAxialStage([
            TemporalAxialBlock(
                dim=c3,
                num_heads=int(num_heads[1]),
                ff_mult=ff_mult,
                conv_kernel_size=conv_kernel_size,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                drop_path=time_dpr[i] if i < len(time_dpr) else 0.0,
                act=act,
                max_positions=max_time_positions,
                use_se=use_se,
            )
            for i in range(depth_time)
        ])

        self.use_stage_fusion = use_stage_fusion
        if use_stage_fusion:
            self.skip1 = StageTransition(c1, c2, act=act, use_se=False)
            self.skip2 = StageTransition(c2, c3, act=act, use_se=False)
            self.fuse1 = CrossStageFusion(c2, use_gated_fusion=use_gated_fusion)
            self.fuse2 = CrossStageFusion(c3, use_gated_fusion=use_gated_fusion)
        else:
            self.skip1 = None
            self.skip2 = None
            self.fuse1 = None
            self.fuse2 = None

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
        if freeze_at >= 1:
            self._freeze_parameters(self.cnn_stage)
        if freeze_at >= 2:
            self._freeze_parameters(self.transition1)
            self._freeze_parameters(self.freq_stage)
            if self.use_stage_fusion:
                self._freeze_parameters(self.skip1)
                self._freeze_parameters(self.fuse1)
        if freeze_at >= 3:
            self._freeze_parameters(self.transition2)
            self._freeze_parameters(self.time_stage)
            if self.use_stage_fusion:
                self._freeze_parameters(self.skip2)
                self._freeze_parameters(self.fuse2)

        if freeze_norm:
            freeze_batch_norm2d(self)

        if pretrained:
            self._load_pretrained(pretrained)

    def _freeze_parameters(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False

    def _load_pretrained(self, pretrained):
        if isinstance(pretrained, str) and os.path.exists(pretrained):
            state = torch.load(pretrained, map_location='cpu')
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            self.load_state_dict(state, strict=False)
            print(f'Load AxialCNNTransformerBackbone state_dict from {pretrained}')
            return

        raise ValueError(
            'AxialCNNTransformerBackbone does not provide built-in pretrained weights. '
            'Please pass a valid checkpoint path or set pretrained=False.'
        )

    def forward(self, x: torch.Tensor):
        feat_cnn = self.cnn_stage(self.stem(x))

        freq_in = self.transition1(feat_cnn)
        feat_freq = self.freq_stage(freq_in)
        if self.use_stage_fusion:
            feat_freq = self.fuse1(feat_freq, self.skip1(feat_cnn))

        time_in = self.transition2(feat_freq)
        feat_time = self.time_stage(time_in)
        if self.use_stage_fusion:
            feat_time = self.fuse2(feat_time, self.skip2(feat_freq))

        outs = [feat_cnn, feat_freq, feat_time]
        return [outs[i] for i in self.return_idx]
