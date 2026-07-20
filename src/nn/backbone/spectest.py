import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops  # 必须引入 torchvision 用于 Deformable Conv
from typing import Optional

from ...core import register
from .layers import trunc_normal_, DropPath, to_2tuple

__all__ = ['SpecDCNBackbone']

"""
SOTA Backbone for Spectrogram Object Detection (Optimized for 128x256 Input)
Features: Deformable CPE, Stable Spectral Gating, Energy-Biased Attention
"""

# =====================================================
# Patch Embedding
# =====================================================

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans=1, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, 7, stride=4, padding=3)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1)
        self.norm = nn.LayerNorm(dim * 2)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W


# =====================================================
# Convolutional Positional Encoding
# =====================================================

class ConvCPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x_img = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        out = self.dwconv(x_img)
        out = out.permute(0, 2, 3, 1).reshape(B, L, C)
        return x + out

class DeformableBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        
        # 偏移量分支 (2 * k * k)
        self.offset_conv = nn.Conv2d(dim, 2 * kernel_size * kernel_size, 
                                     kernel_size=kernel_size, padding=padding)
        # 掩码分支 (1 * k * k)
        self.mask_conv = nn.Conv2d(dim, kernel_size * kernel_size, 
                                   kernel_size=kernel_size, padding=padding)
        
        self.dcn = ops.DeformConv2d(dim, dim, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        # 计算 DCNv2 所需的 offset 和 mask
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        
        x = self.dcn(x, offset, mask)
        x = self.norm(x)
        return self.act(x)
    
class DeformableMixer(nn.Module):
    def __init__(self, dim, kernel_size=3, padding=1):
        super().__init__()
        self.offset_conv = nn.Conv2d(dim, 2 * kernel_size * kernel_size, kernel_size=kernel_size, padding=padding)
        self.mask_conv = nn.Conv2d(dim, kernel_size * kernel_size, kernel_size=kernel_size, padding=padding)
        self.dcn = ops.DeformConv2d(dim, dim, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x_img = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        offset = self.offset_conv(x_img)
        mask = torch.sigmoid(self.mask_conv(x_img))
        
        out = self.dcn(x_img, offset, mask)
        out = out.permute(0, 2, 3, 1).reshape(B, L, C)
        return out 

# =====================================================
# Stable Spectral Gating (固定 16x32 频率分辨率)
# =====================================================

class SpectralGating(nn.Module):
    """
    适配 128x256 输入：
    Stage2 特征尺寸 = 16x32
    rfft 后尺寸 = 16x17
    """
    def __init__(self, dim, h=8, w=16):
        super().__init__()
        # 对应 Stage 3 分辨率 8x16 -> rfft 尺寸为 8x9
        self.h, self.w = h, w
        self.weight = nn.Parameter(torch.ones(dim, h, w // 2 + 1, 2))
        nn.init.truncated_normal_(self.weight, std=.02)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x_img = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        fft = torch.fft.rfft2(x_img, norm="ortho")
        weight = torch.view_as_complex(self.weight)
        fft = fft * weight.unsqueeze(0)

        x = torch.fft.irfft2(fft, s=(H, W), norm="ortho")
        x = x.permute(0, 2, 3, 1).reshape(B, L, C)
        return x


# =====================================================
# Multi-Head Attention
# =====================================================

class GlobalAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# =====================================================
# Transformer Block
# =====================================================

class SpecBlock(nn.Module):
    def __init__(self, dim, num_heads, mixer_type='window', window_size=8, drop_path=0.):
        super().__init__()

        self.mixer_type = mixer_type
        self.window_size = window_size

        self.cpe = ConvCPE(dim)
        self.norm1 = nn.LayerNorm(dim)

        if mixer_type == 'window':
            self.mixer = GlobalAttention(dim, num_heads)
        elif mixer_type == 'global':
            self.mixer = GlobalAttention(dim, num_heads)
        else:
            self.mixer = SpectralGating(dim)

        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, H, W):
        x = self.cpe(x, H, W)
        shortcut = x
        x = self.norm1(x)

        if self.mixer_type == 'window':
            B, L, C = x.shape
            ws = self.window_size

            pad_r = (ws - W % ws) % ws
            pad_b = (ws - H % ws) % ws

            x_img = x.view(B, H, W, C)

            if pad_r or pad_b:
                x_img = F.pad(x_img, (0, 0, 0, pad_r, 0, pad_b))

            Hp, Wp = x_img.shape[1], x_img.shape[2]

            x_win = x_img.view(
                B,
                Hp // ws, ws,
                Wp // ws, ws,
                C
            ).permute(0, 1, 3, 2, 4, 5).contiguous()

            x_win = x_win.view(-1, ws * ws, C)
            attn = self.mixer(x_win)

            attn = attn.view(
                B,
                Hp // ws,
                Wp // ws,
                ws, ws, C
            ).permute(0, 1, 3, 2, 4, 5).contiguous()

            x = attn.view(B, Hp, Wp, C)

            if pad_r or pad_b:
                x = x[:, :H, :W, :]

            x = x.reshape(B, L, C)

        elif self.mixer_type == 'global':
            x = self.mixer(x)
        else:
            x = self.mixer(x, H, W)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# =====================================================
# Backbone
# =====================================================
@register()
class SpecDCNBackbone(nn.Module):
    def __init__(self, in_chans=1, drop_path_rate=0.2):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(in_chans, 64)

        depths = [2, 2, 4, 2]
        dpr = torch.linspace(0, drop_path_rate, sum(depths))
        idx = 0

        # Stage 1: 32x64 (Local-DCN)
        self.stage1 = nn.ModuleList([
            SpecBlock(64, 2, 'deform', dpr[idx+i]) for i in range(depths[0])
        ])
        idx += depths[0]
        self.down1 = Downsample(64)

        # Stage 2: 16x32 (Local-DCN)
        self.stage2 = nn.ModuleList([
            SpecBlock(128, 4, 'deform', dpr[idx+i]) for i in range(depths[1])
        ])
        idx += depths[1]
        self.down2 = Downsample(128)

        # Stage 3: 8x16 (Spectral/FFT - 捕捉谐波)
        # 这是 RT-DETR 的第一个输出尺度 S3
        self.stage3 = nn.ModuleList([
            SpecBlock(256, 8, 'spectral', dpr[idx+i]) for i in range(depths[2])
        ])
        idx += depths[2]
        self.down3 = Downsample(256)

        # Stage 4: 4x8 (Global Attention - 最终分类语义)
        # 这是 RT-DETR 的 S5
        self.stage4 = nn.ModuleList([
            SpecBlock(512, 16, 'global', dpr[idx+i]) for i in range(depths[3])
        ])

    def forward(self, x):
        outs = []
        x, H, W = self.patch_embed(x)

        # Stage 1
        for blk in self.stage1: x = blk(x, H, W)
        
        # Stage 2 -> 16x32
        x, H, W = self.down1(x, H, W)
        for blk in self.stage2: x = blk(x, H, W)
        outs.append(x.view(x.size(0), H, W, -1).permute(0,3,1,2).contiguous())

        # Stage 3 -> 8x16 (FFT 核心层)
        x, H, W = self.down2(x, H, W)
        for blk in self.stage3: x = blk(x, H, W)
        outs.append(x.view(x.size(0), H, W, -1).permute(0,3,1,2).contiguous())

        # Stage 4 -> 4x8 (Global)
        x, H, W = self.down3(x, H, W)
        for blk in self.stage4: x = blk(x, H, W)
        outs.append(x.view(x.size(0), H, W, -1).permute(0,3,1,2).contiguous())

        return outs