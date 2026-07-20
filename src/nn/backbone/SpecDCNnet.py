import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from typing import Optional

from ...core import register
from .layers import trunc_normal_, DropPath, to_2tuple

__all__ = ['SpecDCNBackbone']

# =====================================================
# 1. 基础组件：Overlap Patch Embedding & Downsample
# =====================================================
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=64):
        super().__init__()
        # 128x256 -> 32x64
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=7, stride=4, padding=3)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(dim * 2)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x_img = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x_img = self.conv(x_img)
        B, C, H, W = x_img.shape
        x = x_img.flatten(2).transpose(1, 2)
        return self.norm(x), H, W

# =====================================================
# 2. 核心 Mixer：Deformable (DCNv2)
# =====================================================
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
        return out.permute(0, 2, 3, 1).reshape(B, L, C)

# =====================================================
# 3. 核心 Mixer：Spectral Gating (FFT)
# =====================================================
class SpectralGating(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        # 针对输入尺寸 h, w 进行参数初始化
        self.h = h
        self.w_fft = w // 2 + 1
        self.weight = nn.Parameter(torch.ones(dim, self.h, self.w_fft, 2))
        trunc_normal_(self.weight, std=.02) # 修正：使用 import 的函数

    def forward(self, x, H, W):
        B, L, C = x.shape
        x_img = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        # 实数 FFT
        fft = torch.fft.rfft2(x_img, norm="ortho")
        
        # 频域相乘
        weight = torch.view_as_complex(self.weight)
        fft = fft * weight.unsqueeze(0)

        # 逆 FFT
        x = torch.fft.irfft2(fft, s=(H, W), norm="ortho")
        return x.permute(0, 2, 3, 1).reshape(B, L, C)

# =====================================================
# 4. 辅助组件：CPE & Global Attention
# =====================================================
class ConvCPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x_img = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        out = self.dwconv(x_img)
        return x + out.permute(0, 2, 3, 1).reshape(B, L, C)

class GlobalAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

# =====================================================
# 5. 通用 SpecBlock
# =====================================================
class SpecBlock(nn.Module):
    def __init__(self, dim, num_heads, mixer_type='deform', grid_size=(8, 16), drop_path=0.):
        super().__init__()
        self.mixer_type = mixer_type
        self.cpe = ConvCPE(dim)
        self.norm1 = nn.LayerNorm(dim)

        if mixer_type == 'deform':
            self.mixer = DeformableMixer(dim)
        elif mixer_type == 'spectral':
            self.mixer = SpectralGating(dim, h=grid_size[0], w=grid_size[1])
        elif mixer_type == 'global':
            self.mixer = GlobalAttention(dim, num_heads)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
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

        if self.mixer_type == 'global':
            x = self.mixer(x)
        else:
            x = self.mixer(x, H, W)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(shortcut + self.drop_path(x)))) # 残差连接
        return x

# =====================================================
# 6. 最终 Backbone：SpecDCNBackbone
# =====================================================
@register()
class SpecDCNBackbone(nn.Module):
    def __init__(self, in_chans=3, drop_path_rate=0.2):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(in_chans, 64)

        depths = [2, 2, 4, 2]
        dpr = torch.linspace(0, drop_path_rate, sum(depths))
        idx = 0

        # Stage 1: 32x64 (DCNv2)
        self.stage1 = nn.ModuleList([
            SpecBlock(64, 2, 'deform', drop_path=dpr[idx+i]) for i in range(depths[0])
        ])
        idx += depths[0]
        self.down1 = Downsample(64)

        # Stage 2: 16x32 (DCNv2)
        # RT-DETR S3 输出
        self.stage2 = nn.ModuleList([
            SpecBlock(128, 4, 'deform', drop_path=dpr[idx+i]) for i in range(depths[1])
        ])
        idx += depths[1]
        self.down2 = Downsample(128)

        # Stage 3: 8x16 (Spectral/FFT)
        # RT-DETR S4 输出
        self.stage3 = nn.ModuleList([
            SpecBlock(256, 8, 'spectral', grid_size=(8, 16), drop_path=dpr[idx+i]) for i in range(depths[2])
        ])
        idx += depths[2]
        self.down3 = Downsample(256)

        # Stage 4: 4x8 (Global Attention)
        # RT-DETR S5 输出
        self.stage4 = nn.ModuleList([
            SpecBlock(512, 16, 'global', drop_path=dpr[idx+i]) for i in range(depths[3])
        ])

    def forward(self, x):
        outs = []
        x, H, W = self.patch_embed(x)

        # Stage 1
        for blk in self.stage1:
            x = blk(x, H, W)
        
        # Stage 2 (S3 Output: 16x32)
        x, H, W = self.down1(x, H, W)
        for blk in self.stage2:
            x = blk(x, H, W)
        outs.append(x.view(x.size(0), H, W, -1).permute(0, 3, 1, 2).contiguous())

        # Stage 3 (S4 Output: 8x16)
        x, H, W = self.down2(x, H, W)
        for blk in self.stage3:
            x = blk(x, H, W)
        outs.append(x.view(x.size(0), H, W, -1).permute(0, 3, 1, 2).contiguous())

        # Stage 4 (S5 Output: 4x8)
        x, H, W = self.down3(x, H, W)
        for blk in self.stage4:
            x = blk(x, H, W)
        outs.append(x.view(x.size(0), H, W, -1).permute(0, 3, 1, 2).contiguous())

        return outs