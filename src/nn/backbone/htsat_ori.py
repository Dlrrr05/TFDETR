import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ...core import register
from .layers import trunc_normal_, DropPath, to_2tuple

__all__ = ['EGHTSATBackbone']


class AdvancedWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)) 

        # 生成相对位置坐标
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij')) 
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        
        self.energy_gate = nn.Parameter(torch.zeros(1)) 
        self.band_gate = nn.Parameter(torch.zeros(1))

        # Band Prior
        size = window_size * window_size
        idx = torch.arange(window_size)
        diff = idx.unsqueeze(0) - idx.unsqueeze(1)
        gaussian = torch.exp(-(diff.float() ** 2) / (2 * (window_size / 4) ** 2))
        prior = gaussian.repeat_interleave(window_size, dim=0).repeat_interleave(window_size, dim=1)
        self.register_buffer("band_prior", prior.view(1, 1, size, size))

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, pre_norm_x=None):
        B, N, C = x.shape
        # 1. QKV 计算
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # -----------------------------------------------------------
        # 🛡️ 安全模式开启：强制 Attention 过程使用 FP32
        # -----------------------------------------------------------
        # 混合精度训练(AMP)下，点积容易溢出，转为 float32 计算更安全
        q = q.float()
        k = k.float()
        
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 2. Relative Position Bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0).float()

        # 3. Band Prior
        attn = attn + (torch.tanh(self.band_gate) * self.band_prior.float())

        # 4. Energy Bias (深度加固版)
        if pre_norm_x is not None:
            # 🛡️ 保护措施 A: 限制输入幅度
            # 在深层网络中 pre_norm_x 可能非常大，先做一个数值截断
            x_safe = torch.clamp(pre_norm_x, min=-1e4, max=1e4).float()
            
            # 计算能量
            energy = torch.norm(x_safe, dim=-1) + 1e-6
            
            # 🛡️ 保护措施 B: 对数平滑
            # 如果 energy 是 1000，平方后就是 1,000,000，容易炸。取 log 变平滑。
            # 这里的逻辑是：我们只关心相对能量大小，log 是单调变换，不影响 "谁大谁小"
            energy = torch.log1p(energy) 
            
            energy_bias = energy.unsqueeze(2) @ energy.unsqueeze(1)
            
            # 🛡️ 保护措施 C: 归一化限制在 [-1, 1] 之间
            # 这里的 dim 参数根据你的 PyTorch 版本可能需要调整，通常 normalize 支持多维
            # 简单粗暴的方法：除以最大值
            max_val = energy_bias.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-6
            energy_bias = energy_bias / max_val
            
            attn = attn + (torch.tanh(self.energy_gate) * energy_bias.unsqueeze(1))

        # 5. 防止 Softmax 前溢出
        attn = torch.clamp(attn, min=-100, max=100)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        # 转回原始精度 (例如 fp16) 与 v 进行计算
        attn = attn.to(v.dtype)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# =====================================================
# 2. Conv-Augmented EGBlock (The Solution to Isolation)
# =====================================================

class ConvEGBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AdvancedWindowAttention(dim, window_size=window_size, num_heads=num_heads)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        
        # C. 3x3 Depth-wise Conv: The Bridge between Windows (Critique Fix #1)
        # This allows info to leak to neighbors, solving the receptive field isolation
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def window_partition(self, x):
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
        return windows

    def window_reverse(self, windows, H, W):
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        x = windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x, H, W):
        # x: B, L, C
        B, L, C = x.shape
        shortcut = x
        
        # 1. Convolutional Position Encoding (Global Context Injection)
        # Reshape to Image -> Conv -> Flatten
        x_img = x.view(B, H, W, C).permute(0, 3, 1, 2) # B, C, H, W
        x_img = self.cpe(x_img) # 3x3 DW Conv
        x_img = x_img.permute(0, 2, 3, 1).view(B, L, C)
        x = x + x_img # Add Conv info to stream
        
        # 2. Window Attention
        x_reshaped = x.view(B, H, W, C)
        
        # Padding
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x_reshaped = F.pad(x_reshaped, (0, 0, 0, pad_r, 0, pad_b))
        
        _, Hp, Wp, _ = x_reshaped.shape
        
        # Partition
        x_windows = self.window_partition(x_reshaped) # B*nW, Ws, Ws, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Pre-norm for Energy
        pre_norm_windows = x_windows 
        
        # Attention
        attn_windows = self.attn(self.norm1(x_windows), pre_norm_x=pre_norm_windows)
        
        # Reverse
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x_reshaped = self.window_reverse(attn_windows, Hp, Wp)
        
        if pad_r > 0 or pad_b > 0:
            x_reshaped = x_reshaped[:, :H, :W, :].contiguous()
            
        x = x_reshaped.view(B, H * W, C)

        # Residual + DropPath
        x = shortcut + self.drop_path(x)
        
        # 3. FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# =====================================================
# Patch Merging (Standard)
# =====================================================
class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        pad_r = W % 2
        pad_b = H % 2
        x = x.view(B, H, W, C)
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape
        
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, Hp // 2, Wp // 2

# =====================================================
# Main Backbone (Large Config Optimized)
# =====================================================
@register()
class EGHTSATBackbone(nn.Module):
    __shared__ = ["in_chans"]

    def __init__(self,
                 in_chans=1,
                 embed_dim=192,            # Large Config
                 depths=(2, 2, 18, 2),     # Deep
                 num_heads=(6, 12, 24, 48),# Multi-head
                 window_size=8,
                 return_idx=(1, 2, 3),
                 drop_path_rate=0.2):      
        super().__init__()

        self.in_chans = in_chans
        self.return_idx = return_idx
        self.out_channels = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]
        self.out_strides = [4, 8, 16, 32]

        # ---------------------------------------------------------
        # 🛠️ 修复 1: 拆分 Patch Embed，将 Conv 和 Norm 分开定义
        # ---------------------------------------------------------
        # 1. 卷积层：负责下采样 (B, C_in, H, W) -> (B, C_out, H/4, W/4)
        self.patch_embed_conv = nn.Conv2d(in_chans, embed_dim, kernel_size=4, stride=4)
        
        # 2. 归一化层：将在 Permute 之后使用
        self.patch_embed_norm = nn.LayerNorm(embed_dim)

        # Stochastic Depth Decay Rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        dim = embed_dim
        idx = 0
        for i in range(4):
            # 注意：这里的 ConvEGBlock 需要引入之前提供的 AdvancedWindowAttention
            blocks = nn.ModuleList([
                ConvEGBlock(
                    dim=dim, 
                    num_heads=num_heads[i], 
                    window_size=window_size,
                    drop_path=dpr[idx + j]
                )
                for j in range(depths[i])
            ])
            idx += depths[i]
            self.stages.append(blocks)

            if i < 3:
                self.downsamples.append(PatchMerging(dim))
                dim *= 2
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        outputs = []
        if x.shape[1] != self.in_chans:
             if x.shape[1] == 3 and self.in_chans == 1:
                x = x.mean(dim=1, keepdim=True)
        
        # ---------------------------------------------------------
        # 🛠️ 修复 2: 手动控制前向传播的维度顺序
        # ---------------------------------------------------------
        # 1. 卷积: (B, 1, H, W) -> (B, 192, H/4, W/4)
        x = self.patch_embed_conv(x) 
        
        # 2. 维度置换: NCHW -> NHWC, 即 (B, H/4, W/4, 192)
        # 这样 Channel 维度到了最后，LayerNorm 才能正确工作
        x = x.permute(0, 2, 3, 1) 
        
        # 3. 归一化: 作用在最后一个维度 (192) 上
        x = self.patch_embed_norm(x)
        
        # 4. 展平与 Reshape: 准备进入 Transformer Block
        B, H, W, C = x.shape
        x = x.view(B, H * W, C) # (B, L, C)

        for i in range(4):
            for blk in self.stages[i]:
                x = blk(x, H, W)
            
            # RT-DETR 需要 NCHW 格式的特征图作为输出
            feat = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            
            if i in self.return_idx:
                outputs.append(feat)

            if i < 3:
                x, H, W = self.downsamples[i](x, H, W)

        return outputs