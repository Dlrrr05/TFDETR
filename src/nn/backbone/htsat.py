import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ...core import register
from .layers import trunc_normal_, DropPath, to_2tuple

__all__ = ['EGHTSATBackbone']


class EfficientWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        
        # 1. 标准 QKV
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 🌟 创新 1: SwinV2 风格的 Cosine Attention 缩放因子 (替代暴力的 float 和 clamp)
        # 初始化 logit_scale 使得初始 attention 不会太平缓
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        
        # 🌟 创新 2: O(N) 复杂度的能量门控 (Energy Gate)
        # 替代原先 O(N^2) 的注意力偏置矩阵，改为对 Value (V) 进行空间/通道级别的激发
        self.energy_mlp = nn.Sequential(
            nn.Linear(dim, dim // 4, bias=False),
            nn.GELU(),
            nn.Linear(dim // 4, dim, bias=False),
            nn.Sigmoid() # 输出 0~1 的门控系数
        )
        # 初始时让能量门控失效(全1)，保证训练稳定性
        self.energy_gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        B, N, C = x.shape
        
        # 1. QKV 拆分
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # -----------------------------------------------------------
        # 🛡️ 优雅的安全模式：Cosine Attention (原生防止 FP16 溢出)
        # -----------------------------------------------------------
        # 对 Q 和 K 进行 L2 归一化，点积结果天然在 [-1, 1] 之间！不再需要 .float() 和 clamp
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # 限制 logit_scale 最大值防止指数爆炸
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100) 
        attn = (q @ k.transpose(-2, -1)) * logit_scale
        
        # -----------------------------------------------------------
        # ⚡ 高效的 Energy Gate (O(N) 复杂度)
        # -----------------------------------------------------------
        # 直接计算输入的能量特征，并生成对 Value 的调制权重
        energy_weight = self.energy_mlp(x) # (B, N, C)
        energy_weight = 1.0 + self.energy_gamma * (energy_weight - 0.5)
        energy_weight = energy_weight.view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # 调制 Value
        v = v * energy_weight

        # 2. Attention 计算 (废弃了查表相对位置和 Band Prior，交给外部的 3x3 卷积处理)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 3. 聚合与输出
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# =====================================================
# 2. Conv-Augmented EGBlock (完美的局部-全局协同)
# =====================================================
class ConvGateBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        
        # 🌟 核心保留：3x3 DW 卷积作为跨窗口通信的桥梁 (CPE)
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientWindowAttention(dim, window_size=window_size, num_heads=num_heads)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        
        # 1. 先进行 CPE (局部卷积注入) -> 隐式位置编码与局部特征融合
        x_img = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous() 
        x_img = self.cpe(x_img) 
        x_img = x_img.permute(0, 2, 3, 1).view(B, L, C)
        x = x + x_img # 残差连接
        
        # 2. Window Attention 准备
        x_norm = self.norm1(x)
        x_reshaped = x_norm.view(B, H, W, C)
        
        # Padding (如果分辨率不能被 window_size 整除)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x_reshaped = F.pad(x_reshaped, (0, 0, 0, pad_r, 0, pad_b))
        
        _, Hp, Wp, _ = x_reshaped.shape
        
        # Partition
        x_windows = x_reshaped.view(B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        
        # 🌟 Attention (变得非常干净！)
        attn_windows = self.attn(x_windows)
        
        # Reverse
        attn_windows = attn_windows.view(B, Hp // self.window_size, Wp // self.window_size, self.window_size, self.window_size, C)
        x_reshaped = attn_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)
        
        if pad_r > 0 or pad_b > 0:
            x_reshaped = x_reshaped[:, :H, :W, :].contiguous()
            
        x_attn = x_reshaped.view(B, H * W, C)

        # 3. Residual & FFN
        x = shortcut + self.drop_path(x_attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

# =====================================================
# Patch Merging (保持不变，标准实现)
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
    def __init__(self,
                 in_chans=1, # 假设是医疗或遥感单通道，或者可以改为3
                 embed_dim=192,            
                 depths=(2, 2, 18, 2),     
                 num_heads=(6, 12, 24, 48),
                 window_size=8,
                 return_idx=(1, 2, 3),
                 drop_path_rate=0.3):      
        super().__init__()

        self.in_chans = in_chans
        self.return_idx = return_idx
        
        # 🌟 创新 3: Overlapping Patch Embedding (ConvNeXt 风格)
        # 7x7 卷积带 padding，取代原来的 4x4 non-overlapping 卷积。能极大地保留边界信息。
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=7, stride=4, padding=3),
            nn.LayerNorm(embed_dim) # 注意：这里如果遇到 PyTorch 版本问题，可以在 Forward 里手动 Permute 再 Norm
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        dim = embed_dim
        idx = 0
        for i in range(4):
            blocks = nn.ModuleList([
                ConvGateBlock(
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
                self.downsamples.append(PatchMerging(dim)) # 记得加上这段定义
                dim *= 2
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02) # 使用 trunc_normal 更有利于 Transformer 架构

    def forward(self, x):
        outputs = []
        if x.shape[1] != self.in_chans:
             if x.shape[1] == 3 and self.in_chans == 1:
                x = x.mean(dim=1, keepdim=True)
        
        # 1. 现代化 Patch Embed
        x = self.patch_embed[0](x) # Conv: (B, C, H/4, W/4)
        x = x.permute(0, 2, 3, 1)  # (B, H/4, W/4, C)
        x = self.patch_embed[1](x) # LayerNorm
        
        B, H, W, C = x.shape
        x = x.view(B, H * W, C)

        # 2. 逐层前向传播
        for i in range(4):
            for blk in self.stages[i]:
                x = blk(x, H, W)
            
            # RT-DETR/YOLO 所需特征图格式
            feat = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            
            if i in self.return_idx:
                outputs.append(feat)

            if i < 3:
                x, H, W = self.downsamples[i](x, H, W)

        return outputs