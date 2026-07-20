import math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register


__all__ = ['GlobalTransformerNetBackbone']


# -------------------------
# 2D Patch Embedding (Freq-Time)
# -------------------------
class PatchEmbed2D(nn.Module):
    """
    Input:  x (B, 1, F, T)
    Output: y (B, H, W, D)
    """
    def __init__(self, in_chans: int, d_model: int,
                 patch_size: Tuple[int, int], stride: Tuple[int, int]):
        super().__init__()
        pf, pt = patch_size
        sf, st = stride

        pad_f = (pf - sf) // 2
        pad_t = (pt - st) // 2

        self.proj = nn.Conv2d(
            in_chans, d_model,
            kernel_size=(pf, pt),
            stride=(sf, st),
            padding=(pad_f, pad_t)
        )
        self.norm = nn.GroupNorm(1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,F,T)
        y = self.proj(x)          # (B,D,H,W)
        y = self.norm(y)
        y = F.silu(y)
        y = y.permute(0, 2, 3, 1).contiguous()  # (B,H,W,D)
        return y


# -------------------------
# 2D Sin-Cos Positional Encoding
# -------------------------
class SinCos2DPositionalEncoding(nn.Module):
    """
    Build 2D sinusoidal positional encoding for TF tokens.
    Output: (H, W, D)
    Requirement: d_model % 4 == 0
    """
    def __init__(self, d_model: int, temperature: float = 10000.0):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D sin-cos positional encoding"
        self.d_model = d_model
        self.temperature = temperature

    def forward(self, H: int, W: int, device=None, dtype=None) -> torch.Tensor:
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32

        y = torch.arange(H, device=device, dtype=torch.float32)
        x = torch.arange(W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")  # (H,W), (H,W)

        dim = self.d_model // 4
        omega = torch.arange(dim, device=device, dtype=torch.float32) / dim
        omega = 1.0 / (self.temperature ** omega)  # (dim,)

        out_y = yy[..., None] * omega  # (H,W,dim)
        out_x = xx[..., None] * omega  # (H,W,dim)

        pos = torch.cat(
            [torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)],
            dim=-1
        )  # (H,W,D)

        return pos.to(dtype=dtype)


# -------------------------
# Standard Transformer FFN
# -------------------------
class TransformerFFN(nn.Module):
    def __init__(self, d_model: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        d_ff = d_model * ff_mult
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------
# Standard Transformer Block (Pre-LN)
# -------------------------
class TransformerBlock(nn.Module):
    """
    Standard Transformer block:
      x = x + MHSA(LN(x))
      x = x + FFN(LN(x))
      x = LN_out(x)
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = TransformerFFN(d_model, ff_mult=ff_mult, dropout=dropout)

        self.ln_out = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: (B,N,D)
        key_padding_mask: (B,N) bool, True=keep, False=pad
        """
        mha_kpm = None
        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool or key_padding_mask.dim() != 2:
                raise ValueError("key_padding_mask must be bool tensor of shape (B,N)")
            # MultiheadAttention expects True = mask out
            mha_kpm = ~key_padding_mask

        h = self.ln1(x)
        h, _ = self.mhsa(h, h, h, key_padding_mask=mha_kpm, need_weights=False)
        x = x + self.drop1(h)

        x = x + self.ffn(self.ln2(x))
        x = self.ln_out(x)
        return x


# -------------------------
# Global Transformer Layer
# -------------------------
class GlobalTransformerLayer(nn.Module):
    """
    Apply global self-attention over all TF tokens.
    Input/Output: (B,H,W,D)
    """
    def __init__(self, d_model: int, nhead: int, ff_mult: int,
                 dropout: float, attn_dropout: float):
        super().__init__()
        self.block = TransformerBlock(
            d_model=d_model,
            nhead=nhead,
            ff_mult=ff_mult,
            dropout=dropout,
            attn_dropout=attn_dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,H,W,D)
        B, H, W, D = x.shape
        x = x.reshape(B, H * W, D)   # (B, H*W, D)
        x = self.block(x)            # (B, H*W, D)
        x = x.reshape(B, H, W, D)    # (B, H, W, D)
        return x


# -------------------------
# Frequency Pyramid Fusion
# -------------------------
class FreqPyramidFusion(nn.Module):
    """
    Build a small pyramid along frequency (H axis) and fuse back to base H.
    Input/Output: (B,H,W,D)
    """
    def __init__(self, d_model: int, num_scales: int = 3, down_stride: int = 2):
        super().__init__()
        self.num_scales = num_scales
        self.down_stride = down_stride

        self.down = nn.ModuleList()
        for _ in range(num_scales - 1):
            self.down.append(
                nn.Sequential(
                    nn.Conv2d(d_model, d_model, kernel_size=(3, 1), stride=(down_stride, 1), padding=(1, 0)),
                    nn.GroupNorm(1, d_model),
                    nn.SiLU(),
                )
            )

        self.out = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=1),
                nn.GroupNorm(1, d_model),
                nn.SiLU(),
            ) for _ in range(num_scales)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,H,W,D)
        B, H, W, D = x.shape
        x0 = x.permute(0, 3, 1, 2).contiguous()  # (B,D,H,W)

        feats = [x0]
        cur = x0
        for down in self.down:
            cur = down(cur)
            feats.append(cur)

        out_sum = torch.zeros_like(x0)
        for i, f in enumerate(feats):
            y = self.out[i](f)
            if y.shape[2] != H:
                y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
            out_sum = out_sum + y
        out_sum = out_sum / self.num_scales

        return out_sum.permute(0, 2, 3, 1).contiguous()  # (B,H,W,D)


# ---------------------------------
# Dynamic depth router
# ---------------------------------
class DynamicDepthRouter(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int,
        num_layers: int,
        num_levels: int,
        router_hidden: int = 256,
        temperature: float = 1.0,
        topk: int = 0,
    ):
        super().__init__()
        self.L = num_layers
        self.K = num_levels
        self.temperature = temperature
        self.topk = topk
        self.proj = nn.ModuleList([nn.Linear(d_in, d_model) for _ in range(num_layers)])
        self.routers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(num_layers * d_model),
                nn.Linear(num_layers * d_model, router_hidden),
                nn.SiLU(),
                nn.Linear(router_hidden, num_layers),
            )
            for _ in range(num_levels)
        ])

    def forward(self, layer_feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        assert len(layer_feats) == self.L
        proj_feats = [self.proj[i](layer_feats[i]) for i in range(self.L)]  # (B,N,D)
        cat = torch.cat([F.layer_norm(p, (p.shape[-1],)) for p in proj_feats], dim=-1)  # (B,N,L*D)

        outs, alphas = [], []
        for k in range(self.K):
            logits = self.routers[k](cat)  # (B,N,L)
            if self.topk and self.topk < self.L:
                topv, topi = torch.topk(logits, k=self.topk, dim=-1)
                masked = torch.full_like(logits, float("-inf"))
                masked.scatter_(-1, topi, topv)
                logits = masked
            alpha = F.softmax(logits / max(self.temperature, 1e-6), dim=-1)

            y = 0.0
            for l in range(self.L):
                y = y + proj_feats[l] * alpha[..., l:l+1]
            outs.append(y)
            alphas.append(alpha.unsqueeze(2))

        alphas = torch.cat(alphas, dim=2)  # (B,N,K,L)
        return outs, alphas


# -------------------------
# Frequency Router
# -------------------------
class FreqRouter(nn.Module):
    """
    Input:  tf_feat (B,H,W,D)
    Output:
      memories: list length K, each (B,W,D)
      alphas:   (B,W,K,H)
    """
    def __init__(self, d_model: int, num_levels: int, hidden: int = 256, temperature: float = 1.0, topk: int = 0):
        super().__init__()
        self.K = num_levels
        self.temperature = temperature
        self.topk = topk

        self.routers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1)
            )
            for _ in range(num_levels)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # x: (B,H,W,D)
        B, H, W, D = x.shape
        x_whd = x.permute(0, 2, 1, 3).contiguous()  # (B,W,H,D)

        memories = []
        alpha_all = []

        for k in range(self.K):
            scores = self.routers[k](x_whd).squeeze(-1)  # (B,W,H)

            if self.topk and self.topk < H:
                topv, topi = torch.topk(scores, k=self.topk, dim=-1)
                masked = torch.full_like(scores, float("-inf"))
                masked.scatter_(-1, topi, topv)
                scores = masked

            alpha = F.softmax(scores / max(self.temperature, 1e-6), dim=-1)  # (B,W,H)

            mem = (x_whd * alpha.unsqueeze(-1)).sum(dim=2)  # (B,W,D)
            memories.append(mem)
            alpha_all.append(alpha.unsqueeze(2))

        alpha_all = torch.cat(alpha_all, dim=2)  # (B,W,K,H)
        return memories, alpha_all


# -------------------------
# Backbone: 2D TF + Global Transformer + DepthRouter + FreqFusion
# -------------------------
@register()
class GlobalTransformerNetBackbone(nn.Module):
    """
    Global Transformer backbone that outputs a 3-level frequency pyramid (time W unchanged),
    compatible with RT-DETR HybridEncoder style input: List[Tensor(B,C,H_i,W)].

    Input:
      x: (B, 1, F=128, T)  (log-mel)
    """
    def __init__(self, cfg=None, **kwargs):
        super().__init__()

        if cfg is None:
            cfg = {}
        if isinstance(cfg, dict) and "backbone" in cfg:
            b = cfg["backbone"]
        else:
            b = {**cfg, **kwargs}

        self.d_model = int(b["d_model"])
        self.depth = int(b["depth"])
        self.num_levels = int(b.get("num_levels", 4))

        # 1) 2D patch embed
        pf = int(b.get("patch_f", 8))
        pt = int(b.get("patch_t", 7))
        sf = int(b.get("stride_f", 4))
        st = int(b.get("stride_t", 1))
        self.stride_t = st

        self.patch = PatchEmbed2D(
            in_chans=int(b.get("in_chans", 1)),
            d_model=self.d_model,
            patch_size=(pf, pt),
            stride=(sf, st),
        )

        # 2) 2D positional encoding
        self.pos2d = SinCos2DPositionalEncoding(self.d_model)

        # 3) global transformer layers
        self.layers = nn.ModuleList([
            GlobalTransformerLayer(
                d_model=self.d_model,
                nhead=int(b.get("num_heads", 8)),
                ff_mult=int(b.get("ff_mult", 4)),
                dropout=float(b.get("dropout", 0.1)),
                attn_dropout=float(b.get("attn_dropout", 0.1)),
            )
            for _ in range(self.depth)
        ])

        # 4) dynamic depth router
        self.use_depth_router = bool(b.get("use_depth_router", True))
        if self.use_depth_router:
            self.depth_router = DynamicDepthRouter(
                d_in=self.d_model,
                d_model=self.d_model,
                num_layers=self.depth,
                num_levels=self.num_levels,
                router_hidden=int(b.get("router_hidden", 256)),
                temperature=float(b.get("temperature", 1.0)),
                topk=int(b.get("topk", 0)),
            )

        # 5) multi-frequency fusion
        self.use_freq_pyramid = bool(b.get("use_freq_pyramid", True))
        if self.use_freq_pyramid:
            self.freq_fpn = FreqPyramidFusion(
                d_model=self.d_model,
                num_scales=int(b.get("freq_scales", 3)),
                down_stride=int(b.get("freq_down_stride", 2)),
            )

        # 6) dynamic important bands router
        self.use_freq_router = bool(b.get("use_freq_router", True))
        if self.use_freq_router:
            self.freq_router = FreqRouter(
                d_model=self.d_model,
                num_levels=self.num_levels,
                hidden=int(b.get("freq_router_hidden", 256)),
                temperature=float(b.get("freq_temperature", 1.0)),
                topk=int(b.get("freq_topk", 0)),
            )

        # 7) 3-level frequency-downsample pyramid
        self.pool_f_2 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=False)
        self.pool_f_4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=False),
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=False),
        )

        self.last_tf: Optional[torch.Tensor] = None
        self.last_freq_alphas: Optional[torch.Tensor] = None
        self.last_memories: Optional[List[torch.Tensor]] = None

        self.out_channels = [self.d_model, self.d_model, self.d_model]
        self.out_strides = [self.stride_t, self.stride_t, self.stride_t]

    @staticmethod
    def _pad_freq_to_even(x_bchw: torch.Tensor, factor: int) -> torch.Tensor:
        B, C, H, W = x_bchw.shape
        rem = H % factor
        if rem == 0:
            return x_bchw
        pad = factor - rem
        return F.pad(x_bchw, (0, 0, 0, pad), mode="replicate")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns:
          feats: [f0, f1, f2]
            f0: (B,D,H,   W)
            f1: (B,D,H/2, W)
            f2: (B,D,H/4, W)
        """
        # 1) patch embed -> (B,H,W,D)
        tf = self.patch(x)
        B, H, W, D = tf.shape

        # 2) add 2D positional encoding
        pos = self.pos2d(H, W, device=tf.device, dtype=tf.dtype).unsqueeze(0)  # (1,H,W,D)
        tf = tf + pos

        # 3) global transformer stack
        layer_outs = []
        for blk in self.layers:
            tf = blk(tf)
            if self.use_depth_router:
                layer_outs.append(tf.reshape(B, H * W, D))

        # 4) dynamic depth fusion
        if self.use_depth_router:
            fused_tokens_list, depth_alphas = self.depth_router(layer_outs)
            tf_sum = 0.0
            for tok in fused_tokens_list:
                tf_sum = tf_sum + tok.reshape(B, H, W, D)
            tf = tf_sum / max(len(fused_tokens_list), 1)

        # 5) in-resolution multi-frequency fusion
        if self.use_freq_pyramid:
            tf = self.freq_fpn(tf)

        # 6) side outputs
        if self.use_freq_router:
            memories, freq_alphas = self.freq_router(tf)
            self.last_memories = memories
            self.last_freq_alphas = freq_alphas
        else:
            self.last_memories = None
            self.last_freq_alphas = None

        self.last_tf = tf  # (B,H,W,D)

        # 7) build 3-level feature list for encoder
        feat0 = tf.permute(0, 3, 1, 2).contiguous()  # (B,D,H,W)

        feat0_p2 = self._pad_freq_to_even(feat0, factor=2)
        feat0_p4 = self._pad_freq_to_even(feat0, factor=4)

        feat1 = self.pool_f_2(feat0_p2)  # (B,D,H/2,W)
        feat2 = self.pool_f_4(feat0_p4)  # (B,D,H/4,W)

        feats = [feat0, feat1, feat2]
        return feats