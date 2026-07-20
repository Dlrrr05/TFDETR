import math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer import ConformerBlock  # 你的标准 ConformerBlock（建议带相对位置）
from ...core import register


__all__ = ['ConformerNetBackbone']

# -------------------------
# 2D Patch Embedding (Freq-Time)
# -------------------------
class PatchEmbed2D(nn.Module):
    """
    Input:  x (B, 1, F, T)  (e.g., F=128 mel bins)
    Output: y (B, H, W, D)
      H = floor(F / stride_f)
      W ~ ceil(T / stride_t) (with padding)
    """
    def __init__(self, in_chans: int, d_model: int,
                 patch_size: Tuple[int, int], stride: Tuple[int, int]):
        super().__init__()
        pf, pt = patch_size
        sf, st = stride
        # "same-ish" padding for stable alignment
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
# Axial Conformer (time then freq)
# -------------------------
class AxialConformerLayer(nn.Module):
    """
    Apply Conformer along time axis for each freq row, then along freq axis for each time column.
    Input/Output: (B, H, W, D)
    """
    def __init__(self, d_model: int, nhead: int, ff_mult: int, conv_kernel: int,
                 dropout: float, attn_dropout: float):
        super().__init__()
        self.time_blk = ConformerBlock(
            d_model=d_model, nhead=nhead, ff_mult=ff_mult,
            conv_kernel=conv_kernel, dropout=dropout, attn_dropout=attn_dropout
        )
        self.freq_blk = ConformerBlock(
            d_model=d_model, nhead=nhead, ff_mult=ff_mult,
            conv_kernel=conv_kernel, dropout=dropout, attn_dropout=attn_dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,H,W,D)
        B, H, W, D = x.shape

        # ---- time conformer: treat each freq row as a sequence of length W
        xt = x.view(B * H, W, D)          # (B*H, W, D)
        xt = self.time_blk(xt)            # (B*H, W, D)
        x = xt.view(B, H, W, D)

        # ---- freq conformer: treat each time column as a sequence of length H
        xf = x.permute(0, 2, 1, 3).contiguous().view(B * W, H, D)  # (B*W, H, D)
        xf = self.freq_blk(xf)            # (B*W, H, D)
        x = xf.view(B, W, H, D).permute(0, 2, 1, 3).contiguous()   # (B,H,W,D)

        return x


# -------------------------
# Frequency Pyramid Fusion (multi-frequency resolution)
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
            cur = down(cur)  # (B,D,H/2,W)
            feats.append(cur)

        # bring all to base H and average
        out_sum = torch.zeros_like(x0)
        for i, f in enumerate(feats):
            y = self.out[i](f)  # (B,D,Hi,W)
            if y.shape[2] != H:
                y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
            out_sum = out_sum + y
        out_sum = out_sum / self.num_scales

        return out_sum.permute(0, 2, 3, 1).contiguous()  # (B,H,W,D)


# ---------------------------------
# Dynamic depth router (same as you)
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
        # layer_feats: list length L, each (B,N,d_in)
        assert len(layer_feats) == self.L
        proj_feats = [self.proj[i](layer_feats[i]) for i in range(self.L)]  # (B,N,D)
        cat = torch.cat([F.layer_norm(p, (p.shape[-1],)) for p in proj_feats], dim=-1)  # (B,N,L*D)
        B, N, _ = cat.shape

        outs, alphas = [], []
        for k in range(self.K):
            logits = self.routers[k](cat)  # (B,N,L)
            if self.topk and self.topk < self.L:
                topv, topi = torch.topk(logits, k=self.topk, dim=-1)
                masked = torch.full_like(logits, float("-inf"))
                masked.scatter_(-1, topi, topv)
                logits = masked
            alpha = F.softmax(logits / max(self.temperature, 1e-6), dim=-1)  # (B,N,L)

            y = 0.0
            for l in range(self.L):
                y = y + proj_feats[l] * alpha[..., l:l+1]
            outs.append(y)               # (B,N,D)
            alphas.append(alpha.unsqueeze(2))  # (B,N,1,L)

        alphas = torch.cat(alphas, dim=2)  # (B,N,K,L)
        return outs, alphas


# -------------------------
# Frequency Router: dynamic important bands + K memories for DETR
# -------------------------
class FreqRouter(nn.Module):
    """
    Input:  tf_feat (B,H,W,D)
    Output:
      memories: list length K, each (B,W,D)  # DETR memory (time sequence)
      alphas:   (B,W,K,H)  # per time step and level, importance over freq bins
    """
    def __init__(self, d_model: int, num_levels: int, hidden: int = 256, temperature: float = 1.0, topk: int = 0):
        super().__init__()
        self.K = num_levels
        self.temperature = temperature
        self.topk = topk

        # routers operate on per-(time,freq) tokens -> produce per-time logits over freq
        # we summarize each freq bin token at time t as x[:,f,t,:]
        self.routers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1)  # score for this freq at this time
            )
            for _ in range(num_levels)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # x: (B,H,W,D)
        B, H, W, D = x.shape
        # reshape to (B,W,H,D) for convenience
        x_whd = x.permute(0, 2, 1, 3).contiguous()  # (B,W,H,D)

        memories = []
        alpha_all = []

        for k in range(self.K):
            # compute freq scores per time: (B,W,H,1) -> (B,W,H)
            scores = self.routers[k](x_whd).squeeze(-1)

            if self.topk and self.topk < H:
                topv, topi = torch.topk(scores, k=self.topk, dim=-1)
                masked = torch.full_like(scores, float("-inf"))
                masked.scatter_(-1, topi, topv)
                scores = masked

            alpha = F.softmax(scores / max(self.temperature, 1e-6), dim=-1)  # (B,W,H)

            # freq-weighted sum: (B,W,H,D) * (B,W,H,1) -> (B,W,D)
            mem = (x_whd * alpha.unsqueeze(-1)).sum(dim=2)
            memories.append(mem)
            alpha_all.append(alpha.unsqueeze(2))  # (B,W,1,H)

        alpha_all = torch.cat(alpha_all, dim=2)  # (B,W,K,H)
        return memories, alpha_all


# -------------------------
# Backbone: 2D TF + Axial Conformer + DepthRouter + FreqFusion + DETR memory
# -------------------------
@register()
class ConformerNetBackbone(nn.Module):
    """
    Conformer backbone that outputs a 3-level frequency pyramid (time W unchanged),
    compatible with RT-DETR HybridEncoder style input: List[Tensor(B,C,H_i,W)].

    Input:
      x: (B, 1, F=128, T)  (log-mel)

    Internal:
      tf: (B, H, W, D) after PatchEmbed2D + AxialConformer (+ optional depth/freq fusion)

    Output (to encoder):
      feats: list of 3 tensors:
        feats[0]: (B, D, H,   W)     # highest freq resolution
        feats[1]: (B, D, H/2, W)     # downsample freq by 2
        feats[2]: (B, D, H/4, W)     # downsample freq by 4

    Side outputs (not returned to encoder):
      self.last_tf:         (B, H, W, D)
      self.last_freq_alphas:(B, W, K, H)  (if freq_router exists/used)
      self.last_memories:   list length K, each (B, W, D)
    """
    def __init__(self, cfg=None, **kwargs):
        super().__init__()

        # 兼容两种配置风格：
        # 1) ConformerNetBackbone: { cfg: { backbone: {...} } }
        # 2) ConformerNetBackbone: { in_chans: 1, d_model: ..., ... }
        if cfg is None:
            cfg = {}
        if isinstance(cfg, dict) and "backbone" in cfg:
            b = cfg["backbone"]
        else:
            # kwargs 扁平形式
            b = {**cfg, **kwargs}

        self.d_model = int(b["d_model"])
        self.depth = int(b["depth"])
        self.num_levels = int(b.get("num_levels", 4))

        # 1) 2D patch embed
        pf = int(b.get("patch_f", 8))
        pt = int(b.get("patch_t", 7))
        sf = int(b.get("stride_f", 4))
        st = int(b.get("stride_t", 1))   # keep time precision
        self.stride_t = st

        self.patch = PatchEmbed2D(
            in_chans=int(b.get("in_chans", 1)),
            d_model=self.d_model,
            patch_size=(pf, pt),
            stride=(sf, st),
        )

        # 2) axial conformer layers (2D)
        self.layers = nn.ModuleList([
            AxialConformerLayer(
                d_model=self.d_model,
                nhead=int(b.get("num_heads", 8)),
                ff_mult=int(b.get("ff_mult", 4)),
                conv_kernel=int(b.get("conv_kernel", 31)),
                dropout=float(b.get("dropout", 0.1)),
                attn_dropout=float(b.get("attn_dropout", 0.1)),
            )
            for _ in range(self.depth)
        ])

        # 3) dynamic depth router (optional)
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

        # 4) multi-frequency fusion (within same resolution) (optional)
        self.use_freq_pyramid = bool(b.get("use_freq_pyramid", True))
        if self.use_freq_pyramid:
            self.freq_fpn = FreqPyramidFusion(
                d_model=self.d_model,
                num_scales=int(b.get("freq_scales", 3)),
                down_stride=int(b.get("freq_down_stride", 2)),
            )

        # 5) dynamic important bands router (optional, for DETR memories / interpretability)
        self.use_freq_router = bool(b.get("use_freq_router", True))
        if self.use_freq_router:
            self.freq_router = FreqRouter(
                d_model=self.d_model,
                num_levels=self.num_levels,
                hidden=int(b.get("freq_router_hidden", 256)),
                temperature=float(b.get("freq_temperature", 1.0)),
                topk=int(b.get("freq_topk", 0)),
            )

        # ---- NEW: 3-level frequency-downsample pyramid (time unchanged) ----
        # Use pooling along frequency axis only: kernel=(2,1), stride=(2,1)
        self.pool_f_2 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=False)
        self.pool_f_4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=False),
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=False),
        )

        # Side outputs cache
        self.last_tf: Optional[torch.Tensor] = None
        self.last_freq_alphas: Optional[torch.Tensor] = None
        self.last_memories: Optional[List[torch.Tensor]] = None

        # For compatibility / metadata
        self.out_channels = [self.d_model, self.d_model, self.d_model]
        # Here "stride" refers to time stride in frames; frequency stride isn't used by DETR
        self.out_strides = [self.stride_t, self.stride_t, self.stride_t]

    @staticmethod
    def _pad_freq_to_even(x_bchw: torch.Tensor, factor: int) -> torch.Tensor:
        """
        Pad frequency dimension (H) so it's divisible by `factor`, keeping time (W) unchanged.
        x_bchw: (B,C,H,W)
        """
        B, C, H, W = x_bchw.shape
        rem = H % factor
        if rem == 0:
            return x_bchw
        pad = factor - rem
        # pad format: (pad_left_w, pad_right_w, pad_left_h, pad_right_h)
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

        # 2) axial conformer, optionally collect layer outputs for depth routing
        layer_outs = []
        for blk in self.layers:
            tf = blk(tf)
            if self.use_depth_router:
                B, H, W, D = tf.shape
                layer_outs.append(tf.reshape(B, H * W, D))  # flatten tokens

        # 3) dynamic depth fusion (optional)
        if self.use_depth_router:
            fused_tokens_list, depth_alphas = self.depth_router(layer_outs)  # list of (B,N,D)
            B, H, W, D = tf.shape
            tf_sum = 0.0
            for tok in fused_tokens_list:
                tf_sum = tf_sum + tok.reshape(B, H, W, D)
            tf = tf_sum / max(len(fused_tokens_list), 1)

        # 4) in-resolution multi-frequency fusion (optional)
        if self.use_freq_pyramid:
            tf = self.freq_fpn(tf)  # (B,H,W,D)

        # ---- Side outputs: freq router (optional) ----
        if self.use_freq_router:
            memories, freq_alphas = self.freq_router(tf)  # list (B,W,D), and (B,W,K,H)
            self.last_memories = memories
            self.last_freq_alphas = freq_alphas
        else:
            self.last_memories = None
            self.last_freq_alphas = None

        self.last_tf = tf  # (B,H,W,D)

        # ---- Build 3-level feature list for encoder ----
        # Convert to (B,C,H,W)
        feat0 = tf.permute(0, 3, 1, 2).contiguous()  # (B,D,H,W)

        # Ensure divisible for pooling to avoid odd-size issues
        feat0_p2 = self._pad_freq_to_even(feat0, factor=2)
        feat0_p4 = self._pad_freq_to_even(feat0, factor=4)

        feat1 = self.pool_f_2(feat0_p2)  # (B,D,H/2,W)
        feat2 = self.pool_f_4(feat0_p4)  # (B,D,H/4,W)

        feats = [feat0, feat1, feat2]
        return feats