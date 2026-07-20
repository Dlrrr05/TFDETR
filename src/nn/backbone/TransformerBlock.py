import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        attn_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool or key_padding_mask.dim() != 2:
                raise ValueError("key_padding_mask must be bool tensor of shape (B,N)")
            # nn.MultiheadAttention expects True = ignore
            attn_mask = ~key_padding_mask

        h = self.ln1(x)
        h, _ = self.mhsa(h, h, h, key_padding_mask=attn_mask, need_weights=False)
        x = x + self.drop1(h)

        x = x + self.ffn(self.ln2(x))
        x = self.ln_out(x)
        return x


# -------------------------
# Global Transformer Layer for TF map
# -------------------------
class GlobalTransformerLayer(nn.Module):
    """
    Input/Output: (B,H,W,D)
    Flatten all TF tokens -> global self-attention -> reshape back.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        ff_mult: int,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()
        self.block = TransformerBlock(
            d_model=d_model,
            nhead=nhead,
            ff_mult=ff_mult,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, D = x.shape
        x = x.reshape(B, H * W, D)   # (B,N,D), N=H*W
        x = self.block(x)
        x = x.reshape(B, H, W, D)
        return x