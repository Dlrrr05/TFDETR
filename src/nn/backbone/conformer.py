import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Relative Positional Encoding (Transformer-XL style)
# -------------------------
class RelPositionalEncoding(nn.Module):
    """
    Sinusoidal relative positional encoding used in Transformer-XL / Conformer.
    Returns pe of shape (2T-1, D) for a given input length T.
    """
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.register_buffer("pe", self._build_pe(max_len), persistent=False)

    def _build_pe(self, length: int) -> torch.Tensor:
        # positions: [-L+1, ..., 0, ..., L-1] => (2L-1)
        pos = torch.arange(-(length - 1), length, dtype=torch.float32).unsqueeze(1)  # (2L-1, 1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / self.d_model)
        )  # (D/2,)
        pe = torch.zeros((2 * length - 1, self.d_model), dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe  # (2L-1, D)

    def forward(self, T: int, device=None, dtype=None) -> torch.Tensor:
        if T > self.max_len:
            # extend if needed
            self.max_len = int(T * 1.2)
            self.pe = self._build_pe(self.max_len).to(self.pe.device)
        pe = self.pe[: 2 * T - 1]  # (2T-1, D)
        if device is not None:
            pe = pe.to(device)
        if dtype is not None:
            pe = pe.to(dtype=dtype)
        return pe


def rel_shift(x: torch.Tensor) -> torch.Tensor:
    """
    Efficient relative shift trick from Transformer-XL.
    x: (B, H, T, 2T-1)
    returns: (B, H, T, T)
    """
    B, H, T, _ = x.size()
    # pad on last dim
    x = F.pad(x, (1, 0))  # (B,H,T,2T)
    x = x.view(B, H, -1, T)  # (B,H,2T, T)
    x = x[:, :, 1:, :]  # (B,H,2T-1, T)
    x = x.view(B, H, T, -1)  # (B,H,T,2T-1)
    return x[:, :, :, :T]  # (B,H,T,T)


class RelPositionMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with Transformer-XL relative positional encoding,
    as used in Conformer (paper uses Transformer-XL style RPE).
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = self.d_head ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.r_proj = nn.Linear(d_model, d_model, bias=False)  # project rel-pos to heads space

        # global biases (u,v) in Transformer-XL
        self.u = nn.Parameter(torch.zeros(nhead, self.d_head))
        self.v = nn.Parameter(torch.zeros(nhead, self.d_head))

        self.out = nn.Linear(d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, r: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B,T,D)
        r: (2T-1,D) relative pos embeddings
        attn_mask: optional boolean mask (B,T) or additive mask (B,1,T,T)
        """
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.nhead, self.d_head)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each (B,T,H,dh)
        q = q.transpose(1, 2)  # (B,H,T,dh)
        k = k.transpose(1, 2)  # (B,H,T,dh)
        v = v.transpose(1, 2)  # (B,H,T,dh)

        r = self.r_proj(r).view(2 * T - 1, self.nhead, self.d_head).permute(1, 0, 2)  # (H,2T-1,dh)

        # content-based term: (q+u) @ k^T
        qu = (q + self.u.unsqueeze(1)) * self.scale  # (B,H,T,dh)
        AC = torch.matmul(qu, k.transpose(-2, -1))  # (B,H,T,T)

        # position-based term: (q+v) @ r^T with rel_shift
        qv = (q + self.v.unsqueeze(1)) * self.scale  # (B,H,T,dh)
        BD = torch.matmul(qv, r.transpose(-2, -1))  # (B,H,T,2T-1)
        BD = rel_shift(BD)  # (B,H,T,T)

        attn = AC + BD  # (B,H,T,T)

        if attn_mask is not None:
            # support:
            # - boolean mask (B,T): True means keep; False means mask out
            # - additive mask (B,1,T,T) or (B,H,T,T) with -inf on masked
            if attn_mask.dtype == torch.bool and attn_mask.dim() == 2:
                # key padding mask style
                # mask positions where attn_mask is False
                key_mask = ~attn_mask  # True means masked
                attn = attn.masked_fill(key_mask[:, None, None, :], float("-inf"))
            else:
                attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        y = torch.matmul(attn, v)  # (B,H,T,dh)
        y = y.transpose(1, 2).contiguous().view(B, T, D)  # (B,T,D)
        y = self.out(y)
        y = self.drop(y)
        return y


# -------------------------
# Standard Conformer Conv Module
# -------------------------
class ConformerConvModule(nn.Module):
    """
    Conformer convolution module (standard):
      LN -> PWConv -> GLU -> DWConv -> BN -> SiLU -> PWConv -> Dropout
    """
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd for 'same' padding"
        self.ln = nn.LayerNorm(d_model)

        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)

        self.dw = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()

        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        x = self.ln(x).transpose(1, 2)  # (B,D,T)
        x = self.pw1(x)                 # (B,2D,T)
        x = self.glu(x)                 # (B,D,T)
        x = self.dw(x)                  # (B,D,T)
        x = self.bn(x)
        x = self.act(x)
        x = self.pw2(x)                 # (B,D,T)
        x = self.drop(x)
        return x.transpose(1, 2)        # (B,T,D)


# -------------------------
# Conformer FFN (Macaron style)
# -------------------------
class ConformerFFN(nn.Module):
    def __init__(self, d_model: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        d_ff = d_model * ff_mult
        self.ln = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.ln(x))


# -------------------------
# Standard ConformerBlock (with RelPos MHSA)
# -------------------------
class ConformerBlock(nn.Module):
    """
    A standard Conformer block:
      x = x + 0.5*FFN(x)
      x = x + MHSA_relpos(LN(x))
      x = x + ConvModule(x)
      x = x + 0.5*FFN(x)
      x = LN_out(x)
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        ff_mult: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        pe_max_len: int = 10000,
    ):
        super().__init__()
        self.ffn1 = ConformerFFN(d_model, ff_mult, dropout)
        self.ffn2 = ConformerFFN(d_model, ff_mult, dropout)

        self.ln_mhsa = nn.LayerNorm(d_model)
        self.rpe = RelPositionalEncoding(d_model, max_len=pe_max_len)
        self.mhsa = RelPositionMultiHeadAttention(d_model, nhead, dropout=attn_dropout)

        self.conv = ConformerConvModule(d_model, kernel_size=conv_kernel, dropout=dropout)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B,T,D)
        key_padding_mask: (B,T) bool, True=keep, False=pad (optional)
        """
        x = x + 0.5 * self.ffn1(x)

        h = self.ln_mhsa(x)
        T = h.size(1)
        r = self.rpe(T, device=h.device, dtype=h.dtype)  # (2T-1, D)
        x = x + self.mhsa(h, r, attn_mask=key_padding_mask)

        x = x + self.conv(x)

        x = x + 0.5 * self.ffn2(x)
        x = self.ln_out(x)
        return x