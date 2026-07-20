"""Attention geometry utilities for structured encoder regularization."""

from typing import List, Tuple, Union

import torch


def flatten_index(u: Union[torch.Tensor, int], v: Union[torch.Tensor, int], width: int) -> torch.Tensor:
    """Map 2D coordinates to flattened token indices."""
    return torch.as_tensor(u) * width + torch.as_tensor(v)


def unflatten_index(index: Union[torch.Tensor, int], width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`flatten_index`."""
    index = torch.as_tensor(index)
    return index // width, index % width


def attn_to_4d(attn: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Reshape token attention [..., HW, HW] to [..., H, W, H, W]."""
    expected_tokens = height * width
    if attn.shape[-2:] != (expected_tokens, expected_tokens):
        raise ValueError(
            f"Attention shape {tuple(attn.shape[-2:])} does not match {(expected_tokens, expected_tokens)}."
        )
    return attn.reshape(*attn.shape[:-2], height, width, height, width)


def compute_time_marginal(attn4d: torch.Tensor) -> torch.Tensor:
    """Compute S^tau by averaging over the vertical/frequency axes."""
    if attn4d.dim() < 4:
        raise ValueError("attn4d must end with [H, W, H, W].")
    return attn4d.mean(dim=(-4, -2))


def compute_freq_marginal(attn4d: torch.Tensor) -> torch.Tensor:
    """Compute S^nu by averaging over the horizontal/time axes."""
    if attn4d.dim() < 4:
        raise ValueError("attn4d must end with [H, W, H, W].")
    return attn4d.mean(dim=(-3, -1))


def get_local_neighborhood_indices(
    height: int,
    width: int,
    u: int,
    v: int,
    radius_h: int,
    radius_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Return flattened indices inside the local neighborhood Omega_{u,v}."""
    row_start = max(u - radius_h, 0)
    row_end = min(u + radius_h + 1, height)
    col_start = max(v - radius_w, 0)
    col_end = min(v + radius_w + 1, width)

    rows = torch.arange(row_start, row_end, device=device, dtype=torch.long)
    cols = torch.arange(col_start, col_end, device=device, dtype=torch.long)
    grid_u, grid_v = torch.meshgrid(rows, cols, indexing='ij')
    return flatten_index(grid_u.reshape(-1), grid_v.reshape(-1), width).to(device=device, dtype=torch.long)


def extract_local_block(
    attn: torch.Tensor,
    height: int,
    width: int,
    u: int,
    v: int,
    radius_h: int,
    radius_w: int,
) -> torch.Tensor:
    """Extract B_{u,v} from attention [..., HW, HW]."""
    local_indices = get_local_neighborhood_indices(
        height=height,
        width=width,
        u=u,
        v=v,
        radius_h=radius_h,
        radius_w=radius_w,
        device=attn.device,
    )
    block = attn.index_select(-2, local_indices)
    block = block.index_select(-1, local_indices)
    return block


def sample_anchor_positions(
    height: int,
    width: int,
    num_anchors: int,
    device: torch.device,
) -> List[Tuple[int, int]]:
    """Sample anchor positions P without replacement."""
    total = height * width
    if total == 0:
        return []

    if num_anchors <= 0 or num_anchors >= total:
        flat_indices = torch.arange(total, device=device, dtype=torch.long)
    else:
        flat_indices = torch.randperm(total, device=device, dtype=torch.long)[:num_anchors]

    us, vs = unflatten_index(flat_indices, width)
    return list(zip(us.tolist(), vs.tolist()))
