"""Three-level low-rank regularization on encoder attention."""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .attention_ops import (
    attn_to_4d,
    compute_freq_marginal,
    compute_time_marginal,
    extract_local_block,
    sample_anchor_positions,
)
from .geman import truncated_geman_penalty


LevelSelector = Union[int, Sequence[int], str, None]


def resolve_level_positions(encoder_aux: Dict, level_index: LevelSelector) -> List[int]:
    """Resolve one or more encoder levels by list position or real level id."""
    attn_levels = encoder_aux.get('enc_attn_list', None)
    if not attn_levels:
        return []

    level_ids = encoder_aux.get('enc_level_indices', list(range(len(attn_levels))))
    if level_index is None:
        return [0]

    if isinstance(level_index, str):
        if level_index.lower() == 'all':
            return list(range(len(attn_levels)))
        raise ValueError(f'Unsupported level selector: {level_index}')

    if isinstance(level_index, Sequence) and not isinstance(level_index, (str, bytes)):
        ordered: List[int] = []
        seen = set()
        for item in level_index:
            for pos in resolve_level_positions(encoder_aux, item):
                if pos not in seen:
                    ordered.append(pos)
                    seen.add(pos)
        return ordered

    if level_index in level_ids:
        return [level_ids.index(level_index)]

    if isinstance(level_index, int) and -len(attn_levels) <= level_index < len(attn_levels):
        return [level_index % len(attn_levels)]

    return []


def resolve_level_position(encoder_aux: Dict, level_index: int) -> Optional[int]:
    """Backward-compatible single-level resolver."""
    positions = resolve_level_positions(encoder_aux, level_index)
    if len(positions) == 0:
        return None
    return positions[0]


def get_selected_level_attention_data(
    encoder_aux: Dict,
    level_index: LevelSelector,
) -> List[Tuple[Sequence[torch.Tensor], Tuple[int, int], int]]:
    """Fetch attention layers and spatial metadata for one or more encoder levels."""
    level_positions = resolve_level_positions(encoder_aux, level_index)
    if len(level_positions) == 0:
        raise ValueError(f'Unable to resolve encoder level(s) {level_index}.')

    attn_levels = encoder_aux['enc_attn_list']
    spatial_shapes = encoder_aux['enc_spatial_shapes']
    level_ids = encoder_aux.get('enc_level_indices', list(range(len(attn_levels))))

    return [
        (attn_levels[level_pos], tuple(spatial_shapes[level_pos]), level_ids[level_pos])
        for level_pos in level_positions
    ]


def get_level_attention_data(encoder_aux: Dict, level_index: int) -> Tuple[Sequence[torch.Tensor], Tuple[int, int], int]:
    """Fetch attention layers and spatial metadata for a single encoder level."""
    level_data = get_selected_level_attention_data(encoder_aux, level_index)
    if len(level_data) != 1:
        raise ValueError(f'Expected a single encoder level, got {len(level_data)} from selector {level_index}.')
    return level_data[0]


def compute_local_penalty(
    attn_layers: Sequence[torch.Tensor],
    height: int,
    width: int,
    u: int,
    v: int,
    radius_h: int,
    radius_w: int,
    rank_r: int,
    reduce_heads: bool = True,
    reduce_batch: bool = True,
) -> torch.Tensor:
    """Compute sum_l phi_r(B^{(l)}_{u,v}) with configurable reduction."""
    if len(attn_layers) == 0:
        raise ValueError("attn_layers must be non-empty.")

    total = None
    for attn in attn_layers:
        block = extract_local_block(attn, height, width, u, v, radius_h, radius_w)
        penalty = truncated_geman_penalty(block, rank_r)
        total = penalty if total is None else total + penalty

    if reduce_heads:
        total = total.mean(dim=-1)
    if reduce_batch:
        total = total.mean(dim=0)
    return total


def compute_local_rank_map(
    attn_layers: Sequence[torch.Tensor],
    height: int,
    width: int,
    radius_h: int,
    radius_w: int,
    rank_r: int,
    reduce_heads: bool = True,
    reduce_batch: bool = True,
) -> torch.Tensor:
    """Compute R_tf(u,v) on the full 2D grid."""
    if len(attn_layers) == 0:
        raise ValueError("attn_layers must be non-empty.")

    template = compute_local_penalty(
        attn_layers=attn_layers,
        height=height,
        width=width,
        u=0,
        v=0,
        radius_h=radius_h,
        radius_w=radius_w,
        rank_r=rank_r,
        reduce_heads=reduce_heads,
        reduce_batch=False,
    )
    rank_map = template.new_zeros(*template.shape, height, width)
    for u in range(height):
        for v in range(width):
            rank_map[..., u, v] = compute_local_penalty(
                attn_layers=attn_layers,
                height=height,
                width=width,
                u=u,
                v=v,
                radius_h=radius_h,
                radius_w=radius_w,
                rank_r=rank_r,
                reduce_heads=reduce_heads,
                reduce_batch=False,
            )
    if reduce_batch:
        rank_map = rank_map.mean(dim=0)
    return rank_map


class ThreeLevelLowRankRegularizer(nn.Module):
    """Compute L_tau, L_nu and L_loc from encoder self-attention."""

    def __init__(
        self,
        level_index: int = 0,
        r_tau: int = 1,
        r_nu: int = 1,
        r_loc: int = 1,
        radius_h: int = 4,
        radius_w: int = 4,
        num_anchors: int = 64,
        return_rank_map: bool = False,
        level_reduce: str = 'sum',
    ) -> None:
        super().__init__()
        self.level_index = level_index
        self.r_tau = r_tau
        self.r_nu = r_nu
        self.r_loc = r_loc
        self.radius_h = radius_h
        self.radius_w = radius_w
        self.num_anchors = num_anchors
        self.return_rank_map = return_rank_map
        self.level_reduce = level_reduce

    def forward(
        self,
        encoder_aux: Dict,
        compute_tau: bool = True,
        compute_nu: bool = True,
        compute_local: bool = True,
    ) -> Dict[str, torch.Tensor]:
        level_data = get_selected_level_attention_data(encoder_aux, self.level_index)
        first_attn_layers, _, _ = level_data[0]
        zero = first_attn_layers[0].new_zeros(())
        device = first_attn_layers[0].device

        loss_tau = zero
        loss_nu = zero
        loss_loc = zero
        sampled_anchor_counts = []
        level_ids = []
        spatial_shapes = []
        num_layers = []
        rank_maps: Dict[int, torch.Tensor] = {}

        for attn_layers, spatial_shape, level_id in level_data:
            height, width = spatial_shape
            level_ids.append(level_id)
            spatial_shapes.append(spatial_shape)
            num_layers.append(len(attn_layers))

            if compute_tau or compute_nu:
                for attn in attn_layers:
                    attn4d = attn_to_4d(attn, height, width)
                    if compute_tau:
                        s_tau = compute_time_marginal(attn4d)
                        loss_tau = loss_tau + truncated_geman_penalty(s_tau, self.r_tau).mean()
                    if compute_nu:
                        s_nu = compute_freq_marginal(attn4d)
                        loss_nu = loss_nu + truncated_geman_penalty(s_nu, self.r_nu).mean()

            if compute_local:
                sampled_anchors = sample_anchor_positions(height, width, self.num_anchors, device=device)
                sampled_anchor_counts.append(len(sampled_anchors))
                if len(sampled_anchors) > 0:
                    penalties = []
                    for u, v in sampled_anchors:
                        penalties.append(
                            compute_local_penalty(
                                attn_layers=attn_layers,
                                height=height,
                                width=width,
                                u=u,
                                v=v,
                                radius_h=self.radius_h,
                                radius_w=self.radius_w,
                                rank_r=self.r_loc,
                            )
                        )
                    loss_loc = loss_loc + torch.stack(penalties).mean()
            else:
                sampled_anchor_counts.append(0)

            if self.return_rank_map:
                rank_maps[level_id] = compute_local_rank_map(
                    attn_layers=attn_layers,
                    height=height,
                    width=width,
                    radius_h=self.radius_h,
                    radius_w=self.radius_w,
                    rank_r=self.r_loc,
                )

        num_levels = len(level_data)
        if self.level_reduce == 'mean' and num_levels > 0:
            if compute_tau:
                loss_tau = loss_tau / num_levels
            if compute_nu:
                loss_nu = loss_nu / num_levels
            if compute_local:
                loss_loc = loss_loc / num_levels
        elif self.level_reduce != 'sum':
            raise ValueError(f'Unsupported level_reduce: {self.level_reduce}')

        rank_map = None
        if len(rank_maps) == 1:
            rank_map = next(iter(rank_maps.values()))

        return {
            'loss_tau': loss_tau,
            'loss_nu': loss_nu,
            'loss_loc': loss_loc,
            'rank_map': rank_map,
            'rank_maps': rank_maps,
            'spatial_shape': torch.tensor(spatial_shapes[0], device=device) if len(spatial_shapes) == 1 else None,
            'spatial_shapes': torch.tensor(spatial_shapes, device=device),
            'level_id': torch.tensor(level_ids[0], device=device) if len(level_ids) == 1 else None,
            'level_ids': torch.tensor(level_ids, device=device),
            'num_layers': torch.tensor(num_layers[0], device=device) if len(num_layers) == 1 else None,
            'num_layers_per_level': torch.tensor(num_layers, device=device),
            'num_levels': torch.tensor(num_levels, device=device),
            'num_anchors': torch.tensor(sum(sampled_anchor_counts), device=device),
            'num_anchors_per_level': torch.tensor(sampled_anchor_counts, device=device),
        }
