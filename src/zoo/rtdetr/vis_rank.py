"""Rank-map utilities derived from encoder attention."""

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from .attention_ops import attn_to_4d, compute_freq_marginal, compute_time_marginal
from .geman import truncated_geman_spectral_profile
from .lowrank_regularizer import compute_local_rank_map, get_selected_level_attention_data


LevelSelector = Union[int, Sequence[int], str, None]


def _reduce_level_tensors(level_tensors, reduce: str = 'sum') -> torch.Tensor:
    if len(level_tensors) == 0:
        raise ValueError('At least one level tensor is required.')

    stacked = torch.stack(level_tensors, dim=0)
    if reduce == 'sum':
        return stacked.sum(dim=0)
    if reduce == 'mean':
        return stacked.mean(dim=0)
    raise ValueError(f'Unsupported level reduction: {reduce}')


def _aggregate_1d_profiles(level_profiles, reduce: str = 'sum') -> torch.Tensor:
    if len(level_profiles) == 0:
        raise ValueError('At least one level profile is required.')

    target_length = max(profile.shape[-1] for profile in level_profiles)
    resized = []
    for profile in level_profiles:
        if profile.shape[-1] == target_length:
            resized.append(profile)
            continue
        resized.append(
            F.interpolate(
                profile.unsqueeze(1),
                size=target_length,
                mode='linear',
                align_corners=False,
            ).squeeze(1)
        )

    return _reduce_level_tensors(resized, reduce=reduce)


def _resolve_target_size(rank_maps: Dict[int, torch.Tensor], target_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if target_size is not None:
        return int(target_size[0]), int(target_size[1])

    shapes = [tuple(rank_map.shape[-2:]) for rank_map in rank_maps.values()]
    return max(shapes, key=lambda shape: shape[0] * shape[1])


def aggregate_rank_maps(
    rank_maps: Dict[int, torch.Tensor],
    target_size: Optional[Tuple[int, int]] = None,
    reduce: str = 'mean',
    interpolation: str = 'bilinear',
) -> torch.Tensor:
    """Resize multi-level 2D rank maps to a common grid and aggregate them."""
    if len(rank_maps) == 0:
        raise ValueError('rank_maps must be non-empty.')

    target_h, target_w = _resolve_target_size(rank_maps, target_size)
    resized = []
    for rank_map in rank_maps.values():
        if tuple(rank_map.shape[-2:]) == (target_h, target_w):
            resized.append(rank_map)
            continue

        if interpolation in ('linear', 'bilinear', 'bicubic', 'trilinear'):
            resized_rank_map = F.interpolate(
                rank_map.unsqueeze(1),
                size=(target_h, target_w),
                mode=interpolation,
                align_corners=False,
            ).squeeze(1)
        else:
            resized_rank_map = F.interpolate(
                rank_map.unsqueeze(1),
                size=(target_h, target_w),
                mode=interpolation,
            ).squeeze(1)
        resized.append(resized_rank_map)

    return _reduce_level_tensors(resized, reduce=reduce)


def make_temporal_rank_map(
    encoder_aux: Dict,
    level_index: LevelSelector = 0,
    rank_r: int = 1,
    level_reduce: str = 'sum',
) -> Dict[str, object]:
    """
    Build temporal rank profiles from S^tau using tail-singular-vector attribution.

    The returned profile has shape [B, T].
    """
    per_level = {}
    level_profiles = []
    level_ids = []

    for attn_layers, spatial_shape, level_id in get_selected_level_attention_data(encoder_aux, level_index):
        height, width = spatial_shape
        profile = None
        for attn in attn_layers:
            attn4d = attn_to_4d(attn, height, width)
            s_tau = compute_time_marginal(attn4d)
            layer_profile = truncated_geman_spectral_profile(s_tau, rank_r, reduction='symmetric').mean(dim=1)
            profile = layer_profile if profile is None else profile + layer_profile

        per_level[level_id] = profile
        level_profiles.append(profile)
        level_ids.append(level_id)

    aggregate = _aggregate_1d_profiles(level_profiles, reduce=level_reduce)
    return {
        'rank_map': aggregate,
        'per_level': per_level,
        'level_ids': level_ids,
    }


def make_frequency_rank_map(
    encoder_aux: Dict,
    level_index: LevelSelector = 0,
    rank_r: int = 1,
    level_reduce: str = 'sum',
) -> Dict[str, object]:
    """
    Build frequency rank profiles from S^nu using tail-singular-vector attribution.

    The returned profile has shape [B, F].
    """
    per_level = {}
    level_profiles = []
    level_ids = []

    for attn_layers, spatial_shape, level_id in get_selected_level_attention_data(encoder_aux, level_index):
        height, width = spatial_shape
        profile = None
        for attn in attn_layers:
            attn4d = attn_to_4d(attn, height, width)
            s_nu = compute_freq_marginal(attn4d)
            layer_profile = truncated_geman_spectral_profile(s_nu, rank_r, reduction='symmetric').mean(dim=1)
            profile = layer_profile if profile is None else profile + layer_profile

        per_level[level_id] = profile
        level_profiles.append(profile)
        level_ids.append(level_id)

    aggregate = _aggregate_1d_profiles(level_profiles, reduce=level_reduce)
    return {
        'rank_map': aggregate,
        'per_level': per_level,
        'level_ids': level_ids,
    }


def make_2d_rank_map(
    encoder_aux: Dict,
    level_index: LevelSelector = 0,
    rank_r: int = 1,
    radius_h: int = 4,
    radius_w: int = 4,
    level_reduce: str = 'mean',
    target_size: Optional[Tuple[int, int]] = None,
    interpolation: str = 'bilinear',
) -> Dict[str, object]:
    """
    Build local 2D rank maps R_tf(u, v) from selected encoder levels.

    The returned aggregated map has shape [B, H_ref, W_ref].
    """
    per_level = {}
    spatial_shapes = {}
    level_ids = []

    for attn_layers, spatial_shape, level_id in get_selected_level_attention_data(encoder_aux, level_index):
        height, width = spatial_shape
        rank_map = compute_local_rank_map(
            attn_layers=attn_layers,
            height=height,
            width=width,
            radius_h=radius_h,
            radius_w=radius_w,
            rank_r=rank_r,
            reduce_heads=True,
            reduce_batch=False,
        )
        per_level[level_id] = rank_map
        spatial_shapes[level_id] = spatial_shape
        level_ids.append(level_id)

    aggregate = aggregate_rank_maps(
        per_level,
        target_size=target_size,
        reduce=level_reduce,
        interpolation=interpolation,
    )
    return {
        'rank_map': aggregate,
        'per_level': per_level,
        'spatial_shapes': spatial_shapes,
        'level_ids': level_ids,
    }


def _box_quality_single(
    rank_map: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    image_size: Union[Tuple[int, int], torch.Tensor],
) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.new_zeros(boxes_xyxy.shape[:-1])

    if isinstance(image_size, torch.Tensor):
        image_h = float(image_size[0].item())
        image_w = float(image_size[1].item())
    else:
        image_h = float(image_size[0])
        image_w = float(image_size[1])

    map_h, map_w = rank_map.shape[-2:]
    scale_x = map_w / max(image_w, 1.0)
    scale_y = map_h / max(image_h, 1.0)

    qualities = []
    for box in boxes_xyxy:
        x0, y0, x1, y1 = box.tolist()
        gx0 = max(min(int(math.floor(x0 * scale_x)), map_w - 1), 0)
        gy0 = max(min(int(math.floor(y0 * scale_y)), map_h - 1), 0)
        gx1 = max(min(int(math.ceil(x1 * scale_x)) - 1, map_w - 1), gx0)
        gy1 = max(min(int(math.ceil(y1 * scale_y)) - 1, map_h - 1), gy0)
        qualities.append(rank_map[gy0:gy1 + 1, gx0:gx1 + 1].mean())

    return torch.stack(qualities).to(dtype=rank_map.dtype)


def box_rank_quality(
    rank_map: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    image_size: Union[Tuple[int, int], torch.Tensor],
) -> torch.Tensor:
    """Compute \u0304R(b) by averaging the 2D rank map inside each predicted box."""
    if rank_map.dim() == 2:
        return _box_quality_single(rank_map, boxes_xyxy, image_size)

    if rank_map.dim() != 3 or boxes_xyxy.dim() != 3:
        raise ValueError('Expected rank_map [B, H, W] and boxes_xyxy [B, Q, 4] for batched inputs.')

    if not isinstance(image_size, torch.Tensor) or image_size.dim() != 2:
        raise ValueError('Batched box_rank_quality expects image_size as [B, 2].')

    qualities = []
    for b in range(rank_map.shape[0]):
        qualities.append(_box_quality_single(rank_map[b], boxes_xyxy[b], image_size[b]))
    return torch.stack(qualities, dim=0)


def score_boxes_with_rank(
    scores: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    rank_map: torch.Tensor,
    image_size: Union[Tuple[int, int], torch.Tensor],
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply \u007ep_q = p_q (1 + alpha * \u0304R(b_q))."""
    box_quality = box_rank_quality(rank_map, boxes_xyxy, image_size).to(dtype=scores.dtype)
    rescored = scores * (1.0 + float(alpha) * box_quality)
    return rescored, box_quality
