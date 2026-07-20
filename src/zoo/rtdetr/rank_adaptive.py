"""Adaptive local-rank modulation for boundary-aware regularization."""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .attention_ops import sample_anchor_positions
from .lowrank_regularizer import (
    compute_local_penalty,
    compute_local_rank_map,
    get_selected_level_attention_data,
)


class RankEMA(nn.Module):
    """Track an EMA-smoothed rank map after a warm-up period."""

    def __init__(
        self,
        beta: float = 0.9,
        warmup_steps: int = 0,
        warmup_epochs: float = 0.0,
        update_interval: int = 1,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.warmup_steps = int(warmup_steps)
        self.warmup_epochs = float(warmup_epochs)
        self.update_interval = max(int(update_interval), 1)
        self.register_buffer('ema_rank_map', torch.empty(0), persistent=False)
        self.initialized = False

    def _resolve_warmup_steps(self, steps_per_epoch: Optional[int]) -> int:
        if self.warmup_steps > 0:
            return self.warmup_steps

        if self.warmup_epochs > 0 and steps_per_epoch is not None and steps_per_epoch > 0:
            return int(math.ceil(self.warmup_epochs * steps_per_epoch))

        return 0

    def warmup_finished(
        self,
        global_step: Optional[int],
        epoch: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
    ) -> bool:
        warmup_steps = self._resolve_warmup_steps(steps_per_epoch)
        if warmup_steps <= 0:
            return True

        if global_step is not None:
            return global_step >= warmup_steps

        if epoch is not None and self.warmup_epochs > 0:
            return epoch >= self.warmup_epochs

        return False

    def should_update(self, global_step: Optional[int]) -> bool:
        if self.update_interval <= 1 or global_step is None:
            return True
        return (int(global_step) % self.update_interval) == 0

    def update(
        self,
        rank_map: torch.Tensor,
        global_step: Optional[int],
        epoch: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
    ) -> torch.Tensor:
        rank_map = rank_map.detach()
        if not self.warmup_finished(global_step, epoch=epoch, steps_per_epoch=steps_per_epoch):
            return rank_map

        if (not self.initialized) or self.ema_rank_map.shape != rank_map.shape or self.ema_rank_map.device != rank_map.device:
            self.ema_rank_map = rank_map.clone()
            self.initialized = True
        elif self.should_update(global_step):
            self.ema_rank_map = self.beta * self.ema_rank_map + (1.0 - self.beta) * rank_map

        return self.ema_rank_map


class AdaptiveLocalRankLoss(nn.Module):
    """Compute L_tf using EMA-smoothed rank maps and detached spatial gates."""

    def __init__(
        self,
        level_index: int = 0,
        r_loc: int = 1,
        radius_h: int = 4,
        radius_w: int = 4,
        num_anchors: int = 64,
        ema_beta: float = 0.9,
        warmup_steps: int = 0,
        warmup_epochs: float = 0.0,
        gate_gamma: float = 5.0,
        eps: float = 1e-6,
        level_reduce: str = 'sum',
        stop_gradient: bool = True,
        update_interval: int = 1,
    ) -> None:
        super().__init__()
        self.level_index = level_index
        self.r_loc = r_loc
        self.radius_h = radius_h
        self.radius_w = radius_w
        self.num_anchors = num_anchors
        self.gate_gamma = gate_gamma
        self.eps = eps
        self.level_reduce = level_reduce
        self.stop_gradient = bool(stop_gradient)
        self.update_interval = max(int(update_interval), 1)
        self.ema_beta = ema_beta
        self.warmup_steps = warmup_steps
        self.warmup_epochs = warmup_epochs
        self.rank_ema_modules = nn.ModuleDict()

    def _get_rank_ema(self, level_id: int) -> RankEMA:
        key = str(level_id)
        if key not in self.rank_ema_modules:
            self.rank_ema_modules[key] = RankEMA(
                beta=self.ema_beta,
                warmup_steps=self.warmup_steps,
                warmup_epochs=self.warmup_epochs,
                update_interval=self.update_interval,
            )
        return self.rank_ema_modules[key]

    def _compute_gate_map(self, ema_rank_map: torch.Tensor) -> torch.Tensor:
        mu_r = ema_rank_map.mean()
        sigma_r = ema_rank_map.std(unbiased=False)
        gate = torch.sigmoid(-self.gate_gamma * (ema_rank_map - mu_r) / (sigma_r + self.eps))
        return gate.detach() if self.stop_gradient else gate

    def forward(
        self,
        encoder_aux: Dict,
        global_step: Optional[int] = None,
        epoch: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        level_data = get_selected_level_attention_data(encoder_aux, self.level_index)
        first_attn_layers, _, _ = level_data[0]
        device = first_attn_layers[0].device
        zero = first_attn_layers[0].new_zeros(())

        loss_tf = zero
        rank_maps = {}
        ema_rank_maps = {}
        gate_maps = {}
        mean_ranks = []
        mean_ema_ranks = []
        mean_gates = []
        level_ids = []
        spatial_shapes = []
        anchor_counts = []

        for attn_layers, spatial_shape, level_id in level_data:
            height, width = spatial_shape
            level_ids.append(level_id)
            spatial_shapes.append(spatial_shape)

            rank_map = compute_local_rank_map(
                attn_layers=attn_layers,
                height=height,
                width=width,
                radius_h=self.radius_h,
                radius_w=self.radius_w,
                rank_r=self.r_loc,
            )
            rank_ema = self._get_rank_ema(level_id)
            ema_rank_map = rank_ema.update(
                rank_map,
                global_step=global_step,
                epoch=epoch,
                steps_per_epoch=steps_per_epoch,
            )

            if rank_ema.warmup_finished(global_step, epoch=epoch, steps_per_epoch=steps_per_epoch):
                gate_map = self._compute_gate_map(ema_rank_map)
            else:
                gate_map = torch.ones_like(rank_map)

            anchors = sample_anchor_positions(height, width, self.num_anchors, device=device)
            anchor_counts.append(len(anchors))
            if len(anchors) > 0:
                weighted_penalties = []
                for u, v in anchors:
                    local_penalty = compute_local_penalty(
                        attn_layers=attn_layers,
                        height=height,
                        width=width,
                        u=u,
                        v=v,
                        radius_h=self.radius_h,
                        radius_w=self.radius_w,
                        rank_r=self.r_loc,
                    )
                    weighted_penalties.append(gate_map[u, v] * local_penalty)
                loss_tf = loss_tf + torch.stack(weighted_penalties).mean()

            rank_maps[level_id] = rank_map
            ema_rank_maps[level_id] = ema_rank_map
            gate_maps[level_id] = gate_map
            mean_ranks.append(rank_map.mean())
            mean_ema_ranks.append(ema_rank_map.mean())
            mean_gates.append(gate_map.mean())

        num_levels = len(level_data)
        if self.level_reduce == 'mean' and num_levels > 0:
            loss_tf = loss_tf / num_levels
        elif self.level_reduce != 'sum':
            raise ValueError(f'Unsupported level_reduce: {self.level_reduce}')

        rank_map = next(iter(rank_maps.values())) if len(rank_maps) == 1 else None
        ema_rank_map = next(iter(ema_rank_maps.values())) if len(ema_rank_maps) == 1 else None
        gate_map = next(iter(gate_maps.values())) if len(gate_maps) == 1 else None

        return {
            'loss_tf': loss_tf,
            'rank_map': rank_map,
            'rank_maps': rank_maps,
            'ema_rank_map': ema_rank_map,
            'ema_rank_maps': ema_rank_maps,
            'gate_map': gate_map,
            'gate_maps': gate_maps,
            'mean_rank': torch.stack(mean_ranks).mean() if len(mean_ranks) > 0 else zero,
            'mean_ema_rank': torch.stack(mean_ema_ranks).mean() if len(mean_ema_ranks) > 0 else zero,
            'mean_gate': torch.stack(mean_gates).mean() if len(mean_gates) > 0 else zero,
            'spatial_shape': torch.tensor(spatial_shapes[0], device=device) if len(spatial_shapes) == 1 else None,
            'spatial_shapes': torch.tensor(spatial_shapes, device=device),
            'level_id': torch.tensor(level_ids[0], device=device) if len(level_ids) == 1 else None,
            'level_ids': torch.tensor(level_ids, device=device),
            'num_levels': torch.tensor(num_levels, device=device),
            'num_anchors': torch.tensor(sum(anchor_counts), device=device),
            'num_anchors_per_level': torch.tensor(anchor_counts, device=device),
        }
