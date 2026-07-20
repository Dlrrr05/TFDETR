"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

RT-DETR v2 criterion with structured encoder-attention regularization.
"""

import copy

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .lowrank_regularizer import ThreeLevelLowRankRegularizer
from .rank_adaptive import AdaptiveLocalRankLoss
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ...core import register


@register()
class RTDETRCriterionv2(nn.Module):
    """Compute RT-DETR losses plus optional structured encoder regularization."""

    __share__ = ['num_classes']
    __inject__ = ['matcher']

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        boxes_weight_format=None,
        share_matched_indices=False,
        rank_regularizer_mode='three_level',
        rank_level_index=0,
        rank_r_tau=1,
        rank_r_nu=1,
        rank_r_loc=1,
        rank_window_size=9,
        rank_window_h=None,
        rank_window_w=None,
        rank_num_anchors=64,
        rank_return_maps=False,
        rank_level_reduce='sum',
        adaptive_ema_beta=0.9,
        adaptive_warmup_steps=0,
        adaptive_warmup_epochs=0.0,
        adaptive_gate_gamma=5.0,
        adaptive_eps=1e-6,
        adaptive_stop_gradient=True,
        adaptive_update_interval=1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.rank_regularizer_mode = rank_regularizer_mode

        if rank_window_h is None or rank_window_w is None:
            # The config usually passes a window size. Convert it to symmetric radii.
            rank_radius = max(int(rank_window_size), 1) // 2
            rank_window_h = rank_radius if rank_window_h is None else rank_window_h
            rank_window_w = rank_radius if rank_window_w is None else rank_window_w

        self.lowrank_reg = ThreeLevelLowRankRegularizer(
            level_index=rank_level_index,
            r_tau=rank_r_tau,
            r_nu=rank_r_nu,
            r_loc=rank_r_loc,
            radius_h=int(rank_window_h),
            radius_w=int(rank_window_w),
            num_anchors=rank_num_anchors,
            return_rank_map=rank_return_maps,
            level_reduce=rank_level_reduce,
        )
        self.adaptive_rank_reg = AdaptiveLocalRankLoss(
            level_index=rank_level_index,
            r_loc=rank_r_loc,
            radius_h=int(rank_window_h),
            radius_w=int(rank_window_w),
            num_anchors=rank_num_anchors,
            ema_beta=adaptive_ema_beta,
            warmup_steps=adaptive_warmup_steps,
            warmup_epochs=adaptive_warmup_epochs,
            gate_gamma=adaptive_gate_gamma,
            eps=adaptive_eps,
            level_reduce=rank_level_reduce,
            stop_gradient=adaptive_stop_gradient,
            update_interval=adaptive_update_interval,
        )

        self.rank_loss_names = {'tau', 'nu', 'loc', 'tf', 'rank', 'boundary'}
        self.latest_regularizer_metrics = {}

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        if len(idx[0]) > 0:
            target_classes_o = torch.cat(
                [t["labels"][J] for t, (_, J) in zip(targets, indices)],
                dim=0,
            )
        else:
            target_classes_o = torch.zeros(0, dtype=torch.int64, device=src_logits.device)

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        if len(idx[0]) > 0:
            target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        target = target.to(src_logits.dtype)

        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits,
            target,
            self.alpha,
            self.gamma,
            reduction='none',
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        if values is None:
            if len(idx[0]) > 0:
                src_boxes = outputs['pred_boxes'][idx]
                target_boxes = torch.cat(
                    [t['boxes'][i] for t, (_, i) in zip(targets, indices)],
                    dim=0,
                )
                ious, _ = box_iou(
                    box_cxcywh_to_xyxy(src_boxes),
                    box_cxcywh_to_xyxy(target_boxes),
                )
                ious = torch.diag(ious).detach()
            else:
                ious = torch.zeros(
                    0,
                    dtype=outputs['pred_logits'].dtype,
                    device=outputs['pred_logits'].device,
                )
        else:
            ious = values

        src_logits = outputs['pred_logits']

        if len(idx[0]) > 0:
            target_classes_o = torch.cat(
                [t["labels"][J] for t, (_, J) in zip(targets, indices)],
                dim=0,
            )
        else:
            target_classes_o = torch.zeros(0, dtype=torch.int64, device=src_logits.device)

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        if len(idx[0]) > 0:
            target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        target = target.to(src_logits.dtype)

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        if len(idx[0]) > 0:
            target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(
            src_logits,
            target_score,
            weight=weight,
            reduction='none',
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        if len(idx[0]) == 0:
            zero = outputs['pred_boxes'].sum() * 0.0
            return {
                'loss_bbox': zero,
                'loss_giou': zero,
            }

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)],
            dim=0,
        )

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        if boxes_weight is not None:
            loss_bbox = loss_bbox * boxes_weight.unsqueeze(-1)
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _zero_like_pred(self, outputs):
        return torch.zeros((), device=outputs['pred_logits'].device, dtype=outputs['pred_logits'].dtype)

    def _needs_rank_regularization(self):
        return any(loss_name in self.rank_loss_names for loss_name in self.losses)

    def _compute_rank_regularization(self, outputs, global_step=None, epoch=None, steps_per_epoch=None):
        zero = self._zero_like_pred(outputs)
        rank_stats = {
            'loss_tau': zero,
            'loss_nu': zero,
            'loss_loc': zero,
            'loss_tf': zero,
            'loss_rank': zero,
        }
        self.latest_regularizer_metrics = {}

        if self.rank_regularizer_mode not in ('three_level', 'adaptive'):
            return rank_stats

        encoder_aux = outputs.get('encoder_aux', None)
        if encoder_aux is None or len(encoder_aux.get('enc_attn_list', [])) == 0:
            return rank_stats

        need_tau = ('tau' in self.losses) or ('rank' in self.losses)
        need_nu = ('nu' in self.losses) or ('rank' in self.losses)
        need_loc = self.rank_regularizer_mode == 'three_level' and (('loc' in self.losses) or ('rank' in self.losses))
        need_tf = self.rank_regularizer_mode == 'adaptive' and (('tf' in self.losses) or ('rank' in self.losses))

        def mean_rank_from_maps(rank_map, rank_maps):
            if rank_map is not None:
                return rank_map.mean().detach()
            if isinstance(rank_maps, dict) and len(rank_maps) > 0:
                return torch.stack([m.mean() for m in rank_maps.values()]).mean().detach()
            return zero.detach()

        base_stats = self.lowrank_reg(
            encoder_aux,
            compute_tau=need_tau,
            compute_nu=need_nu,
            compute_local=need_loc,
        )
        rank_stats['loss_tau'] = base_stats['loss_tau']
        rank_stats['loss_nu'] = base_stats['loss_nu']
        if self.rank_regularizer_mode == 'three_level':
            rank_stats['loss_loc'] = base_stats['loss_loc']
            rank_stats['loss_rank'] = (
                rank_stats['loss_tau'] +
                rank_stats['loss_nu'] +
                rank_stats['loss_loc']
            )
            self.latest_regularizer_metrics = {
                'mean_rank': mean_rank_from_maps(base_stats.get('rank_map'), base_stats.get('rank_maps')),
            }
            return rank_stats

        adaptive_stats = None
        if need_tf:
            adaptive_stats = self.adaptive_rank_reg(
                encoder_aux,
                global_step=global_step,
                epoch=epoch,
                steps_per_epoch=steps_per_epoch,
            )
            rank_stats['loss_tf'] = adaptive_stats['loss_tf']
        rank_stats['loss_rank'] = (
            rank_stats['loss_tau'] +
            rank_stats['loss_nu'] +
            rank_stats['loss_tf']
        )
        self.latest_regularizer_metrics = {
            'mean_rank': adaptive_stats['mean_rank'].detach() if adaptive_stats is not None else zero.detach(),
            'mean_ema_rank': adaptive_stats['mean_ema_rank'].detach() if adaptive_stats is not None else zero.detach(),
            'mean_gate': adaptive_stats['mean_gate'].detach() if adaptive_stats is not None else zero.detach(),
        }
        return rank_stats

    def loss_tau(self, outputs, targets, indices, num_boxes, rank_stats=None):
        rank_stats = rank_stats or {}
        return {'loss_tau': rank_stats.get('loss_tau', self._zero_like_pred(outputs))}

    def loss_nu(self, outputs, targets, indices, num_boxes, rank_stats=None):
        rank_stats = rank_stats or {}
        return {'loss_nu': rank_stats.get('loss_nu', self._zero_like_pred(outputs))}

    def loss_loc(self, outputs, targets, indices, num_boxes, rank_stats=None):
        rank_stats = rank_stats or {}
        return {'loss_loc': rank_stats.get('loss_loc', self._zero_like_pred(outputs))}

    def loss_tf(self, outputs, targets, indices, num_boxes, rank_stats=None):
        rank_stats = rank_stats or {}
        return {'loss_tf': rank_stats.get('loss_tf', self._zero_like_pred(outputs))}

    def loss_rank(self, outputs, targets, indices, num_boxes, rank_stats=None):
        rank_stats = rank_stats or {}
        return {'loss_rank': rank_stats.get('loss_rank', self._zero_like_pred(outputs))}

    def loss_boundary(self, outputs, targets, indices, num_boxes, rank_stats=None):
        del targets, indices, num_boxes, rank_stats
        return {'loss_boundary': self._zero_like_pred(outputs)}

    def _get_src_permutation_idx(self, indices):
        if len(indices) == 0:
            return (
                torch.zeros(0, dtype=torch.int64),
                torch.zeros(0, dtype=torch.int64),
            )
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        if len(indices) == 0:
            return (
                torch.zeros(0, dtype=torch.int64),
                torch.zeros(0, dtype=torch.int64),
            )
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'tau': self.loss_tau,
            'nu': self.loss_nu,
            'loc': self.loss_loc,
            'tf': self.loss_tf,
            'rank': self.loss_rank,
            'boundary': self.loss_boundary,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        outputs_for_matcher = {
            'pred_logits': outputs['pred_logits'],
            'pred_boxes': outputs['pred_boxes'],
        }

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes],
            dtype=torch.float,
            device=outputs['pred_logits'].device,
        )
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        matched = self.matcher(outputs_for_matcher, targets)
        indices = matched['indices']
        if self._needs_rank_regularization():
            rank_stats = self._compute_rank_regularization(
                outputs,
                global_step=kwargs.get('global_step'),
                epoch=kwargs.get('epoch'),
                steps_per_epoch=kwargs.get('steps_per_epoch'),
            )
        else:
            zero = self._zero_like_pred(outputs)
            rank_stats = {
                'loss_tau': zero,
                'loss_nu': zero,
                'loss_loc': zero,
                'loss_tf': zero,
                'loss_rank': zero,
            }
            self.latest_regularizer_metrics = {}

        losses = {}
        for loss in self.losses:
            meta = self.get_loss_meta_info(loss, outputs, targets, indices)
            if loss in self.rank_loss_names:
                meta['rank_stats'] = rank_stats
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if not self.share_matched_indices:
                    matched = self.matcher(aux_outputs, targets)
                    indices = matched['indices']

                for loss in self.losses:
                    if loss in self.rank_loss_names:
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    if loss in self.rank_loss_names:
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']

            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                matched = self.matcher(aux_outputs, targets)
                indices = matched['indices']

                for loss in self.losses:
                    if loss in self.rank_loss_names:
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        if loss in self.rank_loss_names:
            return {}

        idx = self._get_src_permutation_idx(indices)
        if len(idx[0]) == 0:
            return {}

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()),
                box_cxcywh_to_xyxy(target_boxes),
            )
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes.detach()),
                    box_cxcywh_to_xyxy(target_boxes),
                )
            )
        else:
            raise AttributeError()

        if loss in ('boxes',):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl',):
            meta = {'values': iou}
        else:
            meta = {}
        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((
                    torch.zeros(0, dtype=torch.int64, device=device),
                    torch.zeros(0, dtype=torch.int64, device=device),
                ))

        return dn_match_indices
