"""
reference: 
https://github.com/facebookresearch/detr/blob/main/models/detr.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Modified to add mathematically grounded spectro-temporal disentanglement losses
for encoder-side TFDisentangler outputs.
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ...core import register


def _find_first_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            out = _find_first_tensor(v)
            if out is not None:
                return out
    if isinstance(obj, (list, tuple)):
        for v in obj:
            out = _find_first_tensor(v)
            if out is not None:
                return out
    return None


@register()
class RTDETRCriterion(nn.Module):
    """This class computes the loss for DETR plus optional TF disentanglement losses.

    Supported disentanglement losses (all optional, controlled by self.losses):
        - tf_global: cross-level consistency of global branch pooled descriptors
        - tf_time  : cross-level consistency of time signatures (avg over frequency)
        - tf_freq  : cross-level consistency of frequency signatures (avg over time)
        - tf_ortho : within-level branch orthogonality between global/time/freq
        - tf_dir   : directional regularization:
                     T branch should be smoother along frequency,
                     F branch should be smoother along time.

    Expected model output contract for TF losses:
        outputs['tf_aux'] is a list of per-level dicts, each containing:
            feat_global: [B, C, H, W]
            feat_time  : [B, C, H, W]
            feat_freq  : [B, C, H, W]
            mask_global/mask_time/mask_freq (optional for visualization only)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        eos_coef=1e-4,
        num_classes=80,
        tf_time_bins=64,
        tf_freq_bins=32,
        tf_eps=1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = alpha
        self.gamma = gamma

        # hyperparameters for TF losses
        self.tf_time_bins = tf_time_bins
        self.tf_freq_bins = tf_freq_bins
        self.tf_eps = tf_eps

    # -----------------------------
    # Standard DETR losses
    # -----------------------------
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target, self.alpha, self.gamma, reduction='none'
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction='none'
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
            )
        )
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    # -----------------------------
    # TF disentanglement helpers
    # -----------------------------
    def _get_tf_levels(self, outputs):
        assert 'tf_aux' in outputs, (
            "TF losses are enabled in criterion.losses, but outputs['tf_aux'] is missing. "
            "Please attach encoder TF auxiliary features to model outputs."
        )
        levels = outputs['tf_aux']
        assert isinstance(levels, (list, tuple)) and len(levels) > 0, "outputs['tf_aux'] must be a non-empty list."
        return levels

    def _l2norm_flat(self, x):
        x = x.reshape(x.shape[0], -1)
        return F.normalize(x, p=2, dim=1, eps=self.tf_eps)

    def _gap_desc(self, x):
        # x: [B, C, H, W] -> [B, C]
        return self._l2norm_flat(F.adaptive_avg_pool2d(x, output_size=1))

    def _time_signature(self, x):
        # x: [B, C, H, W] -> average over frequency -> [B, C, W]
        sig = x.mean(dim=2)
        sig = F.adaptive_avg_pool1d(sig, self.tf_time_bins)
        return self._l2norm_flat(sig)

    def _freq_signature(self, x):
        # x: [B, C, H, W] -> average over time -> [B, C, H]
        sig = x.mean(dim=3)
        sig = F.adaptive_avg_pool1d(sig, self.tf_freq_bins)
        return self._l2norm_flat(sig)

    def _pairwise_mean_mse(self, vecs):
        n = len(vecs)
        if n <= 1:
            return vecs[0].new_zeros(())
        total = vecs[0].new_zeros(())
        cnt = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                total = total + F.mse_loss(vecs[i], vecs[j], reduction='mean')
                cnt += 1
        return total / cnt

    def _pairwise_mean_cos2(self, vecs):
        n = len(vecs)
        if n <= 1:
            return vecs[0].new_zeros(())
        total = vecs[0].new_zeros(())
        cnt = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                vi = self._l2norm_flat(vecs[i])
                vj = self._l2norm_flat(vecs[j])
                cos = (vi * vj).sum(dim=1)
                total = total + (cos ** 2).mean()
                cnt += 1
        return total / cnt

    def _normalized_tv_freq(self, x):
        if x.shape[2] <= 1:
            return x.new_zeros(())
        tv = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        denom = x.abs().mean().clamp_min(self.tf_eps)
        return tv / denom

    def _normalized_tv_time(self, x):
        if x.shape[3] <= 1:
            return x.new_zeros(())
        tv = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        denom = x.abs().mean().clamp_min(self.tf_eps)
        return tv / denom

    # -----------------------------
    # TF disentanglement losses
    # -----------------------------
    def loss_tf_global(self, outputs, targets, indices, num_boxes, **kwargs):
        """
        L_g = avg_{i<j} || N(GAP(G_i)) - N(GAP(G_j)) ||_2^2
        """
        levels = self._get_tf_levels(outputs)
        descs = [self._gap_desc(level['feat_global']) for level in levels]
        return {'loss_tf_global': self._pairwise_mean_mse(descs)}

    def loss_tf_time(self, outputs, targets, indices, num_boxes, **kwargs):
        """
        L_t = avg_{i<j} || N(P_f(T_i)) - N(P_f(T_j)) ||_2^2
        where P_f averages over frequency and adaptive pools to a fixed time length.
        """
        levels = self._get_tf_levels(outputs)
        descs = [self._time_signature(level['feat_time']) for level in levels]
        return {'loss_tf_time': self._pairwise_mean_mse(descs)}

    def loss_tf_freq(self, outputs, targets, indices, num_boxes, **kwargs):
        """
        L_f = avg_{i<j} || N(P_t(F_i)) - N(P_t(F_j)) ||_2^2
        where P_t averages over time and adaptive pools to a fixed frequency length.
        """
        levels = self._get_tf_levels(outputs)
        descs = [self._freq_signature(level['feat_freq']) for level in levels]
        return {'loss_tf_freq': self._pairwise_mean_mse(descs)}

    def loss_tf_ortho(self, outputs, targets, indices, num_boxes, **kwargs):
        """
        L_ortho = avg_l [ cos^2(g_l, t_l) + cos^2(g_l, f_l) + cos^2(t_l, f_l) ] / 3
        where each branch descriptor is GAP pooled then L2 normalized.
        """
        levels = self._get_tf_levels(outputs)
        total = None
        for level in levels:
            g = self._gap_desc(level['feat_global'])
            t = self._gap_desc(level['feat_time'])
            f = self._gap_desc(level['feat_freq'])
            level_loss = self._pairwise_mean_cos2([g, t, f])
            total = level_loss if total is None else total + level_loss
        total = total / max(len(levels), 1)
        return {'loss_tf_ortho': total}

    def loss_tf_dir(self, outputs, targets, indices, num_boxes, **kwargs):
        """
        L_dir = avg_l [ TV_f(T_l)/|T_l| + TV_t(F_l)/|F_l| ]
        This encourages the time branch to be smoother along frequency
        and the frequency branch to be smoother along time.
        """
        levels = self._get_tf_levels(outputs)
        total = None
        for level in levels:
            t = level['feat_time']
            f = level['feat_freq']
            level_loss = self._normalized_tv_freq(t) + self._normalized_tv_time(f)
            total = level_loss if total is None else total + level_loss
        total = total / max(len(levels), 1)
        return {'loss_tf_dir': total}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'cardinality': self.loss_cardinality,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'tf_global': self.loss_tf_global,
            'tf_time': self.loss_tf_time,
            'tf_freq': self.loss_tf_freq,
            'tf_ortho': self.loss_tf_ortho,
            'tf_dir': self.loss_tf_dir,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        num_boxes = sum(len(t["labels"]) for t in targets)
        ref_tensor = _find_first_tensor(outputs)
        assert ref_tensor is not None, "No tensor found in outputs."
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=ref_tensor.device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        indices = self.matcher(outputs_without_aux, targets)['indices']

        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)['indices']
                for loss in self.losses:
                    if loss == 'masks' or loss.startswith('tf_'):
                        continue
                    local_kwargs = {}
                    if loss == 'labels':
                        local_kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **local_kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    if loss == 'masks' or loss.startswith('tf_'):
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, dn_num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

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
                    torch.zeros(0, dtype=torch.int64, device=device)
                ))

        return dn_match_indices


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k."""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
