import copy

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ...core import register


@register()
class RTDETRCriterionv2(nn.Module):
    """RT-DETR v2 criterion with TF disentanglement regularization.

    Standard DETR/RT-DETR process:
        1) Hungarian matching
        2) Detection losses on matched pairs

    Extended with encoder-side TF regularization:
        - tf_global : cross-level global consistency
        - tf_time   : cross-level time-signature consistency
        - tf_freq   : cross-level frequency-signature consistency
        - tf_ortho  : branch orthogonality within each level
        - tf_dir    : directional smoothness
    """
    __share__ = ['num_classes']
    __inject__ = ['matcher']

    def __init__(self,
                 matcher,
                 weight_dict,
                 losses,
                 alpha=0.2,
                 gamma=2.0,
                 num_classes=80,
                 boxes_weight_format=None,
                 share_matched_indices=False,
                 tf_time_bins=64,
                 tf_freq_bins=32,
                 tf_eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma

        # TF disentanglement hyper-params
        self.tf_time_bins = tf_time_bins
        self.tf_freq_bins = tf_freq_bins
        self.tf_eps = tf_eps

    # ----------------------------------------------------
    # Standard detection losses
    # ----------------------------------------------------
    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)],
            dim=0
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        target = F.one_hot(
            target_classes, num_classes=self.num_classes + 1
        )[..., :-1]

        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target, self.alpha, self.gamma, reduction='none'
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat(
                [t['boxes'][i] for t, (_, i) in zip(targets, indices)],
                dim=0
            )
            ious, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)
            )
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)],
            dim=0
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        target = F.one_hot(
            target_classes, num_classes=self.num_classes + 1
        )[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = torch.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction='none'
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)],
            dim=0
        )

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)
            )
        )
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    # ----------------------------------------------------
    # TF disentanglement losses
    # ----------------------------------------------------
    def _find_first_tensor(self, obj):
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                t = self._find_first_tensor(v)
                if t is not None:
                    return t
        if isinstance(obj, (list, tuple)):
            for v in obj:
                t = self._find_first_tensor(v)
                if t is not None:
                    return t
        return None

    def _zero_loss_like_outputs(self, outputs):
        ref = self._find_first_tensor(outputs)
        if ref is None:
            return torch.tensor(0.0)
        return ref.new_tensor(0.0)

    def _get_tf_levels(self, outputs):
        """Return list of per-level tf aux dicts or None."""
        tf_aux = outputs.get('tf_aux', None)
        if tf_aux is None:
            return None
        if isinstance(tf_aux, dict):
            return [tf_aux]
        if isinstance(tf_aux, (list, tuple)):
            return list(tf_aux)
        return None

    def _normalize_flat(self, x):
        return F.normalize(x, dim=-1, eps=self.tf_eps)

    def _pairwise_mean(self, xs, fn, zero):
        if xs is None or len(xs) < 2:
            return zero
        vals = []
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                vals.append(fn(xs[i], xs[j]))
        if len(vals) == 0:
            return zero
        return torch.stack(vals).mean()

    def _mse_mean(self, a, b):
        return F.mse_loss(a, b, reduction='mean')

    def _cosine_sq_mean(self, a, b):
        cos = F.cosine_similarity(a, b, dim=-1, eps=self.tf_eps)
        return (cos ** 2).mean()

    def _global_signature(self, x):
        # x: [B, C, H, W] -> [B, C]
        sig = F.adaptive_avg_pool2d(x, output_size=(1, 1)).flatten(1)
        return self._normalize_flat(sig)

    def _time_signature(self, x):
        # avg over frequency -> [B, C, W] -> fixed bins -> flatten
        sig = x.mean(dim=2)
        sig = F.adaptive_avg_pool1d(sig, self.tf_time_bins)
        sig = sig.flatten(1)
        return self._normalize_flat(sig)

    def _freq_signature(self, x):
        # avg over time -> [B, C, H] -> fixed bins -> flatten
        sig = x.mean(dim=3)
        sig = F.adaptive_avg_pool1d(sig, self.tf_freq_bins)
        sig = sig.flatten(1)
        return self._normalize_flat(sig)

    def loss_tf_global(self, outputs, targets, indices, num_boxes, **kwargs):
        levels = self._get_tf_levels(outputs)
        zero = self._zero_loss_like_outputs(outputs)
        if levels is None:
            return {'loss_tf_global': zero}

        sigs = [self._global_signature(lv['feat_global']) for lv in levels]
        loss = self._pairwise_mean(sigs, self._mse_mean, zero)
        return {'loss_tf_global': loss}

    def loss_tf_time(self, outputs, targets, indices, num_boxes, **kwargs):
        levels = self._get_tf_levels(outputs)
        zero = self._zero_loss_like_outputs(outputs)
        if levels is None:
            return {'loss_tf_time': zero}

        sigs = [self._time_signature(lv['feat_time']) for lv in levels]
        loss = self._pairwise_mean(sigs, self._mse_mean, zero)
        return {'loss_tf_time': loss}

    def loss_tf_freq(self, outputs, targets, indices, num_boxes, **kwargs):
        levels = self._get_tf_levels(outputs)
        zero = self._zero_loss_like_outputs(outputs)
        if levels is None:
            return {'loss_tf_freq': zero}

        sigs = [self._freq_signature(lv['feat_freq']) for lv in levels]
        loss = self._pairwise_mean(sigs, self._mse_mean, zero)
        return {'loss_tf_freq': loss}

    def loss_tf_ortho(self, outputs, targets, indices, num_boxes, **kwargs):
        levels = self._get_tf_levels(outputs)
        zero = self._zero_loss_like_outputs(outputs)
        if levels is None:
            return {'loss_tf_ortho': zero}

        vals = []
        for lv in levels:
            g = lv['feat_global'].flatten(1)
            t = lv['feat_time'].flatten(1)
            f = lv['feat_freq'].flatten(1)

            loss_gt = self._cosine_sq_mean(g, t)
            loss_gf = self._cosine_sq_mean(g, f)
            loss_tf = self._cosine_sq_mean(t, f)
            vals.append((loss_gt + loss_gf + loss_tf) / 3.0)

        return {'loss_tf_ortho': torch.stack(vals).mean() if len(vals) > 0 else zero}

    def loss_tf_dir(self, outputs, targets, indices, num_boxes, **kwargs):
        """
        Directional smoothness:
          - feat_time should be smoother along frequency
          - feat_freq should be smoother along time

        We normalize TV by average magnitude to avoid trivial scale effects.
        """
        levels = self._get_tf_levels(outputs)
        zero = self._zero_loss_like_outputs(outputs)
        if levels is None:
            return {'loss_tf_dir': zero}

        vals = []
        for lv in levels:
            t = lv['feat_time']  # [B, C, H, W]
            f = lv['feat_freq']  # [B, C, H, W]

            # frequency-direction TV for time branch
            tv_f_t = (t[:, :, 1:, :] - t[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
            mag_t = t.abs().mean(dim=(1, 2, 3)).clamp_min(self.tf_eps)
            loss_t = (tv_f_t / mag_t).mean()

            # time-direction TV for freq branch
            tv_t_f = (f[:, :, :, 1:] - f[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
            mag_f = f.abs().mean(dim=(1, 2, 3)).clamp_min(self.tf_eps)
            loss_f = (tv_t_f / mag_f).mean()

            vals.append(loss_t + loss_f)

        return {'loss_tf_dir': torch.stack(vals).mean() if len(vals) > 0 else zero}

    # ----------------------------------------------------
    # Helpers
    # ----------------------------------------------------
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
            'boxes': self.loss_boxes,
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

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        # TF regularizers do not need matcher-derived box weighting
        if loss.startswith('tf_'):
            return {}

        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()),
                box_cxcywh_to_xyxy(target_boxes)
            )
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes.detach()),
                    box_cxcywh_to_xyxy(target_boxes)
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

    def forward(self, outputs, targets, **kwargs):
        """This performs the loss computation."""
        if self.matcher is None:
            raise RuntimeError(
                "RTDETRCriterionv2.matcher is None. Please ensure YAML contains:\n"
                "criterion: RTDETRCriterionv2\n"
                "RTDETRCriterionv2:\n"
                "  matcher: HungarianMatcher"
            )

        # remove aux outputs for main matching
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # robust device resolution because outputs may contain tf_aux (dict/list)
        ref_tensor = self._find_first_tensor(outputs)
        if ref_tensor is None:
            raise RuntimeError('Cannot find any tensor inside outputs.')

        # average number of GT boxes across all nodes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=ref_tensor.device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # main matching
        matched = self.matcher(outputs_without_aux, targets)
        indices = matched['indices']

        # main losses
        losses = {}
        for loss in self.losses:
            meta = self.get_loss_meta_info(loss, outputs, targets, indices)
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # aux decoder losses
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if not self.share_matched_indices:
                    matched = self.matcher(aux_outputs, targets)
                    indices = matched['indices']

                for loss in self.losses:
                    # TF losses belong to encoder-side disentanglement only
                    if loss.startswith('tf_'):
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # dn aux losses
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    if loss.startswith('tf_'):
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # encoder aux losses (RT-DETR v2)
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
                    if loss.startswith('tf_'):
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

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