"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision

from ...core import register
from .vis_rank import make_2d_rank_map, score_boxes_with_rank


__all__ = ['RTDETRPostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class RTDETRPostProcessor(nn.Module):
    __share__ = [
        'num_classes', 
        'use_focal_loss', 
        'num_top_queries', 
        'remap_mscoco_category'
    ]
    
    def __init__(
        self, 
        num_classes=80, 
        use_focal_loss=True, 
        num_top_queries=300, 
        remap_mscoco_category=False,
        rank_rescore_alpha=0.0,
        rank_rescore_level_index=0,
        rank_rescore_r_loc=1,
        rank_rescore_window_size=9,
        rank_rescore_window_h=None,
        rank_rescore_window_w=None,
        rank_rescore_reduce='mean',
        rank_rescore_interpolation='bilinear',
        return_rank_maps=False,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category 
        self.rank_rescore_alpha = float(rank_rescore_alpha)
        self.rank_rescore_level_index = rank_rescore_level_index
        self.rank_rescore_r_loc = int(rank_rescore_r_loc)
        rank_radius = max(int(rank_rescore_window_size), 1) // 2
        self.rank_rescore_window_h = rank_radius if rank_rescore_window_h is None else int(rank_rescore_window_h)
        self.rank_rescore_window_w = rank_radius if rank_rescore_window_w is None else int(rank_rescore_window_w)
        self.rank_rescore_reduce = rank_rescore_reduce
        self.rank_rescore_interpolation = rank_rescore_interpolation
        self.return_rank_maps = return_rank_maps
        self.deploy_mode = False 

    def extra_repr(self) -> str:
        return (
            f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, '
            f'num_top_queries={self.num_top_queries}, rank_rescore_alpha={self.rank_rescore_alpha}'
        )

    @staticmethod
    def _sort_predictions(scores, labels, boxes, box_quality=None):
        order = torch.argsort(scores, dim=-1, descending=True)
        scores = torch.gather(scores, dim=1, index=order)
        labels = torch.gather(labels, dim=1, index=order)
        boxes = torch.gather(
            boxes,
            dim=1,
            index=order.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]),
        )
        if box_quality is None:
            return scores, labels, boxes, None

        box_quality = torch.gather(box_quality, dim=1, index=order)
        return scores, labels, boxes, box_quality
    
    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(bbox_pred, dim=1, index=index.unsqueeze(-1).tile(1, 1, bbox_pred.shape[-1]))
            else:
                boxes = bbox_pred

        rank_map = None
        box_quality = None
        if (self.rank_rescore_alpha > 0 or self.return_rank_maps) and 'encoder_aux' in outputs:
            rank_info = make_2d_rank_map(
                outputs['encoder_aux'],
                level_index=self.rank_rescore_level_index,
                rank_r=self.rank_rescore_r_loc,
                radius_h=self.rank_rescore_window_h,
                radius_w=self.rank_rescore_window_w,
                level_reduce=self.rank_rescore_reduce,
                interpolation=self.rank_rescore_interpolation,
            )
            rank_map = rank_info['rank_map']

            if self.rank_rescore_alpha > 0:
                scores, box_quality = score_boxes_with_rank(
                    scores,
                    boxes,
                    rank_map,
                    orig_target_sizes,
                    alpha=self.rank_rescore_alpha,
                )
                scores, labels, boxes, box_quality = self._sort_predictions(scores, labels, boxes, box_quality)

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for batch_idx, (lab, box, sco) in enumerate(zip(labels, boxes, scores)):
            result = dict(labels=lab, boxes=box, scores=sco)
            if box_quality is not None:
                result['rank_quality'] = box_quality[batch_idx]
            if self.return_rank_maps and rank_map is not None:
                result['rank_map'] = rank_map[batch_idx]
            results.append(result)
        
        return results
        

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 
