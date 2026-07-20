"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import sys
import math
from typing import Iterable

import torch
import torch.amp 
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from .confusion_matrix import DetectionConfusionMatrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.ops import box_iou

confusion_matrix = None


def _infer_num_classes(postprocessor, criterion, num_classes):
    if num_classes is not None:
        return int(num_classes)

    for module in (postprocessor, criterion):
        value = getattr(module, 'num_classes', None)
        if value is not None:
            return int(value)

    return 10


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, steps_per_epoch=len(data_loader))

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)
            
            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        extra_metrics = {}
        if hasattr(criterion, 'latest_regularizer_metrics') and criterion.latest_regularizer_metrics:
            metric_tensors = {
                k: v if isinstance(v, torch.Tensor) else torch.tensor(v, device=device)
                for k, v in criterion.latest_regularizer_metrics.items()
            }
            extra_metrics = {
                k: v.item() for k, v in dist_utils.reduce_dict(metric_tensors).items()
            }

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        if extra_metrics:
            metric_logger.update(**extra_metrics)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process():
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)
            for k, v in extra_metrics.items():
                writer.add_scalar(f'Rank/{k}', v, global_step)
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def plot_confusion_matrix(matrix, class_names):

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names + ["bg"],
        yticklabels=class_names + ["bg"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Detection Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator,
    device,
    num_classes=None,
    class_names=None,
    det_iou_threshold=0.5
):
    model.eval()
    criterion.eval()
    num_classes = _infer_num_classes(postprocessor, criterion, num_classes)
    if coco_evaluator is not None:
        coco_evaluator.cleanup()
        iou_types = coco_evaluator.iou_types
    else:
        iou_types = []

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # 新增：检测混淆矩阵统计器
    det_confmat = DetectionConfusionMatrix(
        num_classes=num_classes,
        iou_threshold=det_iou_threshold
    )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_target_sizes)

        # COCO evaluator
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # 新增：检测分类指标统计
        for output, target in zip(results, targets):
            det_confmat.process_batch(output, target)

    # 同步日志
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 分布式同步 confusion matrix
    det_confmat.reduce_from_all_processes(device)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    # 打印新增指标
    det_confmat.print_summary(class_names=class_names)

    stats = {}

    # 原 COCO 指标
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    # 新增检测指标
    det_result = det_confmat.summary()
    stats['Event F1'] = det_result['Event F1']
    stats['mIoU'] = det_result['mIoU']
    stats['Macro-F1'] = det_result['Macro-F1']
    stats['Prec.'] = det_result['Prec.']
    stats['Recall'] = det_result['Recall']
    stats['MCC'] = det_result['MCC']
    stats['confusion_matrix'] = det_result['matrix'].tolist()

    # 每类 IoU 也一起塞进去
    per_class_iou = {}
    for c in range(num_classes):
        name = class_names[c] if class_names is not None else str(c)
        per_class_iou[name] = det_result['per_class'][c]['iou']
    stats['per_class_iou'] = per_class_iou

    return stats, coco_evaluator
