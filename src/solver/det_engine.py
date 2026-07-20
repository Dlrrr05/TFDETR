"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import sys
import math
from typing import Iterable, Optional, Sequence

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..misc import MetricLogger, SmoothedValue, dist_utils
from .confusion_matrix import DetectionConfusionMatrix


def _device_type(device: torch.device) -> str:
    return device.type if isinstance(device, torch.device) else str(device).split(":")[0]


def _to_float(v):
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return float(v.item())
        raise TypeError("Tensor is not scalar.")
    return float(v)


def _maybe_tensor_to_python(v):
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return v.item()
        return v.detach().cpu().tolist()
    return v


def _default_psds_kwargs():
    return dict(
        dtc_threshold=0.5,
        gtc_threshold=0.5,
        cttc_threshold=0.3,
        alpha_ct=0.0,
        alpha_st=1.0,
        max_efpr=100.0,
        num_operating_points=10,
        merge_gt=True,
        gt_merge_gap=0.0,
        merge_pred=True,
        pred_merge_gap=0.0,
        pred_score_mode="max",
    )


def _fixed_clip_duration_for_psds():
    return 2.56


def _fixed_time_axis_width_for_psds():
    return 512.0


def _extract_file_id(target):
    """
    每个窗口独立作为一个文件。优先用 image_id 作为窗口唯一 ID。
    """
    candidate_keys = ["image_id", "file_id", "filename", "audio_id", "uid"]

    for key in candidate_keys:
        if key not in target:
            continue

        value = target[key]

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return str(value.detach().cpu().item())
            return "_".join([str(x) for x in value.detach().cpu().view(-1).tolist()])

        if isinstance(value, (int, float, str)):
            return str(value)

        if isinstance(value, (list, tuple)) and len(value) > 0:
            return "_".join([str(x) for x in value])

    return None


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    **kwargs
):
    model.train()
    criterion.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)

    amp_device_type = _device_type(device)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)

        if scaler is not None:
            with torch.autocast(device_type=amp_device_type, cache_enabled=True):
                outputs = model(samples, targets=targets)

            with torch.autocast(device_type=amp_device_type, enabled=False):
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

            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(_to_float(loss_value)):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=_to_float(loss_value), **{k: _to_float(v) for k, v in loss_dict_reduced.items()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process():
            writer.add_scalar("Loss/total", _to_float(loss_value), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", _to_float(v), global_step)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def plot_confusion_matrix(matrix, class_names, save_path="confusion_matrix.png"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names + ["bg"],
        yticklabels=class_names + ["bg"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Detection Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator,
    device,
    num_classes: int,
    class_names: Optional[Sequence[str]] = None,
    det_iou_threshold: float = 0.5,
    time_iou_threshold: Optional[float] = None,
    freq_iou_threshold: Optional[float] = None,
    freq_map_iou_threshold: float = 0.5,
    compute_psds: bool = True,
    psds_kwargs: Optional[dict] = None,
    time_scale: float = 1.0,
    psds_export_dir: Optional[str] = None,
):
    if psds_kwargs is None:
        psds_kwargs = _default_psds_kwargs()

    model.eval()
    criterion.eval()

    if coco_evaluator is not None:
        coco_evaluator.cleanup()
        iou_types = coco_evaluator.iou_types
    else:
        iou_types = []

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    det_confmat = DetectionConfusionMatrix(
        num_classes=num_classes,
        iou_threshold=det_iou_threshold,
        time_iou_threshold=time_iou_threshold,
        freq_iou_threshold=freq_iou_threshold,
        freq_map_iou_threshold=freq_map_iou_threshold,
        class_names=list(class_names) if class_names is not None else None,
        psds_gt_box_format="xyxy",
        psds_pred_box_format="xyxy",
    )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_target_sizes)

        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        for output, target in zip(results, targets):
            file_id = _extract_file_id(target)

            if "clip_duration" in target:
                clip_duration = float(target["clip_duration"].detach().cpu().item())
            else:
                clip_duration = _fixed_clip_duration_for_psds()

            time_axis_width = _fixed_time_axis_width_for_psds()

            det_confmat.process_batch(
                detections=output,
                targets=target,
                file_id=file_id,
                clip_duration=clip_duration,
                time_scale=time_scale,
                time_axis_width=time_axis_width,
                update_psds_cache=compute_psds,
            )

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    det_confmat.reduce_from_all_processes(device)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    if compute_psds and dist_utils.is_main_process():
        export_dir = psds_export_dir if psds_export_dir is not None else "./output/psds_tables"
        det_confmat.export_psds_tables(
            save_dir=export_dir,
            merge_gt=True,
            gt_merge_gap=0.0,
            merge_pred_global=False,
            pred_merge_gap=0.0,
            pred_score_mode="max",
        )

    det_confmat.print_summary(
        class_names=list(class_names) if class_names is not None else None,
        compute_psds=compute_psds,
        psds_kwargs=psds_kwargs,
    )

    stats = {}

    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    det_result = det_confmat.summary(
        compute_psds=compute_psds,
        psds_kwargs=psds_kwargs,
    )

    stats["Event F1"] = float(det_result["Event F1"])
    stats["mIoU"] = float(det_result["mIoU"])
    stats["Macro-F1"] = float(det_result["Macro-F1"])
    stats["Prec."] = float(det_result["Prec."])
    stats["Recall"] = float(det_result["Recall"])
    stats["MCC"] = float(det_result["MCC"])

    stats["mTime-IoU"] = float(det_result["mTime-IoU"])
    stats["mFreq-IoU"] = float(det_result["mFreq-IoU"])
    stats["mTF-IoU"] = float(det_result["mTF-IoU"])

    stats["Time-Acc"] = float(det_result["Time-Acc"])
    stats["Freq-Acc"] = float(det_result["Freq-Acc"])
    stats["TF-Acc"] = float(det_result["TF-Acc"])

    stats["overall_time_acc"] = float(det_result["overall_time_acc"])
    stats["overall_freq_acc"] = float(det_result["overall_freq_acc"])
    stats["overall_tf_acc"] = float(det_result["overall_tf_acc"])

    stats["Freq-mAP"] = float(det_result["Freq-mAP"])
    stats["freq_map_iou_threshold"] = float(det_result["freq_map_iou_threshold"])
    stats["confusion_matrix"] = det_result["matrix"].tolist()

    per_class_iou = {}
    per_class_loc = {}
    per_class_freq_ap = {}

    for c in range(num_classes):
        name = class_names[c] if class_names is not None else str(c)
        pc = det_result["per_class"][c]

        per_class_iou[name] = float(pc["iou"])
        per_class_loc[name] = {
            "time_iou_mean": float(pc["time_iou_mean"]),
            "freq_iou_mean": float(pc["freq_iou_mean"]),
            "tf_iou_mean": float(pc["tf_iou_mean"]),
            "time_acc": float(pc["time_acc"]),
            "freq_acc": float(pc["freq_acc"]),
            "tf_acc": float(pc["tf_acc"]),
        }

        ap = det_result["freq_ap_per_class"][c]
        if ap is None:
            per_class_freq_ap[name] = None
        else:
            try:
                ap_is_nan = bool(torch.isnan(torch.tensor(ap)).item())
            except Exception:
                ap_is_nan = False
            per_class_freq_ap[name] = None if ap_is_nan else float(ap)

    stats["per_class_iou"] = per_class_iou
    stats["per_class_localization"] = per_class_loc
    stats["per_class_freq_ap"] = per_class_freq_ap

    if compute_psds:
        stats["PSDS"] = det_result["PSDS"]

    return stats, coco_evaluator
