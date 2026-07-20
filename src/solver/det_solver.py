"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time
import json
import datetime
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from ..misc import dist_utils

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


def _to_python(obj: Any):
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    return obj


def _write_scalar_tree(writer, prefix: str, value: Any, step: int):
    if value is None:
        return

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            writer.add_scalar(prefix, value.item(), step)
        return

    if isinstance(value, (int, float, np.integer, np.floating)):
        writer.add_scalar(prefix, float(value), step)
        return

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in value):
            for i, v in enumerate(value):
                writer.add_scalar(f"{prefix}_{i}", float(v), step)
        return

    if isinstance(value, dict):
        for k, v in value.items():
            _write_scalar_tree(writer, f"{prefix}/{k}", v, step)


def _infer_class_info(evaluator, cfg) -> Tuple[Optional[int], Optional[Sequence[str]]]:
    num_classes = getattr(cfg, "num_classes", None)
    class_names = getattr(cfg, "class_names", None)

    if class_names is None and evaluator is not None and getattr(evaluator, "coco_gt", None) is not None:
        cats = getattr(evaluator.coco_gt, "cats", None)
        if isinstance(cats, dict) and len(cats) > 0:
            ordered_ids = sorted(cats.keys())
            class_names = [cats[k].get("name", str(k)) for k in ordered_ids]

    if num_classes is None and class_names is not None:
        num_classes = len(class_names)

    if num_classes is None and evaluator is not None and getattr(evaluator, "coco_gt", None) is not None:
        cats = getattr(evaluator.coco_gt, "cats", None)
        if isinstance(cats, dict):
            num_classes = len(cats)

    return num_classes, class_names


def _infer_time_scale(cfg) -> float:
    # PSDS 不再使用 x * time_scale 方式换秒，保留为 1.0 即可
    return 1.0


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


def _extract_primary_metric(test_stats: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    if "coco_eval_bbox" in test_stats:
        v = test_stats["coco_eval_bbox"]
        if isinstance(v, (list, tuple)) and len(v) > 0:
            return "coco_eval_bbox/AP@[0.50:0.95]", float(v[0])

    if "PSDS" in test_stats:
        v = test_stats["PSDS"]
        if isinstance(v, dict) and v.get("PSDS", None) is not None:
            return "PSDS", float(v["PSDS"])
        if isinstance(v, (int, float, np.integer, np.floating)):
            return "PSDS", float(v)

    for k in ["Event F1", "TF-Acc", "Freq-mAP", "mIoU", "Macro-F1"]:
        if k in test_stats and isinstance(test_stats[k], (int, float, np.integer, np.floating)):
            return k, float(test_stats[k])

    return None, None


def _append_test_results_txt(
    save_path,
    epoch: int,
    test_stats: Dict[str, Any],
    best_stat: Optional[Dict[str, Any]] = None,
    mode: str = "fit",
):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    test_stats = _to_python(test_stats)
    best_stat = _to_python(best_stat) if best_stat is not None else None

    with save_path.open("a", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"mode: {mode}\n")
        f.write(f"epoch: {epoch}\n")

        if best_stat is not None:
            f.write("best_stat:\n")
            f.write(json.dumps(best_stat, ensure_ascii=False, indent=2))
            f.write("\n")

        f.write("test_stats:\n")
        f.write(json.dumps(test_stats, ensure_ascii=False, indent=2))
        f.write("\n\n")


class DetSolver(BaseSolver):

    def fit(self):
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f"number of trainable parameters: {n_parameters}")

        best_stat = {
            "epoch": -1,
            "metric_name": None,
            "metric_value": float("-inf"),
        }

        start_time = time.time()
        start_epoch = self.last_epoch + 1

        num_classes, class_names = _infer_class_info(self.evaluator, args)
        if num_classes is None:
            raise ValueError("Cannot infer num_classes. Please set cfg.num_classes or provide valid evaluator categories.")

        eval_kwargs = dict(
            num_classes=num_classes,
            class_names=class_names,
            det_iou_threshold=float(getattr(args, "det_iou_threshold", 0.5)),
            time_iou_threshold=getattr(args, "time_iou_threshold", None),
            freq_iou_threshold=getattr(args, "freq_iou_threshold", None),
            freq_map_iou_threshold=float(getattr(args, "freq_map_iou_threshold", 0.5)),
            compute_psds=False,
            psds_kwargs=_default_psds_kwargs(),
            time_scale=_infer_time_scale(args),
            psds_export_dir=str(self.output_dir / "psds_tables") if self.output_dir else "./output/psds_tables",
        )

        test_results_txt = self.output_dir / "test_results.txt" if self.output_dir else None

        for epoch in range(start_epoch, args.epoches):
            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / "last.pth"]
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                **eval_kwargs,
            )

            if self.writer and dist_utils.is_main_process():
                for k, v in test_stats.items():
                    _write_scalar_tree(self.writer, f"Test/{k}", v, epoch)

            metric_name, metric_value = _extract_primary_metric(test_stats)
            if metric_name is not None and metric_value is not None:
                if metric_value > best_stat["metric_value"]:
                    best_stat["epoch"] = epoch
                    best_stat["metric_name"] = metric_name
                    best_stat["metric_value"] = metric_value

                    if self.output_dir:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / "best.pth")

            print(f"best_stat: {best_stat}")

            log_stats = {
                **{f"train_{k}": _to_python(v) for k, v in train_stats.items()},
                **{f"test_{k}": _to_python(v) for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": int(n_parameters),
                "best_stat": _to_python(best_stat),
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats, ensure_ascii=False) + "\n")

                _append_test_results_txt(
                    save_path=test_results_txt,
                    epoch=epoch,
                    test_stats=test_stats,
                    best_stat=best_stat,
                    mode="fit",
                )

                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def val(self):
        self.eval()

        args = self.cfg
        num_classes, class_names = _infer_class_info(self.evaluator, args)
        if num_classes is None:
            raise ValueError("Cannot infer num_classes. Please set cfg.num_classes or provide valid evaluator categories.")

        eval_kwargs = dict(
            num_classes=num_classes,
            class_names=class_names,
            det_iou_threshold=float(getattr(args, "det_iou_threshold", 0.5)),
            time_iou_threshold=getattr(args, "time_iou_threshold", None),
            freq_iou_threshold=getattr(args, "freq_iou_threshold", None),
            freq_map_iou_threshold=float(getattr(args, "freq_map_iou_threshold", 0.5)),
            compute_psds=True,
            psds_kwargs=_default_psds_kwargs(),
            time_scale=_infer_time_scale(args),
            psds_export_dir=str(self.output_dir / "psds_tables") if self.output_dir else "./output/psds_tables",
        )

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            **eval_kwargs,
        )

        if self.output_dir and coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        if self.output_dir and dist_utils.is_main_process():
            _append_test_results_txt(
                save_path=self.output_dir / "test_results.txt",
                epoch=int(getattr(self, "last_epoch", -1)),
                test_stats=test_stats,
                best_stat=None,
                mode="val",
            )

        if dist_utils.is_main_process():
            print("Validation stats:")
            print(json.dumps(_to_python(test_stats), ensure_ascii=False, indent=2))

        return test_stats
