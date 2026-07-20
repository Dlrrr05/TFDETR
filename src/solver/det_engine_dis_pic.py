import sys
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.amp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from .confusion_matrix import DetectionConfusionMatrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# -------------------------------------------------------
# Train
# -------------------------------------------------------
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer: SummaryWriter = kwargs.get('writer', None)

    ema: ModelEMA = kwargs.get('ema', None)
    scaler: GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler: Warmup = kwargs.get('lr_warmup_scheduler', None)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)

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

            loss: torch.Tensor = sum(loss_dict.values())
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

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process():
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# -------------------------------------------------------
# Visualization helpers
# -------------------------------------------------------
def _normalize_01(x: np.ndarray):
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)


def _save_confusion_matrix(matrix, class_names, save_path: Path):
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
    plt.savefig(str(save_path), dpi=300)
    plt.close()


def _global_signature(x: torch.Tensor):
    # [B, C, H, W] -> [B, C]
    sig = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
    sig = F.normalize(sig, dim=-1)
    return sig


def _time_signature(x: torch.Tensor, bins=64):
    # [B, C, H, W] -> avg over freq(H) -> [B, C, W]
    sig = x.mean(dim=2)
    sig = F.adaptive_avg_pool1d(sig, bins).flatten(1)
    sig = F.normalize(sig, dim=-1)
    return sig


def _freq_signature(x: torch.Tensor, bins=32):
    # [B, C, H, W] -> avg over time(W) -> [B, C, H]
    sig = x.mean(dim=3)
    sig = F.adaptive_avg_pool1d(sig, bins).flatten(1)
    sig = F.normalize(sig, dim=-1)
    return sig


def _generic_embed(x: torch.Tensor, out_hw=(8, 8)):
    # same-dim embedding for PCA across branches
    emb = F.adaptive_avg_pool2d(x, out_hw).flatten(1)
    emb = F.normalize(emb, dim=-1)
    return emb


def _cosine_mean(a: torch.Tensor, b: torch.Tensor):
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return 0.0
    return F.cosine_similarity(a[:n], b[:n], dim=-1).mean().item()


def _plot_heatmap(mat, title, save_path: Path, vmin=0.0, vmax=1.0):
    plt.figure(figsize=(5, 4))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="viridis", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel("Level")
    plt.ylabel("Level")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300)
    plt.close()


def _plot_direction_boxplot(rg, rt, rf, save_path: Path):
    plt.figure(figsize=(6, 4))
    plt.boxplot([rg, rt, rf], labels=["Global", "Time", "Freq"], showfliers=False)
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.ylabel("Et / Ef")
    plt.title("Directional ratio")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300)
    plt.close()


def _plot_pca_branch_vis(X, Y, save_path: Path):
    if X.shape[0] < 3:
        return
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    plt.figure(figsize=(6, 5))
    names = {0: "Global", 1: "Time", 2: "Freq"}
    for k in [0, 1, 2]:
        idx = (Y == k)
        if idx.sum() == 0:
            continue
        plt.scatter(Z[idx, 0], Z[idx, 1], s=10, alpha=0.6, label=names[k])
    plt.legend()
    plt.title("Branch embedding PCA")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300)
    plt.close()


def _directional_ratio(x: torch.Tensor):
    # x: [B,C,H,W]
    et = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    ef = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    r = et / (ef + 1e-6)
    return r.detach().cpu().numpy()


def _save_mask_overlay(spec: np.ndarray,
                       mg: np.ndarray,
                       mt: np.ndarray,
                       mf: np.ndarray,
                       save_path: Path,
                       title_prefix=""):
    spec = _normalize_01(spec)
    mg = _normalize_01(mg)
    mt = _normalize_01(mt)
    mf = _normalize_01(mf)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].imshow(spec, origin="lower", aspect="auto")
    axes[0].set_title(f"{title_prefix}Input")

    axes[1].imshow(spec, origin="lower", aspect="auto", alpha=0.7)
    axes[1].imshow(mg, origin="lower", aspect="auto", alpha=0.45)
    axes[1].set_title(f"{title_prefix}Global mask")

    axes[2].imshow(spec, origin="lower", aspect="auto", alpha=0.7)
    axes[2].imshow(mt, origin="lower", aspect="auto", alpha=0.45)
    axes[2].set_title(f"{title_prefix}Time mask")

    axes[3].imshow(spec, origin="lower", aspect="auto", alpha=0.7)
    axes[3].imshow(mf, origin="lower", aspect="auto", alpha=0.45)
    axes[3].set_title(f"{title_prefix}Freq mask")

    for ax in axes:
        ax.set_xlabel("Time")
        ax.set_ylabel("Freq")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300)
    plt.close()


class TFVisualizer:
    def __init__(self,
                 save_dir,
                 enable=True,
                 max_vis_batches=20,
                 max_mask_examples=8,
                 time_bins=64,
                 freq_bins=32):
        self.enable = enable and dist_utils.is_main_process()
        self.save_dir = Path(save_dir) if save_dir is not None else Path("./tf_vis")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.max_vis_batches = max_vis_batches
        self.max_mask_examples = max_mask_examples
        self.time_bins = time_bins
        self.freq_bins = freq_bins

        self._num_levels = None
        self._seen_batches = 0
        self._saved_masks = 0

        self.global_sigs = None
        self.time_sigs = None
        self.freq_sigs = None

        self.rg = []
        self.rt = []
        self.rf = []

        self.pca_X = []
        self.pca_Y = []

    def _ensure_levels(self, num_levels):
        if self._num_levels is None:
            self._num_levels = num_levels
            self.global_sigs = [[] for _ in range(num_levels)]
            self.time_sigs = [[] for _ in range(num_levels)]
            self.freq_sigs = [[] for _ in range(num_levels)]

    @torch.no_grad()
    def collect(self, samples: torch.Tensor, outputs: dict):
        if not self.enable:
            return
        if self._seen_batches >= self.max_vis_batches:
            return
        if not isinstance(outputs, dict) or ('tf_aux' not in outputs):
            return

        tf_aux = outputs['tf_aux']
        if tf_aux is None:
            return

        self._seen_batches += 1
        self._ensure_levels(len(tf_aux))

        # save mask overlay examples
        bsz = samples.shape[0]
        for bi in range(bsz):
            if self._saved_masks >= self.max_mask_examples:
                break

            # input spectrogram, take first channel
            spec = samples[bi, 0].detach().cpu().numpy()

            for li, lv in enumerate(tf_aux):
                mg = lv['mask_global'][bi].detach().cpu().mean(0).numpy()
                mt = lv['mask_time'][bi].detach().cpu().mean(0).numpy()
                mf = lv['mask_freq'][bi].detach().cpu().mean(0).numpy()

                save_path = self.save_dir / f"mask_overlay_sample{self._saved_masks:02d}_level{li}.png"
                _save_mask_overlay(spec, mg, mt, mf, save_path, title_prefix=f"L{li} ")
            self._saved_masks += 1

        # collect signatures and ratios
        for li, lv in enumerate(tf_aux):
            fg = lv['feat_global'].detach().cpu()
            ft = lv['feat_time'].detach().cpu()
            ff = lv['feat_freq'].detach().cpu()

            self.global_sigs[li].append(_global_signature(fg))
            self.time_sigs[li].append(_time_signature(ft, bins=self.time_bins))
            self.freq_sigs[li].append(_freq_signature(ff, bins=self.freq_bins))

            self.rg.extend(_directional_ratio(fg).tolist())
            self.rt.extend(_directional_ratio(ft).tolist())
            self.rf.extend(_directional_ratio(ff).tolist())

            # use level 0 for PCA collection to avoid too many points
            if li == 0:
                eg = _generic_embed(fg).numpy()
                et = _generic_embed(ft).numpy()
                ef = _generic_embed(ff).numpy()
                for i in range(eg.shape[0]):
                    self.pca_X.append(eg[i]); self.pca_Y.append(0)
                for i in range(et.shape[0]):
                    self.pca_X.append(et[i]); self.pca_Y.append(1)
                for i in range(ef.shape[0]):
                    self.pca_X.append(ef[i]); self.pca_Y.append(2)

    def _stack_level_list(self, xs):
        if len(xs) == 0:
            return None
        return torch.cat(xs, dim=0)

    def _build_similarity_matrix(self, sigs_by_level):
        n = len(sigs_by_level)
        mat = np.zeros((n, n), dtype=np.float32)
        stacked = [self._stack_level_list(v) for v in sigs_by_level]
        for i in range(n):
            for j in range(n):
                if stacked[i] is None or stacked[j] is None:
                    mat[i, j] = 0.0
                else:
                    mat[i, j] = _cosine_mean(stacked[i], stacked[j])
        return mat

    def finalize(self):
        if not self.enable or self._num_levels is None:
            return

        # similarity heatmaps
        mat_g = self._build_similarity_matrix(self.global_sigs)
        mat_t = self._build_similarity_matrix(self.time_sigs)
        mat_f = self._build_similarity_matrix(self.freq_sigs)

        _plot_heatmap(mat_g, "Global cross-level similarity",
                      self.save_dir / "tf_global_similarity.png")
        _plot_heatmap(mat_t, "Time cross-level similarity",
                      self.save_dir / "tf_time_similarity.png")
        _plot_heatmap(mat_f, "Freq cross-level similarity",
                      self.save_dir / "tf_freq_similarity.png")

        # directional boxplot
        if len(self.rg) > 0 and len(self.rt) > 0 and len(self.rf) > 0:
            _plot_direction_boxplot(np.array(self.rg), np.array(self.rt), np.array(self.rf),
                                    self.save_dir / "tf_direction_boxplot.png")

        # PCA
        if len(self.pca_X) >= 10:
            X = np.stack(self.pca_X, axis=0)
            Y = np.array(self.pca_Y)
            _plot_pca_branch_vis(X, Y, self.save_dir / "tf_branch_pca.png")


# -------------------------------------------------------
# Evaluate
# -------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator,
    device,
    num_classes=10,
    class_names=None,
    det_iou_threshold=0.5,
    save_dir=None,
    enable_tf_vis=True,
    tf_vis_batches=20,
    tf_mask_examples=8,
):
    model.eval()
    criterion.eval()

    if coco_evaluator is not None:
        coco_evaluator.cleanup()
        iou_types = coco_evaluator.iou_types
    else:
        iou_types = []

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    det_confmat = DetectionConfusionMatrix(
        num_classes=num_classes,
        iou_threshold=det_iou_threshold
    )

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    vis_dir = Path(save_dir) / "tf_vis" if save_dir is not None else Path("./tf_vis")
    tf_visualizer = TFVisualizer(
        save_dir=vis_dir,
        enable=enable_tf_vis,
        max_vis_batches=tf_vis_batches,
        max_mask_examples=tf_mask_examples,
        time_bins=64,
        freq_bins=32,
    )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        # collect tf features/masks for visualization
        tf_visualizer.collect(samples, outputs)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_target_sizes)

        # COCO evaluator
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # confusion matrix stats
        for output, target in zip(results, targets):
            det_confmat.process_batch(output, target)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    det_confmat.reduce_from_all_processes(device)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    # print and save confusion matrix
    det_confmat.print_summary(class_names=class_names)
    if dist_utils.is_main_process():
        _save_confusion_matrix(
            det_confmat.get_matrix(),
            class_names,
            vis_dir / "confusion_matrix.png"
        )
        tf_visualizer.finalize()

    stats = {}

    # COCO stats
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    # custom detection stats
    det_result = det_confmat.summary()
    stats['Event F1'] = det_result['Event F1']
    stats['mIoU'] = det_result['mIoU']
    stats['Macro-F1'] = det_result['Macro-F1']
    stats['Prec.'] = det_result['Prec.']
    stats['Recall'] = det_result['Recall']
    stats['MCC'] = det_result['MCC']
    stats['confusion_matrix'] = det_result['matrix'].tolist()

    per_class_iou = {}
    for c in range(num_classes):
        name = class_names[c] if class_names is not None else str(c)
        per_class_iou[name] = det_result['per_class'][c]['iou']
    stats['per_class_iou'] = per_class_iou

    return stats, coco_evaluator