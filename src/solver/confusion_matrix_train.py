import torch
import numpy as np
import torch.distributed as dist
from torchvision.ops import box_iou
from sklearn.metrics import matthews_corrcoef, roc_auc_score


class DetectionConfusionMatrix:
    def __init__(self, num_classes, iou_threshold=0.5):
        """
        num_classes: 不包含 background 的类别数
        iou_threshold: GT-Pred 的 IoU 匹配阈值
        """
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.bg_index = num_classes
        self.matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

        # 为 AUROC 额外保存每个类别的一组二分类标签与分数
        # class c:
        #   y_true_c: 1 表示该类别 GT 实例，0 表示该类别的错误预测
        #   y_score_c: 对应预测分数；漏检 GT 记为 0
        self.auroc_targets = [[] for _ in range(num_classes)]
        self.auroc_scores = [[] for _ in range(num_classes)]

    def reset(self):
        self.matrix.fill(0)
        self.auroc_targets = [[] for _ in range(self.num_classes)]
        self.auroc_scores = [[] for _ in range(self.num_classes)]

    @staticmethod
    def _safe_div(num, den):
        return float(num) / float(den) if den > 0 else 0.0

    def _append_auroc_sample(self, cls_id, y_true, y_score):
        if 0 <= cls_id < self.num_classes:
            self.auroc_targets[cls_id].append(int(y_true))
            self.auroc_scores[cls_id].append(float(y_score))

    def process_batch(self, detections, targets):
        pred_boxes = detections["boxes"].detach().cpu()
        pred_labels = detections["labels"].detach().cpu()

        # 关键：AUROC 需要分数
        # 若外部没有传 scores，就退化成全 1.0（此时 AUROC 没有太强意义，但代码仍可跑）
        if "scores" in detections:
            pred_scores = detections["scores"].detach().cpu()
        else:
            pred_scores = torch.ones(len(pred_boxes), dtype=torch.float32)

        gt_boxes = targets["boxes"].detach().cpu()
        gt_labels = targets["labels"].detach().cpu()

        num_pred = len(pred_boxes)
        num_gt = len(gt_boxes)

        if num_gt == 0 and num_pred == 0:
            return

        # 没有 GT：所有预测都是背景上的假阳性
        if num_gt == 0:
            for i, pred_label in enumerate(pred_labels):
                cls = int(pred_label.item())
                score = float(pred_scores[i].item())
                self.matrix[self.bg_index, cls] += 1
                self._append_auroc_sample(cls, 0, score)
            return

        # 没有预测：所有 GT 都漏检
        if num_pred == 0:
            for gt_label in gt_labels:
                cls = int(gt_label.item())
                self.matrix[cls, self.bg_index] += 1
                self._append_auroc_sample(cls, 1, 0.0)
            return

        ious = box_iou(gt_boxes, pred_boxes)

        matched_gt = set()
        matched_pred = set()
        matched_pairs = []  # (gt_idx, pred_idx)

        # 与你原来一致：纯按 IoU 做贪心匹配，不要求类别一致
        while True:
            max_iou = torch.max(ious)
            if max_iou < self.iou_threshold:
                break

            inds = torch.where(ious == max_iou)
            gt_idx = inds[0][0].item()
            pred_idx = inds[1][0].item()

            gt_class = int(gt_labels[gt_idx].item())
            pred_class = int(pred_labels[pred_idx].item())

            self.matrix[gt_class, pred_class] += 1

            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            matched_pairs.append((gt_idx, pred_idx))

            ious[gt_idx, :] = -1.0
            ious[:, pred_idx] = -1.0

        # 未匹配 GT -> FN
        for i, gt_label in enumerate(gt_labels):
            if i not in matched_gt:
                self.matrix[int(gt_label.item()), self.bg_index] += 1

        # 未匹配 Pred -> FP
        for i, pred_label in enumerate(pred_labels):
            if i not in matched_pred:
                self.matrix[self.bg_index, int(pred_label.item())] += 1

        # =========================
        # AUROC 统计
        # =========================
        # 规则：
        # 1) 正确匹配（同类）：
        #       对 gt/pred 所属类别 c，加入 (y_true=1, y_score=pred_score)
        # 2) 错分匹配（IoU 够但类别错）：
        #       对预测类别 pred_c，加入负样本 (0, pred_score)
        #       对 GT 类别 gt_c，稍后补一个漏检正样本 (1, 0)
        # 3) 未匹配预测：
        #       对预测类别 pred_c，加入负样本 (0, pred_score)
        # 4) 未被正确识别的 GT（包括未匹配和错分）：
        #       对 GT 类别 gt_c，加入正样本 (1, 0)
        gt_correct_detected = [False] * num_gt

        # 先处理已匹配对
        for gt_idx, pred_idx in matched_pairs:
            gt_class = int(gt_labels[gt_idx].item())
            pred_class = int(pred_labels[pred_idx].item())
            pred_score = float(pred_scores[pred_idx].item())

            if gt_class == pred_class:
                # 正确检测
                self._append_auroc_sample(gt_class, 1, pred_score)
                gt_correct_detected[gt_idx] = True
            else:
                # 错分预测：对预测类来说是负样本
                self._append_auroc_sample(pred_class, 0, pred_score)
                # 对 GT 类来说仍算没有被正确检出，后面补 (1,0)

        # 再处理没有被正确识别的 GT（漏检 + 错分）
        for gt_idx, gt_label in enumerate(gt_labels):
            gt_class = int(gt_label.item())
            if not gt_correct_detected[gt_idx]:
                self._append_auroc_sample(gt_class, 1, 0.0)

        # 最后处理未匹配预测（纯 FP）
        for pred_idx, pred_label in enumerate(pred_labels):
            if pred_idx not in matched_pred:
                pred_class = int(pred_label.item())
                pred_score = float(pred_scores[pred_idx].item())
                self._append_auroc_sample(pred_class, 0, pred_score)

    def reduce_from_all_processes(self, device):
        if not dist.is_available() or not dist.is_initialized():
            return

        # 1) 先规约 confusion matrix
        t = torch.tensor(self.matrix, dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        self.matrix = t.cpu().numpy().astype(np.int64)

        # 2) 再聚合 AUROC 所需的 targets / scores
        world_size = dist.get_world_size()
        gathered = [None for _ in range(world_size)]
        payload = {
            "targets": self.auroc_targets,
            "scores": self.auroc_scores,
        }
        dist.all_gather_object(gathered, payload)

        merged_targets = [[] for _ in range(self.num_classes)]
        merged_scores = [[] for _ in range(self.num_classes)]

        for item in gathered:
            if item is None:
                continue
            part_targets = item["targets"]
            part_scores = item["scores"]
            for c in range(self.num_classes):
                merged_targets[c].extend(part_targets[c])
                merged_scores[c].extend(part_scores[c])

        self.auroc_targets = merged_targets
        self.auroc_scores = merged_scores

    def get_matrix(self):
        return self.matrix.copy()

    def get_fg_matrix(self):
        """
        只取前景类，不要 background
        """
        return self.matrix[:self.num_classes, :self.num_classes].copy()

    def _per_class_stats_fg_only(self):
        """
        只在前景子矩阵上统计
        不考虑 bg 行列
        """
        fg = self.get_fg_matrix()
        per_class = {}

        for c in range(self.num_classes):
            tp = fg[c, c]
            fp = fg[:, c].sum() - tp
            fn = fg[c, :].sum() - tp

            precision = self._safe_div(tp, tp + fp)
            recall = self._safe_div(tp, tp + fn)
            f1 = self._safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
            iou = self._safe_div(tp, tp + fp + fn)

            per_class[c] = {
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou,
            }

        return per_class

    def _macro_metrics_fg_only(self, per_class):
        precisions = [per_class[c]["precision"] for c in range(self.num_classes)]
        recalls = [per_class[c]["recall"] for c in range(self.num_classes)]
        f1s = [per_class[c]["f1"] for c in range(self.num_classes)]
        ious = [per_class[c]["iou"] for c in range(self.num_classes)]

        return {
            "Prec.": float(np.mean(precisions)) if len(precisions) > 0 else 0.0,
            "Recall": float(np.mean(recalls)) if len(recalls) > 0 else 0.0,
            "Macro-F1": float(np.mean(f1s)) if len(f1s) > 0 else 0.0,
            "mIoU": float(np.mean(ious)) if len(ious) > 0 else 0.0,
        }

    def _event_f1_fg_only(self):
        """
        不计算 bg，只在前景类上算 micro-F1
        """
        fg = self.get_fg_matrix()

        tp = np.trace(fg)
        fp = fg.sum(axis=0).sum() - tp
        fn = fg.sum(axis=1).sum() - tp

        precision = self._safe_div(tp, tp + fp)
        recall = self._safe_div(tp, tp + fn)
        f1 = self._safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def _mcc_fg_only(self):
        """
        只基于前景子矩阵计算 MCC
        """
        fg = self.get_fg_matrix()

        y_true = []
        y_pred = []

        for gt_cls in range(self.num_classes):
            for pred_cls in range(self.num_classes):
                count = int(fg[gt_cls, pred_cls])
                if count > 0:
                    y_true.extend([gt_cls] * count)
                    y_pred.extend([pred_cls] * count)

        if len(y_true) == 0:
            return 0.0

        if len(set(y_true)) <= 1 and len(set(y_pred)) <= 1:
            return 0.0

        return float(matthews_corrcoef(y_true, y_pred))

    def _auroc(self):
        """
        返回:
            macro_auroc: 各类别 AUROC 的宏平均
            per_class_auroc: {c: auc_or_none}
        """
        per_class_auroc = {}
        valid_aurocs = []

        for c in range(self.num_classes):
            y_true = self.auroc_targets[c]
            y_score = self.auroc_scores[c]

            if len(y_true) == 0:
                per_class_auroc[c] = None
                continue

            # AUROC 至少需要同时存在正负样本
            uniq = set(y_true)
            if len(uniq) < 2:
                per_class_auroc[c] = None
                continue

            try:
                auc = float(roc_auc_score(y_true, y_score))
                per_class_auroc[c] = auc
                valid_aurocs.append(auc)
            except Exception:
                per_class_auroc[c] = None

        macro_auroc = float(np.mean(valid_aurocs)) if len(valid_aurocs) > 0 else 0.0
        return macro_auroc, per_class_auroc

    def summary(self):
        per_class = self._per_class_stats_fg_only()
        macro_stats = self._macro_metrics_fg_only(per_class)
        event_stats = self._event_f1_fg_only()
        mcc = self._mcc_fg_only()
        macro_auroc, per_class_auroc = self._auroc()

        # 把 AUROC 也写进 per_class
        for c in range(self.num_classes):
            per_class[c]["auroc"] = per_class_auroc[c]

        return {
            "matrix": self.get_fg_matrix(),   # 这里只返回前景矩阵
            "per_class": per_class,
            "Event F1": event_stats["f1"],
            "mIoU": macro_stats["mIoU"],
            "Macro-F1": macro_stats["Macro-F1"],
            "Prec.": macro_stats["Prec."],
            "Recall": macro_stats["Recall"],
            "MCC": mcc,
            "AUROC": macro_auroc,
        }

    def print_summary(self, class_names=None):
        result = self.summary()

        print("[Validation Metrics]")
        print(
            f"{'Event F1':>10s} | {'mIoU':>10s} | {'Macro-F1':>10s} | "
            f"{'Prec.':>10s} | {'Recall':>10s} | {'MCC':>10s} | {'AUROC':>10s}"
        )
        print(
            f"{result['Event F1']:10.4f} | "
            f"{result['mIoU']:10.4f} | "
            f"{result['Macro-F1']:10.4f} | "
            f"{result['Prec.']:10.4f} | "
            f"{result['Recall']:10.4f} | "
            f"{result['MCC']:10.4f} | "
            f"{result['AUROC']:10.4f}"
        )

        print("\n[Confusion Matrix - FG Only]")
        print(result["matrix"])

        print("\n[Per-Class IoU / AUROC]")
        for c in range(self.num_classes):
            name = class_names[c] if class_names is not None else str(c)
            cls_iou = result["per_class"][c]["iou"]
            cls_auc = result["per_class"][c]["auroc"]
            cls_auc_str = f"{cls_auc:.4f}" if cls_auc is not None else "N/A"
            print(f"{name}: IoU={cls_iou:.4f}, AUROC={cls_auc_str}")