import torch
import numpy as np
import torch.distributed as dist
from torchvision.ops import box_iou
from sklearn.metrics import matthews_corrcoef
from typing import Optional

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from psds_eval import PSDSEval
except Exception:
    PSDSEval = None


class DetectionConfusionMatrix:
    def __init__(
        self,
        num_classes,
        iou_threshold=0.5,
        time_iou_threshold=None,
        freq_iou_threshold=None,
        freq_map_iou_threshold=0.5,
        class_names=None,
        psds_gt_box_format="xywh",
        psds_pred_box_format="xyxy",
    ):
        """
        num_classes: 不包含 background 的类别数
        iou_threshold: 时频整体 box 的 IoU 匹配阈值（主流程仍按 xyxy）
        time_iou_threshold: 只看时间轴时的 IoU 阈值
        freq_iou_threshold: 只看频率轴时的 IoU 阈值
        freq_map_iou_threshold: 计算 Freq-mAP 时使用的频率轴 IoU 阈值
        class_names: 类别名列表，可选

        注意：
        - 主流程 process_batch / box_iou 默认仍按 xyxy 处理，不动你现有的检测指标逻辑
        - 只有 PSDS 导出支路单独支持 GT / PRED 不同 box 格式
        """
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.time_iou_threshold = iou_threshold if time_iou_threshold is None else time_iou_threshold
        self.freq_iou_threshold = iou_threshold if freq_iou_threshold is None else freq_iou_threshold
        self.freq_map_iou_threshold = freq_map_iou_threshold

        self.class_names = class_names
        self.bg_index = num_classes

        self.psds_gt_box_format = psds_gt_box_format
        self.psds_pred_box_format = psds_pred_box_format

        self.reset()

    def reset(self):
        self.matrix = np.zeros((self.num_classes + 1, self.num_classes + 1), dtype=np.int64)

        self.per_class_gt = np.zeros(self.num_classes, dtype=np.int64)
        self.per_class_matched_correct = np.zeros(self.num_classes, dtype=np.int64)

        self.per_class_time_correct = np.zeros(self.num_classes, dtype=np.int64)
        self.per_class_freq_correct = np.zeros(self.num_classes, dtype=np.int64)
        self.per_class_tf_correct = np.zeros(self.num_classes, dtype=np.int64)

        self.per_class_time_iou_sum = np.zeros(self.num_classes, dtype=np.float64)
        self.per_class_freq_iou_sum = np.zeros(self.num_classes, dtype=np.float64)
        self.per_class_tf_iou_sum = np.zeros(self.num_classes, dtype=np.float64)

        self.total_gt_events = 0
        self.total_pred_events = 0
        self.total_matched_pairs = 0
        self.total_cls_correct_pairs = 0
        self.total_time_correct = 0
        self.total_freq_correct = 0
        self.total_tf_correct = 0

        self.freq_map_gt_count = np.zeros(self.num_classes, dtype=np.int64)
        self.freq_map_records = {c: [] for c in range(self.num_classes)}

        self.psds_gt_rows = []
        self.psds_pred_rows = []
        self.psds_meta = {}
        self.sample_counter = 0

    # =========================================================
    # 基础检查
    # =========================================================
    def _check_labels(self, labels, name="labels"):
        if len(labels) == 0:
            return
        min_v = int(labels.min().item())
        max_v = int(labels.max().item())
        if min_v < 0 or max_v >= self.num_classes:
            raise ValueError(
                f"{name} out of range: min={min_v}, max={max_v}, "
                f"but valid range is [0, {self.num_classes - 1}]"
            )

    @staticmethod
    def _safe_div(num, den):
        return float(num) / float(den) if den > 0 else 0.0

    # =========================================================
    # 1D IoU：时间轴 / 频率轴
    # 主流程默认按 xyxy
    # =========================================================
    @staticmethod
    def _interval_iou_matrix(starts1, ends1, starts2, ends2):
        if starts1.numel() == 0 or starts2.numel() == 0:
            return torch.zeros((starts1.numel(), starts2.numel()), dtype=torch.float32)

        inter_l = torch.maximum(starts1[:, None], starts2[None, :])
        inter_r = torch.minimum(ends1[:, None], ends2[None, :])
        inter = (inter_r - inter_l).clamp(min=0)

        len1 = (ends1 - starts1).clamp(min=0)[:, None]
        len2 = (ends2 - starts2).clamp(min=0)[None, :]
        union = len1 + len2 - inter
        return inter / union.clamp(min=1e-12)

    @classmethod
    def _time_iou_matrix(cls, gt_boxes, pred_boxes):
        return cls._interval_iou_matrix(
            gt_boxes[:, 0], gt_boxes[:, 2],
            pred_boxes[:, 0], pred_boxes[:, 2]
        )

    @classmethod
    def _freq_iou_matrix(cls, gt_boxes, pred_boxes):
        return cls._interval_iou_matrix(
            gt_boxes[:, 1], gt_boxes[:, 3],
            pred_boxes[:, 1], pred_boxes[:, 3]
        )

    @staticmethod
    def _single_interval_iou(a1, a2, b1, b2):
        inter = max(0.0, min(a2, b2) - max(a1, b1))
        union = max(0.0, a2 - a1) + max(0.0, b2 - b1) - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def _single_time_iou(cls, gt_box, pred_box):
        return cls._single_interval_iou(
            float(gt_box[0]), float(gt_box[2]),
            float(pred_box[0]), float(pred_box[2])
        )

    @classmethod
    def _single_freq_iou(cls, gt_box, pred_box):
        return cls._single_interval_iou(
            float(gt_box[1]), float(gt_box[3]),
            float(pred_box[1]), float(pred_box[3])
        )

    # =========================================================
    # 频率轴 mAP
    # =========================================================
    @staticmethod
    def _compute_ap_from_records(records, gt_count):
        if gt_count <= 0:
            return np.nan
        if len(records) == 0:
            return 0.0

        records = sorted(records, key=lambda x: x["score"], reverse=True)

        tp = np.array([r["tp"] for r in records], dtype=np.float64)
        fp = 1.0 - tp

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recalls = tp_cum / max(float(gt_count), 1e-12)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        return float(ap)

    def _update_freq_map_records(self, detections, targets):
        pred_boxes = detections["boxes"].detach().cpu().float()
        pred_labels = detections["labels"].detach().cpu().long()
        pred_scores = detections.get("scores", None)
        if pred_scores is None:
            pred_scores = torch.ones((len(pred_boxes),), dtype=torch.float32)
        else:
            pred_scores = pred_scores.detach().cpu().float()

        gt_boxes = targets["boxes"].detach().cpu().float()
        gt_labels = targets["labels"].detach().cpu().long()

        for c in range(self.num_classes):
            gt_mask = (gt_labels == c)
            pred_mask = (pred_labels == c)

            gt_cls_boxes = gt_boxes[gt_mask]
            pred_cls_boxes = pred_boxes[pred_mask]
            pred_cls_scores = pred_scores[pred_mask]

            self.freq_map_gt_count[c] += int(len(gt_cls_boxes))

            if len(pred_cls_boxes) == 0:
                continue

            order = torch.argsort(pred_cls_scores, descending=True)
            pred_cls_boxes = pred_cls_boxes[order]
            pred_cls_scores = pred_cls_scores[order]

            if len(gt_cls_boxes) == 0:
                for s in pred_cls_scores:
                    self.freq_map_records[c].append({"score": float(s.item()), "tp": 0})
                continue

            freq_ious = self._freq_iou_matrix(gt_cls_boxes, pred_cls_boxes)
            matched_gt = set()

            for p_idx in range(len(pred_cls_boxes)):
                ious_to_gt = freq_ious[:, p_idx]
                best_iou, best_gt_idx = torch.max(ious_to_gt, dim=0)

                best_iou = float(best_iou.item())
                best_gt_idx = int(best_gt_idx.item())

                is_tp = 0
                if best_iou >= self.freq_map_iou_threshold and best_gt_idx not in matched_gt:
                    is_tp = 1
                    matched_gt.add(best_gt_idx)

                self.freq_map_records[c].append({
                    "score": float(pred_cls_scores[p_idx].item()),
                    "tp": is_tp
                })

    # =========================================================
    # PSDS 辅助函数
    # =========================================================
    def _label_to_name(self, cls_id):
        if self.class_names is not None and 0 <= cls_id < len(self.class_names):
            return self.class_names[cls_id]
        return str(cls_id)

    @staticmethod
    def _sanitize_psds_df(df, has_score=False, min_duration=1e-8):
        if pd is None:
            raise ImportError("pandas is not installed.")

        cols = ["filename", "onset", "offset", "event_label"]
        if has_score:
            cols.append("score")

        if df is None or len(df) == 0:
            return pd.DataFrame(columns=cols)

        df = df.copy()
        df["filename"] = df["filename"].astype(str)
        df["event_label"] = df["event_label"].astype(str)
        df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
        df["offset"] = pd.to_numeric(df["offset"], errors="coerce")

        if has_score:
            df["score"] = pd.to_numeric(df["score"], errors="coerce")
            df = df.dropna(subset=["filename", "event_label", "onset", "offset", "score"]).copy()
        else:
            df = df.dropna(subset=["filename", "event_label", "onset", "offset"]).copy()

        if len(df) == 0:
            return pd.DataFrame(columns=cols)

        df["onset"] = df["onset"].clip(lower=0.0)
        df["offset"] = np.maximum(df["offset"].to_numpy(), df["onset"].to_numpy() + float(min_duration))

        sort_cols = ["filename", "event_label", "onset", "offset"]
        if has_score:
            sort_cols.append("score")

        df = df[cols].sort_values(sort_cols).reset_index(drop=True)
        return df

    @staticmethod
    def _merge_psds_intersections(df, has_score=False, merge_gap=0.0, score_mode="max"):
        if pd is None:
            raise ImportError("pandas is not installed.")

        cols = ["filename", "onset", "offset", "event_label"]
        if has_score:
            cols.append("score")

        if df is None or len(df) == 0:
            return pd.DataFrame(columns=cols)

        df = df.copy().sort_values(["filename", "event_label", "onset", "offset"]).reset_index(drop=True)
        merged_rows = []

        for (fname, label), g in df.groupby(["filename", "event_label"], sort=False):
            g = g.sort_values(["onset", "offset"]).reset_index(drop=True)

            cur_onset = float(g.loc[0, "onset"])
            cur_offset = float(g.loc[0, "offset"])
            if has_score:
                cur_scores = [float(g.loc[0, "score"])]

            for i in range(1, len(g)):
                onset = float(g.loc[i, "onset"])
                offset = float(g.loc[i, "offset"])

                if onset <= cur_offset + float(merge_gap):
                    cur_offset = max(cur_offset, offset)
                    if has_score:
                        cur_scores.append(float(g.loc[i, "score"]))
                else:
                    row = {
                        "filename": fname,
                        "onset": cur_onset,
                        "offset": cur_offset,
                        "event_label": label,
                    }
                    if has_score:
                        row["score"] = float(np.mean(cur_scores)) if score_mode == "mean" else float(np.max(cur_scores))
                    merged_rows.append(row)

                    cur_onset = onset
                    cur_offset = offset
                    if has_score:
                        cur_scores = [float(g.loc[i, "score"])]

            row = {
                "filename": fname,
                "onset": cur_onset,
                "offset": cur_offset,
                "event_label": label,
            }
            if has_score:
                row["score"] = float(np.mean(cur_scores)) if score_mode == "mean" else float(np.max(cur_scores))
            merged_rows.append(row)

        return pd.DataFrame(merged_rows, columns=cols)

    def _get_time_bounds_for_psds(self, boxes: torch.Tensor, box_format: str):
        """
        只给 PSDS 导出用。
        """
        if boxes.numel() == 0:
            empty = torch.zeros((0,), dtype=torch.float32)
            return empty, empty

        boxes = boxes.detach().cpu().float()

        if box_format == "xywh":
            x1 = boxes[:, 0]
            x2 = boxes[:, 0] + boxes[:, 2]
        elif box_format == "xyxy":
            x1 = boxes[:, 0]
            x2 = boxes[:, 2]
        else:
            raise ValueError(f"Unsupported box format: {box_format}")

        onset = torch.minimum(x1, x2)
        offset = torch.maximum(x1, x2)
        return onset, offset

    @staticmethod
    def _pixels_to_seconds(x: torch.Tensor, width: float, clip_duration: float):
        width = max(float(width), 1e-8)
        clip_duration = max(float(clip_duration), 1e-8)
        sec = (x / width) * clip_duration
        return torch.clamp(sec, min=0.0, max=clip_duration)

    # =========================================================
    # PSDS 缓存：把时频框投影到时间轴
    # =========================================================
    def _update_psds_cache(
        self,
        detections,
        targets,
        file_id=None,
        clip_duration=None,
        time_scale=1.0,       # 保留接口，但 PSDS 不再直接用它乘像素坐标
        time_axis_width=512.0
    ):
        pred_boxes = detections["boxes"].detach().cpu().float()
        pred_labels = detections["labels"].detach().cpu().long()
        pred_scores = detections.get("scores", None)
        if pred_scores is None:
            pred_scores = torch.ones((len(pred_boxes),), dtype=torch.float32)
        else:
            pred_scores = pred_scores.detach().cpu().float()

        gt_boxes = targets["boxes"].detach().cpu().float()
        gt_labels = targets["labels"].detach().cpu().long()

        if file_id is None:
            file_id = f"sample_{self.sample_counter:08d}"
            self.sample_counter += 1

        if clip_duration is None:
            clip_duration = 2.56

        width = float(time_axis_width)

        gt_x1, gt_x2 = self._get_time_bounds_for_psds(gt_boxes, self.psds_gt_box_format)
        pred_x1, pred_x2 = self._get_time_bounds_for_psds(pred_boxes, self.psds_pred_box_format)

        gt_onsets = self._pixels_to_seconds(gt_x1, width, clip_duration)
        gt_offsets = self._pixels_to_seconds(gt_x2, width, clip_duration)

        pred_onsets = self._pixels_to_seconds(pred_x1, width, clip_duration)
        pred_offsets = self._pixels_to_seconds(pred_x2, width, clip_duration)

        self.psds_meta[file_id] = max(float(clip_duration), 1e-8)

        for i in range(len(gt_boxes)):
            cls_id = int(gt_labels[i].item())
            self.psds_gt_rows.append({
                "filename": file_id,
                "onset": float(gt_onsets[i].item()),
                "offset": float(gt_offsets[i].item()),
                "event_label": self._label_to_name(cls_id),
            })

        for i in range(len(pred_boxes)):
            cls_id = int(pred_labels[i].item())
            self.psds_pred_rows.append({
                "filename": file_id,
                "onset": float(pred_onsets[i].item()),
                "offset": float(pred_offsets[i].item()),
                "event_label": self._label_to_name(cls_id),
                "score": float(pred_scores[i].item()),
            })

    def build_psds_tables(
        self,
        merge_gt=True,
        gt_merge_gap=0.0,
        merge_pred_global=False,
        pred_merge_gap=0.0,
        pred_score_mode="max",
        min_duration=1e-8,
    ):
        if pd is None:
            raise ImportError("pandas is not installed, cannot build PSDS tables.")

        gt_df = pd.DataFrame(
            self.psds_gt_rows,
            columns=["filename", "onset", "offset", "event_label"]
        )
        pred_df = pd.DataFrame(
            self.psds_pred_rows,
            columns=["filename", "onset", "offset", "event_label", "score"]
        )
        meta_df = pd.DataFrame(
            [{"filename": k, "duration": v} for k, v in self.psds_meta.items()],
            columns=["filename", "duration"]
        )

        gt_df = self._sanitize_psds_df(gt_df, has_score=False, min_duration=min_duration)
        pred_df = self._sanitize_psds_df(pred_df, has_score=True, min_duration=min_duration)

        if merge_gt:
            gt_df = self._merge_psds_intersections(
                gt_df,
                has_score=False,
                merge_gap=gt_merge_gap,
            )

        if merge_pred_global:
            pred_df = self._merge_psds_intersections(
                pred_df,
                has_score=True,
                merge_gap=pred_merge_gap,
                score_mode=pred_score_mode,
            )

        if len(meta_df) > 0:
            meta_df = meta_df.copy()
            meta_df["filename"] = meta_df["filename"].astype(str)
            meta_df["duration"] = pd.to_numeric(meta_df["duration"], errors="coerce")
            meta_df = meta_df.dropna(subset=["filename", "duration"]).copy()
            meta_df["duration"] = meta_df["duration"].clip(lower=float(min_duration))
            meta_df = meta_df.drop_duplicates(subset=["filename"], keep="last").reset_index(drop=True)

        return gt_df, pred_df, meta_df

    def export_psds_tables(
        self,
        save_dir,
        merge_gt=True,
        gt_merge_gap=0.0,
        merge_pred_global=False,
        pred_merge_gap=0.0,
        pred_score_mode="max",
        min_duration=1e-8,
    ):
        if pd is None:
            raise ImportError("pandas is not installed, cannot export PSDS tables.")

        from pathlib import Path

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        gt_df, pred_df, meta_df = self.build_psds_tables(
            merge_gt=merge_gt,
            gt_merge_gap=gt_merge_gap,
            merge_pred_global=merge_pred_global,
            pred_merge_gap=pred_merge_gap,
            pred_score_mode=pred_score_mode,
            min_duration=min_duration,
        )

        gt_path = save_dir / "psds_gt.csv"
        pred_path = save_dir / "psds_pred.csv"
        meta_path = save_dir / "psds_meta.csv"

        gt_df.to_csv(gt_path, index=False, encoding="utf-8-sig")
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
        meta_df.to_csv(meta_path, index=False, encoding="utf-8-sig")

        print(f"[PSDS tables saved]")
        print(f"GT   -> {gt_path}")
        print(f"PRED -> {pred_path}")
        print(f"META -> {meta_path}")

        return gt_df, pred_df, meta_df

    def compute_psds(
        self,
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
        min_duration=1e-8,
    ):
        if pd is None:
            return {"PSDS": None, "reason": "pandas is not installed"}
        if PSDSEval is None:
            return {"PSDS": None, "reason": "psds_eval is not installed"}

        gt_df, pred_df, meta_df = self.build_psds_tables(
            merge_gt=merge_gt,
            gt_merge_gap=gt_merge_gap,
            merge_pred_global=False,
            pred_merge_gap=pred_merge_gap,
            pred_score_mode=pred_score_mode,
            min_duration=min_duration,
        )

        if len(gt_df) == 0:
            return {"PSDS": None, "reason": "empty ground truth"}
        if len(meta_df) == 0:
            return {"PSDS": None, "reason": "empty metadata"}

        try:
            psds_eval = PSDSEval(
                dtc_threshold=dtc_threshold,
                gtc_threshold=gtc_threshold,
                cttc_threshold=cttc_threshold,
                ground_truth=gt_df,
                metadata=meta_df,
            )

            if len(pred_df) == 0:
                return {"PSDS": 0.0, "reason": "empty predictions"}

            unique_scores = np.unique(pred_df["score"].to_numpy(dtype=np.float64))
            unique_scores = np.sort(unique_scores)

            if len(unique_scores) == 0:
                return {"PSDS": 0.0, "reason": "empty predictions"}

            if len(unique_scores) > num_operating_points:
                idx = np.linspace(0, len(unique_scores) - 1, num_operating_points).astype(int)
                thresholds = np.unique(unique_scores[idx])
            else:
                thresholds = unique_scores

            thresholds = thresholds[::-1]
            num_valid_ops = 0

            for thr in thresholds:
                det_df = pred_df[pred_df["score"] >= float(thr)].copy()
                det_df = self._sanitize_psds_df(det_df, has_score=True, min_duration=min_duration)

                if merge_pred:
                    det_df = self._merge_psds_intersections(
                        det_df,
                        has_score=True,
                        merge_gap=pred_merge_gap,
                        score_mode=pred_score_mode,
                    )

                det_df = det_df[["filename", "onset", "offset", "event_label"]].copy()
                if len(det_df) == 0:
                    continue

                psds_eval.add_operating_point(det_df, info={"name": f"thr_{float(thr):.6f}"})
                num_valid_ops += 1

            if num_valid_ops == 0:
                return {"PSDS": 0.0, "reason": "no valid operating points"}

            psds_val = psds_eval.psds(alpha_ct=alpha_ct, alpha_st=alpha_st, max_efpr=max_efpr)
            score = float(psds_val.value) if hasattr(psds_val, "value") else float(psds_val)

            return {
                "PSDS": score,
                "dtc_threshold": dtc_threshold,
                "gtc_threshold": gtc_threshold,
                "cttc_threshold": cttc_threshold,
                "alpha_ct": alpha_ct,
                "alpha_st": alpha_st,
                "max_efpr": max_efpr,
                "num_operating_points": int(num_valid_ops),
                "merge_gt": merge_gt,
                "gt_merge_gap": gt_merge_gap,
                "merge_pred": merge_pred,
                "pred_merge_gap": pred_merge_gap,
                "pred_score_mode": pred_score_mode,
            }
        except Exception as e:
            return {"PSDS": None, "reason": f"psds_eval failed: {repr(e)}"}

    # =========================================================
    # 主处理逻辑
    # =========================================================
    def process_batch(
        self,
        detections,
        targets,
        file_id=None,
        clip_duration=None,
        time_scale=1.0,
        time_axis_width=512.0,
        update_psds_cache=True,
    ):
        """
        主流程仍按 xyxy 假设来算 box_iou / time_iou / freq_iou。
        PSDS 导出支路的格式解释由 psds_gt_box_format / psds_pred_box_format 单独控制。
        """
        pred_boxes = detections["boxes"].detach().cpu().float()
        pred_labels = detections["labels"].detach().cpu().long()

        gt_boxes = targets["boxes"].detach().cpu().float()
        gt_labels = targets["labels"].detach().cpu().long()

        self._check_labels(pred_labels, "pred_labels")
        self._check_labels(gt_labels, "gt_labels")

        num_pred = len(pred_boxes)
        num_gt = len(gt_boxes)

        self.total_gt_events += int(num_gt)
        self.total_pred_events += int(num_pred)

        for gt_label in gt_labels:
            self.per_class_gt[int(gt_label.item())] += 1

        self._update_freq_map_records(detections, targets)

        if update_psds_cache:
            self._update_psds_cache(
                detections=detections,
                targets=targets,
                file_id=file_id,
                clip_duration=clip_duration,
                time_scale=time_scale,
                time_axis_width=time_axis_width,
            )

        if num_gt == 0 and num_pred == 0:
            return

        if num_gt == 0:
            for pred_label in pred_labels:
                self.matrix[self.bg_index, int(pred_label.item())] += 1
            return

        if num_pred == 0:
            for gt_label in gt_labels:
                self.matrix[int(gt_label.item()), self.bg_index] += 1
            return

        ious = box_iou(gt_boxes, pred_boxes)
        matched_gt = set()
        matched_pred = set()

        while True:
            max_iou = torch.max(ious)
            if float(max_iou.item()) < self.iou_threshold:
                break

            inds = torch.where(ious == max_iou)
            gt_idx = int(inds[0][0].item())
            pred_idx = int(inds[1][0].item())

            gt_class = int(gt_labels[gt_idx].item())
            pred_class = int(pred_labels[pred_idx].item())

            self.matrix[gt_class, pred_class] += 1
            self.total_matched_pairs += 1

            gt_box = gt_boxes[gt_idx]
            pred_box = pred_boxes[pred_idx]

            time_iou = self._single_time_iou(gt_box, pred_box)
            freq_iou = self._single_freq_iou(gt_box, pred_box)
            tf_iou = float(max_iou.item())

            if gt_class == pred_class:
                self.total_cls_correct_pairs += 1
                self.per_class_matched_correct[gt_class] += 1

                self.per_class_time_iou_sum[gt_class] += time_iou
                self.per_class_freq_iou_sum[gt_class] += freq_iou
                self.per_class_tf_iou_sum[gt_class] += tf_iou

                if time_iou >= self.time_iou_threshold:
                    self.total_time_correct += 1
                    self.per_class_time_correct[gt_class] += 1

                if freq_iou >= self.freq_iou_threshold:
                    self.total_freq_correct += 1
                    self.per_class_freq_correct[gt_class] += 1

                if (time_iou >= self.time_iou_threshold) and (freq_iou >= self.freq_iou_threshold):
                    self.total_tf_correct += 1
                    self.per_class_tf_correct[gt_class] += 1

            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)

            ious[gt_idx, :] = -1.0
            ious[:, pred_idx] = -1.0

        for i, gt_label in enumerate(gt_labels):
            if i not in matched_gt:
                self.matrix[int(gt_label.item()), self.bg_index] += 1

        for i, pred_label in enumerate(pred_labels):
            if i not in matched_pred:
                self.matrix[self.bg_index, int(pred_label.item())] += 1

    # =========================================================
    # 分布式汇总
    # =========================================================
    def reduce_from_all_processes(self, device):
        if not dist.is_available() or not dist.is_initialized():
            return

        int_stats = np.concatenate([
            self.matrix.reshape(-1),
            self.per_class_gt,
            self.per_class_matched_correct,
            self.per_class_time_correct,
            self.per_class_freq_correct,
            self.per_class_tf_correct,
            np.array([
                self.total_gt_events,
                self.total_pred_events,
                self.total_matched_pairs,
                self.total_cls_correct_pairs,
                self.total_time_correct,
                self.total_freq_correct,
                self.total_tf_correct,
            ], dtype=np.int64)
        ]).astype(np.int64)

        float_stats = np.concatenate([
            self.per_class_time_iou_sum,
            self.per_class_freq_iou_sum,
            self.per_class_tf_iou_sum,
        ]).astype(np.float64)

        freq_map_gt = self.freq_map_gt_count.astype(np.int64)

        t_int = torch.tensor(int_stats, dtype=torch.long, device=device)
        t_float = torch.tensor(float_stats, dtype=torch.float64, device=device)
        t_freq_gt = torch.tensor(freq_map_gt, dtype=torch.long, device=device)

        dist.all_reduce(t_int, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_float, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_freq_gt, op=dist.ReduceOp.SUM)

        int_stats = t_int.cpu().numpy()
        float_stats = t_float.cpu().numpy()
        self.freq_map_gt_count = t_freq_gt.cpu().numpy().astype(np.int64)

        p = 0
        mat_size = (self.num_classes + 1) * (self.num_classes + 1)
        self.matrix = int_stats[p:p + mat_size].reshape(self.num_classes + 1, self.num_classes + 1).astype(np.int64)
        p += mat_size

        self.per_class_gt = int_stats[p:p + self.num_classes].astype(np.int64)
        p += self.num_classes

        self.per_class_matched_correct = int_stats[p:p + self.num_classes].astype(np.int64)
        p += self.num_classes

        self.per_class_time_correct = int_stats[p:p + self.num_classes].astype(np.int64)
        p += self.num_classes

        self.per_class_freq_correct = int_stats[p:p + self.num_classes].astype(np.int64)
        p += self.num_classes

        self.per_class_tf_correct = int_stats[p:p + self.num_classes].astype(np.int64)
        p += self.num_classes

        self.total_gt_events = int(int_stats[p]); p += 1
        self.total_pred_events = int(int_stats[p]); p += 1
        self.total_matched_pairs = int(int_stats[p]); p += 1
        self.total_cls_correct_pairs = int(int_stats[p]); p += 1
        self.total_time_correct = int(int_stats[p]); p += 1
        self.total_freq_correct = int(int_stats[p]); p += 1
        self.total_tf_correct = int(int_stats[p]); p += 1

        q = 0
        self.per_class_time_iou_sum = float_stats[q:q + self.num_classes].astype(np.float64)
        q += self.num_classes
        self.per_class_freq_iou_sum = float_stats[q:q + self.num_classes].astype(np.float64)
        q += self.num_classes
        self.per_class_tf_iou_sum = float_stats[q:q + self.num_classes].astype(np.float64)

        payload = {
            "freq_map_records": self.freq_map_records,
            "psds_gt_rows": self.psds_gt_rows,
            "psds_pred_rows": self.psds_pred_rows,
            "psds_meta": self.psds_meta,
        }
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, payload)

        merged_freq_map_records = {c: [] for c in range(self.num_classes)}
        merged_psds_gt_rows = []
        merged_psds_pred_rows = []
        merged_psds_meta = {}

        for obj in gathered:
            if obj is None:
                continue
            for c in range(self.num_classes):
                merged_freq_map_records[c].extend(obj["freq_map_records"][c])
            merged_psds_gt_rows.extend(obj["psds_gt_rows"])
            merged_psds_pred_rows.extend(obj["psds_pred_rows"])
            merged_psds_meta.update(obj["psds_meta"])

        self.freq_map_records = merged_freq_map_records
        self.psds_gt_rows = merged_psds_gt_rows
        self.psds_pred_rows = merged_psds_pred_rows
        self.psds_meta = merged_psds_meta

    # =========================================================
    # 原有 confusion matrix 指标
    # =========================================================
    def get_matrix(self):
        return self.matrix.copy()

    def get_fg_matrix(self):
        return self.matrix[:self.num_classes, :self.num_classes].copy()

    def _per_class_stats_fg_only(self):
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

            matched_corr = int(self.per_class_matched_correct[c])
            gt_cnt = int(self.per_class_gt[c])

            time_iou_mean = self._safe_div(self.per_class_time_iou_sum[c], matched_corr)
            freq_iou_mean = self._safe_div(self.per_class_freq_iou_sum[c], matched_corr)
            tf_iou_mean = self._safe_div(self.per_class_tf_iou_sum[c], matched_corr)

            time_acc = self._safe_div(self.per_class_time_correct[c], gt_cnt)
            freq_acc = self._safe_div(self.per_class_freq_correct[c], gt_cnt)
            tf_acc = self._safe_div(self.per_class_tf_correct[c], gt_cnt)

            per_class[c] = {
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou,
                "gt_count": gt_cnt,
                "matched_correct": matched_corr,
                "time_iou_mean": time_iou_mean,
                "freq_iou_mean": freq_iou_mean,
                "tf_iou_mean": tf_iou_mean,
                "time_acc": time_acc,
                "freq_acc": freq_acc,
                "tf_acc": tf_acc,
            }

        return per_class

    def _macro_metrics_fg_only(self, per_class):
        precisions = [per_class[c]["precision"] for c in range(self.num_classes)]
        recalls = [per_class[c]["recall"] for c in range(self.num_classes)]
        f1s = [per_class[c]["f1"] for c in range(self.num_classes)]
        ious = [per_class[c]["iou"] for c in range(self.num_classes)]

        time_iou_means = [per_class[c]["time_iou_mean"] for c in range(self.num_classes)]
        freq_iou_means = [per_class[c]["freq_iou_mean"] for c in range(self.num_classes)]
        tf_iou_means = [per_class[c]["tf_iou_mean"] for c in range(self.num_classes)]

        time_accs = [per_class[c]["time_acc"] for c in range(self.num_classes)]
        freq_accs = [per_class[c]["freq_acc"] for c in range(self.num_classes)]
        tf_accs = [per_class[c]["tf_acc"] for c in range(self.num_classes)]

        return {
            "Prec.": float(np.mean(precisions)) if len(precisions) > 0 else 0.0,
            "Recall": float(np.mean(recalls)) if len(recalls) > 0 else 0.0,
            "Macro-F1": float(np.mean(f1s)) if len(f1s) > 0 else 0.0,
            "mIoU": float(np.mean(ious)) if len(ious) > 0 else 0.0,
            "mTime-IoU": float(np.mean(time_iou_means)) if len(time_iou_means) > 0 else 0.0,
            "mFreq-IoU": float(np.mean(freq_iou_means)) if len(freq_iou_means) > 0 else 0.0,
            "mTF-IoU": float(np.mean(tf_iou_means)) if len(tf_iou_means) > 0 else 0.0,
            "Time-Acc": float(np.mean(time_accs)) if len(time_accs) > 0 else 0.0,
            "Freq-Acc": float(np.mean(freq_accs)) if len(freq_accs) > 0 else 0.0,
            "TF-Acc": float(np.mean(tf_accs)) if len(tf_accs) > 0 else 0.0,
        }

    def _event_f1_fg_only(self):
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

    def _overall_localization_acc(self):
        return {
            "overall_time_acc": self._safe_div(self.total_time_correct, self.total_gt_events),
            "overall_freq_acc": self._safe_div(self.total_freq_correct, self.total_gt_events),
            "overall_tf_acc": self._safe_div(self.total_tf_correct, self.total_gt_events),
        }

    def _freq_map_summary(self):
        ap_per_class = {}
        valid_aps = []

        for c in range(self.num_classes):
            ap = self._compute_ap_from_records(self.freq_map_records[c], int(self.freq_map_gt_count[c]))
            ap_per_class[c] = ap
            if not np.isnan(ap):
                valid_aps.append(ap)

        freq_map = float(np.mean(valid_aps)) if len(valid_aps) > 0 else 0.0
        return {
            "Freq-mAP": freq_map,
            "freq_ap_per_class": ap_per_class,
            "freq_map_iou_threshold": self.freq_map_iou_threshold,
        }

    def summary(self, compute_psds=False, psds_kwargs=None):
        per_class = self._per_class_stats_fg_only()
        macro_stats = self._macro_metrics_fg_only(per_class)
        event_stats = self._event_f1_fg_only()
        mcc = self._mcc_fg_only()
        overall_acc = self._overall_localization_acc()
        freq_map_stats = self._freq_map_summary()

        result = {
            "matrix": self.get_fg_matrix(),
            "per_class": per_class,
            "Event F1": event_stats["f1"],
            "mIoU": macro_stats["mIoU"],
            "Macro-F1": macro_stats["Macro-F1"],
            "Prec.": macro_stats["Prec."],
            "Recall": macro_stats["Recall"],
            "MCC": mcc,

            "mTime-IoU": macro_stats["mTime-IoU"],
            "mFreq-IoU": macro_stats["mFreq-IoU"],
            "mTF-IoU": macro_stats["mTF-IoU"],

            "Time-Acc": macro_stats["Time-Acc"],
            "Freq-Acc": macro_stats["Freq-Acc"],
            "TF-Acc": macro_stats["TF-Acc"],

            "overall_time_acc": overall_acc["overall_time_acc"],
            "overall_freq_acc": overall_acc["overall_freq_acc"],
            "overall_tf_acc": overall_acc["overall_tf_acc"],

            "Freq-mAP": freq_map_stats["Freq-mAP"],
            "freq_ap_per_class": freq_map_stats["freq_ap_per_class"],
            "freq_map_iou_threshold": freq_map_stats["freq_map_iou_threshold"],
        }

        psds_kwargs = {} if psds_kwargs is None else psds_kwargs
        result["PSDS"] = self.compute_psds(**psds_kwargs) if compute_psds else None

        return result

    def print_summary(self, class_names=None, compute_psds=False, psds_kwargs=None):
        result = self.summary(compute_psds=compute_psds, psds_kwargs=psds_kwargs)

        print("[Validation Metrics]")
        print(
            f"{'Event F1':>10s} | {'mIoU':>10s} | {'Macro-F1':>10s} | "
            f"{'Prec.':>10s} | {'Recall':>10s} | {'MCC':>10s}"
        )
        print(
            f"{result['Event F1']:10.4f} | "
            f"{result['mIoU']:10.4f} | "
            f"{result['Macro-F1']:10.4f} | "
            f"{result['Prec.']:10.4f} | "
            f"{result['Recall']:10.4f} | "
            f"{result['MCC']:10.4f}"
        )

        print("\n[Localization Metrics]")
        print(
            f"{'mTime-IoU':>10s} | {'mFreq-IoU':>10s} | {'mTF-IoU':>10s} | "
            f"{'Time-Acc':>10s} | {'Freq-Acc':>10s} | {'TF-Acc':>10s} | {'Freq-mAP':>10s}"
        )
        print(
            f"{result['mTime-IoU']:10.4f} | "
            f"{result['mFreq-IoU']:10.4f} | "
            f"{result['mTF-IoU']:10.4f} | "
            f"{result['Time-Acc']:10.4f} | "
            f"{result['Freq-Acc']:10.4f} | "
            f"{result['TF-Acc']:10.4f} | "
            f"{result['Freq-mAP']:10.4f}"
        )

        print("\n[Overall Localization Acc]")
        print(
            f"overall_time_acc = {result['overall_time_acc']:.4f}\n"
            f"overall_freq_acc = {result['overall_freq_acc']:.4f}\n"
            f"overall_tf_acc   = {result['overall_tf_acc']:.4f}"
        )

        print("\n[Confusion Matrix - FG Only]")
        print(result["matrix"])

        print("\n[Per-Class Metrics]")
        for c in range(self.num_classes):
            name = class_names[c] if class_names is not None else (
                self.class_names[c] if self.class_names is not None else str(c)
            )

            pc = result["per_class"][c]
            ap = result["freq_ap_per_class"][c]
            ap_str = "nan" if np.isnan(ap) else f"{ap:.4f}"

            print(
                f"{name}: "
                f"IoU={pc['iou']:.4f}, "
                f"time_iou={pc['time_iou_mean']:.4f}, "
                f"freq_iou={pc['freq_iou_mean']:.4f}, "
                f"tf_iou={pc['tf_iou_mean']:.4f}, "
                f"time_acc={pc['time_acc']:.4f}, "
                f"freq_acc={pc['freq_acc']:.4f}, "
                f"tf_acc={pc['tf_acc']:.4f}, "
                f"freq_ap={ap_str}"
            )

        if result["PSDS"] is not None:
            print("\n[PSDS]")
            print(result["PSDS"])
