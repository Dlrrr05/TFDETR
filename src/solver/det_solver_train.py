"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import json
import datetime

import torch 

from ..misc import dist_utils, profiler_utils

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):
    @staticmethod
    def _infer_num_classes(postprocessor, criterion):
        for module in (postprocessor, criterion):
            value = getattr(module, 'num_classes', None)
            if value is not None:
                return int(value)
        return 10

    @staticmethod
    def _to_scalar(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @classmethod
    def _primary_metric_value(cls, value):
        scalar = cls._to_scalar(value)
        if scalar is not None:
            return scalar

        if isinstance(value, (list, tuple)) and len(value) > 0:
            return cls._to_scalar(value[0])

        return None

    @classmethod
    def _log_test_metric(cls, writer, key, value, epoch):
        if writer is None or (not dist_utils.is_main_process()):
            return

        scalar = cls._to_scalar(value)
        if scalar is not None:
            writer.add_scalar(f'Test/{key}', scalar, epoch)
            return

        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                item_scalar = cls._to_scalar(item)
                if item_scalar is not None:
                    writer.add_scalar(f'Test/{key}_{i}', item_scalar, epoch)
    
    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        best_stat = {'epoch': -1, }

        start_time = time.time()
        start_epcoch = self.last_epoch + 1
        
        for epoch in range(start_epcoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
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
                writer=self.writer
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            
            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
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
                num_classes=self._infer_num_classes(self.postprocessor, self.criterion),
            )

            best_metric_improved = False
            for k, value in test_stats.items():
                self._log_test_metric(self.writer, k, value, epoch)

                metric_value = self._primary_metric_value(value)
                if metric_value is None:
                    continue

                if k in best_stat:
                    if metric_value > best_stat[k]:
                        best_stat[k] = metric_value
                        best_stat['epoch'] = epoch
                        best_metric_improved = True
                else:
                    best_stat[k] = metric_value
                    best_stat['epoch'] = epoch
                    best_metric_improved = True

            if best_metric_improved and self.output_dir:
                dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')

            print(f'best_stat: {best_stat}')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device,
                num_classes=self._infer_num_classes(self.postprocessor, self.criterion))
                
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
