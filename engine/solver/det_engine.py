"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
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
from ..misc.metrics import evaluate_detection_f1


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
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

    cur_iters = epoch * len(data_loader)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace('module.', '')
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

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

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    # For center-based F1 metric
    all_predictions = []
    all_ground_truths = []

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # Collect predictions and ground truths for F1 metric
        for target, output in zip(targets, results):
            image_id = target["image_id"].item()

            # Add predictions
            if len(output["boxes"]) > 0:
                boxes = output["boxes"].float().cpu().numpy()
                # Convert from [x1, y1, x2, y2] to [x, y, w, h]
                boxes_xywh = boxes.copy()
                boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
                boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height

                scores = output["scores"].float().cpu().numpy()
                labels = output["labels"].float().cpu().numpy()

                for box, score, label in zip(boxes_xywh, scores, labels):
                    all_predictions.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": box.tolist(),
                        "score": float(score)
                    })

            # Add ground truths
            gt_boxes = target["boxes"].float().cpu().numpy()
            if len(gt_boxes) > 0:
                # Ground truth boxes are in [x1, y1, x2, y2] format in resized image space (e.g., 640x640)
                # Predictions are already scaled to original image size by postprocessor
                # So we need to scale GT boxes back to original size for matching
                orig_h, orig_w = target["orig_size"].cpu().numpy()

                # Get the current (resized) image size from target or samples
                if "size" in target:
                    resize_h, resize_w = target["size"].cpu().numpy()
                else:
                    # Fallback: get size from samples tensor (C, H, W)
                    resize_h, resize_w = samples.shape[-2:]

                # Scale factor to go from resized to original
                scale_x = orig_w / resize_w
                scale_y = orig_h / resize_h

                # Scale boxes to original size
                gt_boxes_scaled = gt_boxes.copy()
                gt_boxes_scaled[:, [0, 2]] *= scale_x  # x1, x2
                gt_boxes_scaled[:, [1, 3]] *= scale_y  # y1, y2

                # Convert from [x1, y1, x2, y2] to [x, y, w, h]
                gt_boxes_xywh = gt_boxes_scaled.copy()
                gt_boxes_xywh[:, 2] = gt_boxes_scaled[:, 2] - gt_boxes_scaled[:, 0]  # width
                gt_boxes_xywh[:, 3] = gt_boxes_scaled[:, 3] - gt_boxes_scaled[:, 1]  # height

                gt_labels = target["labels"].float().cpu().numpy()
                for box, label in zip(gt_boxes_xywh, gt_labels):
                    all_ground_truths.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": box.tolist()
                    })

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    # Compute center-based F1 metrics
    if all_predictions and all_ground_truths:
        # Get class names from coco_evaluator
        try:
            class_names = {cat['id']: cat['name'] for cat in coco_evaluator.coco_gt.dataset['categories']}
        except:
            class_names = None

        f1_metrics = evaluate_detection_f1(
            predictions=all_predictions,
            ground_truths=all_ground_truths,
            center_threshold=50.0,
            score_threshold=0.3,
            class_names=class_names
        )
        stats['f1_metrics'] = f1_metrics
        print(f"\nCenter-based F1 Metrics (threshold=50px):")
        print(f"  Overall F1: {f1_metrics['overall']['f1_score']:.4f}")
        print(f"  Precision: {f1_metrics['overall']['precision']:.4f}")
        print(f"  Recall: {f1_metrics['overall']['recall']:.4f}")

    return stats, coco_evaluator
