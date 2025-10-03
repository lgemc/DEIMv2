"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
-------------------------------------------------------------------------
Detection metrics utilities including F1 score calculation and W&B logging
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np

try:
    import wandb
except ImportError:
    wandb = None


def calculate_box_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Calculate center of a bounding box in COCO format [x, y, width, height].

    Args:
        bbox: Bounding box as [x, y, width, height]

    Returns:
        Tuple of (center_x, center_y)
    """
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2)


def center_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Euclidean distance between centers of two boxes.

    Args:
        bbox1: First bounding box [x, y, width, height]
        bbox2: Second bounding box [x, y, width, height]

    Returns:
        Euclidean distance between centers
    """
    cx1, cy1 = calculate_box_center(bbox1)
    cx2, cy2 = calculate_box_center(bbox2)
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def match_detections(
    predictions: List[Dict],
    ground_truths: List[Dict],
    center_threshold: float = 50.0,
    score_threshold: float = 0.5
) -> Tuple[int, int, int, Dict]:
    """
    Match predictions to ground truths based on class and center proximity.

    Args:
        predictions: List of predictions with keys: image_id, category_id, bbox, score
        ground_truths: List of ground truths with keys: image_id, category_id, bbox
        center_threshold: Maximum center distance (in pixels) to consider a match
        score_threshold: Minimum confidence score for predictions

    Returns:
        Tuple of (true_positives, false_positives, false_negatives, per_class_stats)
    """
    # Filter predictions by score threshold
    predictions = [p for p in predictions if p.get('score', 1.0) >= score_threshold]

    # Group by image_id for efficient matching
    preds_by_image = defaultdict(list)
    gts_by_image = defaultdict(list)

    for pred in predictions:
        preds_by_image[pred['image_id']].append(pred)

    for gt in ground_truths:
        gts_by_image[gt['image_id']].append(gt)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    per_class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    # Get all unique image_ids
    all_image_ids = set(list(preds_by_image.keys()) + list(gts_by_image.keys()))

    for image_id in all_image_ids:
        img_preds = preds_by_image.get(image_id, [])
        img_gts = gts_by_image.get(image_id, [])

        matched_gts = set()

        # Sort predictions by score (highest first) for greedy matching
        img_preds = sorted(img_preds, key=lambda x: x.get('score', 1.0), reverse=True)

        for pred in img_preds:
            pred_class = pred['category_id']
            pred_bbox = pred['bbox']

            best_match_idx = None
            best_match_dist = center_threshold

            # Find best matching ground truth
            for idx, gt in enumerate(img_gts):
                if idx in matched_gts:
                    continue

                if gt['category_id'] != pred_class:
                    continue

                dist = center_distance(pred_bbox, gt['bbox'])

                if dist < best_match_dist:
                    best_match_dist = dist
                    best_match_idx = idx

            if best_match_idx is not None:
                # True positive
                true_positives += 1
                per_class_stats[pred_class]['tp'] += 1
                matched_gts.add(best_match_idx)
            else:
                # False positive
                false_positives += 1
                per_class_stats[pred_class]['fp'] += 1

        # Count unmatched ground truths as false negatives
        for idx, gt in enumerate(img_gts):
            if idx not in matched_gts:
                false_negatives += 1
                per_class_stats[gt['category_id']]['fn'] += 1

    return true_positives, false_positives, false_negatives, dict(per_class_stats)


def calculate_f1_score(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score.

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives

    Returns:
        Tuple of (precision, recall, f1_score)
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def evaluate_detection_f1(
    predictions: List[Dict],
    ground_truths: List[Dict],
    center_threshold: float = 50.0,
    score_threshold: float = 0.5,
    class_names: Dict[int, str] = None
) -> Dict:
    """
    Evaluate object detection predictions using F1 score with center-based matching.

    Args:
        predictions: List of predictions in COCO format
                    Each dict should have: image_id, category_id, bbox [x,y,w,h], score
        ground_truths: List of ground truths in COCO format
                      Each dict should have: image_id, category_id, bbox [x,y,w,h]
        center_threshold: Maximum center distance in pixels to consider a match
        score_threshold: Minimum confidence score for predictions
        class_names: Optional mapping of category_id to class names

    Returns:
        Dictionary with overall and per-class metrics
    """
    tp, fp, fn, per_class_stats = match_detections(
        predictions, ground_truths, center_threshold, score_threshold
    )

    overall_precision, overall_recall, overall_f1 = calculate_f1_score(tp, fp, fn)

    per_class_f1 = {}
    for class_id, stats in per_class_stats.items():
        precision, recall, f1 = calculate_f1_score(
            stats['tp'], stats['fp'], stats['fn']
        )
        class_label = class_names.get(class_id, f"class_{class_id}") if class_names else f"class_{class_id}"
        per_class_f1[class_label] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': stats['tp'],
            'fp': stats['fp'],
            'fn': stats['fn']
        }

    return {
        'overall': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        },
        'per_class': per_class_f1,
        'config': {
            'center_threshold': center_threshold,
            'score_threshold': score_threshold
        }
    }


class WandbLogger:
    """Weights & Biases logger for training metrics."""

    def __init__(self, project: Optional[str] = None, name: Optional[str] = None,
                 config: Optional[dict] = None, dir: Optional[str] = None):
        """
        Initialize W&B logger.

        Args:
            project: W&B project name
            name: Run name
            config: Configuration dictionary
            dir: Directory for W&B files
        """
        self.enabled = False
        if wandb is None:
            print("W&B not available. Install with: pip install wandb")
            return

        try:
            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                dir=dir
            )
            self.enabled = True
            print(f"W&B logging initialized: {wandb.run.url}")
        except Exception as e:
            print(f"Failed to initialize W&B: {e}")

    def log(self, metrics: dict, step: Optional[int] = None):
        """Log metrics to W&B."""
        if self.enabled:
            wandb.log(metrics, step=step)

    def finish(self):
        """Finish W&B run."""
        if self.enabled:
            wandb.finish()
