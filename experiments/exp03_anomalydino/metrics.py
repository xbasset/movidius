"""Paper metrics with exact score thresholds and bounded curve memory."""

import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter, label
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def image_metrics(gold, scores):
    precision, recall, _ = precision_recall_curve(gold, scores)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(recall), where=precision + recall != 0)
    return {"image_auroc": float(roc_auc_score(gold, scores)),
            "image_ap": float(average_precision_score(gold, scores)),
            "image_f1_max": float(f1.max())}


def pack_scores(scores):
    if len(scores) >= 2**32:
        raise ValueError("Packed pixel indices require fewer than 2**32 pixels per category")
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("Expected finite nonnegative anomaly scores")
    keys = np.asarray(scores, dtype=np.float32).view(np.uint32).astype(np.uint64)
    keys[np.asarray(scores) == 0] = 0
    keys <<= 32
    for start in range(0, len(keys), 1_000_000):
        end = min(start + 1_000_000, len(keys))
        keys[start:end] |= np.arange(start, end, dtype=np.uint64)
    return keys


def sorted_pixel_metrics(scores, target, region_weights, regions):
    return pixel_curves(pack_scores(scores), target, region_weights, regions)


def pixel_curves(keys, target, region_weights, regions):
    """Sort packed score/index pairs in place, then merge thresholds in chunks."""
    positives = int(target.sum())
    negatives = len(target) - positives
    if not positives or not negatives or not regions:
        raise ValueError("Both pixel classes and at least one anomaly region are required")
    keys.sort()
    keys = keys[::-1]
    fp_total, pro_total = 0, 0.
    last_fpr, last_tpr, last_pro = 0., 0., 0.
    auroc, ap, f1_max, aupro = 0., 0., 0., 0.
    for start in range(0, len(keys), 1_000_000):
        end = min(start + 1_000_000, len(keys))
        indices = (keys[start:end] & np.uint64(0xffffffff)).astype(np.uint32)
        score_bits = keys[start:end] >> 32
        fp = np.cumsum(1 - target[indices], dtype=np.uint32) + fp_total
        pro = np.cumsum(region_weights[indices], dtype=np.float64) + pro_total
        fp_total, pro_total = int(fp[-1]), float(pro[-1])
        keep = np.empty(end - start, dtype=bool)
        keep[:-1] = score_bits[:-1] != score_bits[1:]
        keep[-1] = end == len(keys) or score_bits[-1] != keys[end] >> np.uint64(32)
        positions = np.arange(start + 1, end + 1, dtype=np.uint32)[keep]
        tp = positions - fp[keep]
        x = np.r_[last_fpr, fp[keep].astype(np.float64) / negatives]
        y = np.r_[last_tpr, tp.astype(np.float64) / positives]
        p = np.r_[last_pro, np.minimum(pro[keep] / regions, 1.)]
        if len(x) == 1:
            continue
        auroc += float(np.trapezoid(y, x))
        ap += float(np.sum(np.diff(y) * tp / positions))
        f1_max = max(f1_max, float(np.max(2. * tp / (positions + positives))))
        left = x[:-1] < .3
        dx = np.minimum(x[1:], .3) - x[:-1]
        dy = np.divide(dx, np.diff(x), out=np.zeros_like(dx), where=np.diff(x) != 0)
        clipped_pro = p[:-1] + (p[1:] - p[:-1]) * dy
        aupro += float(np.sum((.5 * (p[:-1] + clipped_pro) * dx)[left]))
        last_fpr, last_tpr, last_pro = float(x[-1]), float(y[-1]), float(p[-1])
    return {"pixel_auroc": auroc, "pixel_f1_max": f1_max, "aupro_0_3": aupro / .3}


def pixel_metrics(maps, rows):
    width, height = Image.open(rows[0]["path"]).size
    count = len(rows) * height * width
    keys = np.empty(count, dtype=np.uint64)
    target = np.empty(count, dtype=np.uint8)
    weights = np.zeros(count, dtype=np.float32)
    regions = 0
    for i, (patch_map, row) in enumerate(zip(maps, rows)):
        offset = slice(i * height * width, (i + 1) * height * width)
        prediction = gaussian_filter(cv2.resize(patch_map, (width, height), interpolation=cv2.INTER_LINEAR), sigma=4)
        gold = np.asarray(Image.open(row["mask"])) > 0 if row["gold"] else np.zeros((height, width), dtype=bool)
        components, n = label(gold, structure=np.ones((3, 3), dtype=int))
        regions += n
        areas = np.bincount(components.ravel())
        inverse = np.zeros(n + 1, dtype=np.float32)
        inverse[1:] = 1. / areas[1:]
        keys[offset] = pack_scores(prediction.ravel()) + np.uint64(i * height * width)
        target[offset] = gold.ravel()
        weights[offset] = inverse[components].ravel()
    return pixel_curves(keys, target, weights, regions)
