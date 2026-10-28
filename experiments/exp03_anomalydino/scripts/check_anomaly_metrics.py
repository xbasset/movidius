"""Compare the bounded-memory evaluator with sklearn and direct PRO thresholds."""

import numpy as np
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score, precision_recall_curve

from experiments.exp03_anomalydino.metrics import sorted_pixel_metrics


def check(scores, gold):
    weights = np.zeros_like(scores, dtype=np.float32)
    regions = []
    for i, target in enumerate(gold):
        components, count = label(target, np.ones((3, 3)))
        for component in range(1, count + 1):
            mask = components == component
            weights[i, mask] = 1. / mask.sum()
            regions.append((i, mask))
    result = sorted_pixel_metrics(scores.ravel().copy(), gold.ravel().astype(np.uint8), weights.ravel(), len(regions))
    np.testing.assert_allclose(result['pixel_auroc'], roc_auc_score(gold.ravel(), scores.ravel()), atol=1e-12)
    precision, recall, _ = precision_recall_curve(gold.ravel(), scores.ravel())
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(recall), where=precision+recall != 0)
    np.testing.assert_allclose(result['pixel_f1_max'], f1.max(), atol=1e-12)
    xs, ys = [0.], [0.]
    for threshold in np.unique(scores)[::-1]:
        prediction = scores >= threshold
        xs.append(np.count_nonzero(prediction & ~gold) / np.count_nonzero(~gold))
        ys.append(np.mean([prediction[i][mask].mean() for i, mask in regions]))
    xs, ys = np.array(xs), np.array(ys)
    index = np.searchsorted(xs, .3, side='right')
    x, y = xs[:index], ys[:index]
    if x[-1] < .3:
        y = np.r_[y, ys[index-1] + (ys[index]-ys[index-1]) * (.3-xs[index-1])/(xs[index]-xs[index-1])]
        x = np.r_[x, .3]
    np.testing.assert_allclose(result['aupro_0_3'], np.trapezoid(y, x) / .3, atol=1e-7)


def main():
    rng = np.random.default_rng(42)
    gold = np.zeros((3, 12, 11), dtype=bool)
    gold[0, 1:4, 1:5] = True
    gold[1, 6:9, 7:10] = True
    for _ in range(5):
        check(rng.integers(0, 12, gold.shape).astype(np.float32), gold)
    # Ties spanning the million-element streaming boundary must merge once.
    large_gold = np.zeros((3, 1000, 1100), dtype=bool)
    large_gold[0, 100:400, 100:500] = True
    large_gold[1, 600:900, 700:1000] = True
    large_scores = rng.integers(0, 5, large_gold.shape).astype(np.float32)
    check(large_scores, large_gold)
    print('Pixel AUROC, F1-max and AUPRO match independent references, including ties and chunk boundaries.')


if __name__ == '__main__':
    main()
