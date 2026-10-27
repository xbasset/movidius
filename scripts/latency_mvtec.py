"""Matched single-head latency: ten deterministic images from each category."""

import argparse
import json
import time

import numpy as np
import torch

from movidius_lab.anomalydino import CATEGORIES, MASKED, MacPaper, MyriadPaper, cases
from movidius_lab.dino import ROOT, prepare_image


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--device', choices=['myriad', 'cpu', 'mps'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = MyriadPaper(trials=1) if args.device == 'myriad' else MacPaper(args.device)
    results = []
    for category in CATEGORIES:
        if args.device == 'myriad':
            model.calibrate(category)
        else:
            model.calibrate(category, trials=1)
        rows = cases(category)
        selected = [rows[i] for i in np.linspace(0, len(rows)-1, 10, dtype=int)]
        prepared = [(row, prepare_image(row['path'], 448)[0]) for row in selected]
        for _ in range(3):
            if args.device == 'myriad':
                model.infer(prepared[0][1], category in MASKED)
            else:
                model.infer(prepared[0][1], category in MASKED, trials=1)
        for row, pixels in prepared:
            start = time.perf_counter()
            output = (model.infer(pixels, category in MASKED) if args.device == 'myriad' else
                      model.infer(pixels, category in MASKED, trials=1))
            elapsed = (time.perf_counter() - start) * 1000
            if output.shape != (1, 1025) or not np.isfinite(output).all():
                raise ValueError('Invalid single-head output')
            results.append({'id': row['id'], 'ms': elapsed, 'score': float(output[0, -1])})
        print(args.device, category, f'{np.median([r["ms"] for r in results[-10:]]):.2f} ms', flush=True)
        if args.device == 'myriad' and category == 'capsule':
            model.model.profile(ROOT / 'results/dino448_myriad.tsv')
    times = [r['ms'] for r in results]
    result = {'device': args.device, 'resolution': [448, 448], 'threads': 4, 'shots': 1,
              'rotations': 8, 'samples': len(results), 'selection': 'ten evenly spaced sorted test IDs per category',
              'warmups_per_category': 3, 'timing': 'prepared host tensor to host patch map and score; calibration excluded',
              'pca': ('64 FP16 power iterations on MYRIAD, also computed when mask disabled' if args.device == 'myriad' else
                      'author sklearn PCA on CPU for five categories; other inference on selected backend'),
              'median_ms': float(np.median(times)), 'p95_ms': float(np.percentile(times, 95)), 'rows': results}
    (ROOT / f'results/dino448_{args.device}_latency.json').write_text(json.dumps(result, indent=2) + '\n')
    if hasattr(model, 'close'):
        model.close()


if __name__ == '__main__':
    main()
